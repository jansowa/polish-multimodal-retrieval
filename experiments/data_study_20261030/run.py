"""Session A runner: immutable data, exact P1.A training, shared gallery harness."""
import argparse
import gc
import io
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
from preflight import ROOT, REPORT, SERIES, STORAGE, read, rows
sys.path.insert(0,str(ROOT/'src'))
from common import COORD, artifact, atomic_json, gpu_lock, publish, reserve, sha256, verify_marker
import torch
from pl_mmb_training import source_ablation as historical
from pl_mmb_training.losses import PairLoss, PositivePairs
from pl_mmb_training.retrieval import graph_from_records, evaluate_graph
from pl_mmb_training.features import encode_cached, validate_features
from pl_mmb_training.commercial_backbones import load_backbone
from pl_mmb_benchmark.full_gallery import evaluate_encoder, paired_bootstrap
from guard import filter_verified

RUNS=STORAGE/'runs'
VN=ROOT/'training/runs/tinyclip-visualnews-v1'


def save_tensor(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as stream:torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)


def status(phase,**extra):
    atomic_json(COORD/'A_STATUS.json',dict(schema_version=1,study_id='data-ood-20261030',generation=1,
        owner='A',phase=phase,status='in_progress',pid=os.getpid(),**extra))


def committed():
    return sum(p.stat().st_size for root in (STORAGE,REPORT,Path(__file__).parent,ROOT/'experiments/data_ood_common_20261030')
               for p in root.rglob('*') if p.is_file())


def space(pending=256*1024**2):
    reserve('A','small-code-and-audit',{'C':pending,'D':8*1024**2},committed_bytes=committed())


class Data:
    def __init__(self):
        preparation_started=time.monotonic()
        universe=verify_marker(COORD/'A_UNIVERSE_FROZEN.json')
        self.full=list(rows(ROOT/'training/runs/clean-v1/FULL_CLEAN.jsonl'))
        self.old=list(rows(VN/'data/vn_train.jsonl'));self.dev=list(rows(VN/'data/vn_dev.jsonl'))
        self.new=list(rows(STORAGE/'new_VN_train.jsonl'));self.wiki=list(rows(STORAGE/'Wiki_train.jsonl'))
        self.all=self.full+self.old+self.new+self.wiki
        self.nbase=len(self.full)
        self.meta=read(VN/'native_features.json');old_identity=self.meta['identity']['data']
        native_path=VN/'native_features.pt';assert sha256(native_path)==self.meta['sha256']
        native=torch.load(native_path,weights_only=True,map_location='cpu')
        val=list(rows(ROOT/'training/runs/clean-v1/validation.jsonl'))
        candidate=list(rows(VN/'data/candidates.jsonl'))
        old_images=sorted({r['image'] for r in self.full+val})+[r['image'] for r in candidate]
        assert native.dtype==torch.float32 and native.shape==(len(old_images),512)
        self.image_values=old_images;self.native=native
        self.ii={p:i for i,p in enumerate(old_images)}
        base_cache=read(ROOT/'training/runs/e1-full/cache.json')
        tmeta=old_identity['base']['text_features']
        assert sha256(tmeta['path'])==tmeta['sha256']
        base_text=torch.load(tmeta['path'],weights_only=True,map_location='cpu')
        extra_meta=old_identity['text_features'];assert sha256(extra_meta['path'])==extra_meta['sha256']
        extra_identity=read(Path(extra_meta['path']).with_suffix('.json'))
        extra_text=torch.load(extra_meta['path'],weights_only=True,map_location='cpu')
        self.text_values=base_cache['texts']+extra_identity['values']
        self.texts=torch.cat([base_text,extra_text]);self.ti={t:i for i,t in enumerate(self.text_values)}
        assert self.texts.dtype==torch.float32 and self.texts.shape==(len(self.text_values),384)
        self.dev_graph=graph_from_records(self.dev,self.image_values,self.text_values)
        assert self.dev_graph['sha256']==old_identity['vn_dev_graph']
        self.base_config=read(ROOT/'training/runs/e1-full/config.json')
        self.model_lock=base_cache['model_lock']
        self.vision_lock=old_identity['base']['model']
        self.registration=read(SERIES/'reference_eval_registration_v2_20261029.json')
        self.protocols={}
        for d in self.registration['datasets']:
            p=ROOT/d['protocol_path'];assert sha256(p)==d['protocol_sha256']
            self.protocols[d['name']]=dict(read(p),prompt_mode='v2_no_instructions')
        self.identity=dict(universe_sha256=sha256(COORD/'A_UNIVERSE_FROZEN.json'),
            original_inputs=old_identity, native_sha256=self.meta['sha256'],
            text_prefix='[query]: ', vision_revision=self.vision_lock['revision'],
            text_revision=self.model_lock['text']['revision'])
        self.features()
        self.graph=graph_from_records(self.dev,self.image_values,self.text_values)
        self.dev_i=torch.tensor([self.ii[r['image']] for r in self.dev])
        self.dev_t=self.texts[[self.ti[r['text_pl']] for r in self.dev]].to('cuda')
        self.dev_x=self.native[self.dev_i].to('cuda')
        pairs=PositivePairs(self.dev)
        self.dev_batches=[(p,pairs.mask(p.tolist()).to('cuda')) for p in torch.arange(len(self.dev)).split(256)]
        # Preserve native gallery caches for all variants, including unselected R3.
        self.native_gpu=self.native.to('cuda');self.texts_gpu=self.texts.to('cuda')
        self.identity['feature_preparation_seconds_this_invocation']=time.monotonic()-preparation_started
        # Timing is operational metadata and must not alter run/cache identity.
        atomic_json(REPORT/'feature_preparation_timing.json',dict(seconds=self.identity.pop('feature_preparation_seconds_this_invocation')))

    def features(self):
        folder=STORAGE/'features';folder.mkdir(exist_ok=True)
        # Only existing historical gallery protocols; no B results or new datasets.
        missing_images={r['image']:r for r in self.new+self.wiki if r['image'] not in self.ii}
        historical_by_target={r['target_id']:self.ii[r['image']] for r in self.full+self.old+self.dev}
        for protocol in self.protocols.values():
            for p,uid in zip(protocol['images'],protocol['image_ids']):
                if p in self.ii:continue
                if uid in historical_by_target:self.ii[p]=historical_by_target[uid]
                else:missing_images.setdefault(p,dict(image=p))
        image_values=sorted(missing_images)
        image_path=folder/'native_new.pt';image_meta=folder/'native_new.json'
        identity=dict(values=image_values,universe=self.identity['universe_sha256'],vision=self.vision_lock,
                      precision='fp32-weights-bf16-autocast-cache-fp32',preprocessing='historical native TinyCLIP')
        if image_meta.exists():
            meta=read(image_meta);assert meta['identity']==identity and sha256(image_path)==meta['sha256']
            features=torch.load(image_path,weights_only=True,map_location='cpu')
        else:
            space(128*1024**2)
            model,prep,forward,stats=load_backbone('tinyclip-39m',self.vision_lock)
            assert stats['vision_parameters']==38587392
            model.to('cuda').eval().requires_grad_(False)
            features=torch.empty(len(image_values),512);index={p:i for i,p in enumerate(image_values)};done=set()
            def encode_batch(batch):
                from PIL import Image
                opened=[]
                try:
                    for p,raw in batch:
                        r=missing_images[p]
                        if r.get('sha256'):assert __import__('hashlib').sha256(raw).hexdigest()==r['sha256']
                        with Image.open(io.BytesIO(raw)) as im:opened.append(im.convert('RGB'))
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        result=forward(prep(opened).to('cuda')).float().cpu()
                    validate_features(batch,result)
                    for j,(p,_) in enumerate(batch):features[index[p]]=result[j];done.add(p)
                finally:
                    for im in opened:im.close()
            grouped=defaultdict(dict);plain=[]
            for p,r in missing_images.items():
                if r.get('shard'):grouped[r['shard']][r['shard_row']]=p
                else:plain.append(p)
            import pyarrow.parquet as pq
            for shard,mapping in sorted(grouped.items()):
                parquet=pq.ParquetFile(shard);offset=0;batch=[]
                for group in range(parquet.num_row_groups):
                    length=parquet.metadata.row_group(group).num_rows
                    needed=[i for i in mapping if offset<=i<offset+length]
                    if needed:
                        table=parquet.read_row_group(group,columns=['did','img_path','img'])
                        for i in sorted(needed):
                            a=table.slice(i-offset,1).to_pylist()[0];p=mapping[i];r=missing_images[p]
                            assert a['did']==r['image_did'] and a['img_path']==r['source_image']
                            batch.append((p,a['img']['bytes']))
                            if len(batch)==64:encode_batch(batch);batch=[]
                        del table
                    offset+=length
                if batch:encode_batch(batch)
                print('FEATURE_IMAGES',len(done),'/',len(image_values),flush=True)
            for start in range(0,len(plain),32):encode_batch([(p,Path(p).read_bytes()) for p in plain[start:start+32]])
            assert done==set(image_values)
            validate_features(image_values,features);save_tensor(image_path,features)
            atomic_json(image_meta,dict(identity=identity,sha256=sha256(image_path),stats=stats),immutable=True)
            del model;gc.collect();torch.cuda.empty_cache()
        validate_features(image_values,features)
        offset=len(self.native)
        self.native=torch.cat([self.native,features]);self.image_values+=image_values
        self.ii.update({p:offset+i for i,p in enumerate(image_values)})
        new_texts=sorted(({r['text_pl'] for r in self.new+self.wiki}|{t for p in self.protocols.values() for t in p['texts']})-set(self.ti))
        extra,meta=encode_cached(new_texts,'texts',self.base_config,self.model_lock,folder/'text_cache')
        offset=len(self.text_values);self.text_values+=new_texts;self.texts=torch.cat([self.texts,extra])
        self.ti.update({t:offset+i for i,t in enumerate(new_texts)})
        self.identity['new_native_sha256']=sha256(image_path);self.identity['new_text_cache']=meta
        atomic_json(REPORT/'feature_identity.json',self.identity)

    def retrieve(self,model):
        with torch.inference_mode():
            pred=model(self.native[self.graph['image_indices']].to('cuda')).cpu()
        local=dict(self.graph,image_indices=list(range(len(pred))),text_indices=list(range(len(self.graph['texts']))))
        return evaluate_graph(local,pred,self.texts[self.graph['text_indices']],include_ranks=True)

    def gallery(self,model,name):
        data=self
        class Encoder:
            def encode(self,values,images=False):
                if images:
                    with torch.inference_mode():return model(data.native[[data.ii[p] for p in values]].to('cuda')).cpu()
                return data.texts[[data.ti[t] for t in values]]
            def encode_text_candidates(self,values):return self.encode(values)
        return evaluate_encoder(Encoder(),self.protocols[name],self.registration)


class Sampler:
    def __init__(self,nbase,news,wiki,seed,share):
        self.sampler=historical.Sampler(dict(mode='replace',base_pool=list(range(nbase))),nbase,news,nbase,seed=seed)
        self.wiki=historical.Sampler(dict(mode='half_add',base_pool=[]),0,wiki,nbase+len(news),seed=seed+3000) if share else None
        self.share=share;self.total=0;self.wiki_total=0

    def draw(self,n):
        indices=self.sampler.draw(n)
        number=((self.total+n)*self.share)//100-self.wiki_total
        if number:indices[-number:]=self.wiki.news_rows(number)
        self.total+=n;self.wiki_total+=number
        assert self.wiki_total==self.total*self.share//100
        return indices


def train(data,variant,seed):
    folder=RUNS/variant/f'seed-{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():
        result=read(folder/'result.json');assert result['inputs']==data.identity
        return result
    news=data.old+data.new;wiki=data.wiki if variant in ('R2','R3') else []
    records=data.full+news+wiki;share={'R1':0,'R2':3,'R3':10}[variant]
    torch.manual_seed(seed);model=torch.nn.Linear(512,384).to('cuda')
    criterion=PairLoss('siglip_sigmoid',.07,-10.).to('cuda')
    optimizer=torch.optim.AdamW([dict(name='projector',params=list(model.parameters()),weight_decay=.01),
        dict(name='loss',params=list(criterion.parameters()),weight_decay=0.)],lr=.001)
    sampler=Sampler(data.nbase,news,wiki,seed,share);positive=PositivePairs(records)
    images=torch.tensor([data.ii[r['image']] for r in records],device='cuda')
    texts=torch.tensor([data.ti[r['text_pl']] for r in records],device='cuda')
    presentations=Counter();history=[];best=None;started=time.monotonic()
    config=dict(inputs=data.identity,variant=variant,seed=seed,Wiki_share=share,steps=5500,
        presentations=1405780,batch_size=256,window=275,final_batch=145,selection=read(REPORT/'registration.json')['checkpoint_selection'],
        initialization='fresh torch seed Linear512to384 and SigLIP loss',FP32_head_and_loss=True)
    atomic_json(folder/'config.json',config,immutable=True)
    def monitor(step):
        nonlocal best
        model.eval()
        with torch.inference_mode():
            metrics=data.retrieve(model);metrics.pop('ranks')
            loss=sum(float(criterion(model(data.dev_x[p]),data.dev_t[p],mask))*len(p) for p,mask in data.dev_batches)/len(data.dev)
        row=dict(updates=step,VN_dev=metrics,validation_loss=loss,presentations=dict(presentations),elapsed_seconds=time.monotonic()-started)
        history.append(row)
        with (folder/'history.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        if best is None or historical.key(row)>historical.key(best):
            best=row
            save_tensor(folder/'best.pt',dict(projector=model.state_dict(),loss=criterion.state_dict(),updates=step,
                seed=seed,variant=variant,trained_on_Wiki=bool(wiki),inputs=data.identity))
        model.train()
    monitor(0)
    for step in range(5500):
        n=145 if step%275==274 else 256;indices=sampler.draw(n);gpu=indices.to('cuda')
        optimizer.zero_grad(set_to_none=True)
        loss=criterion(model(data.native_gpu[images[gpu]]),data.texts_gpu[texts[gpu]],positive.mask(indices.tolist()).to('cuda'))
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for g in optimizer.param_groups for p in g['params']):raise RuntimeError('Nonfinite gradient')
        optimizer.step();criterion.constrain()
        full=n//2;wk=(indices>=data.nbase+len(news)).sum().item()
        presentations['FULL']+=full;presentations['VN']+=n-full-wk;presentations['Wiki']+=wk
        if step%275==274:
            monitor(step+1)
            print('TRAIN',variant,seed,step+1,history[-1]['VN_dev']['t2i']['recall@1'],flush=True)
    assert sum(presentations.values())==1405780 and presentations['FULL']==702880
    assert presentations['Wiki']==1405780*share//100
    result=dict(config,best_updates=best['updates'],selected_VN_dev=best['VN_dev'],best_validation_loss=best['validation_loss'],
        checkpoint_path=str(folder/'best.pt'),checkpoint_sha256=sha256(folder/'best.pt'),actual_presentations=dict(presentations),
        history=history,elapsed_seconds=time.monotonic()-started)
    atomic_json(folder/'result.json',result,immutable=True)
    return result


def reload(data,result):
    path=Path(result['checkpoint_path']);assert sha256(path)==result['checkpoint_sha256']
    saved=torch.load(path,weights_only=True,map_location='cpu')
    selected=max(result['history'],key=historical.key)
    assert saved['updates']==result['best_updates']==selected['updates']
    model=torch.nn.Linear(512,384).to('cuda').eval();model.load_state_dict(saved['projector']);model.requires_grad_(False)
    actual=data.retrieve(model);actual.pop('ranks');assert actual==result['selected_VN_dev'],'Reloaded VN metrics differ'
    criterion=PairLoss('siglip_sigmoid',.07,-10.).to('cuda');criterion.load_state_dict(saved['loss'])
    with torch.inference_mode():
        loss=sum(float(criterion(model(data.dev_x[p]),data.dev_t[p],m))*len(p) for p,m in data.dev_batches)/len(data.dev)
    assert abs(loss-result['best_validation_loss'])<2e-5
    return model


def release():
    verify_marker(COORD/'A_UNIVERSE_FROZEN.json')
    # The complete audit index covers immutable components, including image-only aliases.
    # Recheck every admitted Wiki against that protected image/text index.
    wiki=list(rows(STORAGE/'Wiki_train.jsonl'));protected=list(rows(STORAGE/'protected_index.jsonl'))
    g=read(SERIES/'wiki_dev_guard_enforcement.json')
    protected+=[dict(id=uid) for uid in g['ids']]+[dict(text_pl=t) for t in g['texts']]
    protected+=[dict(parent_ids=[p]) for p in g['parent_ids']]+[dict(source_sha1_hex=h) for h in g['source_sha1_hex']]
    kept,rejected=filter_verified(wiki,protected)
    assert len(kept)==len(wiki) and not rejected and wiki
    import subprocess
    check=subprocess.run([sys.executable,str(Path(__file__).with_name('test_guard.py'))],capture_output=True,text=True)
    assert check.returncode==0,check.stderr
    proof=REPORT/('release_guard_tests_'+sha256(__file__)[:12]+'.json')
    if not proof.exists():
        atomic_json(proof,dict(status='passed',tests=check.stderr,all_Wiki_rows_checked=len(wiki),image_only_aliases_tested=True),immutable=True)
    target=REPORT/('Wiki_training_release_'+sha256(__file__)[:12]+'.json')
    value=dict(status='approved',user_approved=True,authorization=read(REPORT/'registration.json')['authorization'],
        train=artifact('Wiki_train',STORAGE/'Wiki_train.jsonl'),dev=artifact('Wiki_dev',SERIES/'wiki_v2/wiki_dev.jsonl'),
        guard=artifact('Wiki_guard',SERIES/'wiki_dev_guard_enforcement.json'),
        pre_wiki_baseline=artifact('pre_Wiki_baseline',SERIES/'wiki_dev_zero_shot_pre_wiki_training.json'),
        tests=artifact('release_tests',proof),loader=artifact('new_verified_loader',Path(__file__)),
        historical_raise_unchanged=True,universe_sha256=sha256(COORD/'A_UNIVERSE_FROZEN.json'))
    if target.exists():
        saved=read(target)
        for k in ('train','dev','guard','pre_wiki_baseline','loader'):
            assert saved[k]['sha256']==value[k]['sha256']
    else:atomic_json(target,value,immutable=True)


def validate_release():
    value=read(REPORT/('Wiki_training_release_'+sha256(__file__)[:12]+'.json'))
    assert value['status']=='approved' and value['user_approved']
    for key in ('train','dev','guard','pre_wiki_baseline','tests','loader'):
        item=value[key];assert sha256(item['path'])==item['sha256'],key
    assert value['universe_sha256']==sha256(COORD/'A_UNIVERSE_FROZEN.json')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    release();validate_release();space()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    status('waiting_for_GPU',waiting_for='canonical worker.lock')
    with gpu_lock(False):
        status('feature_preparation')
        data=Data()
        assert read(REPORT/'sampler_verification.json')['status']=='passed'
        # Exact R0 reload stops the comparison on any numerical/configuration mismatch.
        reg=read(SERIES/'vn_full_finalist_decision.json');r0={}
        for seed in (42,43,44):
            p=Path(reg['seed_checkpoints'][str(seed)]['path']).with_name('result.json')
            r0[seed]=read(p);reload(data,r0[seed])
        atomic_json(REPORT/'R0_reuse_verification.json',dict(status='passed',seeds=[42,43,44],
            reload_metrics_and_loss_exact=True,no_new_legacy_test=True,checkpoint_sha256={str(s):r0[s]['checkpoint_sha256'] for s in r0}))
        if args.prepare_only:return
        # Compare the cached path with the established adapter on original images
        # and texts before any new training score. Same fixed encoder revisions.
        from pl_mmb_benchmark.projector_adapter import ProjectorEncoder
        adapter=ProjectorEncoder(r0[44]['checkpoint_path'],batch_size=32)
        p=data.protocols['wiki_native_dev'];probe_images=p['images'][:8]
        probe_texts=data.text_values[:16];model=reload(data,r0[44])
        # Reproduce extraction batch membership and shape. Diagnostic evidence
        # proved shape-dependent BF16 rounding: matched batches are bit-exact.
        val=list(rows(ROOT/'training/runs/clean-v1/validation.jsonl'))
        candidates=list(rows(VN/'data/candidates.jsonl'))
        known={r['image'] for r in data.full+val+candidates}
        missing={r['image']:r for r in data.new+data.wiki if r['image'] not in known}
        by_target={r['target_id'] for r in data.full+data.old+data.dev}
        for protocol in data.protocols.values():
            for path,uid in zip(protocol['images'],protocol['image_ids']):
                if path not in known and uid not in by_target:missing.setdefault(path,dict(image=path))
        plain=[path for path,row in missing.items() if not row.get('shard')]
        checked=[];image_cos=1.;image_max_abs=0.
        for start in range(0,len(plain),32):
            batch=plain[start:start+32]
            if not set(batch)&set(probe_images):continue
            with torch.inference_mode():
                cached_images=model(data.native[[data.ii[s] for s in batch]].to('cuda')).cpu()
            actual_images=adapter.encode(batch,images=True)
            image_cos=min(image_cos,float(torch.nn.functional.cosine_similarity(cached_images,actual_images).min()))
            image_max_abs=max(image_max_abs,float((cached_images-actual_images).abs().max()))
            checked+=batch
        assert set(probe_images)<=set(checked)
        actual_text=adapter.encode(probe_texts)
        expected_text=data.texts[[data.ti[t] for t in probe_texts]]
        text_cos=float(torch.nn.functional.cosine_similarity(expected_text,actual_text).min())
        assert image_cos>.99999 and text_cos>.99999,'Encoder/cache parity failed; comparison stopped'
        assert image_max_abs==0.,'Matched-batch image parity must be bit-exact'
        assert adapter.components['text_prefix']=='[query]: '
        atomic_json(REPORT/'adapter_cache_parity.json',dict(status='passed',image_min_cosine=image_cos,
            text_min_cosine=text_cos,image_max_abs=image_max_abs,matched_extraction_batches=True,
            checked_images=len(checked),components=adapter.components,head_FP32=True))
        del adapter;gc.collect();torch.cuda.empty_cache()
        results={};galleries={}
        for variant in ('R1','R2','R3'):
            status('training',variant=variant,seed=42);space()
            result=train(data,variant,42);results[(variant,42)]=result;model=reload(data,result)
            score=data.gallery(model,'wiki_native_dev')
            galleries[(variant,42,'wiki_native_dev')]=score
            atomic_json(RUNS/variant/'seed-42'/'Wiki_dev.json',score)
        chosen=max(('R2','R3'),key=lambda v:(galleries[(v,42,'wiki_native_dev')]['directions']['t2i']['metrics']['recall@1'],
            -{'R2':3,'R3':10}[v],-results[(v,42)]['best_updates']))
        choice=dict(chosen_Wiki=chosen,selection_closed_before_seed_confirmation=True,
            seed42_scores={v:galleries[(v,42,'wiki_native_dev')]['directions']['t2i']['metrics']['recall@1'] for v in ('R2','R3')},
            VN_guardrail_seed42_pass=results[(chosen,42)]['selected_VN_dev']['t2i']['recall@1']>=results[('R1',42)]['selected_VN_dev']['t2i']['recall@1']-.02)
        atomic_json(REPORT/'Wiki_variant_choice.json',choice,immutable=True)
        for variant in ('R1',chosen):
            for seed in (43,44):
                status('training',variant=variant,seed=seed);space()
                result=train(data,variant,seed);reload(data,result);results[(variant,seed)]=result
        # All training/selection closed before posthoc galleries or any new B OOD.
        representatives={'R0':44}
        for variant in ('R1',chosen):
            values={s:results[(variant,s)]['selected_VN_dev']['t2i']['recall@1'] for s in (42,43,44)}
            median=statistics.median(values.values());representatives[variant]=min(s for s,x in values.items() if x==median)
        decision=dict(choice,representatives=representatives,all_training_closed=True,no_B_results_read=True,
            variant_mean_sd={v:dict(mean=statistics.mean(results[(v,s)]['selected_VN_dev']['t2i']['recall@1'] for s in (42,43,44)),
                sd=statistics.stdev(results[(v,s)]['selected_VN_dev']['t2i']['recall@1'] for s in (42,43,44))) for v in ('R1',chosen)})
        atomic_json(REPORT/'selection_decision.json',decision,immutable=True)
        status('posthoc_frozen_galleries')
        for variant in ('R0','R1',chosen):
            for seed in (42,43,44):
                result=r0[seed] if variant=='R0' else results[(variant,seed)];model=reload(data,result)
                for name in data.protocols:
                    score=data.gallery(model,name);galleries[(variant,seed,name)]=score
                    p=STORAGE/'evaluation'/variant/f'seed-{seed}'/(name+'.json')
                    atomic_json(p,dict(score,checkpoint_sha256=result['checkpoint_sha256'],
                        source_status='ID' if name=='VN_dev' or (name=='wiki_native_dev' and variant==chosen) else
                            ('source_OOD' if name=='wiki_native_dev' else 'COCO_translation_control')))
        # The unconfirmed Wiki variant keeps its seed42 posthoc galleries too.
        other=next(v for v in ('R2','R3') if v!=chosen)
        model=reload(data,results[(other,42)])
        for name in data.protocols:
            score=data.gallery(model,name);galleries[(other,42,name)]=score
            atomic_json(STORAGE/'evaluation'/other/'seed-42'/(name+'.json'),dict(score,
                checkpoint_sha256=results[(other,42)]['checkpoint_sha256'],
                source_status='ID' if name in ('VN_dev','wiki_native_dev') else 'COCO_translation_control'))
        summary=dict(status='complete',selection=decision,variants={})
        for variant in ('R0','R1',chosen):
            summary['variants'][variant]={}
            for name in data.protocols:
                summary['variants'][variant][name]={}
                for direction in ('t2i','i2t'):
                    summary['variants'][variant][name][direction]={}
                    for metric in galleries[(variant,42,name)]['directions'][direction]['metrics']:
                        values=[galleries[(variant,s,name)]['directions'][direction]['metrics'][metric] for s in (42,43,44)]
                        summary['variants'][variant][name][direction][metric]=dict(values=values,mean=statistics.mean(values),sd=statistics.stdev(values))
        summary['seed42_all_four_variants']={'R0':r0[42],**{v:results[(v,42)] for v in ('R1','R2','R3')}}
        summary['parameter_counts']=dict(V_plus_T=70470528,V=38587392,T=31883136,head_separate=196992)
        atomic_json(REPORT/'training_report.json',summary)
        models=[];representative_ids=[]
        for variant in ('R0','R1',chosen):
            manifest=STORAGE/'manifests'/(variant+'_train.jsonl');manifest.parent.mkdir(exist_ok=True)
            from audit import write_rows
            actual_rows=data.full+data.old+([] if variant=='R0' else data.new)+(data.wiki if variant==chosen else [])
            write_rows(manifest,actual_rows)
            for seed in (42,43,44):
                result=r0[seed] if variant=='R0' else results[(variant,seed)];item=artifact('checkpoint',result['checkpoint_path'])
                mid=f'{variant}_seed{seed}';representative=seed==representatives[variant]
                if representative:representative_ids.append(mid)
                models.append(dict(id=mid,variant=variant,seed=seed,path=item['path'],sha256=item['sha256'],
                    training_manifest_sha256=sha256(manifest),training_manifest_path=artifact('train',manifest)['path'],
                    trained_on_Wiki=variant==chosen,representative=representative,
                    encoder_revisions=dict(vision=data.vision_lock['revision'],text=data.model_lock['text']['revision']),text_prefix='[query]: '))
        publish('A_FINALISTS_FROZEN',[artifact('training_report',REPORT/'training_report.json'),artifact('selection_decision',REPORT/'selection_decision.json')],
            models=models,representative_model_ids=representative_ids,final_OOD_model_ids=representative_ids,
            universe_sha256=sha256(COORD/'A_UNIVERSE_FROZEN.json'))
        del data;gc.collect();torch.cuda.empty_cache()
        publish('A_GPU_DONE',[],A_FINALISTS_FROZEN_sha256=sha256(COORD/'A_FINALISTS_FROZEN.json'),further_A_cuda_planned=False)
    status('complete',further_A_cuda_planned=False)


if __name__=='__main__':main()
