"""Repeat original Wiki variants on generation-2 data; preserve generation 1."""
import gc
import hashlib
import io
import statistics
import time
from collections import defaultdict
from config import *
import run as prior
import torch
from pl_mmb_training.features import encode_cached,validate_features
from pl_mmb_training.commercial_backbones import load_backbone
from pl_mmb_benchmark.projector_adapter import ProjectorEncoder
from common import verify_marker,gpu_lock
from pl_mmb_training.holdouts import require_training_release,verify_release_proof


def configure():
    prior.REPORT=REPORT;prior.STORAGE=STORE;prior.COORD=COORD;prior.RUNS=STORE/'runs'
    prior.status=status;prior.publish=publish;prior.space=lambda pending=None:reserve()
    original_read=prior.read
    def recipe_read(path):
        value=original_read(path)
        if Path(path)==REPORT/'registration.json':return dict(value['training_recipe'],**value)
        return value
    prior.read=recipe_read


class Data(prior.Data):
    def features(self):
        folder=STORE/'features';folder.mkdir(exist_ok=True)
        missing={r['image']:r for r in self.new+self.wiki if r['image'] not in self.ii}
        old_targets={r['target_id']:self.ii[r['image']] for r in self.full+self.old+self.dev}
        for protocol in self.protocols.values():
            for p,uid in zip(protocol['images'],protocol['image_ids']):
                if p in self.ii:continue
                if uid in old_targets:self.ii[p]=old_targets[uid]
                else:missing.setdefault(p,dict(image=p))
        values=sorted(missing);path=folder/'native_new.pt';meta_path=folder/'native_new.json'
        previous_meta=read(OLD_STORE/'features/native_new.json')
        previous_path=OLD_STORE/'features/native_new.pt'
        assert sha256(previous_path)==previous_meta['sha256']
        previous=torch.load(previous_path,weights_only=True,map_location='cpu')
        previous_index={v:i for i,v in enumerate(previous_meta['identity']['values'])}
        fresh=[v for v in values if v not in previous_index]
        identity=dict(values=values,universe=self.identity['universe_sha256'],vision=self.vision_lock,
            precision='fp32-weights-bf16-autocast-cache-fp32',reuse_native_sha256=previous_meta['sha256'],fresh_extraction_values=fresh)
        if meta_path.exists():
            meta=read(meta_path);assert meta['identity']==identity and sha256(path)==meta['sha256']
            native=torch.load(path,weights_only=True,map_location='cpu')
        else:
            reserve();native=torch.empty(len(values),512);index={v:i for i,v in enumerate(values)}
            for v in values:
                if v in previous_index:native[index[v]]=previous[previous_index[v]]
            if fresh:
                model,prep,forward,stats=load_backbone('tinyclip-39m',self.vision_lock)
                model.to('cuda').eval().requires_grad_(False)
                from PIL import Image
                for start in range(0,len(fresh),32):
                    batch=fresh[start:start+32];opened=[]
                    try:
                        for p in batch:
                            r=missing[p]
                            if r.get('shard'):raise RuntimeError('Expected fixed VN native cache is missing')
                            raw=Path(p).read_bytes()
                            expected=r.get('training_sha256') or r.get('sha256')
                            if expected:assert hashlib.sha256(raw).hexdigest()==expected
                            with Image.open(io.BytesIO(raw)) as im:opened.append(im.convert('RGB'))
                        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                            out=forward(prep(opened).to('cuda')).float().cpu()
                        validate_features(batch,out)
                        for j,v in enumerate(batch):native[index[v]]=out[j]
                    finally:
                        for im in opened:im.close()
                    if start%512==0:print('NEW_WIKI_NATIVE',start,len(fresh),flush=True)
                del model;gc.collect();torch.cuda.empty_cache()
            validate_features(values,native);prior.save_tensor(path,native)
            atomic_json(meta_path,dict(identity=identity,sha256=sha256(path)),immutable=True)
        validate_features(values,native)
        offset=len(self.native);self.native=torch.cat([self.native,native]);self.image_values+=values
        self.ii.update({p:offset+i for i,p in enumerate(values)})
        old_text_meta=read(OLD_REPORT/'feature_identity.json')['new_text_cache'];old_text_path=Path(old_text_meta['path'])
        assert sha256(old_text_path)==old_text_meta['sha256']
        old_text_identity=read(old_text_path.with_suffix('.json'))
        old_text=torch.load(old_text_path,weights_only=True,map_location='cpu')
        old_ti={v:i for i,v in enumerate(old_text_identity['values'])}
        needed=sorted(({r['text_pl'] for r in self.new+self.wiki}|{t for p in self.protocols.values() for t in p['texts']})-set(self.ti))
        fresh_texts=[v for v in needed if v not in old_ti]
        extra,extra_meta=encode_cached(fresh_texts,'texts',self.base_config,self.model_lock,folder/'text_cache')
        fresh_ti={v:i for i,v in enumerate(fresh_texts)}
        merged=torch.stack([old_text[old_ti[v]] if v in old_ti else extra[fresh_ti[v]] for v in needed])
        offset=len(self.text_values);self.texts=torch.cat([self.texts,merged]);self.text_values+=needed
        self.ti.update({v:offset+i for i,v in enumerate(needed)})
        self.identity.update(new_native_sha256=sha256(path),new_text_cache=extra_meta,
            reused_text_cache_sha256=old_text_meta['sha256'],generation=2)
        atomic_json(REPORT/'feature_identity.json',self.identity)


def parity(data,result):
    proof=read(OLD_REPORT/'adapter_cache_parity.json');assert proof['status']=='passed' and proof['image_max_abs']==0
    meta=read(STORE/'features/native_new.json');images=meta['identity']['fresh_extraction_values'][:32]
    assert len(images)==32
    model=prior.reload(data,result);adapter=ProjectorEncoder(result['checkpoint_path'],batch_size=32)
    with torch.inference_mode():cached=model(data.native[[data.ii[p] for p in images]].to('cuda')).cpu()
    actual=adapter.encode(images,images=True)
    assert torch.equal(cached,actual),'Matched-batch new Wiki feature cache differs from adapter'
    values=data.text_values[:16];texts=adapter.encode(values)
    assert torch.equal(texts,data.texts[[data.ti[t] for t in values]])
    atomic_json(REPORT/'adapter_cache_parity.json',dict(status='passed',image_max_abs=0,text_max_abs=0,
        checked_new_Wiki_images=32,generation_1_proof_sha256=sha256(OLD_REPORT/'adapter_cache_parity.json')))
    del adapter;gc.collect();torch.cuda.empty_cache()


def train_or_reuse(data,variant,seed):
    if variant=='R1':
        assert sha256(STORE/'new_VN_train.jsonl')==sha256(OLD_STORE/'new_VN_train.jsonl')
        result=read(OLD_STORE/'runs/R1'/f'seed-{seed}'/'result.json')
        prior.reload(data,result);return result
    folder=STORE/'runs'/variant/f'seed-{seed}'
    if folder.exists() and (folder/'config.json').exists() and not (folder/'result.json').exists():
        # Keep a failed attempt intact. Its deterministic replacement has the
        # identical seed/config; no result-based restart or checkpoint cherry-pick.
        target=folder.with_name(folder.name+'.interrupted-'+str(time.time_ns()))
        assert folder.resolve().is_relative_to((STORE/'runs').resolve())
        folder.rename(target)
    return prior.train(data,variant,seed)


def main():
    if (COORD/'A_GPU_DONE.json').exists():
        verify_marker(COORD/'A_FINALISTS_FROZEN.json');verify_marker(COORD/'A_GPU_DONE.json')
        print('GENERATION_2_ALREADY_CLOSED_NO_ADDITIONAL_CUDA',flush=True)
        return
    configure();verify_marker(COORD/'A_UNIVERSE_FROZEN.json');reserve()
    holdout_proof=require_training_release(STORE,REPORT)
    proof_path=REPORT/'global_holdout_training_release.json'
    if proof_path.exists():assert read(proof_path)==holdout_proof
    else:atomic_json(proof_path,holdout_proof,immutable=True)
    # Match old loader registration keys and guard tests without changing the
    # original registration or any generation-1 artifact.
    prior.release();prior.validate_release()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    status('waiting_for_GPU',GPU_planned=True)
    with gpu_lock(True):
        verify_release_proof(holdout_proof,STORE)
        status('preparing_generation_2_features',GPU_active=True)
        data=Data();reg=read(SERIES/'vn_full_finalist_decision.json');r0={};results={};galleries={}
        for seed in (42,43,44):
            r0[seed]=read(Path(reg['seed_checkpoints'][str(seed)]['path']).with_name('result.json'))
            prior.reload(data,r0[seed]);results[('R1',seed)]=train_or_reuse(data,'R1',seed)
        parity(data,r0[44])
        atomic_json(REPORT/'baseline_reuse.json',dict(status='passed',R0_seeds=[42,43,44],R1_seeds=[42,43,44],
            fixed_VN_manifest_sha256=sha256(STORE/'new_VN_train.jsonl'),reload_metrics_and_loss_exact=True,
            explanation='Only Wiki pool changed; R0/R1 rows, captions, native/text features, precision and samplers unchanged.'))
        for variant in ('R2','R3'):
            status('training',variant=variant,seed=42,GPU_active=True);reserve()
            result=train_or_reuse(data,variant,42);results[(variant,42)]=result
            model=prior.reload(data,result);score=data.gallery(model,'wiki_native_dev')
            galleries[(variant,42,'wiki_native_dev')]=score
            atomic_json(STORE/'runs'/variant/'seed-42/Wiki_dev.json',score)
        chosen=max(('R2','R3'),key=lambda v:(galleries[(v,42,'wiki_native_dev')]['directions']['t2i']['metrics']['recall@1'],
            -{'R2':3,'R3':10}[v],-results[(v,42)]['best_updates']))
        choice=dict(chosen_Wiki=chosen,seed42_scores={v:galleries[(v,42,'wiki_native_dev')]['directions']['t2i']['metrics']['recall@1'] for v in ('R2','R3')},
            selection_rule='Wiki t2i R1; tie smaller Wiki share then earlier saved update',no_new_B_results_read=True)
        if not (REPORT/'Wiki_variant_choice.json').exists():atomic_json(REPORT/'Wiki_variant_choice.json',choice,immutable=True)
        else:assert read(REPORT/'Wiki_variant_choice.json')==choice
        for seed in (43,44):
            status('training',variant=chosen,seed=seed,GPU_active=True);reserve()
            results[(chosen,seed)]=train_or_reuse(data,chosen,seed);prior.reload(data,results[(chosen,seed)])
        representatives={'R0':44}
        for v in ('R1',chosen):
            vals={s:results[(v,s)]['selected_VN_dev']['t2i']['recall@1'] for s in (42,43,44)}
            median=statistics.median(vals.values());representatives[v]=min(s for s,x in vals.items() if x==median)
        selection=dict(choice,representatives=representatives,all_training_closed=True,
            variant_mean_sd={v:dict(mean=statistics.mean(results[(v,s)]['selected_VN_dev']['t2i']['recall@1'] for s in (42,43,44)),
                sd=statistics.stdev(results[(v,s)]['selected_VN_dev']['t2i']['recall@1'] for s in (42,43,44))) for v in ('R1',chosen)})
        if not (REPORT/'selection_decision.json').exists():atomic_json(REPORT/'selection_decision.json',selection,immutable=True)
        else:assert read(REPORT/'selection_decision.json')==selection
        status('frozen_gallery_evaluation',GPU_active=True)
        evaluation_runs=[(v,s) for v in ('R0','R1',chosen) for s in (42,43,44)]+[(next(v for v in ('R2','R3') if v!=chosen),42)]
        for v,s in evaluation_runs:
            result=r0[s] if v=='R0' else results[(v,s)];model=prior.reload(data,result)
            for name in data.protocols:
                score=data.gallery(model,name);galleries[(v,s,name)]=score
                atomic_json(STORE/'evaluation'/v/f'seed-{s}'/(name+'.json'),dict(score,
                    checkpoint_sha256=result['checkpoint_sha256'],source_status='ID' if name=='VN_dev' or (name=='wiki_native_dev' and v in ('R2','R3')) else 'source_OOD' if name=='wiki_native_dev' else 'COCO_translation_control'))
        summary=dict(status='complete',generation=2,selection=selection,variants={},global_holdout_release=artifact('holdout_release',proof_path),
            seed42_all_four_variants={'R0':r0[42],**{v:results[(v,42)] for v in ('R1','R2','R3')}},
            parameter_counts=dict(V_plus_T=70470528,V=38587392,T=31883136,head_separate=196992))
        for v in ('R0','R1',chosen):
            summary['variants'][v]={}
            for name in data.protocols:
                summary['variants'][v][name]={d:{metric:dict(values=[galleries[(v,s,name)]['directions'][d]['metrics'][metric] for s in (42,43,44)],
                    mean=statistics.mean(galleries[(v,s,name)]['directions'][d]['metrics'][metric] for s in (42,43,44)),
                    sd=statistics.stdev(galleries[(v,s,name)]['directions'][d]['metrics'][metric] for s in (42,43,44)))
                    for metric in galleries[(v,42,name)]['directions'][d]['metrics']} for d in ('t2i','i2t')}
        atomic_json(REPORT/'training_report.json',summary)
        models=[];reps=[]
        for v in ('R0','R1',chosen):
            manifest=STORE/'manifests'/(v+'_train.jsonl')
            actual=data.full+data.old+([] if v=='R0' else data.new)+(data.wiki if v==chosen else [])
            if not manifest.exists():write_rows(manifest,actual)
            for s in (42,43,44):
                result=r0[s] if v=='R0' else results[(v,s)];a=artifact('checkpoint',result['checkpoint_path']);mid=f'{v}_seed{s}'
                rep=s==representatives[v]
                if rep:reps.append(mid)
                models.append(dict(id=mid,variant=v,seed=s,path=a['path'],sha256=a['sha256'],
                    training_manifest_path=artifact('train',manifest)['path'],training_manifest_sha256=sha256(manifest),
                    trained_on_Wiki=v==chosen,representative=rep,text_prefix='[query]: ',
                    encoder_revisions=dict(vision=data.vision_lock['revision'],text=data.model_lock['text']['revision'])))
        publish('A_FINALISTS_FROZEN',[artifact('training_report',REPORT/'training_report.json'),artifact('selection_decision',REPORT/'selection_decision.json')],
            models=models,representative_model_ids=reps,final_OOD_model_ids=reps,requires_generation_2_B_overlap_audit=True)
        del data;gc.collect();torch.cuda.empty_cache()
        publish('A_GPU_DONE',[],A_FINALISTS_FROZEN_sha256=sha256(COORD/'A_FINALISTS_FROZEN.json'),further_A_cuda_planned=False)
    print('GENERATION_2_TRAINING_COMPLETE',chosen,flush=True)


if __name__=='__main__':main()
