"""Frozen-projector source ablations, fresh initialization, VN-only selection."""
import copy
import gc
import math
import os
import time
from collections import Counter,defaultdict
from pathlib import Path

import torch

from . import tinyclip_visualnews as vn
from . import tinyclip_adaptation as base
from . import tinyclip_optimization as opt
from .common import ROOT,digest,file_sha,read_json,write_json
from .losses import PairLoss,PositivePairs
from .retrieval import evaluate_graph

OUT=ROOT/'training/runs/tinyclip-source-ablation-v1'
SOURCES={'A':'aro_visual_attribution','R':'aro_visual_relation','S':'sugarcrepe','C':'mbeir_mscoco'}
STEPS=5500
WINDOW=275
SEED=42


class UnionFind:
    def __init__(self):self.parent={}
    def find(self,key):
        self.parent.setdefault(key,key)
        while self.parent[key]!=key:
            self.parent[key]=self.parent[self.parent[key]];key=self.parent[key]
        return key
    def join(self,keys):
        if not keys:return
        root=self.find(keys[0])
        for key in keys[1:]:self.parent[self.find(key)]=root


def no_coco_indices(rows,vg_rows,protected,external_non_coco=()):
    """Follow all official VG/Flickr/COCO links and observed file/pixel aliases."""
    uf=UnionFind();vg={str(r['image_id']):r for r in vg_rows};coco_nodes=set()
    for r in vg_rows:
        nodes=['vg:'+str(r['image_id'])]
        for field,prefix in [('flickr_id','flickr:'),('coco_id','coco:')]:
            if r.get(field) is not None:nodes.append(prefix+str(r[field]))
        uf.join(nodes);coco_nodes.update(k for k in nodes if k.startswith('coco:'))
    def add(r):
        nodes=['pixels:'+r.get('target_id',r.get('pixels_sha256','')),'file:'+r['sha256']]+r['parent_ids']
        uf.join(nodes);coco_nodes.update(k for k in nodes if k.startswith('coco:'))
        if r.get('dataset') in ('mbeir_mscoco','sugarcrepe'):coco_nodes.add(nodes[0])
    for r in rows+protected+list(external_non_coco):add(r)
    forbidden={uf.find(key) for key in coco_nodes}
    for r in external_non_coco:
        if uf.find('pixels:'+r['target_id']) in forbidden:
            raise ValueError('Known COCO alias in external non-COCO collection: '+r['id'])
    kept=[];reasons=Counter()
    for i,r in enumerate(rows):
        if r['dataset'] not in ('aro_visual_attribution','aro_visual_relation'):
            reasons['non_ARO_source']+=1;continue
        vg_ids=[p[3:] for p in r['parent_ids'] if p.startswith('vg:')]
        if not r['lineage_verified'] or not vg_ids or any(k not in vg for k in vg_ids):
            reasons['missing_official_VG_identity']+=1;continue
        if any(vg[k].get('coco_id') is not None for k in vg_ids):
            reasons['official_COCO_id']+=1;continue
        if uf.find('pixels:'+r['target_id']) in forbidden:
            reasons['transitive_COCO_Flickr_pixel_or_file_alias']+=1;continue
        kept.append(i)
    return kept,dict(reasons)


def counts(rows):
    return dict(rows=len(rows),images=len({r['target_id'] for r in rows}),
        image_text_pairs=len({(r['target_id'],r['text_pl']) for r in rows}),
        datasets=dict(Counter(r['dataset'] for r in rows)))


def variants(source_indices,strict_indices,nbase):
    full=list(range(nbase));strict=set(strict_indices)
    plans={'VN_only':dict(mode='half_add',base_pool=[])}
    for code in ('A','R'):
        plans['VN_nonCOCO_'+code]=dict(mode='half_add',base_pool=[i for i in source_indices[code] if i in strict])
    plans['VN_nonCOCO_AR']=dict(mode='half_add',base_pool=sorted(strict))
    for code in SOURCES:plans['VN_'+code]=dict(mode='half_add',base_pool=source_indices[code])
    plans['VN_FULL']=dict(mode='replace',base_pool=full)
    for code in SOURCES:
        removed=set(source_indices[code])
        plans['VN_FULL_drop_'+code]=dict(mode='replace',base_pool=[i for i in full if i not in removed])
    plans['FULL_only']=dict(mode='full_only',base_pool=full)
    return plans


class Sampler:
    """LOO replaces dropped original slots by VN, without changing retained slots."""
    def __init__(self,plan,nbase,vn_rows,vn_offset,seed=SEED):
        self.mode=plan['mode'];self.pool=torch.tensor(plan['base_pool'],dtype=torch.long)
        self.keep=torch.zeros(nbase,dtype=torch.bool);self.keep[self.pool]=True
        self.base=vn.Bag(nbase,seed)
        self.add=vn.Bag(len(self.pool),seed) if len(self.pool) else None
        grouped=defaultdict(list)
        for i,r in enumerate(vn_rows,vn_offset):grouped[r['target_id']].append(i)
        self.image_rows=[grouped[k] for k in sorted(grouped)]
        self.news=vn.Bag(len(self.image_rows),seed+1000)
        self.caption=torch.Generator().manual_seed(seed+2000)
    def news_rows(self,n):
        if not n:return torch.empty(0,dtype=torch.long)
        images=self.news.take(n)
        return torch.tensor([self.image_rows[j][int(torch.randint(len(self.image_rows[j]),(),generator=self.caption))]
            for j in images.tolist()],dtype=torch.long)
    def draw(self,n):
        if self.mode=='full_only':return self.base.take(n)
        if self.mode=='half_add':
            if self.add is None:return self.news_rows(n)
            return torch.cat([self.pool[self.add.take(n//2)],self.news_rows(n-n//2)])
        original=self.base.take(n//2);keep=self.keep[original]
        result=original.clone();result[~keep]=self.news_rows(int((~keep).sum()))
        return torch.cat([result,self.news_rows(n-n//2)])
    def state(self):
        return dict(base=self.base.state(),add=self.add.state() if self.add else None,
            news=self.news.state(),caption=self.caption.get_state())
    def restore(self,state):
        self.base.restore(state['base']);self.news.restore(state['news']);self.caption.set_state(state['caption'])
        if self.add:self.add.restore(state['add'])


class Data:
    def __init__(self):
        OUT.mkdir(parents=True,exist_ok=True)
        self.ctx=vn.Context();self.rows=self.ctx.base.train+self.ctx.vn;self.nbase=len(self.ctx.base.train)
        blacklist=read_json(ROOT/'training/runs/e1-full/blacklist.json')
        vg_path=Path.home()/'.cache/huggingface/pl_mmb_training/visual_genome_image_data.json'
        assert file_sha(vg_path)==blacklist['lineage_metadata']['visual_genome']
        protected=self.ctx.base.val+[dict(info,parent_ids=info['parent_ids']) for info in blacklist['images'].values()]
        strict,rejected=no_coco_indices(self.ctx.base.train,read_json(vg_path),protected,self.ctx.vn+self.ctx.dev)
        self.strict=set(strict)
        assert self.strict
        original=base.records(base.CLEAN/'NON_COCO_CLEANROOM.jsonl')
        assert {self.ctx.base.train[i]['id'] for i in strict}.issubset(r['id'] for r in original)
        self.source_indices={code:[i for i,r in enumerate(self.ctx.base.train) if r['dataset']==source]
            for code,source in SOURCES.items()}
        self.plans=variants(self.source_indices,strict,self.nbase)
        meta=read_json(vn.OUT/'native_features.json')
        assert meta['identity']['data']==self.ctx.identity
        assert file_sha(vn.OUT/'native_features.pt')==meta['sha256']
        self.native=torch.load(vn.OUT/'native_features.pt',weights_only=True,map_location='cpu')
        assert self.native.shape==(len(self.ctx.images),512)
        self.row_images=torch.tensor([self.ctx.ii[r['image']] for r in self.rows])
        self.row_texts=torch.tensor([self.ctx.ti[r['text_pl']] for r in self.rows])
        self.identity=dict(data=self.ctx.identity,native_features_sha256=meta['sha256'],
            VG_metadata_sha256=file_sha(vg_path),strict_base_indices_sha256=digest(strict),
            selection='VN_dev_t2i_r1_then_mean_r1_then_earliest',COCO_used_for_selection=False,
            wiki_used=False,encoders_frozen=True,projector='Linear512to384',seed=SEED)
        self.audit=dict(status='passed',strict_ARO=counts([self.rows[i] for i in strict]),strict_rejections=rejected,
            full=counts(self.ctx.base.train),VN=counts(self.ctx.vn),dev=counts(self.ctx.dev),
            original_NON_COCO_rows=len(original),lineage='All official VG-Flickr-COCO components plus observed file/pixel aliases',
            caveat='No known COCO ancestry in our supervised training. Does not prove absence of unidentified copies or COCO exposure during encoder pretraining.')
        write_json(OUT/'data_audit.json',self.audit)
        self.x=self.native[self.row_images].to('cuda')
        self.t=self.ctx.texts[self.row_texts].to('cuda')
        self.dev_i=torch.tensor([self.ctx.ii[r['image']] for r in self.ctx.dev])
        self.dev_t=self.ctx.texts[[self.ctx.ti[r['text_pl']] for r in self.ctx.dev]].to('cuda')
        self.dev_x=self.native[self.dev_i].to('cuda')
        self.dev_pairs=PositivePairs(self.ctx.dev)
        self.dev_batches=[(part,self.dev_pairs.mask(part.tolist()).to('cuda')) for part in torch.arange(len(self.ctx.dev)).split(256)]

    def eligible(self,plan):
        indices=sorted(plan['base_pool']+(list(range(self.nbase,len(self.rows))) if plan['mode']!='full_only' else []))
        return indices

    def retrieve(self,model,graph):
        with torch.inference_mode():
            pred=model(self.native[graph['image_indices']].to('cuda')).cpu()
        local=dict(graph,image_indices=list(range(len(pred))),text_indices=list(range(len(graph['texts']))))
        return evaluate_graph(local,pred,self.ctx.texts[graph['text_indices']],include_ranks=True)

    def verify(self):
        self.ctx.verify()
        assert file_sha(vn.OUT/'native_features.pt')==self.identity['native_features_sha256']


def key(row):
    score=row['VN_dev']
    return score['t2i']['recall@1'],score['mean_r1'],-row['updates']


def initialize():
    path=OUT/'random_initialization.pt'
    torch.manual_seed(SEED)
    model=base.projector(512,0);loss=PairLoss('siglip_sigmoid',.07,-10.)
    expected=dict(projector=model.state_dict(),loss=loss.state_dict(),seed=SEED,
        initialized_from='random_torch_seed_42_no_data_no_pretrained_projector')
    if path.exists():
        saved=torch.load(path,weights_only=True)
        for group in ('projector','loss'):
            for name,value in expected[group].items():assert torch.equal(saved[group][name],value)
    else:opt.atomic_save(expected,path)
    return path,expected


def train(data,name,plan,initial_path,initial):
    assert not (OUT/'evaluation_started.json').exists(),'No new training after COCO evaluation starts'
    folder=OUT/name;folder.mkdir(exist_ok=True)
    indices=data.eligible(plan);rows=[data.rows[i] for i in indices]
    no_coco=all(i>=data.nbase or i in data.strict for i in indices)
    config=dict(inputs=data.identity,experiment=name,plan=plan,train=counts(rows),
        train_indices_sha256=digest(indices),no_known_COCO_training=no_coco,
        initial_sha256=file_sha(initial_path),initialization=initial['initialized_from'],
        source={p:file_sha(ROOT/p) for p in ['src/pl_mmb_training/source_ablation.py','src/pl_mmb_training/tinyclip_visualnews.py',
            'src/pl_mmb_training/losses.py','src/pl_mmb_training/retrieval.py','training/source_ablation.py']},
        optimization=dict(steps=STEPS,batch_size=256,window=WINDOW,final_batch_per_window=145,
            learning_rate=.001,weight_decay=.01,scheduler='none',augmentation='none_native_frozen_features'),
        selection='VN_dev_t2i_r1_then_mean_r1_then_earliest',COCO_used_for_training=not no_coco,
        COCO_used_for_checkpoint_or_configuration_selection=False,wiki_used=False)
    run_id=digest(config);result_path=folder/'result.json'
    if result_path.exists():
        result=read_json(result_path)
        assert result['run_id']==run_id and result['checkpoint_sha256']==file_sha(folder/'best.pt')
        return result
    write_json(folder/'config.json',config)
    write_json(OUT/'status.json',dict(status='training',experiment=name,COCO_scored=False,pid=os.getpid()))
    model=base.projector(512,0).to('cuda');model.load_state_dict(initial['projector'])
    criterion=PairLoss('siglip_sigmoid',.07,-10.).to('cuda');criterion.load_state_dict(initial['loss'])
    optimizer=torch.optim.AdamW([dict(name='projector',params=list(model.parameters()),weight_decay=.01),
        dict(name='loss',params=list(criterion.parameters()),weight_decay=0.)],lr=.001)
    # Excluded annotations must not enter the multi-positive label graph.
    positives=PositivePairs(rows)
    mapping=torch.full((len(data.rows),),-1,dtype=torch.long);mapping[indices]=torch.arange(len(indices))
    sampler=Sampler(plan,data.nbase,data.ctx.vn,data.nbase)
    history=[];best=None;first=0;presentations=Counter();seen=set();elapsed_before=0.
    if (folder/'last.pt').exists():
        saved=torch.load(folder/'last.pt',weights_only=True,map_location='cpu');assert saved['run_id']==run_id
        model.load_state_dict(saved['projector']);criterion.load_state_dict(saved['loss']);optimizer.load_state_dict(saved['optimizer'])
        sampler.restore(saved['sampler']);history=saved['history'];best=saved['best'];first=saved['updates']
        presentations=Counter(saved['presentations']);seen=set(saved['seen']);elapsed_before=history[-1]['elapsed_seconds']
    (folder/'epochs.jsonl').write_text(''.join(__import__('json').dumps(r)+'\n' for r in history),encoding='utf-8')
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()

    def checkpoint(update):
        return dict(projector=model.state_dict(),loss=criterion.state_dict(),run_id=run_id,updates=update,
            initialization_sha256=config['initial_sha256'],no_known_COCO_training=no_coco)

    def monitor(update,total=0.,count=0,norms=None):
        nonlocal best
        model.eval()
        with torch.inference_mode():
            val_loss=sum(float(criterion(model(data.dev_x[p]),data.dev_t[p],mask))*len(p)
                for p,mask in data.dev_batches)/len(data.ctx.dev)
            metrics=data.retrieve(model,data.ctx.dev_graph);metrics.pop('ranks')
        row=dict(updates=update,budget_epoch=update/WINDOW,train_loss=total/count if count else None,
            validation_loss=val_loss,VN_dev=metrics,gradient_norm=norms,
            trainable_params=sum(p.numel() for p in model.parameters())+sum(p.numel() for p in criterion.parameters()),
            peak_vram_bytes=torch.cuda.max_memory_allocated(),presentations=dict(presentations),
            unique_train_images_seen=len(seen),elapsed_seconds=elapsed_before+time.monotonic()-started,**criterion.scalars())
        history.append(row)
        with (folder/'epochs.jsonl').open('a',encoding='utf-8') as stream:stream.write(__import__('json').dumps(row,allow_nan=False)+'\n')
        if best is None or key(row)>key(best):
            best=row;opt.atomic_save(checkpoint(update),folder/'best.pt')
        print(f'{name} update={update}/{STEPS} VNdev_t2i={metrics["t2i"]["recall@1"]:.5f} loss={val_loss:.4f}',flush=True)
        model.train()
    if not history:monitor(0)
    for step in range(first,STEPS):
        position=step%WINDOW
        if position==0:
            total=0.;count=0;grad_sum={g['name']:0. for g in optimizer.param_groups};grad_max=dict(grad_sum)
        size=145 if position==WINDOW-1 else 256
        global_indices=sampler.draw(size);local=mapping[global_indices]
        assert (local>=0).all()
        if no_coco:assert all(i>=data.nbase or i in data.strict for i in global_indices.tolist())
        optimizer.zero_grad(set_to_none=True)
        loss=criterion(model(data.x[global_indices.to('cuda')]),data.t[global_indices.to('cuda')],positives.mask(local.tolist()).to('cuda'))
        if not torch.isfinite(loss):raise RuntimeError('Non-finite train loss')
        loss.backward()
        for group in optimizer.param_groups:
            norm=float(opt.norm(group['params'],gradient=True))
            if not math.isfinite(norm):raise RuntimeError('Non-finite gradient')
            grad_sum[group['name']]+=norm;grad_max[group['name']]=max(grad_max[group['name']],norm)
        optimizer.step();criterion.constrain()
        total+=float(loss.detach())*size;count+=size
        for i in global_indices.tolist():
            presentations[data.rows[i]['dataset']]+=1;seen.add(data.rows[i]['target_id'])
        if position==WINDOW-1:
            monitor(step+1,total,count,{k:dict(mean=v/WINDOW,maximum=grad_max[k]) for k,v in grad_sum.items()})
            opt.atomic_save(dict(checkpoint(step+1),optimizer=optimizer.state_dict(),sampler=sampler.state(),history=history,
                best=best,presentations=dict(presentations),seen=sorted(seen)),folder/'last.pt')
    result=dict(experiment=name,run_id=run_id,config=config,history=history,train=counts(rows),
        best_updates=best['updates'],selected_VN_dev=best['VN_dev'],best_validation_loss=best['validation_loss'],
        checkpoint_path=str(folder/'best.pt'),checkpoint_sha256=file_sha(folder/'best.pt'),
        no_known_COCO_training=no_coco,COCO_used_for_selection=False,COCO_scored=False,
        wiki_used=False,xm3600_used=False,steps=STEPS,example_presentations=sum(presentations.values()))
    assert result['example_presentations']==1405780
    write_json(result_path,result)
    return result


def export(data,results):
    report=dict(protocol='source-ablation-v1',status='training' if len(results)<len(data.plans) else 'training_complete',
        selection='VN_dev_only',COCO_used_for_selection=False,COCO_scored=False,
        data_audit=data.audit,results=results)
    write_json(ROOT/'reports/tinyclip_source_ablation.json',report)
    lines=['# Ablacja źródeł: tylko projector, wybór po VN dev','',
        'Oba encodery frozen, Linear 512→384 od losowych wag. Świeży sigmoid loss. Seed 42.',
        '5500 kroków / 1 405 780 prezentacji na wariant. LR 0.001, AdamW, bez schedulera/augmentacji.',
        'COCO nie wybiera checkpointów ani konfiguracji. Dane wiki i retrieval XM3600 nieużywane.','',
        '| Wariant | Bez znanego COCO w treningu | Train pary | Train obrazy | VN dev t2i R@1 | VN dev i2t R@1 | Best krok |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in results:
        v=r['selected_VN_dev']
        lines.append(f'| {r["experiment"]} | {"tak" if r["no_known_COCO_training"] else "nie"} | {r["train"]["rows"]:,} | {r["train"]["images"]:,} | {100*v["t2i"]["recall@1"]:.2f}% | {100*v["i2t"]["recall@1"]:.2f}% | {r["best_updates"]} |')
    (ROOT/'reports/tinyclip_source_ablation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    with (vn.OUT/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        data=Data();initial_path,initial=initialize()
        plan=dict(inputs=data.identity,variants=data.plans,optimization=dict(steps=STEPS,window=WINDOW,learning_rate=.001,
            batch_size=256,initialization='random',seed=SEED),COCO_used_for_selection=False,
            eligible_finalists=['VN_only','VN_nonCOCO_AR','VN_FULL','best_strict_by_VN_dev'],
            no_automatic_seed_grid=True,wiki_used=False,source_meaning=SOURCES,
            leave_one_out='Replace removed slots with VN; preserve every retained original-source draw and total compute.')
        path=OUT/'plan.json'
        if path.exists():assert read_json(path)==plan
        else:write_json(path,plan)
        results=[]
        for name,spec in data.plans.items():
            results.append(train(data,name,spec,initial_path,initial));export(data,results)
            gc.collect();torch.cuda.empty_cache()
        strict=[r for r in results if r['no_known_COCO_training']]
        best=max(strict,key=lambda r:(r['selected_VN_dev']['t2i']['recall@1'],r['selected_VN_dev']['mean_r1'],-r['best_updates']))
        finalists=list(dict.fromkeys(['VN_only','VN_nonCOCO_AR','VN_FULL',best['experiment']]))
        registry=dict(status='all_training_closed_before_COCO_evaluation',best_strict_by_VN_dev=best['experiment'],
            finalists=finalists,checkpoints={r['experiment']:dict(sha256=r['checkpoint_sha256'],path=r['checkpoint_path'],
                updates=r['best_updates'],no_known_COCO_training=r['no_known_COCO_training']) for r in results},
            COCO_used_for_selection=False,wiki_used=False,xm3600_used=False)
        write_json(OUT/'selection_registry.json',registry);data.verify()
        write_json(OUT/'status.json',dict(status='training_complete',experiments=len(results),COCO_scored=False))


if __name__=='__main__':main()
