"""One-seed wiki dose pilot; frozen native encoders, VN-only checkpoint selection."""
import copy
import gc
import json
import math
import os
import time
from collections import defaultdict,Counter

import numpy as np
import torch
from . import wiki_pilot_data as wd
from . import source_ablation as source
from . import tinyclip_visualnews as vn
from . import tinyclip_adaptation as base
from . import tinyclip_optimization as opt
from .common import ROOT,digest,file_sha,read_json,write_json
from .losses import PairLoss,PositivePairs
from .retrieval import evaluate_graph,graph_from_records

OUT=wd.OUT
STEPS=5500
WINDOW=275
SEED=42


class UniformImages:
    def __init__(self,rows,offset,seed,caption_seed):
        groups=defaultdict(list)
        for i,r in enumerate(rows,offset):groups[r['target_id']].append(i)
        self.groups=[groups[k] for k in sorted(groups)]
        self.bag=vn.Bag(len(self.groups),seed)
        self.caption=torch.Generator().manual_seed(caption_seed)
    def take(self,n):
        if not n:return torch.empty(0,dtype=torch.long)
        return torch.tensor([self.groups[j][int(torch.randint(len(self.groups[j]),(),generator=self.caption))]
            for j in self.bag.take(n).tolist()],dtype=torch.long)
    def state(self):return dict(bag=self.bag.state(),caption=self.caption.get_state())
    def restore(self,state):self.bag.restore(state['bag']);self.caption.set_state(state['caption'])


class Sampler:
    """Wiki replaces a nested subset of the same baseline VN slots at fixed compute."""
    def __init__(self,news,wiki,percent):
        self.news=UniformImages(news,0,SEED+1000,SEED+2000)
        self.wiki=UniformImages(wiki,len(news),SEED+4000,SEED+5000)
        self.slots=torch.Generator().manual_seed(SEED+3000)
        self.percent=percent;self.presentations=0
    def draw(self,n):
        rows=self.news.take(n)
        slots=torch.randperm(n,generator=self.slots)
        wiki=(self.presentations+n)*self.percent//100-self.presentations*self.percent//100
        rows[slots[:wiki]]=self.wiki.take(wiki)
        self.presentations+=n
        return rows
    def state(self):return dict(news=self.news.state(),wiki=self.wiki.state(),slots=self.slots.get_state(),presentations=self.presentations)
    def restore(self,state):
        self.news.restore(state['news']);self.wiki.restore(state['wiki']);self.slots.set_state(state['slots']);self.presentations=state['presentations']


class Data:
    def __init__(self):
        self.ctx=vn.Context();self.audit=read_json(wd.FOLDER/'audit.json');self.features=read_json(wd.FOLDER/'features.json')
        assert self.audit['status']=='passed' and self.features['audit_sha256']==file_sha(wd.FOLDER/'audit.json')
        for name,sha in self.audit['files'].items():assert file_sha(wd.FOLDER/name)==sha
        assert file_sha(wd.FOLDER/'vision_features.pt')==self.features['vision_sha256']
        assert file_sha(self.features['text_meta']['path'])==self.features['text_meta']['sha256']
        native_meta=read_json(vn.OUT/'native_features.json')
        assert native_meta['identity']['data']==self.ctx.identity
        assert file_sha(vn.OUT/'native_features.pt')==native_meta['sha256']
        native=torch.load(vn.OUT/'native_features.pt',weights_only=True,map_location='cpu')
        wiki_x=torch.load(wd.FOLDER/'vision_features.pt',weights_only=True,map_location='cpu')
        wiki_t=torch.load(self.features['text_meta']['path'],weights_only=True,map_location='cpu')
        wt={s:i for i,s in enumerate(self.features['text_values'])}
        self.news=self.ctx.vn;self.wiki=base.records(wd.FOLDER/'wiki_train.jsonl');self.wiki_dev=base.records(wd.FOLDER/'wiki_dev.jsonl')
        self.rows=self.news+self.wiki;self.offset=len(self.news)
        self.x=torch.cat([native[[self.ctx.ii[r['image']] for r in self.news]],wiki_x[[r['cache_index'] for r in self.wiki]]]).to('cuda')
        self.t=torch.cat([self.ctx.texts[[self.ctx.ti[r['text_pl']] for r in self.news]],wiki_t[[wt[r['text_pl']] for r in self.wiki]]]).to('cuda')
        self.graphs={};self.gx={};self.gt={};self.loss_batches={}
        self.probe=self.make_probe()
        for split,rows in [('VN_dev',self.ctx.dev),('wiki_dev',self.wiki_dev),('wiki_train_probe',self.probe)]:
            image_values=sorted({r['image'] for r in rows});text_values=sorted({r['text_pl'] for r in rows})
            graph=graph_from_records(rows,image_values,text_values)
            by_image={r['image']:r for r in rows}
            if split=='VN_dev':
                x=native[[self.ctx.ii[p] for p in image_values]]
                t=self.ctx.texts[[self.ctx.ti[s] for s in text_values]]
            else:
                x=wiki_x[[by_image[p]['cache_index'] for p in image_values]]
                t=wiki_t[[wt[s] for s in text_values]]
            self.graphs[split]=graph;self.gx[split]=x.to('cuda');self.gt[split]=t
            im={p:i for i,p in enumerate(image_values)};tm={s:i for i,s in enumerate(text_values)}
            row_x=x[[im[r['image']] for r in rows]].to('cuda');row_t=t[[tm[r['text_pl']] for r in rows]].to('cuda')
            positive=PositivePairs(rows)
            self.loss_batches[split]=[(row_x[p],row_t[p],positive.mask(p.tolist()).to('cuda')) for p in torch.arange(len(rows)).split(256)]
        self.identity=dict(old_VN=self.ctx.identity,wiki_audit_sha256=file_sha(wd.FOLDER/'audit.json'),
            wiki_features_sha256=file_sha(wd.FOLDER/'features.json'),native_VN_sha256=native_meta['sha256'],
            graphs={k:v['sha256'] for k,v in self.graphs.items()},seed=SEED,selection='VN_dev_only',
            training_sources=['visualnews','wikimedia'],COCO_used_for_training=False,COCO_used_for_selection=False,
            COCO_scored=False,xm3600_scored=False)
        write_json(OUT/'inputs.json',self.identity)
    def make_probe(self):
        groups=defaultdict(list)
        for r in self.wiki:groups[r['target_id']].append(r)
        chosen=sorted(groups,key=lambda target:digest(['wiki-train-probe',42,target]))[:min(200,len(groups))]
        return [r for target in chosen for r in groups[target]]
    def retrieve(self,model,split,ranks=False):
        with torch.inference_mode():pred=model(self.gx[split]).cpu()
        return evaluate_graph(self.graphs[split],pred,self.gt[split],include_ranks=ranks)
    def losses(self,model,criterion,split):
        with torch.inference_mode():
            return sum(float(criterion(model(x),t,p))*len(x) for x,t,p in self.loss_batches[split])/sum(len(x) for x,_,_ in self.loss_batches[split])
    def verify(self):
        self.ctx.verify()
        assert file_sha(vn.OUT/'native_features.pt')==self.identity['native_VN_sha256']
        assert file_sha(wd.FOLDER/'vision_features.pt')==self.features['vision_sha256']
        assert file_sha(self.features['text_meta']['path'])==self.features['text_meta']['sha256']
        for name,sha in self.audit['files'].items():assert file_sha(wd.FOLDER/name)==sha
        assert file_sha(wd.FOLDER/'features.json')==self.identity['wiki_features_sha256']
        for path,sha in self.audit['protected_inputs'].items():assert file_sha(path)==sha


def selection_key(row):
    value=row['VN_dev']
    return value['t2i']['recall@1'],value['mean_r1'],-row['updates']


def initialize():
    torch.manual_seed(SEED)
    model=base.projector(512,0);criterion=PairLoss('siglip_sigmoid',.07,-10.)
    initial=dict(projector=model.state_dict(),loss=criterion.state_dict(),seed=42,initialization='fresh_random_no_data')
    old=torch.load(source.OUT/'random_initialization.pt',weights_only=True,map_location='cpu')
    for group in ('projector','loss'):
        for name,tensor in initial[group].items():assert torch.equal(tensor,old[group][name])
    path=OUT/'random_initialization.pt'
    if path.exists():
        saved=torch.load(path,weights_only=True,map_location='cpu')
        for group in ('projector','loss'):
            for name,tensor in initial[group].items():assert torch.equal(tensor,saved[group][name])
    else:opt.atomic_save(initial,path)
    return initial,path


def selected(model,criterion,data):
    return {split:dict(retrieval=data.retrieve(model,split,ranks=True),loss=data.losses(model,criterion,split))
        for split in ('VN_dev','wiki_dev','wiki_train_probe')}


def train(data,percent,initial,initial_path):
    name=f'WIKI_{percent:02d}pct';folder=OUT/name;folder.mkdir(exist_ok=True)
    sources=['src/pl_mmb_training/wiki_pilot.py','src/pl_mmb_training/wiki_pilot_data.py',
        'src/pl_mmb_training/losses.py','src/pl_mmb_training/retrieval.py','training/wiki_pilot.py']
    config=dict(experiment=name,wiki_percent=percent,inputs=data.identity,initial_sha256=file_sha(initial_path),
        seed=SEED,initialization='fresh_random_no_trained_projector',source={p:file_sha(ROOT/p) for p in sources},
        optimization=dict(steps=STEPS,window=WINDOW,batch_size=256,final_batch_per_window=145,
            learning_rate=.001,weight_decay=.01,scheduler='none',augmentation='none_native_features'),
        checkpoint_selection='VN_dev_t2i_R1_then_mean_then_earliest',wiki_metrics_used_for_checkpoint=False,
        COCO_scored=False,xm3600_scored=False)
    run_id=digest(config);result_path=folder/'result.json'
    if result_path.exists():
        result=read_json(result_path);assert result['run_id']==run_id and result['checkpoint_sha256']==file_sha(folder/'best.pt');return result
    write_json(folder/'config.json',config)
    write_json(OUT/'status.json',dict(status='training',experiment=name,wiki_percent=percent))
    model=base.projector(512,0).to('cuda');model.load_state_dict(initial['projector'])
    criterion=PairLoss('siglip_sigmoid',.07,-10.).to('cuda');criterion.load_state_dict(initial['loss'])
    optimizer=torch.optim.AdamW([dict(name='projector',params=list(model.parameters()),weight_decay=.01),
        dict(name='loss',params=list(criterion.parameters()),weight_decay=0.)],lr=.001)
    sampler=Sampler(data.news,data.wiki,percent)
    eligible=data.rows if percent else data.news
    positives=PositivePairs(eligible)
    history=[];best=None;first=0;presentations=Counter();seen=defaultdict(set);elapsed_before=0.
    if (folder/'last.pt').exists():
        saved=torch.load(folder/'last.pt',weights_only=True,map_location='cpu');assert saved['run_id']==run_id
        model.load_state_dict(saved['projector']);criterion.load_state_dict(saved['loss']);optimizer.load_state_dict(saved['optimizer'])
        sampler.restore(saved['sampler']);history=saved['history'];best=saved['best'];first=saved['updates']
        presentations=Counter(saved['presentations']);seen={k:set(v) for k,v in saved['seen'].items()};elapsed_before=history[-1]['elapsed_seconds']
        seen=defaultdict(set,seen)
    (folder/'epochs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history),encoding='utf-8')
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    def checkpoint(update):
        return dict(projector=model.state_dict(),loss=criterion.state_dict(),run_id=run_id,updates=update,
            wiki_percent=percent,initial_sha256=config['initial_sha256'])
    def monitor(update,total=0.,count=0,grad=None):
        nonlocal best
        model.eval()
        metrics={split:data.retrieve(model,split) for split in ('VN_dev','wiki_dev','wiki_train_probe')}
        row=dict(updates=update,budget_epoch=update/WINDOW,train_loss=total/count if count else None,
            **metrics,validation_loss=data.losses(model,criterion,'VN_dev'),wiki_dev_loss=data.losses(model,criterion,'wiki_dev'),
            gradient_norm=grad,peak_vram_bytes=torch.cuda.max_memory_allocated(),trainable_params=196994,
            presentations=dict(presentations),unique_images_seen={k:len(v) for k,v in seen.items()},
            elapsed_seconds=elapsed_before+time.monotonic()-started,**criterion.scalars())
        history.append(row)
        with (folder/'epochs.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        if best is None or selection_key(row)>selection_key(best):best=row;opt.atomic_save(checkpoint(update),folder/'best.pt')
        print(f'{name} update={update}/{STEPS} VN_R1={metrics["VN_dev"]["t2i"]["recall@1"]:.5f} WIKI_R1={metrics["wiki_dev"]["t2i"]["recall@1"]:.5f}',flush=True)
        model.train()
    if not history:monitor(0)
    for step in range(first,STEPS):
        pos=step%WINDOW
        if pos==0:
            total=0.;count=0;norm_sum={g['name']:0. for g in optimizer.param_groups};norm_max=dict(norm_sum)
        n=145 if pos==WINDOW-1 else 256
        rows=sampler.draw(n)
        assert (rows<len(eligible)).all()
        optimizer.zero_grad(set_to_none=True)
        loss=criterion(model(data.x[rows.to('cuda')]),data.t[rows.to('cuda')],positives.mask(rows.tolist()).to('cuda'))
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward()
        for group in optimizer.param_groups:
            norm=float(opt.norm(group['params'],gradient=True))
            if not math.isfinite(norm):raise RuntimeError('Nonfinite gradient')
            norm_sum[group['name']]+=norm;norm_max[group['name']]=max(norm_max[group['name']],norm)
        optimizer.step();criterion.constrain();total+=float(loss.detach())*n;count+=n
        for i in rows.tolist():
            r=data.rows[i];presentations[r['dataset']]+=1;seen[r['dataset']].add(r['target_id'])
        if pos==WINDOW-1:
            monitor(step+1,total,count,{k:dict(mean=v/WINDOW,maximum=norm_max[k]) for k,v in norm_sum.items()})
            opt.atomic_save(dict(checkpoint(step+1),optimizer=optimizer.state_dict(),sampler=sampler.state(),history=history,
                best=best,presentations=dict(presentations),seen={k:sorted(v) for k,v in seen.items()}),folder/'last.pt')
    final=selected(model.eval(),criterion,data)
    saved=torch.load(folder/'best.pt',weights_only=True,map_location='cpu')
    model.load_state_dict(saved['projector']);criterion.load_state_dict(saved['loss']);model.eval()
    best_metrics=selected(model,criterion,data)
    result=dict(experiment=name,run_id=run_id,config=config,wiki_percent=percent,history=history,best_updates=best['updates'],
        selected=best_metrics,final_fixed_budget=final,checkpoint_path=str(folder/'best.pt'),checkpoint_sha256=file_sha(folder/'best.pt'),
        steps=STEPS,example_presentations=sum(presentations.values()),presentations=dict(presentations),
        train_sources=source.counts(eligible),wiki_used_for_selection=False,COCO_used_for_selection=False,
        COCO_scored=False,xm3600_scored=False)
    assert result['example_presentations']==1405780
    assert presentations.get('wikimedia',0)==1405780*percent//100
    write_json(result_path,result);return result


def export(data,results):
    write_json(ROOT/'reports/tinyclip_wiki_pilot.json',dict(status='training' if len(results)<3 else 'training_complete',
        data_audit=data.audit,results=results,checkpoint_selection='VN_dev_only',COCO_scored=False,xm3600_scored=False))
    lines=['# Wiki pilot: 0%, 3%, 10% w batchu','',
        'Frozen TinyCLIP-39M i PolDense, losowy Linear 512→384, świeży sigmoid loss. Seed 42; 5500 kroków, LR .001.',
        'Checkpoint tylko po VN dev. Wiki dev i train probe są diagnostyczne. Brak oceny COCO i XM3600.','',
        '| Wariant | VN dev t2i R@1 | Wiki dev t2i R@1 | Wiki train probe R@1 | Best krok |',
        '|---|---:|---:|---:|---:|']
    for r in results:
        values=[100*r['selected'][s]['retrieval']['t2i']['recall@1'] for s in ('VN_dev','wiki_dev','wiki_train_probe')]
        lines.append(f'| {r["experiment"]} | '+ ' | '.join(f'{v:.2f}%' for v in values)+f' | {r["best_updates"]} |')
    (ROOT/'reports/tinyclip_wiki_pilot.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    from .series_holdout import require_wiki_training_release
    require_wiki_training_release()
    import fcntl
    with (vn.OUT/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        data=Data();initial,path=initialize()
        plan=dict(inputs=data.identity,wiki_percents=[0,3,10],steps=STEPS,seed=SEED,initial_sha256=file_sha(path),
            checkpoint_selection='VN_dev_only',wiki_dev_diagnostic=True,train_probe_diagnostic=True,
            fixed_final_step_diagnostic=True,COCO_scored=False,xm3600_scored=False,
            recommendation_rule='Prefer the lowest added wiki share with positive wiki-dev gain and <=0.5pp VN-dev point degradation; treat conditional intervals and one seed as a pilot, not definitive evidence.')
        if (OUT/'plan.json').exists():assert read_json(OUT/'plan.json')==plan
        else:write_json(OUT/'plan.json',plan)
        results=[]
        for percent in (0,3,10):
            results.append(train(data,percent,initial,path));export(data,results)
            gc.collect();torch.cuda.empty_cache()
        data.verify();write_json(OUT/'status.json',dict(status='training_complete',experiments=3,COCO_scored=False,xm3600_scored=False))


if __name__=='__main__':main()
