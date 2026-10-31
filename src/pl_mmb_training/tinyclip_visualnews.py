"""Matched data-volume experiments: frozen/last-two-block TinyCLIP, fixed compute."""
import copy
import gc
import json
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from . import tinyclip_adaptation as base
from . import tinyclip_generalization as gen
from . import tinyclip_optimization as opt
from . import visualnews_data as vd
from .common import ROOT, digest, file_sha, read_json, write_json
from .features import encode_cached
from .losses import PairLoss, PositivePairs
from .retrieval import evaluate_graph, graph_from_records

OUT = vd.OUT


def copy_suffix(vision, blocks, prefix_hook):
    """Deepcopy must not copy the reference tower's observation hook into the student."""
    suffix=base.Suffix(vision,'projected',blocks)
    for layer in suffix.layers:
        layer._forward_pre_hooks.pop(prefix_hook.id,None)
    return suffix


class Bag:
    """Uniform no-replacement draws, with independent reproducible cycles."""
    def __init__(self, size, seed):
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.empty(0, dtype=torch.long)
        self.cursor = 0

    def take(self, number):
        chunks = []
        while number:
            if self.cursor == len(self.order):
                self.order = torch.randperm(self.size, generator=self.generator)
                self.cursor = 0
            count = min(number, len(self.order)-self.cursor)
            chunks.append(self.order[self.cursor:self.cursor+count])
            self.cursor += count
            number -= count
        return torch.cat(chunks)

    def state(self):
        return dict(size=self.size, order=self.order, cursor=self.cursor, generator=self.generator.get_state())

    def restore(self, state):
        assert state['size'] == self.size
        self.order, self.cursor = state['order'], state['cursor']
        self.generator.set_state(state['generator'])


class Context:
    def __init__(self):
        self.base = base.Inputs(OUT)
        folder = OUT/'data'
        self.audit = read_json(folder/'audit.json')
        assert self.audit['status']=='passed'
        for name, sha in self.audit['files'].items():
            assert file_sha(folder/name)==sha
        self.extraction = read_json(folder/'extraction.json')
        assert file_sha(folder/'extraction.json')==self.audit['extraction_sha256']
        for name, sha in self.extraction['files'].items():
            assert file_sha(folder/name)==sha
        self.candidates = base.records(folder/'candidates.jsonl')
        self.vn = base.records(folder/'vn_train.jsonl')
        self.small = base.records(folder/'vn_10k.jsonl')
        self.dev = base.records(folder/'vn_dev.jsonl')
        self.offset = len(self.base.images)
        self.images = self.base.images+[r['image'] for r in self.candidates]
        self.ii = {p:i for i,p in enumerate(self.images)}
        assert len(self.ii)==len(self.images)
        old_manifest = read_json(ROOT/'training/runs/e1-full/cache.json')
        self.text_values = old_manifest['texts']
        old_set = set(self.text_values)
        new_values = sorted({r['text_pl'] for r in self.vn+self.dev}-old_set)
        config = read_json(ROOT/'training/runs/e1-full/config.json')
        extra, self.text_meta = encode_cached(new_values,'texts',config,old_manifest['model_lock'],OUT/'text_cache')
        self.text_values = self.text_values+new_values
        self.texts = torch.cat([self.base.texts,extra])
        self.ti = {s:i for i,s in enumerate(self.text_values)}
        self.graph = self.base.graph
        self.dev_graph = graph_from_records(self.dev,self.images,self.text_values)
        self.vi,self.vt,self.vp = self.base.tensors['validation']
        self.identity = dict(base=self.base.identity, audit_sha256=file_sha(folder/'audit.json'),
            extraction_sha256=self.audit['extraction_sha256'], text_features=self.text_meta,
            vn_dev_graph=self.dev_graph['sha256'], primary_selection='unchanged_COCO_val_t2i_r1_then_mean_then_earliest',
            vn_dev_used_for_checkpoint_selection=False, test_used=False, xm3600_scored=False)
        write_json(OUT/'context.json',self.identity)

    def verify(self):
        self.base.verify()
        assert file_sha(OUT/'data/audit.json')==self.identity['audit_sha256']
        for name,sha in self.audit['files'].items():
            assert file_sha(OUT/'data'/name)==sha


class Engine:
    def __init__(self, ctx):
        self.ctx=ctx
        self.vision,_,self.stats=base.load_vision(ctx.base)
        assert not any(p.requires_grad for p in self.vision.parameters())
        self.mean=torch.tensor(self.stats['preprocessing']['image_mean'],device='cuda')[None,:,None,None]
        self.std=torch.tensor(self.stats['preprocessing']['image_std'],device='cuda')[None,:,None,None]
        native_path=OUT/'native_features.pt'
        native_meta=OUT/'native_features.json'
        old_rgb_folder=gen.OUT/'pixels'
        old_meta=read_json(old_rgb_folder/'complete.json')
        assert old_meta['identity']['images']==ctx.base.images
        assert file_sha(old_rgb_folder/'native_rgb.npy')==old_meta['sha256']
        old_rgb=np.load(old_rgb_folder/'native_rgb.npy',mmap_mode='r')
        new_rgb=np.load(OUT/'data/native_rgb.npy',mmap_mode='r')
        # Allocate once; avoid a second complete GPU copy during concatenation.
        self.rgb=torch.empty((len(ctx.images),3,224,224),dtype=torch.uint8,device='cuda')
        for array,offset in [(old_rgb,0),(new_rgb,ctx.offset)]:
            for start in range(0,len(array),512):
                part=torch.from_numpy(np.array(array[start:start+512],copy=True))
                self.rgb[offset+start:offset+start+len(part)].copy_(part)
        cache_meta=read_json(base.OUT/'cache/complete.json')
        assert cache_meta['identity']['inputs']==ctx.base.identity
        for name,sha in cache_meta['files'].items():
            assert file_sha(base.OUT/'cache'/name)==sha
        old_native=torch.load(base.OUT/'cache/features.pt',weights_only=True)['projected']
        active=sorted({ctx.ii[r['image']] for r in ctx.vn+ctx.dev})
        identity=dict(data=ctx.identity, base_rgb=old_meta['sha256'],
            vn_rgb=ctx.extraction['files']['native_rgb.npy'], active_vn_indices=active,
            model=ctx.base.lock, base_native=cache_meta['files']['features.pt'])
        if native_meta.exists():
            meta=read_json(native_meta)
            assert meta['identity']==identity and file_sha(native_path)==meta['sha256']
            self.native=torch.load(native_path,weights_only=True).to('cuda')
        else:
            result=torch.zeros(len(ctx.images),512)
            result[:ctx.offset]=old_native
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                probe=self.vision(pixel_values=self.pixels(torch.arange(32))).image_embeds.float().cpu()
                assert float(F.cosine_similarity(probe,old_native[:32]).min())>.999
                for start in range(0,len(active),128):
                    part=torch.tensor(active[start:start+128])
                    result[part]=self.vision(pixel_values=self.pixels(part)).image_embeds.float().cpu()
                    if start%4096==0:
                        print(f'VN native features {start}/{len(active)}',flush=True)
            opt.atomic_save(result,native_path)
            write_json(native_meta,dict(identity=identity,sha256=file_sha(native_path)))
            self.native=result.to('cuda')
        self.prefix=np.load(base.OUT/'cache/prefix.npy',mmap_mode='r')
        self.captured=[]
        self.hook=self.vision.vision_model.encoder.layers[-2].register_forward_pre_hook(lambda module,args:self.captured.append(args[0].detach()))

    def pixels(self,indices):
        return (self.rgb[indices.to('cuda')].float()/255-self.mean)/self.std

    def forward(self,indices,suffix,aug_generator=None):
        pixels=self.pixels(indices)
        if aug_generator is not None:
            pixels=gen.augment(pixels,aug_generator)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            original=self.vision(pixel_values=pixels).image_embeds.float()
        tokens=self.captured.pop()
        assert not self.captured and not tokens.requires_grad
        if suffix is None:
            return original,original
        with torch.autocast('cuda',dtype=torch.bfloat16):
            features=suffix(tokens).float()
        return features,original

    def coco_features(self,suffix):
        if suffix is None:
            return self.native
        current=torch.zeros_like(self.native)
        for part in torch.unique(self.ctx.vi).split(128):
            tokens=torch.from_numpy(np.array(self.prefix[part.numpy()],copy=True)).to('cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                current[part.to('cuda')]=suffix(tokens).float()
        return current

    def retrieval(self,graph,current,model):
        pred=model(current[graph['image_indices']]).cpu()
        texts=self.ctx.texts[graph['text_indices']]
        local=dict(graph,image_indices=list(range(len(pred))),text_indices=list(range(len(texts))))
        return evaluate_graph(local,pred,texts,include_ranks=True)

    def selected_dev(self,suffix,model):
        with torch.inference_mode():
            current=self.native if suffix is None else torch.zeros_like(self.native)
            if suffix is not None:
                needed=torch.tensor(self.ctx.dev_graph['image_indices'])
                for part in needed.split(128):
                    features,_=self.forward(part,suffix)
                    current[part.to('cuda')]=features
            return self.retrieval(self.ctx.dev_graph,current,model)


def specs():
    common=dict(seed=42,steps=3300,window_steps=275,warmup_steps=275,
        projector_lr=1e-4,vision_lr=1e-5,cosine_floor=.1,weight_decay=.01,
        init='A1_epoch9',augmentation='zoom_1_to_1.08_translation_3pct',
        batch_size=256,full_clean_fraction=.5,anchor_lambda=50.)
    return {name:dict(common,volume=volume,blocks=blocks,anchor_lambda=50. if blocks else 0.)
        for volume in ('10k','all') for blocks in (0,2)
        for name in [f'VN_{volume}_'+('last2_anchor' if blocks else 'frozen')]}


def load_initial(spec):
    if spec['init']=='A1_epoch9':
        path=opt.WARM_START
        saved=torch.load(path,weights_only=True,map_location='cpu')
        return path,saved['projector'],saved['loss']
    path=base.CLEAN/'FULL_CLEAN/seed-42/checkpoints/epoch-02.pt'
    saved=torch.load(path,weights_only=True,map_location='cpu')
    return path,saved['state_dict'],saved['loss_state_dict']


def train(ctx,engine,name,spec):
    folder=OUT/name
    folder.mkdir(exist_ok=True)
    initial_path,initial_model,initial_loss=load_initial(spec)
    source={p:file_sha(ROOT/p) for p in ['src/pl_mmb_training/tinyclip_visualnews.py',
        'src/pl_mmb_training/visualnews_data.py','src/pl_mmb_training/tinyclip_generalization.py',
        'src/pl_mmb_training/tinyclip_adaptation.py','src/pl_mmb_training/tinyclip_optimization.py',
        'src/pl_mmb_training/losses.py','src/pl_mmb_training/retrieval.py']}
    identity=dict(data=ctx.identity,spec=spec,source=source,initial_sha256=file_sha(initial_path),
        primary_selection='existing_COCO_validation_only',vn_dev_used_for_selection=False,torch_version=torch.__version__)
    run_id=digest(identity)
    result_path=folder/'result.json'
    if result_path.exists():
        result=read_json(result_path)
        assert result['run_id']==run_id and file_sha(folder/'best.pt')==result['checkpoint_sha256']
        return result
    write_json(folder/'config.json',identity)
    write_json(OUT/'status.json',dict(status='training',experiment=name,pid=os.getpid()))
    torch.manual_seed(spec['seed'])
    model=base.projector(512,0).to('cuda')
    model.load_state_dict(initial_model)
    criterion=PairLoss('siglip_sigmoid').to('cuda')
    criterion.load_state_dict(initial_loss)
    fixed_criterion=copy.deepcopy(criterion).requires_grad_(False)
    suffix=copy_suffix(engine.vision,spec['blocks'],engine.hook).to('cuda') if spec['blocks'] else None
    groups=[dict(name='projector',params=list(model.parameters()),lr=spec['projector_lr'],weight_decay=.01),
        dict(name='loss',params=list(criterion.parameters()),lr=spec['projector_lr'],weight_decay=0.)]
    if suffix is not None:
        groups.append(dict(name='vision',params=[p for p in suffix.parameters() if p.requires_grad],lr=spec['vision_lr'],weight_decay=.01))
    optimizer=torch.optim.AdamW(groups)
    params=sum(p.numel() for g in optimizer.param_groups for p in g['params'])
    vn=ctx.small if spec['volume']=='10k' else ctx.vn
    rows=ctx.base.train+vn
    ni=len(ctx.base.train)
    train_i=torch.tensor([ctx.ii[r['image']] for r in rows])
    train_t=ctx.texts[[ctx.ti[r['text_pl']] for r in rows]].to('cuda')
    positives=PositivePairs(rows)
    image_rows=defaultdict(list)
    for i,r in enumerate(vn,ni):
        image_rows[r['target_id']].append(i)
    image_rows=[image_rows[k] for k in sorted(image_rows)]
    base_bag=Bag(ni,spec['seed'])
    vn_bag=Bag(len(image_rows),spec['seed']+1000)
    caption_gen=torch.Generator().manual_seed(spec['seed']+2000)
    aug_gen=torch.Generator().manual_seed(spec['seed']+10000)
    val_text=ctx.vt.to('cuda')
    val_batches=[(part,ctx.vp.mask(part.tolist()).to('cuda')) for part in torch.arange(len(ctx.vi)).split(256)]
    history,best,first,seen=[],None,0,set()
    elapsed_before=0.
    if (folder/'last.pt').exists():
        saved=torch.load(folder/'last.pt',weights_only=True,map_location='cpu')
        assert saved['run_id']==run_id
        model.load_state_dict(saved['projector']);criterion.load_state_dict(saved['loss'])
        if suffix is not None:suffix.load_state_dict(saved['vision_suffix'])
        optimizer.load_state_dict(saved['optimizer'])
        base_bag.restore(saved['base_bag']);vn_bag.restore(saved['vn_bag'])
        caption_gen.set_state(saved['caption_generator']);aug_gen.set_state(saved['augmentation_generator'])
        first=saved['updates'];history=saved['history'];best=saved['best'];seen=set(saved['seen_vn'])
        elapsed_before=history[-1]['elapsed_seconds']
    (folder/'epochs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history),encoding='utf-8')
    started=time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    def checkpoint(updates):
        return dict(projector=model.state_dict(),loss=criterion.state_dict(),vision_suffix=suffix.state_dict() if suffix is not None else None,
            run_id=run_id,spec=spec,updates=updates,epoch=updates/spec['window_steps'])

    def monitor(updates,totals,count,norms):
        nonlocal best
        model.eval()
        if suffix is not None:suffix.eval()
        with torch.inference_mode():
            current=engine.coco_features(suffix)
            val,fixed=0.,0.
            for part,mask in val_batches:
                pred=model(current[ctx.vi[part].to('cuda')])
                val+=float(criterion(pred,val_text[part],mask))*len(part)
                fixed+=float(fixed_criterion(pred,val_text[part],mask))*len(part)
            metrics=engine.retrieval(ctx.graph,current,model)
            metrics.pop('ranks')
            needed=torch.unique(ctx.vi).to('cuda')
            drift=float(gen.geometry_loss(current[needed],engine.native[needed]))
        row=dict(epoch=updates/spec['window_steps'],updates=updates,train_loss=totals['gold']/count if count else None,
            train_anchor_loss=totals['anchor']/count if count else None,train_total_loss=totals['total']/count if count else None,
            validation_loss=val/len(ctx.vi),validation_loss_fixed_scalars=fixed/len(ctx.vi),validation=metrics,
            validation_feature_cosine_drift=drift,gradient_norm=norms,peak_vram_bytes=torch.cuda.max_memory_allocated(),
            trainable_params=params,unique_vn_images_seen=len(seen),elapsed_seconds=elapsed_before+time.monotonic()-started,**criterion.scalars())
        history.append(row)
        with (folder/'epochs.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        print(f'{name} update={updates} t2i={metrics["t2i"]["recall@1"]:.5f} i2t={metrics["i2t"]["recall@1"]:.5f} val={row["validation_loss"]:.4f} drift={drift:.5f}',flush=True)
        if best is None or base.checkpoint_key(row)>base.checkpoint_key(best):
            best=row
            opt.atomic_save(checkpoint(updates),folder/'best.pt')
        model.train()
        if suffix is not None:suffix.train()

    if not history:monitor(0,dict(gold=0.,anchor=0.,total=0.),0,{})
    totals,count,norm_sums,norm_max=dict(gold=0.,anchor=0.,total=0.),0,{},{}
    for step in range(first,spec['steps']):
        position=step%spec['window_steps']
        if position==0:
            totals,count=dict(gold=0.,anchor=0.,total=0.),0
            norm_sums={g['name']:0. for g in optimizer.param_groups};norm_max=dict(norm_sums)
        number=len(ctx.base.train)%256 if position==spec['window_steps']-1 else 256
        nbase=number//2
        bi=base_bag.take(nbase)
        images=vn_bag.take(number-nbase)
        news=torch.tensor([image_rows[j][int(torch.randint(len(image_rows[j]),(),generator=caption_gen))] for j in images.tolist()])
        seen.update(images.tolist())
        part=torch.cat([bi,news])
        multiplier=opt.lr_factor(step,spec['steps'],spec['warmup_steps'],True,spec['cosine_floor'])
        for g in optimizer.param_groups:g['lr']=(spec['vision_lr'] if g['name']=='vision' else spec['projector_lr'])*multiplier
        optimizer.zero_grad(set_to_none=True)
        features,original=engine.forward(train_i[part],suffix,aug_gen)
        if step==0 and suffix is not None:
            assert float(F.cosine_similarity(features.detach(),original).min())>.9999
        anchor=gen.geometry_loss(features,original) if spec['anchor_lambda'] else features.new_zeros(())
        gold=criterion(model(features),train_t[part],positives.mask(part.tolist()).to('cuda'))
        loss=gold+spec['anchor_lambda']*anchor
        if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
        loss.backward()
        for g in optimizer.param_groups:
            norm=float(opt.norm(g['params'],gradient=True))
            if not math.isfinite(norm):raise RuntimeError('Non-finite gradient')
            norm_sums[g['name']]+=norm;norm_max[g['name']]=max(norm_max[g['name']],norm)
        optimizer.step();criterion.constrain()
        for k,v in [('gold',gold),('anchor',anchor),('total',loss)]:totals[k]+=float(v.detach())*number
        count+=number
        update=step+1
        if update%50==0:print(f'{name} update={update}/{spec["steps"]} gold={float(gold.detach()):.4f}',flush=True)
        within=position+1
        dense={math.ceil(spec['window_steps']*f) for f in (.25,.5,.75,1.)}
        if within==spec['window_steps'] or (step<550 and within in dense):
            monitor(update,totals,count,{k:dict(mean=v/within,maximum=norm_max[k]) for k,v in norm_sums.items()})
        if within==spec['window_steps']:
            opt.atomic_save(dict(checkpoint(update),optimizer=optimizer.state_dict(),history=history,best=best,
                base_bag=base_bag.state(),vn_bag=vn_bag.state(),caption_generator=caption_gen.get_state(),
                augmentation_generator=aug_gen.get_state(),seen_vn=sorted(seen)),folder/'last.pt')
    saved=torch.load(folder/'best.pt',weights_only=True,map_location='cpu')
    model.load_state_dict(saved['projector']);model.eval()
    if suffix is not None:suffix.load_state_dict(saved['vision_suffix']);suffix.eval()
    dev=engine.selected_dev(suffix,model)
    result=dict(experiment=name,run_id=run_id,spec=spec,history=history,best_epoch=best['epoch'],
        selected_validation=best['validation'],selected_vn_dev=dev,trainable_params=params,
        vn_train_images=len(image_rows),vn_train_rows=len(vn),checkpoint_path=str(folder/'best.pt'),
        checkpoint_sha256=file_sha(folder/'best.pt'),train_sampling='half original FULL_CLEAN rows / half uniform VN images; same source mixture for 10k and all',
        selection_uses_test=False,vn_dev_used_for_selection=False,test_mscoco=None,xm3600_used=False)
    write_json(result_path,result)
    return result


def reuse_zero(ctx,engine):
    results=[]
    for blocks,old_name in [(0,'G1_frozen_aug'),(2,'G3_last2_aug_anchor')]:
        old=read_json(gen.OUT/old_name/'result.json')
        config=read_json(gen.OUT/old_name/'config.json')
        assert config['inputs']==ctx.base.identity
        assert old['spec']['blocks']==blocks and old['spec']['augmentation']
        assert old['spec']['epochs']==12 and old['history'][-1]['updates']==3300
        assert old['spec']['projector_lr']==1e-4 and old['spec']['vision_lr']==1e-5
        assert old['spec']['anchor_lambda']==(50. if blocks else 0.)
        checkpoint=gen.OUT/old_name/'best.pt'
        assert file_sha(checkpoint)==old['checkpoint_sha256']
        saved=torch.load(checkpoint,weights_only=True,map_location='cpu')
        model=base.projector(512,0).to('cuda').eval();model.load_state_dict(saved['projector'])
        suffix=copy_suffix(engine.vision,2,engine.hook).to('cuda').eval() if blocks else None
        if suffix is not None:suffix.load_state_dict(saved['vision_suffix'])
        with torch.inference_mode():
            actual=engine.retrieval(ctx.graph,engine.coco_features(suffix),model)
        actual.pop('ranks')
        assert actual==old['selected_validation']
        dev=engine.selected_dev(suffix,model)
        result=dict(old,experiment='VN_0_'+('last2_anchor' if blocks else 'frozen'),
            reused_reference=True,source_experiment=old_name,selected_vn_dev=dev,vn_train_images=0,vn_train_rows=0,
            checkpoint_path=str(checkpoint),vn_dev_used_for_selection=False,
            reference_note='Original exact batch permutation and 3300 updates retained; mixed-data runs preserve batch sizes/compute but halve FULL_CLEAN exposures.')
        results.append(result)
    return results


def export(ctx,results):
    write_json(ROOT/'reports/tinyclip_visualnews.json',dict(results=results,data_audit=ctx.audit,
        primary_selection='unchanged_COCO_validation',vn_dev_used_for_selection=False,test_used=False,xm3600_used=False))
    lines=['# TinyCLIP: liczba obrazów VisualNews a partial finetuning','',
        'FULL_CLEAN i dotychczasowa walidacja niezmienione. Osobny, zamrożony dodatek VN oraz VN dev.',
        'Projector Linear 512→384, PolDense frozen, oryginalne projected TinyCLIP-39M. Warm start A1 ep. 9.',
        '3300 aktualizacji, batch 256 (145 co 275 kroków), head LR 1e-4, vision LR 1e-5, warmup 275, cosine do 10%.',
        'TRAIN z VN: 50% oryginalne wiersze FULL_CLEAN / 50% uniform VN images. Ta sama proporcja dla 10k i all.',
        'Kontrole 0 VN odtworzone z G1/G3, bez powtarzania identycznego treningu. Ich ekspozycja na FULL_CLEAN jest dwukrotnie większa.',
        'FT: ostatnie 2 bloki + anchor 50×(1−cos), augmentacja identyczna dla frozen i FT. Seed 42.',
        'Checkpoint wyłącznie po dotychczasowym COCO val retrieval. VN dev oceniany dopiero w wybranym checkpointcie.', '',
        '| Experiment | VN train images | Feature | Projector | Vision blocks | Trainable params | COCO val t2i R@1 | COCO val i2t R@1 | VN dev t2i R@1 | Best budget epoch |',
        '|---|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        v=r['selected_validation'];dev=r['selected_vn_dev']
        lines.append(f'| {r["experiment"]} | {r["vn_train_images"]:,} | projected | Linear | {r["spec"]["blocks"]} | {r["trainable_params"]:,} | {100*v["t2i"]["recall@1"]:.2f}% | {100*v["i2t"]["recall@1"]:.2f}% | {100*dev["t2i"]["recall@1"]:.2f}% | {r["best_epoch"]:.3f} |')
    lines+=['','Jedna epoka budżetu = 275 kroków; liczba kroków i prezentacji jest wspólna, nie liczba obiegów każdego zbioru.',
        'VN dev ma osobną galerię i rozkład captionów; jego procentów nie porównywać bezpośrednio z COCO.',
        'Seed scout. Test MSCOCO i XM3600 nieuruchamiane. SHA snapshotów sprawdzane przed i po.']
    (ROOT/'reports/tinyclip_visualnews.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    import fcntl
    OUT.mkdir(exist_ok=True)
    with (OUT/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        ctx=Context()
        write_json(OUT/'plan.json',dict(data=ctx.identity,experiments=specs(),references=['G1_frozen_aug','G3_last2_aug_anchor'],
            followup_rule='Replicate a >=1pp FT gain against matched frozen on three fine-tuning seeds; otherwise inspect earlier LP-FT with matched control before adding a grid.',
            test_used=False,xm3600_used=False))
        engine=Engine(ctx)
        results=reuse_zero(ctx,engine);export(ctx,results)
        for name,spec in specs().items():
            results.append(train(ctx,engine,name,spec));export(ctx,results)
            gc.collect();torch.cuda.empty_cache()
        ctx.verify()
        write_json(OUT/'status.json',dict(status='complete',experiments=[r['experiment'] for r in results],snapshot_unchanged=True))

