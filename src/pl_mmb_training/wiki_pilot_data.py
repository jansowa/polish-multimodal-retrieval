"""Audit the immutable wiki addendum; no model fitting and no benchmark scoring."""
import base64
import io
import json
import re
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.fft import dctn

from . import tinyclip_visualnews as vn
from . import tinyclip_adaptation as base
from . import visualnews_data as vd
from .source_ablation import UnionFind,counts
from .common import ROOT,digest,file_sha,read_json,write_json
from .features import encode_cached
from .leakage import normalized_text,image_identity

OUT=ROOT/'training/runs/tinyclip-wiki-pilot-v1'
FOLDER=OUT/'data'


def sha1number(value):
    value=value.lower()
    return int(value,16) if len(value)==40 and all(c in '0123456789abcdef' for c in value) else int(value,36)


def gray_hash(encoded):
    gray=np.frombuffer(base64.b64decode(encoded),dtype=np.uint8).reshape(32,32)
    coefficients=dctn(gray.astype(float),type=2)[:8,:8]
    bits=coefficients>np.median(coefficients)
    return int.from_bytes(np.packbits(bits.reshape(-1)).tobytes(),'big')


def entities(source):
    values={a.get('wikidata_qid') for a in source.get('associations',[])}
    values.add(source.get('wikidata_qid'))
    return sorted(v for v in values if v and re.fullmatch(r'Q\d+',v))


def flickr_ids(source):
    text=json.dumps(source,ensure_ascii=False)
    values=set(re.findall(r'flickr\.com/(?:photos/[^/]+/|photo\.gne\?id=)(\d+)',text))
    values.update(re.findall(r'staticflickr\.com/[^/]+/(\d+)_',text))
    return sorted(values)


def parity(ctx):
    from transformers import AutoImageProcessor
    processor=AutoImageProcessor.from_pretrained('wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M',
        revision=ctx.base.lock['revision'],local_files_only=True,use_fast=False)
    rgb=np.load(FOLDER/'wiki_native_rgb.npy',mmap_mode='r')
    mean=torch.tensor(processor.image_mean)[None,:,None,None]
    std=torch.tensor(processor.image_std)[None,:,None,None]
    checks={}
    for path in sorted((FOLDER/'parity_raw').glob('*.bin')):
        index=int(path.stem);meta=read_json(FOLDER/'decoded'/f'{index:05d}.json')
        raw=path.read_bytes();assert image_identity(payload=raw)=={k:meta[k] for k in ('sha256','pixels_sha256')}
        with Image.open(io.BytesIO(raw)) as original:
            image=original.convert('RGB')
            expected=processor(images=[image],return_tensors='pt')['pixel_values']
        current=torch.from_numpy(np.array(rgb[index],copy=True))[None].float()/255
        restored=(current-mean)/std
        error=float((expected-restored).abs().max())
        assert error<2e-6,(index,error)
        checks[str(index)]=dict(normalized_max_error=error,sha256=meta['sha256'])
    assert len(checks)>=8
    write_json(FOLDER/'preprocessing_parity.json',dict(status='passed',checks=checks,
        algorithm='native HF CLIP slow processor parity including full source identities'))
    print('WIKI_PREPROCESSING_PARITY',len(checks),flush=True)
    return processor


def protection(ctx):
    paths=[ROOT/'training/runs/e1-full/blacklist.json',ROOT/'training/runs/xm3600-pl/blacklist.json']
    xm=read_json(ROOT/'reports/xm3600_pl_preparation.json')
    assert file_sha(paths[1])==xm['blacklist_sha256']
    images={};texts=set();parents=set()
    for path in paths:
        value=read_json(path);images.update(value['images']);texts.update(value['texts'])
        parents.update(p for r in value['images'].values() for p in r.get('parent_ids',[]))
    for r in ctx.base.train+ctx.base.val:
        images[r['image']]=r
    for r in ctx.base.train+ctx.base.val+ctx.vn+ctx.dev:
        texts.update(normalized_text(r[f]) for f in ('text_en','text_pl') if r.get(f))
    old_path=ROOT/'training/runs/clean-v1/perceptual_hashes.json'
    old=read_json(old_path)['hashes']
    hashes={p:int(old[p],16) for p in images if p in old}
    missing=[p for p in images if p not in hashes]
    cached=FOLDER/'additional_protected_phashes.json'
    identity=digest({p:images[p]['sha256'] for p in missing})
    if cached.exists():
        saved=read_json(cached);assert saved['identity']==identity
        hashes.update({p:int(h,16) for p,h in saved['hashes'].items()})
    else:
        def compute(p):
            assert file_sha(p)==images[p]['sha256']
            with Image.open(p) as im:return p,vd.phash_image(im)
        with ThreadPoolExecutor(max_workers=8) as pool:
            values=dict(pool.map(compute,missing))
        hashes.update(values);write_json(cached,dict(identity=identity,hashes={p:f'{h:016x}' for p,h in values.items()}))
    exact={h for v in images.values() for h in (v['sha256'],v['pixels_sha256'])}
    index=vd.HammingIndex()
    for p,h in hashes.items():index.add(p,h)
    for r in ctx.candidates:
        index.add(r['id'],int(r['phash'],16));exact.update([r['sha256'],r['pixels_sha256']])
    vg_path=Path.home()/'.cache/huggingface/pl_mmb_training/visual_genome_image_data.json'
    legacy=read_json(paths[0]);assert file_sha(vg_path)==legacy['lineage_metadata']['visual_genome']
    coco_flickr={str(r['flickr_id']) for r in read_json(vg_path) if r.get('coco_id') is not None and r.get('flickr_id') is not None}
    return dict(index=index,exact=exact,texts=texts,parents=parents,coco_flickr=coco_flickr,
        inputs={str(p):file_sha(p) for p in paths+[old_path,vg_path]},reference_phashes=len(index.hashes))


def split_groups(records,rejected,target=200,maximum=250,minimum=150,min_groups=15):
    uf=UnionFind();near=vd.HammingIndex();seen={}
    for i,r in enumerate(records):
        keys=['pixel:'+r['target_id'],'originalSHA1:'+r['source_sha1_hex']]
        keys+=['entity:'+v for v in r['entity_ids']]
        keys+=['caption:'+normalized_text(r[f]) for f in ('text_pl','source_caption') if r.get(f)]
        uf.join(['row:'+str(i)]+keys)
        for j,_ in near.near(int(r['phash'],16)):uf.join(['row:'+str(i),'row:'+str(j)])
        near.add(i,int(r['phash'],16))
    groups=defaultdict(list)
    for i in range(len(records)):groups[uf.find('row:'+str(i))].append(i)
    bad={uf.find('row:'+str(i)) for i in rejected}
    kept=[v for k,v in groups.items() if k not in bad]
    ordered=sorted(kept,key=lambda g:digest(['wiki-pilot-split-v1',1701,sorted(records[i]['id'] for i in g)]))
    train=[];dev=[];dev_images=set()
    for group in ordered:
        n=len({records[i]['target_id'] for i in group})
        if len(dev_images)<target and n<=100 and len(dev_images)+n<=maximum:
            dev+=group;dev_images.update(records[i]['target_id'] for i in group)
        else:train+=group
    group_id={i:digest(sorted(records[j]['id'] for j in group)) for group in kept for i in group}
    def materialize(indices,split):
        return [dict(records[i],split=split,component=group_id[i]) for i in sorted(indices,key=lambda k:records[k]['id'])]
    train_rows=materialize(train,'train');dev_rows=materialize(dev,'wiki_dev')
    assert {r['target_id'] for r in train_rows}.isdisjoint(r['target_id'] for r in dev_rows)
    assert {v for r in train_rows for v in r['entity_ids']}.isdisjoint(v for r in dev_rows for v in r['entity_ids'])
    assert {normalized_text(r[f]) for r in train_rows for f in ('text_pl','source_caption') if r.get(f)}.isdisjoint(
        normalized_text(r[f]) for r in dev_rows for f in ('text_pl','source_caption') if r.get(f))
    assert minimum<=len(dev_images)<=maximum
    assert len({r['component'] for r in dev_rows})>=min_groups,'Insufficient independent wiki dev groups'
    summary=dict(total_components=len(groups),excluded_components=len(bad),kept_components=len(kept),
        train_components=len({r['component'] for r in train_rows}),dev_components=len({r['component'] for r in dev_rows}),
        largest_component_images=max(len({records[i]['target_id'] for i in group}) for group in kept),
        direct_rejected=len(rejected),propagated_rejected_rows=sum(len(v) for k,v in groups.items() if k in bad))
    return train_rows,dev_rows,summary


def audit(ctx):
    complete=FOLDER/'audit.json'
    if complete.exists():
        result=read_json(complete)
        for name,sha in result['files'].items():assert file_sha(FOLDER/name)==sha
        return result
    snapshot=read_json(FOLDER/'snapshot.json');download=read_json(FOLDER/'download_complete.json')
    assert snapshot==download['snapshot']
    assert file_sha(FOLDER/'wiki_pairs_snapshot.jsonl')==snapshot['pairs_sha256']
    assert file_sha(FOLDER/'wiki_native_rgb.npy')==download['native_sha256']
    pairs=base.records(FOLDER/'wiki_pairs_snapshot.jsonl');reference=protection(ctx)
    records=[];rejected={};failures=[];reason_counts=Counter();examples=[]
    for i,pair in enumerate(pairs):
        meta=read_json(FOLDER/'decoded'/f'{i:05d}.json')
        if meta['status']!='ready':failures.append(dict(id=pair['id'],reason=meta['error']));continue
        source=pair['source_record'];expected=sha1number(source['source_sha1'])
        assert expected==sha1number(meta['source_sha1_verified'])
        qids=entities(source);flickr=flickr_ids(source);h=gray_hash(meta['gray32_base64'])
        row=dict(id=pair['id'],dataset='wikimedia',text_pl=pair['text_pl'],text_en=pair.get('text_en'),
            source_caption=pair['source_caption'],source_language=pair['source_language'],route=pair['route'],
            image='wiki-cache://'+pair['id'],target_id=meta['pixels_sha256'],pixels_sha256=meta['pixels_sha256'],
            sha256=meta['sha256'],source_sha1_hex=f'{expected:040x}',phash=f'{h:016x}',cache_index=i,
            entity_ids=qids,parent_ids=['commons:'+pair['id']]+['wikidata:'+q for q in qids]+['flickr:'+f for f in flickr],
            lineage_verified=True,preview=str(FOLDER/'previews'/f'{i:05d}.jpg'),
            source_page_url=source['source_page_url'],source_image=source['image_original_url'],
            license=source['license'],license_url=source['license_url'],attribution=source.get('attribution'),
            acquisition_mode=meta.get('acquisition_mode','original_SHA1_verified'))
        reasons=[]
        if not qids:reasons.append('no_entity_group_identity')
        if set(flickr)&reference['coco_flickr']:reasons.append('known_Flickr_COCO_identity')
        if set(row['parent_ids'])&reference['parents']:reasons.append('protected_parent')
        if row['sha256'] in reference['exact'] or row['pixels_sha256'] in reference['exact']:reasons.append('protected_or_VN_exact_content')
        if any(normalized_text(row[f]) in reference['texts'] for f in ('text_pl','text_en') if row.get(f)):
            reasons.append('protected_or_existing_caption')
        near=reference['index'].near(h)
        if near:
            reasons.append('protected_or_VN_phash_le6')
            examples.append(dict(id=row['id'],matches=sorted(near,key=lambda x:x[1])[:3]))
        if reasons:rejected[len(records)]=reasons;reason_counts.update(reasons)
        records.append(row)
    from .series_holdout import workspace_guard
    wiki_guard=workspace_guard()
    if wiki_guard:
        for index,row in enumerate(records):
            extra=wiki_guard.reasons(row)
            if extra:
                rejected.setdefault(index,[]).extend(extra);reason_counts.update(extra)
    train,dev,groups=split_groups(records,rejected)
    for name,rows in [('candidates.jsonl',records),('wiki_train.jsonl',train),('wiki_dev.jsonl',dev)]:vd.write_rows(FOLDER/name,rows)
    write_json(FOLDER/'rejections.json',dict(direct={records[i]['id']:r for i,r in rejected.items()},
        failures=failures,near_examples=examples,reason_counts=dict(reason_counts)))
    value=dict(status='passed',snapshot=snapshot,download_ready=download['ready'],download_failed=download['failed'],
        candidates=counts(records),train=counts(train),dev=counts(dev),groups=groups,
        train_entities=len({q for r in train for q in r['entity_ids']}),dev_entities=len({q for r in dev for q in r['entity_ids']}),
        rejection_counts=dict(reason_counts),protected_inputs=reference['inputs'],reference_phashes=reference['reference_phashes'],
        source_sha1_verified=True,COCO_scored=False,xm3600_scored=False,
        acquisition_final_sources={p:file_sha(ROOT/p) for p in ('training/fetch_wiki_pilot_images.py',
            'src/pl_mmb_collection/sources/http.py','src/pl_mmb_collection/sources/rate_limit.py')},
        files={p:file_sha(FOLDER/p) for p in [*('snapshot.json','wiki_pairs_snapshot.jsonl','pipeline_manifest.json',
            'download_complete.json','render_plan.json','wiki_native_rgb.npy','candidates.jsonl','wiki_train.jsonl',
            'wiki_dev.jsonl','rejections.json','preprocessing_parity.json'),
            *[str(p.relative_to(FOLDER)) for p in sorted((FOLDER/'decoded').glob('*.json'))]]},
        caveat='Known source/parent/hash and pHash<=6 checks, not a guarantee against unidentified crops/copies. Wiki dev grouped by actual entity IDs, source SHA1, captions, pixels and pHash.')
    write_json(complete,value);write_json(ROOT/'reports/tinyclip_wiki_pilot_data_audit.json',value)
    print('WIKI_SPLIT',json.dumps(dict(train=value['train'],dev=value['dev'],groups=groups)),flush=True)
    return value


def encode(ctx,processor,audited):
    complete=FOLDER/'features.json'
    if complete.exists():
        result=read_json(complete)
        assert result['audit_sha256']==file_sha(FOLDER/'audit.json')
        assert file_sha(FOLDER/'vision_features.pt')==result['vision_sha256']
        assert file_sha(result['text_meta']['path'])==result['text_meta']['sha256']
        return result
    records=base.records(FOLDER/'wiki_train.jsonl')+base.records(FOLDER/'wiki_dev.jsonl')
    indices=sorted({r['cache_index'] for r in records});rgb=np.load(FOLDER/'wiki_native_rgb.npy',mmap_mode='r')
    vision,_,_=base.load_vision(ctx.base);assert not any(p.requires_grad for p in vision.parameters())
    mean=torch.tensor(processor.image_mean,device='cuda')[None,:,None,None]
    std=torch.tensor(processor.image_std,device='cuda')[None,:,None,None]
    result=torch.zeros(len(rgb),512)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for start in range(0,len(indices),64):
            part=indices[start:start+64]
            pixels=torch.from_numpy(np.array(rgb[part],copy=True)).to('cuda').float()/255
            result[part]=vision(pixel_values=(pixels-mean)/std).image_embeds.float().cpu()
    assert torch.isfinite(result).all() and (result[indices].norm(dim=1)>0).all()
    torch.save(result,FOLDER/'vision_features.pt');del vision;torch.cuda.empty_cache()
    texts=sorted({r['text_pl'] for r in records});old=read_json(ROOT/'training/runs/e1-full/cache.json')
    config=read_json(ROOT/'training/runs/e1-full/config.json')
    vectors,text_meta=encode_cached(texts,'texts',config,old['model_lock'],OUT/'text_cache')
    # Length audit uses exactly the same pinned tokenizer/prefix, with no text edits.
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(config['text']['model'],revision=old['model_lock']['text']['revision'],local_files_only=True)
    lengths=[len(v) for v in tokenizer([config['text']['prefix']+t for t in texts],truncation=False)['input_ids']]
    meta=dict(audit_sha256=file_sha(FOLDER/'audit.json'),vision_model=ctx.base.lock,
        vision_sha256=file_sha(FOLDER/'vision_features.pt'),text_values=texts,text_meta=text_meta,
        text_tokens=dict(max_length=config['text']['max_length'],exceeding_max=sum(n>config['text']['max_length'] for n in lengths),
            median=float(np.median(lengths)),maximum=max(lengths)),frozen_encoders=True)
    write_json(complete,meta);print('WIKI_FEATURES_COMPLETE',len(indices),len(texts),flush=True)
    return meta


def prepare(parity_only=False):
    ctx=vn.Context();processor=parity(ctx)
    if parity_only:return
    audited=audit(ctx);meta=encode(ctx,processor,audited)
    ctx.verify()
    write_json(OUT/'status.json',dict(status='data_ready',train=audited['train'],dev=audited['dev']))
    return meta
