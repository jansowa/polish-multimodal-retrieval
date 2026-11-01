"""Freeze XTD10 full originals, exact parent/hash overlap and paired cached styles."""
import concurrent.futures
import hashlib
import io
import json
import re
import shutil
import ssl
import sys
import urllib.request
from collections import defaultdict,Counter
from pathlib import Path
from PIL import Image,ImageOps
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pl_mmb_training.series_20261023 import ROOT,OUT,read,rows,freeze,file_sha,require_space,local_path
from pl_mmb_training.leakage import normalized_text

DEST=OUT/'xtd10_v2'
ASSETS=Path('C:/Users/jsowa/benchmark-translations/series-2026-10-23/xtd10/images')


def source_groups():
    groups={name:rows(ROOT/path) for name,path in (
        ('FULL_CLEAN','training/runs/clean-v1/FULL_CLEAN.jsonl'),
        ('validation','training/runs/clean-v1/validation.jsonl'),
        ('VN_train','training/runs/tinyclip-visualnews-v1/data/vn_train.jsonl'),
        ('VN_dev','training/runs/tinyclip-visualnews-v1/data/vn_dev.jsonl'))}
    for name,path in [('legacy_eval','training/runs/e1-full/blacklist.json'),('XM_blacklist','training/runs/xm3600-pl/blacklist.json')]:
        groups[name]=[dict(r,image=p) for p,r in read(ROOT/path)['images'].items()]
    return groups


def resolve_source(value):
    if value.startswith('/home/'):
        return Path('\\\\wsl.localhost\\Ubuntu'+value.replace('/','\\'))
    return local_path(value)


def main():
    DEST.mkdir(exist_ok=True)
    context=ssl.create_default_context(cafile=str(ROOT/'combined-ca.pem'))
    original=rows(OUT/'xtd10_annotation_manifest.jsonl')
    groups=source_groups()
    by_name=defaultdict(list)
    for name,records in groups.items():
        if name=='XM_blacklist':continue  # leakage checks only; never resolve assets from XM
        for r in records:
            value=r.get('image','')
            basename=Path(value.replace('\\','/')).name
            if basename.startswith('COCO_'):by_name[basename].append(r)
    decision=dict(status='registered_before_asset_acquisition',script_sha256=file_sha(__file__),
        annotation_manifest_sha256=file_sha(OUT/'xtd10_annotation_manifest.jsonl'),
        asset_root=str(ASSETS),workers=4,max_image_bytes=20_000_000,
        source='Reuse exact-name full images after SHA verification; otherwise literal upstream COCO name from official images.cocodataset.org S3 bucket',
        URL='https://s3.amazonaws.com/images.cocodataset.org/{split}/{literal_upstream_filename}',
        no_guessed_image_mapping=True,XM3600_used_only_for_overlap_detection=True,
        no_model_encoding_or_retrieval=True,reserve_bytes=4*2**30)
    freeze(DEST/'acquisition_decision.json',decision)
    require_space(['C:/','D:/'],expected_bytes=200*2**20)
    ASSETS.mkdir(parents=True,exist_ok=True)
    parent_sets={k:{p for r in rs for p in r.get('parent_ids',[])} for k,rs in groups.items()}
    hash_sets={k:{r[f] for r in rs for f in ('sha256','pixels_sha256') if r.get(f)} for k,rs in groups.items()}
    def acquire(row):
        name=row['upstream_image_name'];path=ASSETS/name
        receipt=DEST/'receipts'/(row['id'].split(':')[1]+'.json');receipt.parent.mkdir(exist_ok=True)
        if receipt.exists():
            saved=read(receipt)
            if saved.get('status')=='ready' and file_sha(path)==saved['sha256']:return saved
        require_space(['C:/','D:/'],expected_bytes=20*2**20)
        source_url=None;reused=None
        if not path.exists():
            for r in by_name[name]:
                source=resolve_source(r['image'])
                if source.is_file() and file_sha(source)==r['sha256']:
                    shutil.copyfile(source,path);reused=r['image'];break
            else:
                match=re.fullmatch(r'COCO_((?:val|train)2014)_\d{12}\.jpg',name)
                if not match:raise ValueError('No explicit literal COCO 2014 filename')
                source_url=f'https://s3.amazonaws.com/images.cocodataset.org/{match[1]}/{name}'
                request=urllib.request.Request(source_url,headers={'User-Agent':'pl-mmb-research/2026-10-23'})
                with urllib.request.urlopen(request,context=context,timeout=45) as response:
                    raw=response.read(20_000_001)
                if len(raw)>20_000_000:raise ValueError('Image exceeds registered per-file size')
                with path.open('xb') as output:output.write(raw)
        raw=path.read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            rgb=ImageOps.exif_transpose(image).convert('RGB')
            pixels=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
            dimensions=list(rgb.size)
        sha=hashlib.sha256(raw).hexdigest();hashes={sha,pixels};parents=set(row['parent_ids'])
        overlaps={name:dict(parent=bool(parents & parent_sets[name]),hashes=bool(hashes & hash_sets[name])) for name in groups}
        linux_image='/mnt/c'+str(path)[2:].replace('\\','/')
        result=dict(row,status='ready',image=linux_image,windows_image=str(path),source_url=source_url,reused_from=reused,
            sha256=sha,pixels_sha256=pixels,target_id=pixels,dimensions=dimensions,bytes=len(raw),
            overlap=overlaps,component=next(iter(parents)),hash_audit='verified_original_asset',
            mapping_provenance='literal official XTD10 filename -> explicit COCO parent ID')
        freeze(receipt,result);return result
    acquired=[];failed=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(acquire,row):row['id'] for row in original}
        for index,future in enumerate(concurrent.futures.as_completed(futures),1):
            try:acquired.append(future.result())
            except Exception as error:failed.append(dict(id=futures[future],error=type(error).__name__+': '+str(error)))
            if index%50==0:print(f'XTD_ASSETS {index}/1000 ready={len(acquired)} failed={len(failed)}',flush=True)
    if failed:
        freeze(DEST/'acquisition_failures.json',failed)
        print(json.dumps(dict(status='incomplete',ready=len(acquired),failures=failed[:10])),flush=True);return
    acquired.sort(key=lambda r:r['id'])
    if len({r['target_id'] for r in acquired})!=1000:raise ValueError('Canonical duplicate images: explicit gallery/qrels review required')
    # Join repeated normalized human captions into bootstrap components.
    root={}
    def find(key):
        root.setdefault(key,key)
        if root[key]!=key:root[key]=find(root[key])
        return root[key]
    for r in acquired:
        a=find(r['target_id']);b=find('text:'+normalized_text(r['text_pl']));root[b]=a
    component_ids=defaultdict(list)
    for r in acquired:component_ids[find(r['target_id'])].append(r['id'])
    for r in acquired:r['component']=hashlib.sha256(json.dumps(sorted(component_ids[find(r['target_id'])])).encode()).hexdigest()
    full_sha=freeze(DEST/'full_1k.jsonl',acquired,jsonl=True)
    clean=[r for r in acquired if not any(v['parent'] or v['hashes'] for v in r['overlap'].values())]
    clean_sha=freeze(DEST/'no_overlap.jsonl',clean,jsonl=True)
    # Only exact EN matches in already frozen translation-derived training rows.
    translation_map=defaultdict(set)
    for split in ('FULL_CLEAN','validation'):
        for r in groups[split]:
            if r.get('text_en') and r.get('text_pl'):
                translation_map[normalized_text(r['text_en'])].add(r['text_pl'])
    paired=[];ambiguous=[]
    for r in acquired:
        candidates=translation_map[normalized_text(r['text_en'])]
        if len(candidates)==1:paired.append(dict(id=r['id'],image=r['image'],target_id=r['target_id'],component=r['component'],
            human_pl=r['text_pl'],cached_translation_pl=next(iter(candidates)),original_en=r['text_en'],
            pairing='exact normalized EN in frozen FULL_CLEAN/validation translation rows; no generated translations'))
        elif len(candidates)>1:ambiguous.append(r['id'])
    paired_sha=freeze(DEST/'paired_styles.jsonl',paired,jsonl=True)
    result=dict(status='frozen_image_verified',full_images=1000,full_manifest_sha256=full_sha,
        clean_images=len(clean),clean_manifest_sha256=clean_sha,
        paired_exact_EN_images=len(paired),paired_manifest_sha256=paired_sha,ambiguous_exact_EN_ids=ambiguous,
        overlap_images={name:dict(parent=sum(r['overlap'][name]['parent'] for r in acquired),
            hashes=sum(r['overlap'][name]['hashes'] for r in acquired),
            either=sum(r['overlap'][name]['parent'] or r['overlap'][name]['hashes'] for r in acquired)) for name in groups},
        original_asset_bytes=sum(r['bytes'] for r in acquired),role='translation_style_control_not_holdout',
        source_revision=read(OUT/'xtd10_source/upstream.json')['revision'],license_sha256=file_sha(OUT/'xtd10_source/LICENSE.md'),
        no_retrieval_scores_computed=True,XM3600_scored=False)
    freeze(DEST/'preparation.json',result);print('XTD_FROZEN '+json.dumps(result),flush=True)


if __name__=='__main__':main()
