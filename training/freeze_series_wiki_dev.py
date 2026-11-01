"""CPU-only component leakage audit and immutable native Wiki holdout freeze."""
import hashlib
import json
import math
import re
import sys
from collections import Counter,defaultdict
from pathlib import Path
from urllib.parse import unquote
import numpy as np
from PIL import Image,ImageOps
from scipy.fft import dctn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pl_mmb_training.series_20261023 import ROOT,OUT,VN,WIKI,read,rows,freeze,file_sha,local_path,require_space
from pl_mmb_training.leakage import normalized_text
from prepare_series_xtd_assets import source_groups,resolve_source

DEST=OUT/'wiki_v2'


class UnionFind:
    def __init__(self):self.parents={}
    def find(self,key):
        self.parents.setdefault(key,key)
        while self.parents[key]!=key:
            self.parents[key]=self.parents[self.parents[key]];key=self.parents[key]
        return key
    def join(self,keys):
        if keys:
            root=self.find(keys[0])
            for key in keys[1:]:self.parents[self.find(key)]=root


class HammingIndex:
    """Seven disjoint bands guarantee an exact matching band for radius <=6."""
    def __init__(self):self.hashes={};self.buckets=defaultdict(list)
    @staticmethod
    def bands(value):
        shift=0
        for i,width in enumerate((10,9,9,9,9,9,9)):
            yield i,(value>>shift)&((1<<width)-1);shift+=width
    def add(self,key,value):
        self.hashes[key]=value
        for band in self.bands(value):self.buckets[band].append(key)
    def near(self,value):
        candidates={k for band in self.bands(value) for k in self.buckets[band]}
        return [(k,(self.hashes[k]^value).bit_count()) for k in candidates if (self.hashes[k]^value).bit_count()<=6]


def source_parents(source,uid):
    values=['commons:'+uid]
    if source.get('source_file_id'):values.append('commons-file:'+str(source['source_file_id']))
    if source.get('source_page_url'):values.append('source-page:'+unquote(source['source_page_url']).replace('_',' '))
    qids={source.get('wikidata_qid')}|{a.get('wikidata_qid') for a in source.get('associations',[])}
    values.extend('wikidata:'+q for q in qids if q and re.fullmatch(r'Q\d+',q))
    text=json.dumps(source,ensure_ascii=False)
    flickr=set(re.findall(r'flickr\.com/(?:photos/[^/]+/|photo\.gne\?id=)(\d+)',text))
    flickr.update(re.findall(r'staticflickr\.com/[^/]+/(\d+)_',text))
    values.extend('flickr:'+key for key in sorted(flickr))
    return sorted(set(values))


def text_keys(row):
    source=row.get('source_record',{})
    return {normalized_text(text) for text in [row.get('text_pl'),row.get('text_en'),row.get('source_caption'),
            source.get('caption_en'),source.get('caption_pl')] if text}


def main():
    require_space(['C:/','D:/'],expected_bytes=16*2**20)
    if read(DEST/'download_state.json')['status']!='complete':raise ValueError('Acquisition not complete')
    candidates=rows(DEST/'candidates.jsonl')
    pilot_pairs={r['id']:r for r in rows(WIKI/'wiki_pairs_snapshot.jsonl')}
    pilot=rows(WIKI/'candidates.jsonl')
    groups=source_groups();protected=[r for rs in groups.values() for r in rs]
    legacy=read(ROOT/'training/runs/e1-full/blacklist.json');xm=read(ROOT/'training/runs/xm3600-pl/blacklist.json')
    protected_texts={normalized_text(t) for black in (legacy,xm) for t in black['texts']}
    protected_texts.update(normalized_text(r[f]) for r in protected for f in ('text_en','text_pl') if r.get(f))
    protected_parents={p for r in protected for p in r.get('parent_ids',[])}
    protected_hashes={r[k] for r in protected for k in ('sha256','pixels_sha256') if r.get(k)}
    phash_sources=[ROOT/'training/runs/clean-v1/perceptual_hashes.json',WIKI/'additional_protected_phashes.json']
    phashes={}
    for path in phash_sources:phashes.update(read(path)['hashes'])
    reference=HammingIndex();missing=[];seen=set()
    for r in protected:
        key=(r['sha256'],r['pixels_sha256'])
        if key in seen:continue
        seen.add(key)
        value=r.get('phash') or phashes.get(r.get('image'))
        if not value:missing.append(r);continue
        reference.add(key,int(value,16))
    if missing:
        # No new reference cache until every original can be SHA-verified.
        for r in missing:
            path=resolve_source(r.get('image',''))
            if not path.is_file() or file_sha(path)!=r['sha256']:
                raise ValueError('Protected pHash unavailable: '+str(path))
            with Image.open(path) as im:
                gray=ImageOps.exif_transpose(im).convert('RGB').convert('L').resize((32,32),Image.Resampling.LANCZOS)
                dct=dctn(np.asarray(gray,dtype=float),type=2)[:8,:8]
                value=int.from_bytes(np.packbits((dct>np.median(dct)).reshape(-1)).tobytes(),'big')
            reference.add((r['sha256'],r['pixels_sha256']),value)
    metadata_path=resolve_source('/home/jsowa/.cache/huggingface/pl_mmb_training/visual_genome_image_data.json')
    if file_sha(metadata_path)!=legacy['lineage_metadata']['visual_genome']:raise ValueError('Official VG metadata differs')
    coco_flickr={str(r['flickr_id']) for r in read(metadata_path) if r.get('coco_id') is not None and r.get('flickr_id') is not None}
    all_rows=[];candidate_indices=[];failures=[]
    for index,pair in enumerate(candidates):
        receipt=read(DEST/'decoded'/f'{index:05d}.json')
        source=pair['source_record']
        row=dict(pair,parent_ids=source_parents(source,pair['id']),source_sha1_hex=f"{int(source['source_sha1'],16):040x}" if len(source['source_sha1'])==40 else source['source_sha1'])
        if receipt['status']=='ready':
            if file_sha(receipt['image'])!=receipt['sha256']:raise ValueError('Acquired asset changed')
            row.update(receipt);row['image']='/mnt/c'+receipt['image'][2:].replace('\\','/');row['target_id']=receipt['pixels_sha256']
        else:row['asset_failure']=receipt['error'];failures.append(pair['id'])
        candidate_indices.append(len(all_rows));all_rows.append(row)
    for row in pilot:
        source=pilot_pairs[row['id']]['source_record']
        all_rows.append(dict(row,source_record=source,parent_ids=sorted(set(row['parent_ids']+source_parents(source,row['id'])))))
    uf=UnionFind();near=HammingIndex();direct={};near_examples=[]
    for index,row in enumerate(all_rows):
        keys=['row:'+str(index),'id:'+row['id']]+row.get('parent_ids',[])
        keys.extend('text:'+text for text in text_keys(row))
        for field,prefix in [('pixels_sha256','pixel:'),('source_sha1_hex','sourceSHA1:'),('sha256','file:')]:
            if row.get(field):keys.append(prefix+row[field])
        uf.join(keys)
        reasons=[]
        if row.get('asset_failure'):reasons.append('unverified_candidate_original')
        if set(row.get('parent_ids',[]))&protected_parents:reasons.append('protected_parent')
        if any(row.get(k) in protected_hashes for k in ('sha256','pixels_sha256')):reasons.append('protected_exact_content')
        if text_keys(row)&protected_texts:reasons.append('protected_normalized_text')
        if any(p.startswith('flickr:') and p[7:] in coco_flickr for p in row.get('parent_ids',[])):reasons.append('known_COCO_Flickr_ancestry')
        if row.get('phash'):
            value=int(row['phash'],16)
            for match,_ in near.near(value):uf.join(['row:'+str(index),'row:'+str(match)])
            near.add(index,value)
            matches=reference.near(value)
            if matches:reasons.append('protected_phash_le6');near_examples.append(dict(id=row['id'],matches=matches[:3]))
        if reasons:direct[index]=reasons
    components=defaultdict(list)
    for index in range(len(all_rows)):components[uf.find('row:'+str(index))].append(index)
    bad={uf.find('row:'+str(index)) for index in direct}
    component_id={root:hashlib.sha256(json.dumps(sorted({all_rows[i]['id'] for i in members})).encode()).hexdigest()
                  for root,members in components.items()}
    eligible=[]
    for index in candidate_indices:
        root=uf.find('row:'+str(index))
        if root not in bad:
            row=dict(all_rows[index],component=component_id[root],dataset='wiki_native_dev',lineage_verified=True)
            eligible.append(row)
    unique={}
    for row in sorted(eligible,key=lambda r:(r['selected_candidate'].get('quality','D'),r['id'])):
        unique.setdefault(row['target_id'],row)
    available=list(unique.values())
    audit=dict(status='audited',candidate_snapshot_sha256=file_sha(DEST/'snapshot.json'),original_candidates=len(candidates),
        acquired_originals=len(candidates)-len(failures),asset_failures=failures,protected_images=len(reference.hashes),
        extra_reference_phashes_computed=len(missing),eligible_native_images=len(available),
        direct_reasons=dict(Counter(reason for rs in direct.values() for reason in rs)),
        total_components=len(components),excluded_components=len(bad),
        eligible_classes=dict(Counter(r['selected_candidate'].get('quality') for r in available)),
        old_pilot_rows=len(pilot),no_model_calls=True,XM3600_used_only_for_overlap_detection=True,
        protected_phash_inputs={str(p.relative_to(ROOT)):file_sha(p) for p in phash_sources},
        near_examples=near_examples[:30])
    freeze(DEST/'component_audit.json',audit)
    if len(available)<300:
        freeze(DEST/'holdout_status.json',dict(status='stopped_below_300_after_component_audit',
            eligible_native_images=len(available),no_translated_or_recycled_substitution=True,audit_sha256=file_sha(DEST/'component_audit.json')))
        print('WIKI_NATIVE_STOP '+json.dumps(audit),flush=True);return
    target=min(500,len(available));classes=Counter(r['selected_candidate'].get('quality') for r in available)
    quota={key:math.floor(target*count/len(available)) for key,count in classes.items()}
    remainder=target-sum(quota.values())
    order=sorted(classes,key=lambda k:(-(target*classes[k]/len(available)-quota[k]),k))
    for key in order[:remainder]:quota[key]+=1
    selected=[]
    for key,n in sorted(quota.items()):
        pool=[r for r in available if r['selected_candidate'].get('quality')==key]
        pool.sort(key=lambda r:hashlib.sha256(('wiki-native-dev-v2:20261023:'+r['target_id']).encode()).hexdigest())
        selected.extend(pool[:n])
    selected.sort(key=lambda r:r['target_id'])
    selected_components={r['component'] for r in selected}
    heldout=[r for i,r in enumerate(all_rows) if component_id[uf.find('row:'+str(i))] in selected_components]
    dev_sha=freeze(DEST/'wiki_dev.jsonl',selected,jsonl=True)
    heldout_ids={r['id'] for r in heldout}
    old_train=rows(WIKI/'wiki_train.jsonl')
    new_train=[r for r in old_train if r['id'] not in heldout_ids]
    train_sha=freeze(DEST/'pilot_wiki_train_disjoint.jsonl',new_train,jsonl=True)
    protected_images={str(i):{k:r[k] for k in ('sha256','pixels_sha256','parent_ids','phash') if r.get(k)}
                      for i,r in enumerate(heldout) if r.get('sha256') and r.get('pixels_sha256')}
    guard=dict(schema=1,status='frozen',dataset='wiki_native_dev_20261023_v2',dev_manifest_sha256=dev_sha,
        ids=sorted(heldout_ids),texts=sorted({text for r in heldout for text in text_keys(r)}),images=protected_images,
        parent_ids=sorted({p for r in heldout for p in r.get('parent_ids',[])}),phash_radius=6,
        entire_components_reserved=True,manifest_path='reports/series_2026-10-23/wiki_v2/wiki_dev.jsonl')
    guard_sha=freeze(OUT/'wiki_dev_guard.json',guard)
    image_ids=[r['target_id'] for r in selected];images=[r['image'] for r in selected]
    text_map={}
    for r in selected:text_map.setdefault(normalized_text(r['text_pl']),r['text_pl'])
    text_keys_sorted=sorted(text_map);texts=[text_map[key] for key in text_keys_sorted];ti={key:i for i,key in enumerate(text_keys_sorted)}
    edges=sorted({(ti[normalized_text(r['text_pl'])],i) for i,r in enumerate(selected)})
    text_components={normalized_text(r['text_pl']):r['component'] for r in selected}
    graph=dict(status='frozen_image_verified',id='wiki_native_dev',manifest_sha256=dev_sha,
        images=images,image_ids=image_ids,texts=texts,query_ids=[hashlib.sha256(t.encode()).hexdigest() for t in text_keys_sorted],
        edges=edges,image_components=[r['component'] for r in selected],text_components=[text_components[t] for t in text_keys_sorted],
        labels='native human PL accepted directly; no translation or recycling',
        gallery='all dev images',bootstrap='transitive source/page/text/pixel/pHash components',metrics=['R@1','R@5','R@10','MRR','NDCG@10'])
    freeze(DEST/'retrieval_protocol.json',graph)
    result=dict(status='frozen_image_verified',dev_images=len(selected),caption_rows=len(selected),source_class_quota=quota,
        dev_components=len(selected_components),entire_heldout_component_rows=len(heldout),dev_manifest_sha256=dev_sha,guard_sha256=guard_sha,
        original_dev_asset_bytes=sum(r['bytes'] for r in selected),all_acquired_asset_bytes=read(DEST/'download_state.json')['asset_bytes'],
        original_pilot_train_rows=len(old_train),new_disjoint_pilot_train_rows=len(new_train),
        removed_pilot_train_rows=len(old_train)-len(new_train),new_pilot_train_sha256=train_sha,
        historical_1322_snapshot_unchanged=True,source_export_sha256=read(DEST/'snapshot.json')['source_export_sha256'],
        zero_shot_baseline_pending=True,new_Wiki_training_allowed=False,guard_integration_pending=True)
    freeze(DEST/'holdout_status.json',result)
    print('WIKI_NATIVE_FROZEN '+json.dumps(result),flush=True)


if __name__=='__main__':main()
