"""Bounded CPU audit of frozen exports, with all historical protected components.

No network, no SQLite, no model evaluation. Original files never copied.
"""
import hashlib
import io
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from preflight import ROOT, REPORT, SERIES, STORAGE, read, rows
sys.path.insert(0,str(ROOT/'src'))
from common import artifact, atomic_json, local_path, reserve, sha256, verify_marker, publish, COORD
from guard import Components, normalized


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def fingerprints(path=None,raw=None):
    import numpy as np
    from PIL import Image,ImageOps
    from scipy.fft import dctn
    if raw is None:raw=Path(path).read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        rgb=ImageOps.exif_transpose(image).convert('RGB')
        pixel=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
        # Match historical RGB-independent grayscale pHash convention.
        gray=ImageOps.exif_transpose(image).convert('L').resize((32,32),Image.Resampling.LANCZOS)
        coeff=dctn(np.asarray(gray,dtype=float),type=2)[:8,:8]
        h=int.from_bytes(np.packbits((coeff>np.median(coeff)).reshape(-1)).tobytes(),'big')
    return dict(sha256=hashlib.sha256(raw).hexdigest(),pixels_sha256=pixel,phash=f'{h:016x}')


def write_rows(path,records):
    # Own audit artifacts; atomic publication with no half-written consumers.
    tmp=path.with_suffix('.tmp')
    with tmp.open('x') as stream:
        for r in records:stream.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
        stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)


def main():
    start=time.monotonic()
    verify_marker(COORD/'A_INPUTS_FROZEN.json')
    graph=Components();evidence={};protected=[];train=[];old=[]
    def load(name,relative):
        p=ROOT/relative;evidence[relative]=sha256(p)
        data=list(rows(p))
        for r in data:r['_group']=name
        return data
    train+=load('FULL_train','training/runs/clean-v1/FULL_CLEAN.jsonl')
    train+=load('VN_train','training/runs/tinyclip-visualnews-v1/data/vn_train.jsonl')
    protected+=load('COCO_val','training/runs/clean-v1/validation.jsonl')
    protected+=load('VN_dev','training/runs/tinyclip-visualnews-v1/data/vn_dev.jsonl')
    for label,relative in [('legacy_eval','training/runs/e1-full/blacklist.json'),('XM_blacklist_only','training/runs/xm3600-pl/blacklist.json')]:
        p=ROOT/relative;v=read(p);evidence[relative]=sha256(p)
        protected.extend(dict(r,image=p,id='image:'+p,_group=label) for p,r in v['images'].items())
        for text in v['texts']:graph.add(dict(text_pl=text),[label])
        for uid in v['ids']:graph.add(dict(id=uid),[label])
    guard=read(SERIES/'wiki_dev_guard_enforcement.json')
    protected.extend(dict(r,id='reserved:'+i,_group='Wiki_reserved') for i,r in guard['images'].items())
    for text in guard['texts']:graph.add(dict(text_pl=text),['Wiki_reserved'])
    for uid in guard['ids']:graph.add(dict(id=uid),['Wiki_reserved'])
    for parent in guard['parent_ids']:graph.add(dict(parent_ids=[parent]),['Wiki_reserved'])
    for h in guard['source_sha1_hex']:graph.add(dict(source_sha1_hex=h),['Wiki_reserved'])
    xtd=load('XTD_full','reports/series_2026-10-23/xtd10_v2/full_1k.jsonl')
    protected+=xtd
    # Verify historical pHash cache identity against original complete inputs.
    phash_path=ROOT/'training/runs/clean-v1/perceptual_hashes.json';pc=read(phash_path)
    legacy=read(ROOT/'training/runs/e1-full/blacklist.json')
    expected={p:r['sha256'] for p,r in legacy['images'].items()}
    expected.update({r['image']:r['sha256'] for r in rows(ROOT/'training/runs/e1-full/data.jsonl')})
    assert pc['identity_sha256']==digest(dict(algorithm='phash-dct32-low8-median-including-dc-exif-v1',files=expected))
    wiki_audit=read(SERIES/'wiki_v2/component_audit.json')
    extra_path=ROOT/'training/runs/tinyclip-wiki-pilot-v1/data/additional_protected_phashes.json'
    assert sha256(phash_path)==wiki_audit['protected_phash_inputs']['training\\runs\\clean-v1\\perceptual_hashes.json']
    assert sha256(extra_path)==wiki_audit['protected_phash_inputs']['training\\runs\\tinyclip-wiki-pilot-v1\\data\\additional_protected_phashes.json']
    evidence[str(phash_path)]=sha256(phash_path);evidence[str(extra_path)]=sha256(extra_path)
    cache={**pc['hashes'],**read(extra_path)['hashes']}
    # Reuse VN fingerprints only after checking immutable candidate manifest and
    # every original parquet shard SHA; embedding cache is never identity proof.
    extraction=read(ROOT/'training/runs/tinyclip-visualnews-v1/data/extraction.json')
    candidates_path=ROOT/'training/runs/tinyclip-visualnews-v1/data/candidates.jsonl'
    assert sha256(candidates_path)==extraction['files']['candidates.jsonl']
    vn_candidates={r['id']:r for r in rows(candidates_path)}
    for p,h in extraction['verified_shards'].items():
        assert sha256(p)==h,p
    evidence['VN_verified_shards']=extraction['verified_shards']
    print('HISTORICAL_SHARDS_VERIFIED',len(extraction['verified_shards']),flush=True)
    checked_raw={};missing=[]
    for r in train+protected:
        if r.get('dataset')=='visualnews':
            saved=vn_candidates[r['id']]
            assert all(saved[k]==r[k] for k in ('sha256','pixels_sha256','phash','shard','shard_row'))
            r['verified_location']=dict(path=r['shard'],row=r['shard_row'],image_did=r['image_did'])
            r['fingerprint_evidence']='verified historical extraction manifest + current original shard SHA'
        elif r.get('image') and local_path(r['image']).is_file():
            p=r['image']
            if p not in checked_raw:
                assert sha256(p)==r['sha256'],p
                checked_raw[p]=True
            if not r.get('phash'):
                if p in cache:r['phash']=cache[p]
                else:r.update(fingerprints(path=local_path(p)))
            r['verified_location']=dict(path=str(local_path(p)))
            r['fingerprint_evidence']='current raw SHA; historical manifest-bound pHash or new single decode'
        elif r['_group']=='Wiki_reserved':
            r['fingerprint_evidence']='immutable enforced Wiki guard with original verification evidence'
        else:missing.append(dict(id=r.get('id'),group=r['_group'],image=r.get('image')))
        r['_node']=graph.add(r,[r['_group']])
    if missing:
        atomic_json(REPORT/'historical_asset_gaps.json',dict(missing=missing))
        raise RuntimeError('Historical original/fingerprint gaps prevent universe freeze: '+str(len(missing)))
    print('PROTECTED_COMPONENTS_ADDED',len(train),len(protected),flush=True)
    exports=read(SERIES/'expanded_corpus_metadata_audit_20261029.json')['inputs']
    availability=read(REPORT/'original_availability.json')
    originals={r['id']:r for r in availability['unprotected_local_candidates']}
    # All export metadata participates in transitive closure, even where originals
    # are unavailable. Never claim undecoded records are admitted training images.
    selected=[];counts=Counter();wiki_records=[];vn_meta={};vn_nodes={}
    for kind,item in exports.items():
        for line,r in enumerate(rows(local_path(item['path'])),1):
            record={k:r[k] for k in ('id','text_en','text_pl','original_en','source_caption','accepted_text','source_image','image_did','source_record','route') if k in r}
            node=graph.add(record);counts[kind+'_metadata_rows']+=1
            selected.append((kind,r['id'],node,line, r.get('source_image')))
            if kind=='visualnews':
                vn_meta[r['id']]={k:r[k] for k in ('id','source_image','image_did','original_en','text_en','text_pl','route')}
                vn_nodes[r['id']]=node
            if kind=='wikimedia' and r['id'] in originals:
                p=Path(originals[r['id']]['paths'][0]);raw=p.read_bytes()
                assert hashlib.sha1(raw).hexdigest()==originals[r['id']]['sha1']
                r.update(fingerprints(raw=raw));r['image']=str(p);r['target_id']=r['pixels_sha256']
                source=r['source_record'];r['source_sha1_hex']=originals[r['id']]['sha1']
                r['parent_ids']=['commons-file:'+source['source_file_id']]+['wikidata:'+q for q in {source.get('wikidata_qid')}|{a.get('wikidata_qid') for a in source.get('associations',[])} if q]
                r['lineage_verified']=True;r['_node']=graph.add(r);graph.join(node,r['_node'])
                r['verified_location']=dict(path=str(p));r['fingerprint_evidence']='current original SHA1 against frozen source; one decode raw/pixel/pHash'
                wiki_records.append(r)
    # Official caption -> original image joins and all article aliases. No DB.
    assets=local_path('C:/Users/jsowa/benchmark-translations/visualnews-image-cache')
    asset_manifest=read(assets/'manifest.json')
    assert sha256(assets/'original-data.json')==asset_manifest['metadata_sha256']
    sys.path.insert(0,str(ROOT/'experiments'))
    from visualnews_image_assets import json_array_items, format_caption
    pool=ROOT/'.cache/visualnews-image-sources'
    official={r['did']:r['img_path'] for r in rows(pool/'mbeir_visualnews_task0_cand_pool.jsonl')}
    official_text={r['did']:r['txt'] for r in rows(pool/'mbeir_visualnews_task3_cand_pool.jsonl')}
    evidence['official_image_pool']=artifact('official_image_pool',pool/'mbeir_visualnews_task0_cand_pool.jsonl')
    evidence['official_text_pool']=artifact('official_text_pool',pool/'mbeir_visualnews_task3_cand_pool.jsonl')
    by_path=defaultdict(list);by_caption=defaultdict(list);joins=set();articles=defaultdict(set)
    for uid,r in vn_meta.items():
        assert official[r['image_did']]==r['source_image']
        assert official_text[uid.removeprefix('MBeirvisualnews:')]==r['original_en']
        by_path[r['source_image']].append(uid);by_caption[r['original_en']].append(uid)
    for r in train+protected:
        if r.get('source_image') and r.get('article_ids'):
            graph.add(dict(source_image=r['source_image'],article_ids=r['article_ids']))
    for record in json_array_items(assets/'original-data.json'):
        path='mbeir_images/visualnews_images/'+record['image_path'].replace('\\','/').removeprefix('./')
        article=record.get('article_path')
        if article:article=str(article).removeprefix('./')
        if path in by_path:
            if article:
                graph.add(dict(source_image=path,article_ids=[article]));articles[path].add(article)
            caption=format_caption(record.get('caption'))
            for uid in by_caption.get(caption,[]):
                if vn_meta[uid]['source_image']==path:joins.add(uid)
    assert joins==set(vn_meta),'Official VN caption/image mapping incomplete'
    evidence['VN_original_metadata_sha256']=asset_manifest['metadata_sha256']
    print('VN_OFFICIAL_JOINS',len(joins),flush=True)
    # Evaluate after adding every row: aliases through missing-original records
    # still reserve their complete metadata components.
    wiki=[];reasons=Counter()
    for r in wiki_records:
        labels=graph.reasons(r['_node'])
        if labels:reasons.update(labels)
        else:wiki.append(r)
    summary=dict(status='metadata_closure_and_local_Wiki_image_audit_complete',
        all_export_metadata=dict(counts),historical_train_rows=len(train),protected_image_rows=len(protected),
        current_raw_files_verified=len(checked_raw),all_historical_VN_shards_verified=len(extraction['verified_shards']),
        local_Wiki_not_directly_reserved=len(wiki_records),Wiki_after_full_transitive_closure=len(wiki),
        Wiki_component_rejection_labels=dict(reasons),evidence=evidence,elapsed_seconds=time.monotonic()-start,
        training_ready=False,universe_frozen=False)
    atomic_json(REPORT/'data_audit_progress.json',summary)
    write_rows(STORAGE/'wiki_audited_candidates.jsonl',[dict(r,component=graph.find(r['_node'])) for r in wiki])
    print('WIKI_CLOSURE',len(wiki),'REASONS',dict(reasons),flush=True)
    # Persist metadata selection for the deterministic VN audit, never model results.
    clean=[(kind,uid,node,line,path) for kind,uid,node,line,path in selected if kind=='visualnews' and not graph.reasons(node)]
    groups=defaultdict(list)
    for entry in clean:groups[graph.find(entry[2])].append(entry)
    ordered=sorted(groups.values(),key=lambda g:hashlib.sha256(('data-ood-20261030:pilot:'+min(e[4] for e in g)).encode()).hexdigest())
    chosen=[];n=0
    for group in ordered:
        images=len({e[4] for e in group})
        if n+images<=20000:chosen+=group;n+=images
    atomic_json(REPORT/'VN_pilot_selection.json',dict(status='metadata_selected_before_model_results',
        candidates=len(clean),selected_rows=len(chosen),selected_source_images=n,
        selection='whole metadata components ordered by sha256(data-ood-20261030:pilot: + smallest canonical source path)',
        entries=[dict(id=e[1],source_line=e[3],source_image=e[4]) for e in chosen]))
    print('VN_SELECTED',len(chosen),n,flush=True)
    reserve('A','small-code-and-audit',{'C':256*1024**2,'D':2*1024**2})
    # Full original shard mirror is already local. Verify every shard used here;
    # stream small batches so neither embedded images nor full metadata grow RAM.
    import pyarrow.parquet as pq
    requested={vn_meta[e[1]]['image_did']:vn_meta[e[1]] for e in chosen}
    new_vn=[];found=set();decode_failures=[]
    for item in asset_manifest['image_shards']:
        shard=local_path(item['path'])
        if str(shard) not in extraction['verified_shards']:
            assert sha256(shard)==item['sha256'],str(shard)
        evidence.setdefault('new_VN_verified_shards',{})[str(shard)]=item['sha256']
        offset=0
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=128,columns=['did','img_path','img']):
            for j,asset in enumerate(batch.to_pylist()):
                if asset['did'] not in requested:continue
                r=dict(requested[asset['did']]);assert asset['img_path']==r['source_image']
                assert official[asset['did']]==asset['img_path']
                found.add(r['id'])
                try:r.update(fingerprints(raw=asset['img']['bytes']))
                except Exception as exc:
                    decode_failures.append(dict(id=r['id'],reason=type(exc).__name__+': '+str(exc)));continue
                r.update(target_id=r['pixels_sha256'],image='parquet-original://'+r['image_did'],
                    shard=str(shard),shard_row=offset+j,article_ids=sorted(articles[r['source_image']]),
                    parent_ids=['visualnews-image:'+r['image_did'],'visualnews-path:'+r['source_image']],
                    lineage_verified=bool(articles[r['source_image']]),dataset='visualnews',
                    verified_location=dict(path=str(shard),row=offset+j,image_did=r['image_did']),
                    fingerprint_evidence='verified official caption/path/DID + shard SHA; new original single decode',
                    source_revision=asset_manifest['image_revision'])
                r['_node']=graph.add(r);graph.join(r['_node'],vn_nodes[r['id']]);new_vn.append(r)
            offset+=len(batch)
        print('VN_SHARD',shard.name,len(found),'/',len(requested),flush=True)
    assert found=={e[1] for e in chosen}
    eligible_vn=[r for r in new_vn if not graph.reasons(r['_node']) and r['lineage_verified']]
    # VN pHash/source aliases can propagate into Wiki, so finalize both together.
    wiki=[r for r in wiki_records if not graph.reasons(r['_node'])]
    vn_reject=Counter(label for r in new_vn for label in graph.reasons(r['_node']))
    wiki_reject=Counter(label for r in wiki_records for label in graph.reasons(r['_node']))
    def final(r,role):
        return {**{k:v for k,v in r.items() if not k.startswith('_')},
            'role':role,'canonical_image_id':r.get('pixels_sha256'),
            'normalized_texts':sorted({normalized(r[f]) for f in ('text_en','text_pl','original_en','source_caption') if r.get(f)}),
            'component':str(graph.find(r['_node']))}
    # Distinct new components include all available rows for those images. No
    # replacement of historical captions. Native/recycled provenance is retained.
    manifest=STORAGE/'new_VN_train.jsonl';wiki_manifest=STORAGE/'Wiki_train.jsonl'
    write_rows(manifest,[final(r,'new_VN_train') for r in eligible_vn])
    write_rows(wiki_manifest,[final(r,'Wiki_train') for r in wiki])
    write_rows(STORAGE/'training_universe.jsonl',[final(r,r['_group']) for r in train]+
               [final(r,'new_VN_train') for r in eligible_vn]+[final(r,'Wiki_train') for r in wiki])
    write_rows(STORAGE/'protected_index.jsonl',[final(r,r['_group']) for r in protected]+[final(r,r['_group']) for r in train])
    summary.update(status='image_audit_complete',training_ready=bool(eligible_vn) and bool(wiki),
        new_VN_selected_images=n,new_VN_decoded=len(new_vn),new_VN_accepted=len(eligible_vn),
        new_VN_rejection_labels=dict(vn_reject),decode_failures=decode_failures,
        Wiki_accepted=len(wiki),Wiki_rejection_labels=dict(wiki_reject),
        Wiki_provenance_routes=dict(Counter(r['route'] for r in wiki)),
        protected_metadata_texts_and_ids_in_guard_and_blacklists=True,
        no_gold_labels_created_from_phash=True,evidence=evidence,elapsed_seconds=time.monotonic()-start)
    atomic_json(REPORT/'image_audit.json',summary)
    if not summary['training_ready']:raise RuntimeError('No eligible VN or Wiki image pool; training remains gated')
    publish('A_UNIVERSE_FROZEN',[artifact('training_universe',STORAGE/'training_universe.jsonl'),
        artifact('protected_index',STORAGE/'protected_index.jsonl'),artifact('image_audit',REPORT/'image_audit.json'),
        artifact('study_registration',REPORT/'registration.json'),artifact('new_VN_train',manifest),artifact('Wiki_train',wiki_manifest)],
        counts=dict(old_train=len(train),new_VN=len(eligible_vn),Wiki=len(wiki)),
        normalized_text_guard_artifacts=[artifact('Wiki_guard',SERIES/'wiki_dev_guard_enforcement.json'),
            artifact('legacy_blacklist',ROOT/'training/runs/e1-full/blacklist.json'),
            artifact('XM_blacklist_only',ROOT/'training/runs/xm3600-pl/blacklist.json')])
    print('UNIVERSE_FROZEN',len(eligible_vn),len(wiki),flush=True)


if __name__=='__main__':main()
