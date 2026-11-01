"""Freeze growing exports and audit metadata components; never declare image audit passed."""
import hashlib
import json
import os
import sys
from collections import Counter,defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pl_mmb_training.series_20261023 import ROOT,OUT,VN,read,rows,freeze,file_sha,require_space
from pl_mmb_training.leakage import normalized_text

TARGET=Path('/mnt/c/Users/jsowa/benchmark-translations/series-2026-10-23/expanded-inputs-20261029')


def text_token(value):
    return 'text:'+hashlib.sha256(normalized_text(value).encode()).hexdigest()


def sha1(value):
    try:return f'{int(value,16 if len(value)==40 else 36):040x}'
    except (ValueError,TypeError):return None


def tokens(row,dataset):
    values={'id:'+row['id']}
    for key in ('text_en','text_pl','original_en','accepted_text','source_caption'):
        if row.get(key):values.add(text_token(row[key]))
    for parent in row.get('parent_ids',[]):values.add('parent:'+parent)
    for key in ('sha256','pixels_sha256'):
        if row.get(key):values.add('content:'+row[key])
    if row.get('source_image'):values.add('vn_image:'+row['source_image'])
    if row.get('image_did'):values.add('vn_did:'+str(row['image_did']))
    source=row.get('source_record',{})
    if dataset=='wikimedia' or source:
        for identifier in (source.get('id'),source.get('source_file_id')):
            if identifier:values.add('id:'+str(identifier))
        identity=sha1(row.get('source_sha1_hex') or source.get('source_sha1') or '')
        if identity:values.add('source_sha1:'+identity)
        if source.get('source_page_url'):values.add('page:'+source['source_page_url'])
    return values


def main():
    require_space(['/mnt/c','/mnt/d','/'])
    sources={kind:ROOT/f'reports/{kind}_pipeline/v1/final_pairs.jsonl' for kind in ('visualnews','wikimedia')}
    require_space(['/mnt/c'],expected_bytes=sum(p.stat().st_size for p in sources.values())+256*2**20)
    decision=OUT/'expanded_corpus_preparation_decision_20261029.json'
    freeze(decision,dict(status='registered_before_new_corpus_preparation',scope='freeze exports and metadata-only component audit',
        authorization='User asked to resume work including preparation of the larger corpus on 2026-10-29',
        source_exports={k:str(p) for k,p in sources.items()},output=str(TARGET),
        script_sha256=file_sha(__file__),no_live_SQLite=True,no_training=True,
        old_VN_snapshot_unchanged=True,old_FULL_CLEAN_unchanged=True,
        image_hash_phash_lineage_and_license_audits_still_required=True,
        no_new_Wiki_training_release=True,automatic_deletion=False))
    TARGET.mkdir(parents=True,exist_ok=True)
    inputs_path=TARGET/'inputs_complete.json'
    if inputs_path.exists():
        inputs=read(inputs_path)
        for value in inputs.values():
            if file_sha(Path(value['path']))!=value['sha256']:raise ValueError('Frozen input changed')
    else:
        inputs={}
        for kind,source in sources.items():
            target=TARGET/(kind+'_final_pairs.jsonl')
            digest=hashlib.sha256()
            with source.open('rb') as original,target.open('xb') as destination:
                for chunk in iter(lambda:original.read(8*2**20),b''):
                    require_space(['/mnt/c','/mnt/d','/'])
                    require_space(['/mnt/c'],expected_bytes=len(chunk))
                    destination.write(chunk);digest.update(chunk)
                destination.flush();os.fsync(destination.fileno())
            if file_sha(target)!=digest.hexdigest():raise ValueError('Frozen export copy differs')
            inputs[kind]=dict(path=str(target),sha256=digest.hexdigest(),bytes=target.stat().st_size,
                note='single open descriptor of atomically published JSONL; no live database access')
        freeze(inputs_path,inputs)
    protected=defaultdict(set)
    def protect(values,label):
        for value in values:protected[value].add(label)
    for name,path in [('validation',ROOT/'training/runs/clean-v1/validation.jsonl'),('VN_dev',VN/'vn_dev.jsonl')]:
        for row in rows(path):protect(tokens(row,name),name)
    for name,path in [('legacy_eval',ROOT/'training/runs/e1-full/blacklist.json'),('XM_blacklist_only',ROOT/'training/runs/xm3600-pl/blacklist.json')]:
        value=read(path)
        for text in value['texts']:protect([text_token(text)],name)
        for identifier in value['ids']:protect(['id:'+identifier],name)
        for row in value['images'].values():protect(tokens(dict(row,id=row.get('id','')),name)-{'id:'},name)
    guard=read(OUT/'wiki_dev_guard_enforcement.json')
    for key,prefix in [('ids','id:'),('parent_ids','parent:'),('source_sha1_hex','source_sha1:')]:
        protect([prefix+str(v) for v in guard.get(key,[])],'Wiki_dev_reserved_component')
    protect([text_token(t) for t in guard['texts']],'Wiki_dev_reserved_component')
    for row in guard['images'].values():
        protect(tokens(dict(row,id=row.get('id','')),'wikimedia')-{'id:'},'Wiki_dev_reserved_component')
    existing_ids={r['id'] for r in rows(VN/'pipeline_pairs_snapshot.jsonl')}
    known_train_tokens=set()
    for path in (ROOT/'training/runs/clean-v1/FULL_CLEAN.jsonl',VN/'vn_train.jsonl'):
        for row in rows(path):known_train_tokens.update(tokens(row,'train'))
    parent=[];token_owner={};records=[]
    def find(index):
        while parent[index]!=index:
            parent[index]=parent[parent[index]];index=parent[index]
        return index
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[b]=a
    routes=defaultdict(Counter)
    for kind,item in inputs.items():
        with Path(item['path']).open(encoding='utf-8') as stream:
            for line_number,line in enumerate(stream,1):
                if not line.strip():continue
                row=json.loads(line);index=len(parent);parent.append(index)
                values=tokens(row,kind);reasons=set()
                for token in values:
                    reasons.update(protected.get(token,()))
                    if token in token_owner:union(index,token_owner[token])
                    else:token_owner[token]=index
                route=row['route'];routes[kind][route]+=1
                records.append(dict(dataset=kind,id=row['id'],route=route,source_line=line_number,
                    source_export_sha256=item['sha256'],direct_protected_matches=sorted(reasons),
                    already_in_frozen_VN_snapshot=kind=='visualnews' and row['id'] in existing_ids,
                    exact_metadata_link_to_existing_train=bool(values & known_train_tokens)))
    component_reasons=defaultdict(set)
    for index,row in enumerate(records):component_reasons[find(index)].update(row['direct_protected_matches'])
    counts=defaultdict(Counter)
    candidate=TARGET/'metadata_candidates.jsonl'
    with candidate.open('x',encoding='utf-8') as stream:
        for index,row in enumerate(records):
            reasons=sorted(component_reasons[find(index)])
            row.update(provisional_component=find(index),component_protected_matches=reasons,
                status='quarantine_pending_verified_image_hash_phash_lineage_and_license_audit')
            counts[row['dataset']]['rows']+=1
            counts[row['dataset']]['known_protected_component_rows']+=bool(reasons)
            counts[row['dataset']]['existing_VN_snapshot_ID_rows']+=row['already_in_frozen_VN_snapshot']
            counts[row['dataset']]['exact_metadata_link_to_existing_train_rows']+=row['exact_metadata_link_to_existing_train']
            counts[row['dataset']]['provisionally_new_nonprotected_rows']+=not reasons and not row['already_in_frozen_VN_snapshot'] and not row['exact_metadata_link_to_existing_train']
            stream.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    summary=dict(status='metadata_audit_complete_image_audit_pending',inputs=inputs,
        candidate_manifest=str(candidate),candidate_manifest_sha256=file_sha(candidate),
        dataset_counts={k:dict(v) for k,v in counts.items()},routes={k:dict(v) for k,v in routes.items()},
        provisional_components=len({find(i) for i in range(len(parent))}),
        methodology='transitive components over exact IDs, source identities, claimed exact hashes and normalized texts; not a pHash/image verification pass',
        training_ready_rows=0,no_training_allowed_from_this_manifest=True,
        pending=['verified image mapping and local original assets','canonical pixel SHA and pHash<=6 grouping',
            'complete source/parent lineage','license/attribution audit','new training decision; Wiki release gate'],
        guard_sha256=file_sha(OUT/'wiki_dev_guard_enforcement.json'),decision_sha256=file_sha(decision))
    freeze(TARGET/'metadata_audit_summary.json',summary)
    freeze(OUT/'expanded_corpus_metadata_audit_20261029.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
