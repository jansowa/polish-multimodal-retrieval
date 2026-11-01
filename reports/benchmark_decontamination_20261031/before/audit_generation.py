"""Full generation-2 closure, original fingerprints, deterministic Wiki pilot."""
import hashlib
from collections import Counter,defaultdict
from config import *
from guard import Components,normalized
from audit import fingerprints


def main():
    marker=COORD/'A_UNIVERSE_FROZEN.json'
    if marker.exists():
        from common import verify_marker
        verify_marker(marker);return read(REPORT/'image_audit.json')
    reserve();status('auditing_originals',GPU_planned_after_audit=True)
    graph=Components();protected=[]
    # Reuse historical fingerprints only after checking the immutable generation-1
    # marker and all its bound artifacts. No embedding cache is identity evidence.
    from common import verify_marker
    previous=verify_marker(COORD.parent/'A_UNIVERSE_FROZEN.json')
    for r in rows(OLD_STORE/'protected_index.jsonl'):
        node=graph.add(r,['protected_or_existing_train'])
        if r.get('component') is not None:
            other=graph.add(dict(parent_ids=['generation1-component:'+str(r['component'])]))
            graph.join(node,other)
        protected.append(r)
    guard=read(SERIES/'wiki_dev_guard_enforcement.json')
    for key in ('ids','texts','parent_ids','source_sha1_hex'):
        for value in guard.get(key,[]):
            r={'ids':dict(id=value),'texts':dict(text_pl=value),'parent_ids':dict(parent_ids=[value]),
               'source_sha1_hex':dict(source_sha1_hex=value)}[key]
            graph.add(r,['Wiki_reserved'])
    for rel in ('training/runs/e1-full/blacklist.json','training/runs/xm3600-pl/blacklist.json'):
        v=read(ROOT/rel)
        for t in v['texts']:graph.add(dict(text_pl=t),['protected_query_text'])
        for uid in v['ids']:graph.add(dict(id=uid),['protected_source_id'])
    # Keep fixed VN from generation 1. All export metadata participates in closure.
    vn_export=local_path('C:/Users/jsowa/benchmark-translations/series-2026-10-23/expanded-inputs-20261029/visualnews_final_pairs.jsonl')
    assert sha256(vn_export)=='5cbb3b399d77029f1758619232b186b6d1e5b7a304df51b1a4936bd50c7df6f5'
    for r in rows(vn_export):graph.add(r)
    for r in rows(OLD_STORE/'new_VN_train.jsonl'):graph.add(r,['fixed_new_VN_train'])
    records=[];ready=[];missing=0;decode_failures=[];route_counts=Counter()
    for i,row in enumerate(rows(EXPORT)):
        src=row['source_record'];h=source_sha1(src.get('source_sha1'));receipt=STORE/'compact_receipts'/(h+'.json')
        record=dict(row);node=graph.add(record);record['_node']=node;records.append(record)
        route_counts[row['route']]+=1
        value=read(receipt) if receipt.exists() else {}
        if value.get('status')!='ready':missing+=1;continue
        path=Path(value['native_path'])
        try:
            assert value['original_sha1']==h and value['preprocessing_bit_exact']
            assert sha256(path)==value['native_sha256']
            assert sha256(value['master_path'])==value['master_sha256']
            fp={k:value[k] for k in ('sha256','pixels_sha256','phash')}
            qs={src.get('wikidata_qid')}|{a.get('wikidata_qid') for a in src.get('associations',[])}
            parents=['commons-file:'+str(src['source_file_id'])]+['wikidata:'+q for q in qs if q]
            record.update(fp,image=str(path),training_sha256=value['native_sha256'],target_id=fp['pixels_sha256'],source_sha1_hex=h,parent_ids=parents,
                lineage_verified=True,dataset='wikimedia',verified_location=dict(path=str(path)),
                fingerprint_evidence='Original stream SHA1, raw/pixel SHA256, pHash before resize; pinned exact224 preprocessing proof; derivative SHA reverified',
                original_receipt=artifact('original_verification',receipt),master_image=value['master_path'],
                role='Wiki_train',canonical_image_id=fp['pixels_sha256'])
            enriched=graph.add(record);graph.join(node,enriched);ready.append(record)
        except Exception as exc:
            decode_failures.append(dict(id=row['id'],reason=type(exc).__name__+': '+str(exc)))
        if i%250==0:
            status('auditing_originals',rows_scanned=i,verified_original_rows=len(ready),missing_original_rows=missing)
            print('AUDIT_PROGRESS',i,len(ready),flush=True)
    groups=defaultdict(list)
    for r in ready:
        if not graph.reasons(r['_node']):groups[graph.find(r['_node'])].append(r)
    # A selected whole component must have verified images for every exported row
    # belonging to it. Missing or decode-failed aliases never silently enter train.
    total_by_component=Counter(graph.find(r['_node']) for r in records)
    eligible=[group for root,group in groups.items() if len(group)==total_by_component[root]]
    eligible.sort(key=lambda group:hashlib.sha256(('wiki-recovery-20261031:'+','.join(sorted(r['source_sha1_hex'] for r in group))).encode()).hexdigest())
    selected=[];image_ids=set();kept_components=0
    for group in eligible:
        ids={r['pixels_sha256'] for r in group}
        if len(image_ids|ids)>MAX_WIKI:continue
        selected+=group;image_ids.update(ids);kept_components+=1
    rejection=Counter(label for r in ready for label in graph.reasons(r['_node']))
    summary=dict(status='image_audit_complete',generation=2,export_rows=len(records),download_ready_rows=len(ready),
        missing_original_rows=missing,decode_failures=decode_failures,metadata_routes=dict(route_counts),
        rejection_labels=dict(rejection),eligible_complete_components=len(eligible),
        eligible_unique_images=len({r['pixels_sha256'] for g in eligible for r in g}),
        selected_Wiki_rows=len(selected),selected_Wiki_images=len(image_ids),selected_Wiki_components=kept_components,
        selected_routes=dict(Counter(r['route'] for r in selected)),minimum_gate_images=MIN_WIKI,
        minimum_gate_pass=len(image_ids)>=MIN_WIKI,previous_universe_sha256=sha256(COORD.parent/'A_UNIVERSE_FROZEN.json'))
    atomic_json(REPORT/'image_audit.json',summary)
    if not summary['minimum_gate_pass']:
        raise RuntimeError(f'Only {len(image_ids)} eligible Wiki images; require at least {MIN_WIKI}. No training.')
    def clean(r):
        return dict({k:v for k,v in r.items() if not k.startswith('_')},
            component='g2:'+str(graph.find(r['_node'])),
            normalized_texts=sorted({normalized(r[f]) for f in ('text_en','text_pl','source_caption') if r.get(f)}))
    selected=sorted(selected,key=lambda r:r['id'])
    write_rows(STORE/'Wiki_train.jsonl',[clean(r) for r in selected])
    # Manifest copies only; original image assets remain in one location.
    write_rows(STORE/'new_VN_train.jsonl',rows(OLD_STORE/'new_VN_train.jsonl'))
    write_rows(STORE/'protected_index.jsonl',protected)
    write_rows(STORE/'training_universe.jsonl',__import__('itertools').chain(rows(OLD_STORE/'training_universe.jsonl'),[clean(r) for r in selected]))
    active=local_path(read(REPORT/'active_catalog.json')['path'])
    publish('A_INPUTS_FROZEN',[artifact('Wiki_export',EXPORT),artifact('download_catalog',active),
        artifact('download_summary',REPORT/'download_summary.json'),artifact('registration',REPORT/'registration.json')])
    publish('A_UNIVERSE_FROZEN',[artifact('training_universe',STORE/'training_universe.jsonl'),
        artifact('protected_index',STORE/'protected_index.jsonl'),artifact('image_audit',REPORT/'image_audit.json'),
        artifact('study_registration',REPORT/'registration.json'),artifact('archive_policy',REPORT/'archive_policy.json'),artifact('new_VN_train',STORE/'new_VN_train.jsonl'),
        artifact('Wiki_train',STORE/'Wiki_train.jsonl'),artifact('Wiki_guard',SERIES/'wiki_dev_guard_enforcement.json')],
        generation_1_union_retained=True,requires_new_B_overlap_audit=True)
    reserve();print('GENERATION_2_UNIVERSE_FROZEN',len(image_ids),kept_components,flush=True)
    return summary


if __name__=='__main__':main()
