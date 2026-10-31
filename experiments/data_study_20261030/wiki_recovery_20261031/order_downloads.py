"""Before bulk acquisition: keep metadata components contiguous under disk cap."""
import hashlib
from collections import defaultdict
from config import *
from guard import Components


def main():
    path=STORE/'download_catalog_components.jsonl'
    if path.exists():return
    graph=Components()
    for r in rows(OLD_STORE/'protected_index.jsonl'):
        r.pop('phash',None);graph.add(r,['protected'])
    g=read(SERIES/'wiki_dev_guard_enforcement.json')
    for field in ('ids','texts','parent_ids','source_sha1_hex'):
        for v in g[field]:
            row={'ids':{'id':v},'texts':{'text_pl':v},'parent_ids':{'parent_ids':[v]},'source_sha1_hex':{'source_sha1_hex':v}}[field]
            graph.add(row,['protected'])
    nodes={}
    for r in rows(EXPORT):nodes[source_sha1(r['source_record']['source_sha1'])]=graph.add(r)
    groups=defaultdict(list)
    for item in rows(STORE/'download_catalog.jsonl'):
        root=graph.find(nodes[item['sha1']]);groups[root].append(item)
    ordered=[]
    for root,group in groups.items():
        key=hashlib.sha256(('wiki-recovery-20261031:'+','.join(sorted(r['sha1'] for r in group))).encode()).hexdigest()
        protected=bool(graph.reasons(root))
        for item in sorted(group,key=lambda r:r['sha1']):
            ordered.append(dict(item,metadata_component=key,metadata_component_size=len(group),metadata_protected=protected))
    ordered.sort(key=lambda r:(r['metadata_protected'],r['metadata_component'],r['sha1']))
    write_rows(path,ordered)
    decision=dict(status='registered_before_bulk_download_and_model_results',
        reason='Observed originals can exceed total 51GB free; prioritize complete unprotected metadata components so capacity cutoff does not fragment every component.',
        sample_originals_retained=True,no_model_results_used=True,unchanged_export_sha256=sha256(EXPORT),
        catalog=artifact('component_ordered_catalog',path),metadata_components=len(groups),
        unprotected_rows=sum(not r['metadata_protected'] for r in ordered))
    atomic_json(REPORT/'component_download_order.json',decision,immutable=True)
    atomic_json(REPORT/'active_catalog.json',decision['catalog'])
    print('COMPONENT_ORDER_READY',decision['metadata_components'],decision['unprotected_rows'],flush=True)


if __name__=='__main__':main()
