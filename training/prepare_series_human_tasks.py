"""Extend immutable human-task manifests to the frozen native Wiki holdout."""
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pl_mmb_training.series_20261023 import ROOT,OUT,VN,WIKI,read,rows,freeze,file_sha,deterministic_images,local_path


def sha1hex(value):
    return f"{int(value,16 if len(value)==40 else 36):040x}"


def main():
    wiki=OUT/'wiki_v2';status=read(wiki/'holdout_status.json')
    if status['status']!='frozen_image_verified':raise ValueError('Native Wiki dev unavailable')
    guard=read(OUT/'wiki_dev_guard.json')
    wanted=set(guard['ids']);source_records={}
    for path in (wiki/'candidates.jsonl',WIKI/'wiki_pairs_snapshot.jsonl'):
        for r in rows(path):
            if r['id'] in wanted:source_records[r['id']]=r['source_record']
    enforcement=dict(guard,source_sha1_hex=sorted({sha1hex(r['source_sha1']) for r in source_records.values()}),
        original_guard_sha256=file_sha(OUT/'wiki_dev_guard.json'))
    freeze(OUT/'wiki_dev_guard_enforcement.json',enforcement)
    train=rows(wiki/'pilot_wiki_train_disjoint.jsonl')
    train_ids={r['id'] for r in train}
    raw=[r for r in rows(WIKI/'wiki_pairs_snapshot.jsonl') if r['id'] in train_ids]
    raw_sha=freeze(wiki/'pilot_pairs_disjoint_snapshot.jsonl',raw,jsonl=True)
    freeze(wiki/'pilot_snapshot_v2.json',dict(status='frozen_disjoint_train_pool',rows=len(raw),
        pairs_sha256=raw_sha,train_manifest_sha256=file_sha(wiki/'pilot_wiki_train_disjoint.jsonl'),
        prior_original_snapshot_rows=1322,prior_original_snapshot_sha256=file_sha(WIKI/'wiki_pairs_snapshot.jsonl'),
        dev_guard_sha256=file_sha(OUT/'wiki_dev_guard_enforcement.json'),training_allowed=False,
        removed_prior_train_rows=status['removed_pilot_train_rows']))
    folder=OUT/'annotation_v2';folder.mkdir(exist_ok=True)
    query=rows(OUT/'annotation/query_tasks.jsonl');audit=rows(OUT/'annotation/audit_tasks.jsonl')
    chosen=deterministic_images(rows(wiki/'wiki_dev.jsonl'),150,'20261023:short-query:Wiki')
    for r in chosen:
        query.append(dict(id=r['id'],target_id=r['target_id'],component=r['component'],dataset='Wiki',
            source='wikimedia',split='wiki_native_dev',asset='wiki_original',image=r['image'],
            sha256=r['sha256'],asset_view='verified original image'))
    audit_rows=deterministic_images(train,100,'20261023:audit:Wiki-outside-dev')
    for r in audit_rows:
        if r['id'] in wanted:raise ValueError('Wiki audit overlaps reserved dev component')
        preview=local_path(r['preview'])
        if not preview.is_file():raise ValueError('Frozen pilot preview missing')
        audit.append(dict(id=r['id'],target_id=r['target_id'],component=r['component'],dataset='Wiki',
            source='wikimedia',split='outside_native_dev',asset='wiki_pilot_preview',image=r['preview'],
            sha256=file_sha(preview),text_pl=r['text_pl'],asset_view='frozen pilot preview 224x224'))
    query_sha=freeze(folder/'query_tasks.jsonl',query,jsonl=True)
    audit_sha=freeze(folder/'audit_tasks.jsonl',audit,jsonl=True)
    freeze(folder/'asset_verification.json',read(OUT/'annotation/asset_verification.json'))
    sampling=dict(status='VN_and_Wiki_ready',query=dict(VN=150,Wiki=150,manifest_sha256=query_sha),
        audit=dict(VN=100,Wiki=100,manifest_sha256=audit_sha),
        query_captions_hidden=True,no_model_suggestions=True,production_annotations_created=0,
        wiki_dev_sha256=status['dev_manifest_sha256'],wiki_guard_sha256=file_sha(OUT/'wiki_dev_guard_enforcement.json'),
        original_VN_sampling_sha256=file_sha(OUT/'annotation/sampling.json'),
        Wiki_audit_components_disjoint_from_dev=True,
        server='python experiments/series_annotation.py --port 8768')
    freeze(folder/'sampling.json',sampling)
    print(json.dumps(sampling,indent=2))


if __name__=='__main__':main()
