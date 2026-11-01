"""Read-only generation-2 OOD audit against the exact checkpoint train manifest."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'experiments/ood_eval_20261030')]
from retrieval_benchmark.core import HERE,local_path,read,sha,write
from leakage import ProtectedIndex


def main():
    checksum='094122a574d8f2322b1434bb0eb0a5d539b510c36b5cfc74c2eb140c0e1f7636'
    marker=ROOT/'reports/data_ood_coordination_20261030/generation_2/A_FINALISTS_FROZEN.json'
    model=next(m for m in read(marker)['models'] if m['sha256']==checksum)
    training=local_path(model['training_manifest_path'])
    if sha(training)!=model['training_manifest_sha256'] or sha(local_path(model['path']))!=checksum:
        raise ValueError('Frozen checkpoint/training manifest changed')
    registry=read(HERE/'holdouts/registry-v2.json')
    evidence=next(item for item in registry['inputs'] if item['key']=='verified_original_fingerprints')
    cache=local_path(evidence['path'])
    if sha(cache)!=evidence['sha256']:raise ValueError('Original fingerprint evidence changed')
    phashes={}
    for line in cache.open(encoding='utf-8'):
        if not line.strip():continue
        row=json.loads(line)
        if row.get('phash'):
            for field in ('sha256','pixels_sha256'):
                if row.get(field):phashes[row[field]]=row['phash']
    index=ProtectedIndex();count=0
    for line in training.open(encoding='utf-8'):
        if not line.strip():continue
        row=json.loads(line)
        fingerprint=row.get('phash') or phashes.get(row.get('sha256')) or phashes.get(row.get('pixels_sha256'))
        if not fingerprint:raise ValueError('Missing original training-image pHash: '+str(row.get('id')))
        index.add(dict(row,phash=fingerprint));count+=1
    config=read(HERE/'local/config.json');datasets={}
    for name in ('flickr30k_pl_local_test','vizwiz_pl_local_validation','xm3600_pl'):
        path=local_path(config['datasets'][name]['image_manifest'])
        rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()] if path.suffix=='.jsonl' else read(path)
        matches=[]
        for row in rows:
            reasons=index.reasons(row)
            if reasons:matches.append(dict(id=row['id'],image=row['image'],reasons=reasons))
        datasets[name]=dict(manifest_sha256=sha(path),caption_rows=len(rows),
            matched_rows=len(matches),matched_images=len({r['image'] for r in matches}),matches=matches)
    proof=dict(status='passed' if not any(d['matched_rows'] for d in datasets.values()) else 'overlap_found',
        checkpoint_sha256=checksum,finalists_marker_sha256=sha(marker),
        training_manifest_sha256=sha(training),training_rows=count,original_phash_rows=count,
        fingerprint_cache_sha256=evidence['sha256'],datasets=datasets,
        method='All actual train rows: source IDs/aliases, normalized EN/PL texts, original raw/pixel hashes and pHash<=6',
        galleries_changed=False,pretraining_exposure='unknown')
    out=local_path('C:/Users/jsowa/benchmark-translations/retrieval-benchmark-v1/wiki2-r3-seed43-20261031/ood_overlap_audit_v2.json')
    write(out,proof,immutable=True)
    print(json.dumps(dict(status=proof['status'],training_rows=count,
        datasets={k:{f:v[f] for f in ('caption_rows','matched_rows','matched_images')} for k,v in datasets.items()})),flush=True)


if __name__=='__main__':main()
