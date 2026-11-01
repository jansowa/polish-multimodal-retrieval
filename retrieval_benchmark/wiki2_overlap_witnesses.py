"""Explain OOD audit flags without changing galleries or labelling pHash as gold."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'experiments/ood_eval_20261030')]
from retrieval_benchmark.core import HERE,read,local_path,sha,write
from leakage import tokens
from pl_mmb_training.leakage import normalized_text


def main():
    base=local_path('C:/Users/jsowa/benchmark-translations/retrieval-benchmark-v1/wiki2-r3-seed43-20261031')
    audit=read(base/'ood_overlap_audit_v2.json')
    reg=read(HERE/'holdouts/registry-v2.json')
    evidence=next(v for v in reg['inputs'] if v['key']=='verified_original_fingerprints')
    cache=local_path(evidence['path']);assert sha(cache)==evidence['sha256']
    phashes={}
    for line in cache.open(encoding='utf-8'):
        row=json.loads(line)
        for field in ('sha256','pixels_sha256'):
            if row.get(field) and row.get('phash'):phashes[row[field]]=row['phash']
    config=read(HERE/'local/config.json');manifest=local_path(config['datasets']['xm3600_pl']['image_manifest'])
    wanted={r['id'] for r in audit['datasets']['xm3600_pl']['matches']}
    targets=[json.loads(l) for l in manifest.open(encoding='utf-8') if json.loads(l)['id'] in wanted]
    train=local_path('C:/Users/jsowa/benchmark-translations/data-ood-20261030/A/wiki-recovery-20261031/manifests/R3_train.jsonl')
    fields=('text_pl','text_en','original_en','source_caption','accepted_text')
    def texts(row):return {normalized_text(row[k]) for k in fields if row.get(k)}|set(row.get('normalized_texts',[]))
    prepared=[(row,tokens(row),texts(row)) for row in targets];witnesses={r['id']:[] for r in targets}
    seen=set()
    for line in train.open(encoding='utf-8'):
        row=json.loads(line);rt=tokens(row);text=texts(row)
        h=row.get('phash') or phashes.get(row.get('sha256')) or phashes.get(row.get('pixels_sha256'))
        for target,tt,tx in prepared:
            distance=(int(h,16)^int(target['phash'],16)).bit_count()
            if rt.isdisjoint(tt) and distance>6:continue
            hashes=[k for k in ('sha256','pixels_sha256') if row.get(k)==target.get(k)]
            shared_text=sorted(text&tx);parents=sorted(set(row.get('parent_ids',[]))&set(target.get('parent_ids',[])))
            key=(target['id'],row['image'],tuple(shared_text))
            if key in seen:continue
            seen.add(key)
            witnesses[target['id']].append(dict(train_id=row['id'],train_dataset=row.get('dataset'),
                train_image=row['image'],target_image=target['image'],target_text=target['text_pl'],
                train_text=row.get('text_pl'),same_original_hash_fields=hashes,shared_parents=parents,
                shared_normalized_text=shared_text,phash_hamming_distance=distance))
    write(base/'ood_overlap_witnesses.json',dict(checkpoint_sha256=audit['checkpoint_sha256'],
        exact_image_matches=sum(bool(r['same_original_hash_fields']) for values in witnesses.values() for r in values),
        witnesses=witnesses),immutable=True)
    print(json.dumps(witnesses,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
