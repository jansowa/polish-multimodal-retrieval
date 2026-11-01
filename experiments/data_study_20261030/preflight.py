"""Freeze exact historical inputs and inventory local Wiki originals. CPU only."""
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/data_ood_common_20261030'))
from common import COORD, artifact, atomic_json, disk_snapshot, local_path, publish, reserve, sha256

REPORT = ROOT / 'reports/data_study_20261030'
SERIES = ROOT / 'reports/series_2026-10-23'
STORAGE = local_path('C:/Users/jsowa/benchmark-translations/data-ood-20261030/A')


def read(p):
    return json.loads(Path(p).read_text())


def rows(p):
    with Path(p).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main():
    start = time.monotonic()
    REPORT.mkdir(parents=True, exist_ok=True)
    STORAGE.mkdir(parents=True, exist_ok=True)
    reserve('A', 'small-code-and-audit', {'C': 8*1024**2, 'D': 2*1024**2})
    registration = dict(status='registered_before_any_new_model_result', generation=1,
        authorization='User 2026-10-30: execute session A; Wiki pilot authorized only after data audit, guard and baseline gates.',
        export_policy='Reuse exact 2026-10-29 frozen 132875 VN / 24125 Wiki; no newer export.',
        pilot=dict(max_new_VN_images=20000,max_Wiki_images=5000,
                   order='sha256(data-ood-20261030:pilot: + canonical source identity), ascending; whole components only',
                   reason='Bound materialization/cache to 8 GiB A; physical D has only about 4.07 GiB free; use local originals only'),
        variants=dict(R0=[50,50,0],R1=[50,50,0],R2=[50,47,3],R3=[50,40,10]),
        variant_columns=['FULL','VN','Wiki'], percentages='pair presentations; FULL historical slots unchanged',
        old_VN_captions='retain immutable original VN train; never overwrite captions by new export',
        seeds_first=[42],confirmation_seeds=[43,44],steps=5500,window=275,batch_size=256,final_batch=145,
        total_presentations=1405780,projector='fresh Linear512to384',encoders_frozen=True,
        optimizer='AdamW lr0.001; head wd0.01; fresh SigLIP sigmoid loss wd0; no augmentation/scheduler',
        precision='Exact approved P1.A cache/head/loss FP32; BF16 encoder extraction only. Verify parity before reuse.',
        checkpoint_selection='VN dev t2i R1, then mean R1, then earliest; every 275 updates including 0',
        Wiki_variant_selection='R2/R3 maximal Wiki dev t2i R1; tie smaller Wiki share then earlier saved update',
        VN_guardrail='drop > 2 percentage points versus R1: fail recommendation, retain results',
        representatives='median VN dev t2i R1 over seeds; tie lower seed; R0 remains original seed44',
        read_B_results=False, XM3600_scoring=False, new_legacy_test=False,
        publication=False, human_annotations_required=False)
    if not (REPORT/'registration.json').exists():
        atomic_json(REPORT/'registration.json', registration, immutable=True)
    preservation = read(SERIES/'preservation_before.json')
    checked=[]
    for name, expected in preservation.items():
        assert sha256(ROOT/name) == expected, name
        checked.append(artifact('legacy_train_inputs' if 'train' in name or 'FULL' in name else 'preserved_snapshot', ROOT/name))
    audit = read(SERIES/'expanded_corpus_metadata_audit_20261029.json')
    exports=[]
    for kind,item in audit['inputs'].items():
        p=local_path(item['path'])
        assert sha256(p)==item['sha256'] and p.stat().st_size==item['bytes']
        count=sum(1 for _ in rows(p))
        assert count==audit['dataset_counts'][kind]['rows']
        exports.append(artifact('VN_export' if kind=='visualnews' else 'Wiki_export',p))
    candidate=local_path(audit['candidate_manifest'])
    assert sha256(candidate)==audit['candidate_manifest_sha256']
    guard=read(SERIES/'wiki_dev_guard_enforcement.json')
    assert sha256(ROOT/guard['manifest_path'])=='e761b644719b8d8795bc56716fa6f67bdc1f5531b9f0e2f76955a1ca91e2c4a3'
    assert sha256(SERIES/'wiki_dev_guard_enforcement.json')==audit['guard_sha256']
    baseline=read(SERIES/'wiki_dev_zero_shot_pre_wiki_training.json')
    assert baseline['never_trained_on_Wiki'] and baseline['checkpoint_sha256']=='4bf977049098ca6d8bcc7abc177183aace8c03c5f08e0ecaf011d82aad5b6844'
    finalists=read(SERIES/'vn_full_finalist_decision.json')
    p=local_path(finalists['checkpoint_path'])
    assert sha256(p)==baseline['checkpoint_sha256']
    safeguards=[artifact('Wiki_guard',SERIES/'wiki_dev_guard_enforcement.json'),artifact('Wiki_dev',ROOT/guard['manifest_path']),
                artifact('pre_Wiki_baseline',SERIES/'wiki_dev_zero_shot_pre_wiki_training.json'),artifact('original_finalist',p)]
    if not (COORD/'A_INPUTS_FROZEN.json').exists():
        publish('A_INPUTS_FROZEN',exports+checked+safeguards+[artifact('metadata_candidate_index',candidate)],
                counts={'VN':132875,'Wiki':24125},export_policy=registration['export_policy'])
    # Inventory only original byte assets; never use resized RGB/previews/embedding caches.
    dirs=[local_path('C:/Users/jsowa/benchmark-translations/series-2026-10-23/wiki-native-dev/source-assets'),
          ROOT/'training/runs/tinyclip-wiki-pilot-v1/data/parity_raw']
    original_by_sha1={}; files=0
    for folder in dirs:
        for p in sorted(folder.glob('*.bin')):
            raw=p.read_bytes(); h=hashlib.sha1(raw).hexdigest()
            original_by_sha1.setdefault(h,[]).append(str(p));files+=1
    protected_sha1=set(guard['source_sha1_hex']);protected_ids=set(guard['ids'])
    local_counts=Counter(); routes=Counter();licenses=Counter();examples=[]; unprotected=[]
    p=local_path(audit['inputs']['wikimedia']['path'])
    for r in rows(p):
        source=r['source_record']; value=source.get('source_sha1') or ''
        try: h=f'{int(value,16 if len(value)==40 else 36):040x}'
        except ValueError: h=''
        available=h in original_by_sha1
        protected=h in protected_sha1 or r['id'] in protected_ids
        local_counts['rows']+=1;local_counts['local_verified_source_sha1']+=available
        local_counts['missing_local_original']+=not available
        local_counts['local_protected_source_identity']+=available and protected
        local_counts['local_not_directly_protected_source_identity']+=available and not protected
        routes[r['route']]+=1;licenses[source.get('license') or 'missing']+=1
        if available and not protected:
            unprotected.append(dict(id=r['id'],sha1=h,paths=original_by_sha1[h],route=r['route']))
        if not available and len(examples)<5:examples.append(dict(id=r['id'],source_sha1=h))
    report=dict(status='original_availability_inventory_complete_not_full_image_audit',
        originals_searched=[str(p) for p in dirs],original_files=files,unique_original_sha1=len(original_by_sha1),
        counts=dict(local_counts),unprotected_local_candidates=unprotected,missing_examples=examples,
        provenance_routes=dict(routes),license_metadata_counts=dict(licenses),
        metadata_is_not_training_clearance=True, no_images_downloaded=True,no_SQLite_access=True,
        protected_inputs_verified=True, baseline_preserved=True, elapsed_seconds=time.monotonic()-start,
        disk=disk_snapshot())
    atomic_json(REPORT/'original_availability.json',report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__': main()
