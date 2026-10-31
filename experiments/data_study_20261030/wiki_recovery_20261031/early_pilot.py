"""Freeze available inputs once; audit/train/report while archive continues."""
import hashlib
import json
import os
import traceback
from collections import Counter
from config import *


def pilot_status(phase,**extra):
    value=dict(schema_version=1,generation=2,owner='A',phase=phase,status='in_progress',updated_at=utc(),**extra)
    atomic_json(REPORT/'TRAINING_STATUS.json',value)
    atomic_json(COORD/'A_STATUS.json',value)


def freeze_available():
    target=STORE/'early_receipts_snapshot.jsonl'
    if target.exists():return
    assert not (COORD/'A_UNIVERSE_FROZEN.json').exists(),'Universe already frozen; no change allowed'
    assert not any((STORE/'runs').rglob('result.json')) if (STORE/'runs').exists() else True
    catalog={r['sha1'] for r in rows(STORE/'download_catalog.jsonl')}
    captured=[];total_bytes=0
    for path in sorted((STORE/'compact_receipts').glob('*.json')):
        if path.stem not in catalog:continue
        with path.open('rb') as stream:payload=stream.read()
        value=json.loads(payload)
        if value.get('status')!='ready':continue
        captured.append(dict(sha1=path.stem,receipt=value,receipt_sha256=hashlib.sha256(payload).hexdigest()))
        total_bytes+=value['stored_bytes']
    assert len(captured)>=MIN_WIKI,'Too few archived inputs even before leakage audit'
    write_rows(target,captured)
    summary=dict(status='available_input_snapshot_frozen_before_training',updated_at=utc(),
        counts={'ready':len(captured)},expected_unique_originals=len(catalog),stored_asset_bytes=total_bytes,
        original_fingerprints_computed_before_resize=True,full_frame_master_edge=1536,native224_lossless=True)
    atomic_json(REPORT/'frozen_download_summary.json',summary,immutable=True)
    decision=dict(status='registered_before_generation2_model_results',generation=2,created_at=utc(),
        authorization='User: Może warto już teraz zaczynać z pierwszymi treningami uwzględniającymi dane z wiki?',
        change='Freeze currently available verified receipts; run existing audit/training stages now while acquisition continues independently.',
        snapshot=artifact('available_receipts',target),archived_candidate_images=len(captured),
        selection_rule_unchanged=read(REPORT/'registration.json')['selection'],minimum_Wiki_images=MIN_WIKI,maximum_Wiki_images=MAX_WIKI,
        no_adding_training_data_after_universe_freeze=True,original_experiments_unchanged=True,
        later_downloads='Archive only; never enter this closed model comparison',no_new_B_results_read=True)
    atomic_json(REPORT/'early_start_decision.json',decision,immutable=True)
    print('AVAILABLE_RECEIPTS_FROZEN',len(captured),flush=True)


def main():
    with lock(STORE/'early_pilot.lock',blocking=False):
        freeze_available()
        import audit_generation
        import train_generation
        import finish
        audit_generation.status=pilot_status;train_generation.status=pilot_status
        audit_generation.main()
        train_generation.main()
        finish.main()
        print('EARLY_PILOT_COMPLETE_ARCHIVE_CONTINUES',flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json(REPORT/'EARLY_FAILURE.json',dict(status='failed',traceback=traceback.format_exc(),updated_at=utc()))
        raise
