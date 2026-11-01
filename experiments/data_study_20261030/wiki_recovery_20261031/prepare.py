"""Freeze an executable queue and source inventory; no live database reads."""
import hashlib
from collections import Counter
from config import *


def main():
    for p in (STORE,REPORT,COORD,STORE/'originals',STORE/'receipts',STORE/'logs'):p.mkdir(parents=True,exist_ok=True)
    assert sha256(EXPORT)=='415d2f7373ad41bd4654ac9ab62a2da8ed0619e978c5112a5d68addc07a81b08'
    registration=dict(schema_version=1,generation=2,status='registered_before_download_and_new_training',created_at=utc(),
        authorization='User: koniecznie je pobierz; priority; queue download originals then repeat training experiments overnight; C has 51GB.',
        supersedes='Generation 1 remains immutable; new data require generation-2 universe and new B overlap audit before any new OOD evaluation.',
        scope=['download original Wiki assets for exact frozen 24125-record export','full source/raw/pixel/pHash/text audit',
               'R0/R1 baseline reuse after exact verification; R2/R3 seed42 then chosen Wiki seed43/44; original training recipe and selection'],
        export=artifact('Wiki_export',EXPORT),originals_cap_bytes=RAW_CAP,total_new_bytes_cap=TOTAL_CAP,
        physical_free_floor_each_volume_bytes=4*GIB,shared_limiter='outputs/wikimedia_http_state.json',
        download_workers=1,decode_workers=1,no_SQLite=True,no_deletion=True,
        min_verified_Wiki_images=MIN_WIKI,max_verified_Wiki_images=MAX_WIKI,
        selection='whole eligible components ordered by sha256(wiki-recovery-20261031: + sorted source SHA1 identities), before model results; no replacement images',
        insufficient_data_policy='Stop training if fewer than 1000 distinct verified Wiki images remain; publish failure report, never repeat 28-image study.',
        partial_download_policy='Keep every completed original and explicit failure receipt; capacity-limited deterministic prefix allowed only if minimum gate passes.',
        training_recipe=read(OLD_REPORT/'registration.json'),generation_1_markers={
            n:sha256(COORD.parent/(n+'.json')) for n in ('A_UNIVERSE_FROZEN','A_FINALISTS_FROZEN','A_GPU_DONE')})
    target=REPORT/'registration.json'
    if not target.exists():atomic_json(target,registration,immutable=True)
    else:assert read(target)['export']['sha256']==registration['export']['sha256']
    reserve()
    existing={}
    directories=[local_path('C:/Users/jsowa/benchmark-translations/series-2026-10-23/wiki-native-dev/source-assets'),
                 ROOT/'training/runs/tinyclip-wiki-pilot-v1/data/parity_raw']
    for folder in directories:
        for p in folder.glob('*.bin'):
            with p.open('rb') as stream:h=hashlib.file_digest(stream,'sha1').hexdigest()
            existing[h]=str(p)
    values=[];seen=set();mime=Counter()
    for row in rows(EXPORT):
        src=row['source_record'];h=source_sha1(src.get('source_sha1'))
        if h in seen:continue
        seen.add(h);url=src.get('image_original_url')
        if not url or not url.startswith('https://upload.wikimedia.org/'):raise ValueError('Invalid original URL')
        mime[src.get('mime_type','unknown')]+=1
        values.append(dict(sha1=h,id=row['id'],url=url,source_page_url=src.get('source_page_url'),
            existing_path=existing.get(h),width=src.get('width'),height=src.get('height'),mime_type=src.get('mime_type'),
            expected_size=src.get('size') or src.get('image_size'),
            order=hashlib.sha256(('wiki-recovery-20261031:'+h).encode()).hexdigest()))
    values.sort(key=lambda r:r['order'])
    catalog=STORE/'download_catalog.jsonl'
    if not catalog.exists():write_rows(catalog,values)
    summary=dict(unique_originals=len(values),already_local=sum(bool(r['existing_path']) for r in values),
        need_download=sum(not r['existing_path'] for r in values),mime_types=dict(mime),
        known_size_bytes=sum(r['expected_size'] or 0 for r in values),known_size_count=sum(bool(r['expected_size']) for r in values),
        hard_storage_cap_GiB=36,originals_cap_GiB=32,physical_disk=disk_snapshot(),catalog=artifact('download_catalog',catalog))
    atomic_json(REPORT/'download_plan.json',summary)
    status('queued',queue=['download_originals','audit_and_freeze_generation_2','encoder_cache_and_training','bootstrap_and_report'],**summary)
    print(__import__('json').dumps(summary))


if __name__=='__main__':main()
