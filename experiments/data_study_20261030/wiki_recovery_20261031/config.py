"""Generation 2: explicitly authorized original acquisition and repeated Wiki study."""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OLD_CODE=ROOT/'experiments/data_study_20261030'
sys.path.insert(0,str(OLD_CODE))
sys.path.insert(0,str(ROOT/'experiments/data_ood_common_20261030'))
sys.path.insert(0,str(ROOT/'src'))
from common import atomic_json,artifact,sha256,local_path,lock,disk_snapshot,GIB,GPU_LOCK

OLD_REPORT=ROOT/'reports/data_study_20261030'
OLD_STORE=local_path('C:/Users/jsowa/benchmark-translations/data-ood-20261030/A')
STORE=OLD_STORE/'wiki-recovery-20261031'
REPORT=OLD_REPORT/'wiki_recovery_20261031'
COORD=ROOT/'reports/data_ood_coordination_20261030/generation_2'
SERIES=ROOT/'reports/series_2026-10-23'
EXPORT=local_path('C:/Users/jsowa/benchmark-translations/series-2026-10-23/expanded-inputs-20261029/wikimedia_final_pairs.jsonl')
RAW_CAP=32*GIB
TOTAL_CAP=36*GIB
MIN_WIKI=1000
MAX_WIKI=5000
JOB_OWNER='WIKI_RECOVERY_20261031'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def utc():
    from datetime import datetime,timezone
    return datetime.now(timezone.utc).isoformat()


def status(phase,**extra):
    value=dict(schema_version=1,study_id='data-ood-20261031-wiki-recovery',generation=2,owner='A',
        phase=phase,status='in_progress',updated_at=utc(),**extra)
    atomic_json(COORD/'A_STATUS.json',value)
    atomic_json(REPORT/'STATUS.json',value)
    # Mutable coordination status explicitly distinguishes this new authorization
    # from completed, immutable generation-1 readiness markers.
    atomic_json(COORD.parent/'A_STATUS.json',dict(value,
        generation_1_markers_unchanged=True,generation_1_GPU_DONE_applies_only_to_generation_1=True,
        generation_2_requires_new_OOD_overlap_audit=True,new_coordination_path=str(COORD)))


def publish(phase,artifacts,**extra):
    value=dict(schema_version=1,study_id='data-ood-20261031-wiki-recovery',generation=2,owner='A',phase=phase,
        status='complete',contract_sha256=sha256(COORD.parent/'COORDINATION.md'),artifacts=artifacts,
        authorization_sha256=sha256(REPORT/'registration.json'),**extra)
    target=COORD/(phase+'.json')
    if target.exists():
        assert read(target)==value,'Immutable generation-2 marker differs: '+phase
    else:atomic_json(target,value,immutable=True)
    return value


def write_rows(path,values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',encoding='utf-8') as stream:
        for value in values:stream.write(json.dumps(value,ensure_ascii=False,sort_keys=True)+'\n')
        stream.flush();__import__('os').fsync(stream.fileno())
    tmp.replace(path)


def source_sha1(value):
    if not value:raise ValueError('Missing original SHA1')
    return f'{int(value,16 if len(value)==40 else 36):040x}'


def reserve(remaining=None,used_hint=None):
    """Compatible extension: original A/B caps and common helper SHA unchanged.

    Existing helper counts ALL pending reservations against physical volume free
    space. This separately authorized job gets its own owner and 36-GiB cap.
    """
    used=used_hint if used_hint is not None else sum(p.stat().st_size for root in (STORE,REPORT) if root.exists() for p in root.rglob('*') if p.is_file())
    if used>TOTAL_CAP:raise RuntimeError('New job exceeded 36 GiB cap')
    pending=TOTAL_CAP-used if remaining is None else remaining
    if pending<0 or used+pending>TOTAL_CAP:raise ValueError('Invalid reservation')
    parent=COORD.parent;parent.mkdir(parents=True,exist_ok=True)
    with lock(parent/'disk_reservations.lock'):
        p=parent/'DISK_RESERVATIONS.json'
        value=read(p) if p.exists() else dict(schema_version=1,reservations={},committed={})
        value['reservations'][JOB_OWNER+':overnight']=dict(owner=JOB_OWNER,remaining={'C':pending,'D':2*1024**2},temporary_bytes=0)
        value['committed'][JOB_OWNER]=used
        physical=disk_snapshot()
        for drive in ('C','D'):
            outstanding=sum(r['remaining'].get(drive,0) for r in value['reservations'].values())
            if physical[drive]['free']-outstanding<4*GIB:
                raise RuntimeError('Capacity gate: 4 GiB physical reserve on '+drive+' would be violated')
        value['physical_snapshot']=physical
        value.setdefault('authorized_extensions',{})[JOB_OWNER]=dict(cap_bytes=TOTAL_CAP,registration=str(REPORT/'registration.json'))
        atomic_json(p,value)
    return used
