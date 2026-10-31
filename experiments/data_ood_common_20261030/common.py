"""Coordination API v1. Run in WSL; no CUDA operation outside gpu_lock()."""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]
COORD = ROOT / 'reports/data_ood_coordination_20261030'
GPU_LOCK = ROOT / 'training/runs/tinyclip-visualnews-v1/worker.lock'
GIB = 1024 ** 3


def local_path(value):
    value = str(value).replace('\\', '/')
    if len(value) > 2 and value[1:3] == ':/':
        return Path('/mnt/' + value[0].lower() + '/' + value[3:])
    return Path(value)


def windows_path(value):
    value = str(Path(value).resolve())
    if value.startswith('/mnt/') and len(value) > 7:
        return value[5].upper() + ':/' + value[7:]
    raise ValueError('Artifact must reside on a Windows volume: ' + value)


def sha256(path):
    h = hashlib.sha256()
    with local_path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@contextlib.contextmanager
def lock(path, blocking=True):
    path = local_path(path)
    with path.open('a+b') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield stream
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def gpu_lock(blocking=True):
    return lock(GPU_LOCK, blocking)


def atomic_json(path, payload, immutable=False):
    path = local_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8')
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            # link is an atomic create-if-absent, unlike exists()+replace().
            os.link(tmp, path)
        else:
            os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def artifact(role, path):
    path = local_path(path)
    return dict(role=role, path=windows_path(path), sha256=sha256(path), bytes=path.stat().st_size)


def publish(phase, artifacts, **extra):
    payload = dict(schema_version=1, study_id='data-ood-20261030', generation=1,
                   owner='A', phase=phase, status='complete',
                   contract_sha256=sha256(COORD / 'COORDINATION.md'),
                   artifacts=artifacts, **extra)
    atomic_json(COORD / (phase + '.json'), payload, immutable=True)
    return payload


def verify_marker(path):
    data = json.loads(local_path(path).read_text())
    if data['contract_sha256'] != sha256(COORD / 'COORDINATION.md'):
        raise ValueError('Contract changed')
    for item in data['artifacts']:
        p = local_path(item['path'])
        if p.stat().st_size != item['bytes'] or sha256(p) != item['sha256']:
            raise ValueError('Artifact changed: ' + str(p))
    return data


def disk_snapshot():
    return {drive: dict(zip(('total', 'used', 'free'), shutil.disk_usage('/mnt/' + drive.lower())))
            for drive in ('C', 'D')}


def reserve(owner, key, remaining, *, committed_bytes=0, temporary_bytes=0):
    """remaining={'C': bytes, 'D': bytes}, includes temporary bytes.

    Replace an owner's named reservation after each write; free space already
    accounts for written bytes. committed_bytes is total persistent study data
    written by this owner (all reservations), never a second free-space debit.
    Temporary allocations across all owners are capped at 4 GiB.
    """
    if owner not in ('A', 'B') or not key or any(d not in ('C', 'D') for d in remaining):
        raise ValueError('Invalid owner/key/volume')
    if any(type(n) is not int or n < 0 for n in [*remaining.values(), committed_bytes, temporary_bytes]):
        raise ValueError('Byte counts must be nonnegative integers')
    if temporary_bytes > sum(remaining.values()):
        raise ValueError('Temporary bytes must be part of remaining')
    with lock(COORD / 'disk_reservations.lock'):
        path = COORD / 'DISK_RESERVATIONS.json'
        registry = json.loads(path.read_text()) if path.exists() else dict(schema_version=1, reservations={}, committed={})
        reservations = registry['reservations']
        token = owner + ':' + key
        reservations[token] = dict(owner=owner, remaining=remaining, temporary_bytes=temporary_bytes)
        registry['committed'][owner] = committed_bytes
        for who, budget in [('A', 8 * GIB), ('B', 3 * GIB)]:
            pending = sum(sum(r['remaining'].values()) - r['temporary_bytes'] for r in reservations.values() if r['owner'] == who)
            if registry['committed'].get(who, 0) + pending > budget:
                raise ValueError('Persistent budget exceeded: ' + who)
        if sum(r['temporary_bytes'] for r in reservations.values()) > 4 * GIB:
            raise ValueError('Shared temporary budget exceeded')
        physical = disk_snapshot()
        for drive in ('C', 'D'):
            pending = sum(r['remaining'].get(drive, 0) for r in reservations.values())
            if physical[drive]['free'] - pending < 4 * GIB:
                raise ValueError('Physical ' + drive + ': 4 GiB reserve would be violated')
        registry['physical_snapshot'] = physical
        atomic_json(path, registry)
        return registry
