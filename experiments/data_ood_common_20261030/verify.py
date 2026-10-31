"""Real subprocess exclusion and budget/publication checks; CPU only."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from common import COORD, GPU_LOCK, GIB, ROOT, artifact, atomic_json, gpu_lock, publish, reserve, sha256


def main():
    report = ROOT / 'reports/data_study_20261030'
    report.mkdir(parents=True, exist_ok=True)
    checks = {}
    child = 'from common import gpu_lock\ntry:\n with gpu_lock(False): pass\nexcept BlockingIOError: raise SystemExit(23)\n'
    try:
        with gpu_lock(False):
            result = subprocess.run([sys.executable, '-c', child], cwd=Path(__file__).parent, capture_output=True)
            assert result.returncode == 23, result.stderr
        result = subprocess.run([sys.executable, '-c', child], cwd=Path(__file__).parent, capture_output=True)
        assert result.returncode == 0, result.stderr
        checks['real_two_process_gpu_exclusion_and_release'] = True
    except BlockingIOError:
        # Do not interfere with a real worker. Retry verification after it ends.
        raise RuntimeError('Existing GPU worker holds canonical lock; no COMMON_READY yet')
    before = sha256(COORD / 'DISK_RESERVATIONS.json') if (COORD / 'DISK_RESERVATIONS.json').exists() else None
    try:
        reserve('A', 'verification-overflow', {'C': 9 * GIB})
        raise AssertionError('Excess reservation accepted')
    except ValueError:
        checks['over_budget_rejected'] = True
    after = sha256(COORD / 'DISK_RESERVATIONS.json') if (COORD / 'DISK_RESERVATIONS.json').exists() else None
    assert before == after
    with patch('common.disk_snapshot', return_value={d: dict(free=4*GIB+100) for d in ('C','D')}):
        try:
            reserve('A', 'verification-overflow', {'D': 101})
            raise AssertionError('Physical reserve violation accepted')
        except ValueError:
            checks['physical_volume_floor_rejected'] = True
    with tempfile.TemporaryDirectory() as folder:
        p = Path(folder) / 'marker.json'
        atomic_json(p, {'first': True}, immutable=True)
        try:
            atomic_json(p, {'second': True}, immutable=True)
            raise AssertionError('Immutable marker replaced')
        except FileExistsError:
            assert json.loads(p.read_text()) == {'first': True}
            checks['immutable_marker_cannot_be_overwritten'] = True
    reserve('A', 'small-code-and-audit', {'C': 8*1024*1024, 'D': 2*1024*1024}, committed_bytes=0)
    atomic_json(report / 'common_verification.json', dict(status='passed', checks=checks, canonical_lock=str(GPU_LOCK)))
    publish('COMMON_READY', [artifact('common_helper', Path(__file__).with_name('common.py')),
                            artifact('helper_usage', Path(__file__).with_name('USAGE.md')),
                            artifact('helper_verification', report / 'common_verification.json'),
                            artifact('helper_verification_code', Path(__file__))], api_version=1)
    print(json.dumps(checks))


if __name__ == '__main__':
    main()
