"""Record hashes of F01-a validation evidence (WSL, standard library only)."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/f_series/S03-F01-a'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


summary = json.loads((OUT / 'summary.json').read_text())
for path, expected in {**summary['protected_sha256'], **summary['artifacts_sha256']}.items():
    assert sha(ROOT / path) == expected, path
registry = json.loads((OUT / 'validation/registry.log').read_text())
assert registry['status'] == 'registry_verified'
assert registry['registry_sha256'] == summary['registry_sha256']
assert json.loads((OUT / 'validation/reproduce.log').read_text())['status'] == 'passed'
tests = (OUT / 'validation/unittest.log').read_text()
assert 'Ran 8 tests' in tests and tests.rstrip().endswith('OK'), tests
paths = list((OUT / 'validation').glob('*.log')) + [OUT / 'summary.json', OUT / 'split-summary.json', OUT / 'PLAN.md']
paths += list((ROOT / 'experiments/f_series').glob('*f01*.py'))
paths += [ROOT / 'tests/test_f01_protocol.py']
value = dict(status='passed', session='S03-F01-a', validation_environment='WSL ~/.venvs/pl-mmb-benchmark/bin/python; CPU only',
             base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
             initial_checkout='clean; observed by git status --short before first edit',
             commands=[
                 'PYTHONPATH=src ~/.venvs/pl-mmb-benchmark/bin/python experiments/f_series/f01_protocol.py --write',
                 'PYTHONPATH=src ~/.venvs/pl-mmb-benchmark/bin/python experiments/f_series/f01_protocol.py --check',
                 'PYTHONPATH=src ~/.venvs/pl-mmb-benchmark/bin/python tests/test_f01_protocol.py -v',
                 'PYTHONPATH=src ~/.venvs/pl-mmb-benchmark/bin/python -m pl_mmb_training.current_holdouts'],
             tests=dict(passed=8, failed=0, runner='unittest',
                        pytest_attempt='not installed in required WSL environment; no dependency installation'),
             sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)},
             new_reads=summary['new_reads'], gpu_jobs=[])
payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
target = OUT / 'validation-evidence.json'
if target.exists():
    assert target.read_bytes() == payload
else:
    target.write_bytes(payload)
print(json.dumps(dict(status='passed', evidence_sha256=sha(target))))
