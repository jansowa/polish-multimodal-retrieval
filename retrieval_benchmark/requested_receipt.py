"""Snapshot job evidence without reading full scores or model logs."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrieval_benchmark.core import read, write, sha, public_check
from retrieval_benchmark.requested_queue import REG, REPORT, RUN

if __name__ == '__main__':
    doc = public_check(read(ROOT/'retrieval_benchmark/space/results.json'))
    evidence = dict(registration_sha256=sha(REG), ticket=read(RUN/'JOB.json'), state=read(RUN/'STATE.json'),
                    publisher=read(RUN/'PUBLISH_STATUS.json'),
                    completed=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p), value=read(p))
                               for p in sorted(RUN.glob('*/complete.json'))],
                    cleanup=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p), value=read(p))
                             for p in sorted(RUN.glob('*/cleanup.json'))],
                    public_results_sha256=sha(ROOT/'retrieval_benchmark/space/results.json'),
                    new_results=sum(r.get('registration_sha256')==sha(REG) for r in doc['results']),
                    tests='python -m unittest retrieval_benchmark.tests.test_requested_queue -v',
                    git_history='Remote initial snapshot had no remote-only files; local changes in six shared files predated this stage. Merged remote ancestry with ours strategy, preserved current tree, then fast-forward pushed main.')
    write(REPORT/'launch-receipt.json',evidence)
    print('RECEIPT',sha(REPORT/'launch-receipt.json'), 'new_results',evidence['new_results'],
          'completed',len(evidence['completed']), 'publisher',evidence['publisher']['status'])
