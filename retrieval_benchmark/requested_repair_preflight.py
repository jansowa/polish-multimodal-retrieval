"""Preserve failed launch evidence and refreeze before the first inference."""
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrieval_benchmark.core import read, write, sha
from retrieval_benchmark.requested_queue import REG, REPORT, RUN, verify

if __name__ == '__main__':
    # Only this queue's own supervisor may be stopped; do not touch unrelated jobs.
    ticket = read(RUN/'JOB.json')
    proc = Path('/proc')/str(ticket['pid'])
    if proc.exists():
        command = (proc/'cmdline').read_bytes()
        if b'requested_queue.py\x00run' not in command:
            raise RuntimeError('Recorded PID no longer belongs to our supervisor')
        state = read(RUN/'STATE.json')
        worker = Path('/proc')/str(state.get('worker_pid',0))
        if worker.exists():
            raise RuntimeError('Own worker still running; wait for its disk preflight to finish')
        os.kill(ticket['pid'], signal.SIGTERM)
    if list(RUN.glob('*/smoke.json')) or list(RUN.glob('*/complete.json')):
        raise RuntimeError('Never amend after inference')
    reg = read(REG)
    write(REPORT/'registration-launch-preflight.json', reg, immutable=True)
    write(REPORT/'failed-launch-ticket.json',ticket,immutable=True)
    baseline_bytes = subprocess.check_output(['git','show','HEAD:retrieval_benchmark/space/results.json'], cwd=ROOT)
    import json
    baseline = json.loads(baseline_bytes)
    current = read(ROOT/'retrieval_benchmark/space/results.json')
    key = lambda r:(r['model_id'],r['dataset_id'],r['prompt_mode'],r['profile'])
    if {key(r):r for r in baseline['results']} != {key(r):r for r in current['results']} or baseline['models'] != current['models']:
        raise RuntimeError('Historical export differs beyond ordering; investigate')
    # Revert only the ordering introduced by our own earlier sync, preserving bytes.
    (ROOT/'retrieval_benchmark/space/results.json').write_bytes(baseline_bytes)
    reg['implementation'] = {name:sha(ROOT/name) for name in reg['implementation']}
    reg['amendment'] += ' Failed launch before inference: D needs only 128MiB for logs; use Windows publisher; append-only export.'
    write(REG,reg)
    verify(reg)
    summary = read(REPORT/'summary.json')
    summary['registration_sha256'] = sha(REG)
    summary['failed_preflight_had_no_inference'] = True
    write(REPORT/'summary.json',summary)
    print('READY_TO_RESTART',sha(REG))
