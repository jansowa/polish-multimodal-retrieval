"""Start durable WSL workers without changing translation/reviewer processes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'benchmark/results'
PYTHON = Path.home() / '.venvs/pl-mmb-legacy-cu130/bin/python'
if not PYTHON.is_file():
    PYTHON = Path.home() / '.venvs/pl-mmb-benchmark/bin/python'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['start', 'worker', 'serve', 'status'])
    p.add_argument('--wait', action='store_true', help='Queue after the current worker finishes')
    p.add_argument('--python', type=Path, default=PYTHON, help='Benchmark environment interpreter')
    p.add_argument('--preflight', action='store_true', help='Check selected encoders before full GPU evaluation')
    args, extra = p.parse_known_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.action == 'status':
        state = RESULTS / 'worker.json'
        print(state.read_text() if state.exists() else 'No worker state')
        return
    if args.action == 'start':
        with (RESULTS / 'worker.log').open('a') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker', '--python', str(args.python), *(['--wait'] if args.wait else []), *(['--preflight'] if args.preflight else []), *extra],
                                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print('Started worker supervisor PID', child.pid)
        return
    if args.action == 'serve':
        from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
        from functools import partial
        server = ThreadingHTTPServer(('0.0.0.0', 8766), partial(SimpleHTTPRequestHandler, directory=str(RESULTS)))
        print('Leaderboard: http://localhost:8766', flush=True)
        server.serve_forever()
        return
    with (RESULTS / 'worker.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | (0 if args.wait else fcntl.LOCK_NB))
        except BlockingIOError:
            raise SystemExit('Benchmark worker already running')
        command = [str(args.python), '-u', str(ROOT / 'benchmark.py'), *extra]
        state = dict(pid=os.getpid(), command=command, started=datetime.now(timezone.utc).isoformat(), status='running')
        path = RESULTS / 'worker.json'
        history = RESULTS / 'workers' / f'{os.getpid()}.json'
        history.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(state, indent=2))
        history.write_text(json.dumps(state, indent=2))
        print('Starting', command, flush=True)
        rc = 0
        if args.preflight:
            option = '--models' if '--models' in extra else '--model'
            start = extra.index(option) + 1
            selected = []
            for value in extra[start:]:
                if value.startswith('--'):
                    break
                selected.append(value)
            rc = subprocess.call([str(args.python), '-u', str(ROOT / 'benchmark/check_models.py'), '--models', *selected], cwd=ROOT)
        if rc == 0:
            rc = subprocess.call(command, cwd=ROOT)
        state.update(status='complete' if rc == 0 else 'failed', exit_code=rc, finished=datetime.now(timezone.utc).isoformat())
        path.write_text(json.dumps(state, indent=2))
        history.write_text(json.dumps(state, indent=2))
        raise SystemExit(rc)


if __name__ == '__main__':
    main()
