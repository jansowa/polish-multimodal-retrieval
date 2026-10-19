import argparse
import gc
import subprocess
import sys
import hashlib
import json
from pathlib import Path

from .data import SPECS, ROOT, load, translation_path
from .models import MODELS, REVISIONS, Encoder, check_cache, worker_python
from .protocol import PROTOCOL
from .leaderboard import build


def main():
    p = argparse.ArgumentParser(description='Polish CLIP benchmark (WSL CUDA)')
    p.add_argument('--datasets', nargs='+', default=['mscoco_t2i'], choices=[*SPECS, 'all'])
    p.add_argument('--models', '--model', nargs='+', default=list(MODELS))
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--limit', type=int, help='Pilot only; separated from full results')
    p.add_argument('--output', type=Path, default=ROOT / 'benchmark/results')
    p.add_argument('--list', action='store_true')
    p.add_argument('--leaderboard', action='store_true')
    p.add_argument('--internal-worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.list:
        for k, spec in SPECS.items():
            print(k, spec[3])
        print('Models:', MODELS)
        return
    if args.leaderboard:
        print(build(args.output))
        return
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        p.error('Batch size and limit must be positive')
    check_cache()
    datasets = list(SPECS) if 'all' in args.datasets else list(dict.fromkeys(args.datasets))
    datasets.sort(key=lambda k: SPECS[k][3])
    if not args.internal_worker:
        # Fresh CUDA context for each model/dataset avoids WSL driver allocation buildup.
        for key in datasets:
            sha = hashlib.sha256(translation_path(key).read_bytes()).hexdigest()
            for model in args.models:
                repo = MODELS.get(model, model)
                completed = []
                for path in args.output.glob('*/result.json'):
                    result = json.loads(path.read_text())
                    if (result.get('protocol'), result.get('model'), result.get('dataset'),
                        result.get('revision'), result.get('translations_sha256'), result.get('limit')) == (
                        PROTOCOL, repo, key, REVISIONS.get(repo), sha, args.limit):
                        completed.append(path)
                if completed:
                    print(f'Skip completed {repo} / {key}', flush=True)
                    continue
                executable = worker_python(repo, sys.executable)
                command = [executable, '-u', str(ROOT/'benchmark.py'), '--internal-worker', '--datasets', key,
                           '--models', repo, '--batch-size', str(min(args.batch_size, 16) if repo == MODELS['embeddinggemma-2'] else args.batch_size),
                           '--output', str(args.output)]
                if args.limit:
                    command += ['--limit', str(args.limit)]
                print(f'Worker: {repo} / {key}', flush=True)
                subprocess.run(command, check=True)
        print('Leaderboard:', build(args.output), flush=True)
        return
    from .evaluate import evaluate
    import torch
    torch.set_num_threads(4)
    # Dataset first: all models finish the smallest dataset before moving to the next.
    for key in datasets:
        examples, sha = load(key)
        for model in args.models:
            repo = MODELS.get(model, model)
            encoder = Encoder(repo, min(args.batch_size, 16) if repo == MODELS['embeddinggemma-2'] else args.batch_size)
            try:
                evaluate(encoder, key, examples, sha, args.output, args.limit)
                print('Leaderboard:', build(args.output), flush=True)
            finally:
                del encoder
                gc.collect()
                torch.cuda.empty_cache()
