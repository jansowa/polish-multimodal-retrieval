import argparse
import gc
from pathlib import Path

from .data import SPECS, ROOT, load
from .models import MODELS, Encoder, check_cache
from .leaderboard import build


def main():
    p = argparse.ArgumentParser(description='Polish CLIP benchmark (WSL CUDA)')
    p.add_argument('--datasets', nargs='+', default=['mscoco_t2i'], choices=[*SPECS, 'all'])
    p.add_argument('--models', '--model', nargs='+', default=list(MODELS))
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--limit', type=int, help='Pilot only; separated from full results')
    p.add_argument('--output', type=Path, default=ROOT / 'benchmark/results')
    p.add_argument('--list', action='store_true')
    p.add_argument('--leaderboard', action='store_true')
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
    from .evaluate import evaluate
    import torch
    torch.set_num_threads(4)
    datasets = list(SPECS) if 'all' in args.datasets else list(dict.fromkeys(args.datasets))
    datasets.sort(key=lambda k: SPECS[k][3])
    # Dataset first: all models finish the smallest dataset before moving to the next.
    for key in datasets:
        examples, sha = load(key)
        for model in args.models:
            encoder = Encoder(MODELS.get(model, model), args.batch_size)
            try:
                evaluate(encoder, key, examples, sha, args.output, args.limit)
                print('Leaderboard:', build(args.output), flush=True)
            finally:
                del encoder
                gc.collect()
                torch.cuda.empty_cache()
