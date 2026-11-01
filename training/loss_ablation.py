"""Run P0 and E2a/E2b offline, using the frozen full E1 data/features."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pl_mmb_training.common import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['p0', 'all'], nargs='?', default='all')
    parser.add_argument('--config', type=Path, default=ROOT / 'training/configs/p0-e2.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'training/runs/p0-e2')
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    import torch
    torch.set_num_threads(4)
    from pl_mmb_training.ablation import Context, run_p0, run_suite
    write_json(args.output / 'status.json', dict(status='running', command=args.command))
    try:
        ctx = Context(read_json(args.config), args.output)
        run_p0(ctx) if args.command == 'p0' else run_suite(ctx)
    except Exception as exc:
        write_json(args.output / 'status.json', dict(status='failed', command=args.command, error=repr(exc)))
        raise
    write_json(args.output / 'status.json', dict(status='complete', command=args.command))


if __name__ == '__main__':
    main()
