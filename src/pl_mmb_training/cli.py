import argparse
import os
from pathlib import Path

from .common import ROOT, read_config, write_json


def main():
    parser = argparse.ArgumentParser(description='Frozen PolDense multimodal experiments: E0 and E1')
    parser.add_argument('command', choices=['fetch', 'fetch-data', 'e0', 'cache', 'e1', 'all'])
    parser.add_argument('--config', type=Path, default=ROOT / 'training/configs/e1.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'training/runs/e1-full')
    parser.add_argument('--cache-root', type=Path, default=Path.home() / '.cache/huggingface/pl_mmb_training')
    args = parser.parse_args()
    config = read_config(args.config)
    from .runner import bind_config
    bind_config(config, args.output)
    if args.command in ('fetch', 'fetch-data', 'all'):
        from .fetch import fetch
        fetch(config, args.output, args.cache_root, data_only=args.command == 'fetch-data')
    if args.command in ('fetch', 'fetch-data'):
        return
    # Downloads are explicit. No encoder or optimizer silently reaches the network.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    from huggingface_hub import constants
    constants.HF_HUB_OFFLINE = True  # fetch may already have imported the Hub in `all`.
    import datasets
    datasets.config.HF_DATASETS_OFFLINE = True
    import torch
    torch.set_num_threads(4)
    from .runner import run_e0, run_cache, run_e1
    operations = [('e0', run_e0), ('cache', run_cache), ('e1', run_e1)]
    for name, operation in operations:
        if args.command in (name, 'all'):
            write_json(args.output / 'status.json', dict(stage=name, status='running'))
            try:
                operation(config, args.output, args.cache_root) if name != 'e1' else operation(config, args.output)
            except Exception as exc:
                write_json(args.output / 'status.json', dict(stage=name, status='failed', error=str(exc)))
                raise
            write_json(args.output / 'status.json', dict(stage=name, status='complete'))
