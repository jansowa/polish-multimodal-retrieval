"""Run the five authorized commercially oriented frozen-vision experiments."""
import argparse
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pl_mmb_training.common import read_json, write_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['smoke', 'all'])
    parser.add_argument('--config', type=Path, default=ROOT / 'training/configs/vision-commercial-five.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'training/runs/vision-commercial-five')
    parser.add_argument('--cache-root', type=Path, default=Path.home() / '.cache/huggingface/pl_mmb_training/vision_commercial/embeddings')
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    import torch
    torch.set_num_threads(4)
    from pl_mmb_training.commercial_study import smoke, run, validate_study
    config = read_json(args.config)
    validate_study(config)
    target = args.output / 'study_config.json'
    if target.exists() and read_json(target) != config:
        raise ValueError('Output belongs to another immutable study configuration')
    write_json(target, config)
    try:
        smoke(args.output) if args.command == 'smoke' else run(config, args.output, args.cache_root)
    except Exception as exc:
        write_json(args.output / 'status.json', dict(status='failed', error=repr(exc)))
        raise
