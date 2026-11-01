"""Run only DINOv2-S, MobileCLIP2-S2, ConvNeXt-T and official iBOT ViT-S/16."""
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
    parser.add_argument('--config', type=Path, default=ROOT / 'training/configs/vision-four.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'training/runs/vision-four')
    parser.add_argument('--cache-root', type=Path, default=Path.home() / '.cache/huggingface/pl_mmb_training/vision_four/embeddings')
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    import torch
    torch.set_num_threads(4)
    from pl_mmb_training.vision_study import smoke, run
    config = read_json(args.config)
    path = args.output / 'study_config.json'
    if path.exists() and read_json(path) != config:
        raise ValueError('Output belongs to another vision study')
    write_json(path, config)
    try:
        smoke(args.output) if args.command == 'smoke' else run(config, args.output, args.cache_root)
    except Exception as exc:
        write_json(args.output / 'status.json', dict(status='failed', error=repr(exc)))
        raise
