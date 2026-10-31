"""Isolate a frozen image cache in its own CUDA process; no training changes."""
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pl_mmb_training.backbones import cache_backbone
from pl_mmb_training.commercial_backbones import load_backbone
from pl_mmb_training.common import read_json, write_json

if __name__ == '__main__':
    torch.set_num_threads(4)
    key, input_path, output_path, cache_root, batch, deterministic = sys.argv[1:]
    torch.use_deterministic_algorithms(deterministic == 'True')
    inputs = read_json(input_path)
    features, metadata = cache_backbone(key, inputs['lock'], inputs['images'], inputs['expected_sha'],
                                        Path(cache_root), int(batch), loader=load_backbone)
    write_json(Path(output_path), metadata)
    print(f'ISOLATED CACHE COMPLETE {key}', flush=True)
