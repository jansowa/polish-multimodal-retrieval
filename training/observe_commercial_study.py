"""Observability only: run unchanged study with memory/phase diagnostics."""
import os
import resource
import runpy
import sys
import gc
import ctypes
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pl_mmb_training.commercial_study import CommercialContext
from pl_mmb_training import commercial_study
from pl_mmb_training.common import read_json, file_sha, digest, write_json
from pl_mmb_training.features import validate_features
import torch

original_cache = commercial_study.cache_backbone

def trim():
    gc.collect()
    ctypes.CDLL('libc.so.6').malloc_trim(0)

def cache_backbone(key, lock, image_paths, expected_sha, cache_root, batch_size=32, loader=None):
    # Reuse exact audited tensors without instantiating an unused image/text model.
    # This changes allocation only, and returns the identical original metadata.
    path = ROOT / 'training/runs/vision-commercial-five/models' / key / 'feature_cache.json'
    trim()
    if path.exists():
        meta = read_json(path)
        tensor_path = Path(meta['path'])
        identity_path = tensor_path.parent / 'identity.json'
        if identity_path.exists() and identity_path.stat().st_size:
            identity = read_json(identity_path)
        else:
            # VM restart can lose an unflushed JSON on ext4. Reconstruct only
            # if its canonical hash matches the already recorded immutable hash.
            identity = dict(schema=1, stats=meta['model_stats'], environment=meta['environment'],
                            precision='fp32-weights-bf16-autocast', images=list(image_paths),
                            image_sha256=[expected_sha[p] for p in image_paths])
        if (identity['images'] != list(image_paths) or identity['image_sha256'] != [expected_sha[p] for p in image_paths]
                or digest(identity) != meta['identity_sha256'] or file_sha(tensor_path) != meta['sha256']
                or meta['model_stats']['checkpoint_files'] != lock['files']):
            raise ValueError('Recovery cache provenance changed')
        for artifact, sha in lock['files'].items():
            if file_sha(artifact) != sha:
                raise ValueError('Recovery checkpoint changed')
        from importlib.metadata import version
        if meta['environment'] != {p: version(p) for p in meta['environment']}:
            raise ValueError('Recovery feature extraction environment changed')
        tensor = validate_features(image_paths, torch.load(tensor_path, weights_only=True))
        print(f'RECOVERY CACHE {key}: exact tensors and metadata verified', flush=True)
        return tensor, meta
    recovery = ROOT / 'training/runs/vision-commercial-five/recovery'
    recovery.mkdir(parents=True, exist_ok=True)
    input_path, output_path = recovery / f'{key}-input.json', recovery / f'{key}-features.json'
    write_json(input_path, dict(lock=lock, images=list(image_paths), expected_sha=expected_sha))
    subprocess.run([sys.executable, '-u', str(ROOT / 'training/cache_commercial_backbone.py'), key,
                    str(input_path), str(output_path), str(cache_root), str(batch_size),
                    str(torch.are_deterministic_algorithms_enabled())], check=True)
    meta = read_json(output_path)
    if file_sha(meta['path']) != meta['sha256']:
        raise ValueError('Isolated feature tensor checksum mismatch')
    result = (validate_features(image_paths, torch.load(meta['path'], weights_only=True)), meta)
    trim()
    return result

commercial_study.cache_backbone = cache_backbone

def trace(label):
    print(f'PHASE {label} pid={os.getpid()} peak_rss_MiB={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.1f}', flush=True)

original_load = CommercialContext.load_test
original_benchmarks = CommercialContext.benchmarks

def load_test(self):
    trace('before_compact_snapshot')
    original_load(self)
    trace('after_compact_snapshot')

def benchmarks(self, model):
    trim()
    trace('before_benchmarks')
    value = original_benchmarks(self, model)
    trim()
    trace('after_benchmarks')
    return value

CommercialContext.load_test = load_test
CommercialContext.benchmarks = benchmarks
sys.argv = [str(ROOT / 'training/commercial_vision_study.py'), 'all']
runpy.run_path(sys.argv[0], run_name='__main__')
