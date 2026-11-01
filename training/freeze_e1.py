"""Seal original E1 artifacts and source before adding new experiments."""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pl_mmb_training.common import read_json, write_json, file_sha, digest


def freeze(baseline, output):
    baseline, output = baseline.resolve(), output.resolve()
    lock_path = output / 'baseline_lock.json'
    if lock_path.exists():
        lock = read_json(lock_path)
        for path, sha in lock['files'].items():
            if file_sha(baseline / path) != sha:
                raise ValueError(f'Frozen E1 artifact changed: {path}')
        for path, sha in lock['archived'].items():
            if file_sha(output / path) != sha:
                raise ValueError(f'Frozen E1 archive changed: {path}')
        print('Original E1 lock verified', flush=True)
        return lock
    config = read_json(baseline / 'config.json')
    if config['seeds'] != [42, 43, 44] or config['data']['max_pairs'] is not None:
        raise ValueError('Expected the complete three-seed E1 baseline')
    files = {str(p.relative_to(baseline)): file_sha(p) for p in sorted(baseline.rglob('*')) if p.is_file()}
    archived = {}
    for seed in config['seeds']:
        identity = read_json(baseline / f'E1-seed-{seed}/config.json')
        result = read_json(baseline / f'E1-seed-{seed}/result.json')
        if digest(identity) != result['run_id']:
            raise ValueError('E1 run identity mismatch')
        for path, sha in identity['source'].items():
            if file_sha(ROOT / path) != sha:
                raise ValueError(f'Archive original code before editing it: {path}')
            target = output / 'source' / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / path, target)
            archived[str(target.relative_to(output))] = sha
    for path, sha in files.items():
        if path == 'config.json' or path == 'summary.json' or path.startswith('E1-seed-'):
            target = output / 'artifacts' / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(baseline / path, target)
            archived[str(target.relative_to(output))] = sha
    manifest = read_json(baseline / 'cache.json')
    native = {manifest[key]['path']: manifest[key]['sha256'] for key in ('image_features', 'text_features')}
    for path, sha in native.items():
        if file_sha(path) != sha:
            raise ValueError('E1 feature cache changed')
    lock = dict(schema=1, baseline=str(baseline), files=files, archived=archived,
                native_features=native, scores=read_json(baseline / 'summary.json')['scores'],
                std=read_json(baseline / 'summary.json')['std'], seeds=config['seeds'])
    write_json(lock_path, lock)
    print(f'Frozen E1: {len(files)} artifacts, three checkpoints, original source; {lock_path}', flush=True)
    return lock


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=ROOT / 'training/runs/e1-full')
    parser.add_argument('--output', type=Path, default=ROOT / 'training/runs/p0-e2/baseline')
    args = parser.parse_args()
    freeze(args.baseline, args.output)
