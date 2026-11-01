"""Archive the three completed sigmoid/SigLIP2 runs before vision ablations."""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pl_mmb_training.common import read_json, write_json, file_sha


def main():
    base = ROOT / 'training/runs/p0-e2'
    dest = ROOT / 'training/runs/vision-four/reference'
    lock_path = dest / 'lock.json'
    if lock_path.exists():
        lock = read_json(lock_path)
        for name, sha in lock['files'].items():
            if file_sha(base / name) != sha:
                raise ValueError('Sigmoid reference changed: ' + name)
        for name, sha in lock['archive'].items():
            if file_sha(dest / name) != sha:
                raise ValueError('Reference archive changed: ' + name)
        print('Sigmoid reference lock verified', flush=True)
        return
    source = read_json(base / 'E2b-seed-42/config.json')['source']
    archived = {}
    for name, sha in source.items():
        if file_sha(ROOT / name) != sha:
            raise ValueError('Original sigmoid source changed: ' + name)
        target = dest / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        archived[str(target.relative_to(dest))] = sha
    files = {}
    paths = [base / name for name in ('config.json', 'summary.json', 'evaluation_examples.json', 'benchmark_graph.json', 'validation_graph.json')]
    paths += [p for p in base.glob('E2b-seed-*/*') if p.is_file()]
    for path in paths:
        name = str(path.relative_to(base))
        files[name] = file_sha(path)
        target = dest / 'artifacts' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        archived[str(target.relative_to(dest))] = files[name]
    write_json(lock_path, dict(files=files, archive=archived, baseline=str(base), seeds=[42, 43, 44]))
    print('Archived original sigmoid/SigLIP2 code, checkpoints and results', flush=True)


if __name__ == '__main__':
    main()
