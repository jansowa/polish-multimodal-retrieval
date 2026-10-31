"""Read-only inventory of production translation checkpoints and image sources."""
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pl_mmb_training.common import write_json


def main():
    extra = Path('/mnt/c/Users/jsowa/benchmark-translations')
    rows = []
    for parent in (ROOT / 'outputs', extra):
        for path in sorted(parent.glob('**/checkpoint.sqlite3')):
            if '-pilot' in str(path) or 'review' in str(path):
                continue
            with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
                completed = db.execute('SELECT COUNT(*) FROM completed').fetchone()[0]
                failed = db.execute('SELECT COUNT(*) FROM failures').fetchone()[0]
                first = db.execute('SELECT source_json, metadata_json FROM completed LIMIT 1').fetchone()
            entry = dict(path=str(path), completed=completed, failed=failed,
                         dataset=path.parent.parent.name, run=path.parents[2].name)
            if first:
                metadata = json.loads(first[1])
                entry.update(revision=metadata.get('source_revision'), hf_path=metadata.get('hf_path'))
            rows.append(entry)
            print(json.dumps(entry), flush=True)
    plan = json.loads((ROOT / 'reports/absence_extension_plan.json').read_text())
    for name, info in plan['original_files'].items():
        with Path(info['path']).open() as stream:
            first = json.loads(next(stream))
        print('mbeir_source', name, json.dumps(first)[:1800], flush=True)
    from huggingface_hub import HfApi
    from pl_mmb_translate.registry import DATASETS
    files = {}
    for key in ('aro_visual_relation', 'aro_visual_attribution', 'sugarcrepe'):
        adapter = DATASETS[key].factory()
        files[key] = HfApi().list_repo_files(adapter.path, revision=adapter.revision, repo_type='dataset')
        print('files', key, files[key], flush=True)
    files['mbeir'] = HfApi().list_repo_files('TIGER-Lab/M-BEIR',
                       revision='023f7cc5425e15e3c93f6146f7295c28eae767dc', repo_type='dataset')
    print('mbeir_assets', [f for f in files['mbeir'] if any(s in f.lower() for s in ('zip', 'tar', 'image', 'readme'))], flush=True)
    write_json(ROOT / 'reports/training_data_inventory.json', dict(checkpoints=rows, source_files=files))


if __name__ == '__main__':
    main()
