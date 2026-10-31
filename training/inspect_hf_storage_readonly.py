"""Inspect shared HF caches and local provenance. Never delete any cache."""
import datetime
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF = Path('/home/jsowa/.cache/huggingface')


def disk_bytes(path):
    run = subprocess.run(['du', '-s', '-B1', str(path)], capture_output=True, text=True, check=True)
    return int(run.stdout.split()[0])


def main():
    datasets = []
    metadata_errors = []
    for kind in ('json', 'parquet'):
        base = HF / 'datasets' / kind
        for info in sorted(base.glob('**/dataset_info.json')):
            try:
                value = json.loads(info.read_text())
            except (ValueError, OSError) as exc:
                metadata_errors.append(dict(path=str(info), error=str(exc), config_bytes=disk_bytes(info.parent)))
                continue
            sources = []
            for name, recorded in value.get('download_checksums', {}).items():
                path = Path(name)
                sources.append(dict(path=name, exists=path.is_file(),
                                    current_bytes=path.stat().st_size if path.is_file() else None,
                                    recorded=recorded))
            datasets.append(dict(kind=kind, path=str(info.parent),
                                 bytes=disk_bytes(info.parent),
                                 sources=sources, features=list(value.get('features', {})),
                                 splits=value.get('splits'),
                                 files=[dict(name=p.name, bytes=p.stat().st_size)
                                        for p in info.parent.iterdir() if p.is_file()]))
    models = []
    for directory in sorted((HF / 'hub').glob('models--*')):
        refs = {}
        if (directory / 'refs').is_dir():
            for path in (directory / 'refs').rglob('*'):
                if path.is_file():
                    refs[str(path.relative_to(directory / 'refs'))] = path.read_text().strip()
        models.append(dict(path=str(directory), repo_id=directory.name[8:].replace('--', '/'),
                           bytes=disk_bytes(directory), refs=refs))
    result = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  mode='read-only metadata and file-size inspection; no deletions',
                  metadata_errors=metadata_errors,
                  dataset_totals={kind: disk_bytes(HF / 'datasets' / kind) for kind in ('json', 'parquet')},
                  datasets=datasets, models=sorted(models, key=lambda m: -m['bytes']))
    output = ROOT / 'reports/series_2026-10-23/hf_storage_inspection_2026-10-24.json'
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    for kind in ('json', 'parquet'):
        entries = [d for d in datasets if d['kind'] == kind]
        print(json.dumps(dict(kind=kind, total_bytes=result['dataset_totals'][kind],
                              configs=len(entries), sources=sorted({s['path'] for d in entries for s in d['sources']})), ensure_ascii=False))
        if kind == 'parquet':
            print(json.dumps(entries, ensure_ascii=False))
    print(json.dumps(dict(models=result['models'], metadata_errors=metadata_errors, report=str(output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
