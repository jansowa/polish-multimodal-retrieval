"""Check whether frozen dataset inputs depend on an original HF source cache."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CACHES = {
    'sugarcrepe': 'SUGARCREPE_fmt',
    'aro_visual_relation': 'ARO-Visual-Relation',
    'aro_visual_attribution': 'ARO-Visual-Attribution',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main(dataset='sugarcrepe'):
    cache = Path('/home/jsowa/.cache/huggingface/hub/datasets--mteb--' + SOURCE_CACHES[dataset])
    counts = {}
    images = {}
    manifest_hashes = {}
    for name in ('FULL_CLEAN.jsonl', 'validation.jsonl'):
        path = ROOT / 'training/runs/clean-v1' / name
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        selected = [r for r in rows if r['dataset'] == dataset]
        counts[name] = len(selected)
        manifest_hashes[str(path)] = sha(path)
        for row in selected:
            if row['image'] in images and images[row['image']] != row['sha256']:
                raise ValueError('Conflicting frozen image SHA')
            images[row['image']] = row['sha256']
    missing, mismatches, in_source_cache = [], [], []
    for name, expected in images.items():
        path = Path(name)
        if path.is_relative_to(cache):
            in_source_cache.append(name)
        if not path.is_file():
            missing.append(name)
        elif sha(path) != expected:
            mismatches.append(name)
    audit = json.loads((ROOT / 'training/runs/clean-v1/audit.json').read_text())
    direct_dependencies = [p for p in audit['inputs'] if cache.name in p]
    result = dict(mode='read-only; no cache changes or deletions',
                  source_cache=str(cache), source_role='original downloaded Parquet shards',
                  frozen_rows=counts, unique_extracted_images=len(images),
                  image_directories=dict(Counter(str(Path(name).parent) for name in images)),
                  all_frozen_image_SHAs_verified=not missing and not mismatches,
                  missing_images=missing, SHA_mismatches=mismatches,
                  frozen_image_paths_inside_source_cache=in_source_cache,
                  direct_frozen_audit_dependencies_on_source_cache=direct_dependencies,
                  manifests_SHA256=manifest_hashes,
                  source_loader_rebuild_requires_redownload_if_removed=True)
    output = ROOT / f'reports/series_2026-10-23/{dataset}_source_cache_review_2026-10-24.json'
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', nargs='+', choices=tuple(SOURCE_CACHES), default=['sugarcrepe'])
    for dataset in parser.parse_args().dataset:
        main(dataset)
