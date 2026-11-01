"""Create an immutable local identity registry from the current audited inputs."""
import hashlib
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from pl_mmb_training.current_holdouts import DEFAULT_REGISTRY, GUARD, local_path, read_jsonl
from retrieval_benchmark.core import CATALOG


def validation_caption_pairs():
    validation = read_jsonl((ROOT / 'training/runs/clean-v1/validation.jsonl').read_bytes())
    pairs, sources = [], []
    for dataset, positive, negative in [('aro_visual_relation', 'true_caption', 'false_caption'),
                                        ('aro_visual_attribution', 'true_caption', 'false_caption'),
                                        ('sugarcrepe', 'caption', 'negative_caption')]:
        source = ROOT / 'training/runs/e1-full/translation_snapshots' / (dataset + '.jsonl')
        sources.append(source)
        wanted = {r['translation_id']: r for r in validation if r['dataset'] == dataset}
        found = set()
        for patch in read_jsonl(source.read_bytes()):
            if patch['id'] not in wanted:
                continue
            row = wanted[patch['id']]
            if patch['translated'][positive] != row['text_pl']:
                raise RuntimeError('Validation caption lineage mismatch: ' + patch['id'])
            pairs.append(dict(id=row['id'], text_pl=row['text_pl'], text_en=row['text_en'],
                              normalized_texts=[patch['translated'][negative], patch['source'][negative]],
                              dataset=dataset))
            found.add(patch['id'])
        if found != set(wanted):
            raise RuntimeError('Missing held-out negative captions: ' + dataset)
    if len(pairs) != 5602:
        raise RuntimeError('Unexpected validation caption-pair count')
    target = ROOT / 'retrieval_benchmark/local/holdout-validation-caption-pairs-v1.jsonl'
    target.parent.mkdir(exist_ok=True)
    payload = ('\n'.join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in sorted(pairs, key=lambda r: r['id'])) + '\n').encode('utf-8')
    if target.exists():
        if target.read_bytes() != payload:
            raise RuntimeError('Frozen caption pairs changed: create a new version')
    else:
        target.write_bytes(payload)
    return target, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    config = json.loads((ROOT / 'retrieval_benchmark/local/config.json').read_text(encoding='utf-8'))
    inputs = []

    def add(key, path, fmt, role):
        path = local_path(path)
        try:
            location = path.relative_to(ROOT).as_posix()
        except ValueError:
            location = path.as_posix()
        payload = path.read_bytes()
        inputs.append(dict(key=key, path=location, format=fmt, role=role,
                           sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload)))

    for key, value in sorted(config['datasets'].items()):
        fmt = 'jsonl' if value['image_manifest'].endswith('.jsonl') else 'json_array'
        role = 'selection_dev' if key in ('VN_dev', 'wiki_native_dev') else 'test'
        add(key, value['image_manifest'], fmt, role)
    add('benchmark_catalog', CATALOG, 'catalog', 'evidence_only')
    add('retention_validation', 'training/runs/clean-v1/validation.jsonl', 'jsonl', 'selection_dev')
    captions, sources = validation_caption_pairs()
    add('validation_negative_captions', captions, 'jsonl', 'selection_dev')
    for source in sources:
        add('caption_evidence_' + source.stem, source, 'evidence', 'evidence_only')
    add('VN_original_dev', 'training/runs/tinyclip-visualnews-v1/data/vn_dev.jsonl', 'jsonl', 'selection_dev')
    add('Wiki_original_pilot_dev', 'training/runs/tinyclip-wiki-pilot-v1/data/wiki_dev.jsonl', 'jsonl', 'selection_dev')
    add('Wiki_entire_reserved_components', 'reports/series_2026-10-23/wiki_dev_guard_enforcement.json', 'guard', 'selection_dev')
    add('XM3600_test', 'training/runs/xm3600-pl/blacklist.json', 'guard', 'test')
    add('historical_eval_reservation', 'training/runs/e1-full/blacklist.json', 'guard', 'historical_protection')
    add('verified_original_fingerprints', 'C:/Users/jsowa/benchmark-translations/data-ood-20261030/A/protected_index.jsonl', 'fingerprint_cache', 'evidence_only')
    add('historical_phashes', 'training/runs/clean-v1/perceptual_hashes.json', 'phash_cache', 'evidence_only')
    add('XM_phashes', 'training/runs/tinyclip-wiki-pilot-v1/data/additional_protected_phashes.json', 'phash_cache', 'evidence_only')
    value = dict(schema=1, version=args.output.stem + '-20261031',
                 scope='Protect all listed dev/test images, image-only candidates, captions and reserved source components from NEW training admissions',
                 historical_exposure='Immutable historical base training is retained; known XTD-full exposure must remain labelled',
                 phash_radius=6, guard_sha256=hashlib.sha256(GUARD.read_bytes()).hexdigest(), inputs=inputs)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8')
    if args.output.exists():
        if args.output.read_bytes() != payload:
            raise RuntimeError('Registry already exists with different inputs: create a NEW version, do not overwrite')
    else:
        args.output.write_bytes(payload)
    print(args.output)


if __name__ == '__main__':
    main()
