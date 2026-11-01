"""Verify resume metadata without loading encoders or active databases."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pl_mmb_training.series_20261023 import local_path

OUT = ROOT / 'reports/series_2026-10-23'


def lines(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main():
    image_counts = {}
    frozen_ids = {r['id'] for r in lines(ROOT / 'training/runs/tinyclip-visualnews-v1/data/pipeline_pairs_snapshot.jsonl')}
    for kind in ('visualnews', 'wikimedia'):
        keys = set()
        counts = Counter()
        digest = hashlib.sha256()
        path = ROOT / f'reports/{kind}_pipeline/v1/final_pairs.jsonl'
        with path.open('rb') as stream:
            for raw in stream:
                digest.update(raw)
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if kind == 'visualnews':
                    key = row['source_image']
                    counts['ids_already_in_frozen_VN_snapshot'] += row['id'] in frozen_ids
                else:
                    source = row['source_record']
                    key = source.get('source_sha1') or source.get('source_file_id') or source['id']
                keys.add(str(key))
                counts['rows'] += 1
        image_counts[kind] = dict(counts, unique_source_image_keys=len(keys), export_sha256=digest.hexdigest(),
                                  uniqueness_caveat='source identities only; not canonical-pixel or perceptual deduplication')
    assets = {}
    for name in ('wiki_v2/wiki_dev.jsonl', 'xtd10_v2/full_1k.jsonl'):
        records = list(lines(OUT / name))
        missing = [r['id'] for r in records if not local_path(r['image']).is_file()]
        assets[name] = dict(rows=len(records), all_image_paths_exist=not missing, missing=missing,
                           note='file existence only; no repeat of full pixel/hash audit')
    protocols = {name: (OUT / name).is_file() for name in (
        'wiki_v2/retrieval_protocol.json', 'vn_dev_protocol/retrieval_protocol.json',
        'xtd10_v2/full_retrieval_protocol.json', 'xtd10_v2/clean_retrieval_protocol.json',
        'xtd10_v2/cached_style_audit.json', 'wiki_dev_zero_shot_pre_wiki_training.json',
        'wiki_training_release_decision.json')}
    seed = json.loads((OUT / 'vn_full_seeds_posthoc.json').read_text())
    result = dict(no_pipeline_changes=True, no_databases_read=True,
                  frozen_VN_snapshot_rows=len(frozen_ids), exported_images=image_counts,
                  original_eval_assets=assets, execution_artifact_exists=protocols,
                  P1_A_posthoc_keys=list(seed), P1_A_posthoc_summary={k: v for k, v in seed.items()
                      if k not in ('COCO_val', 'legacy_test', 'fixed_gallery', 'COCO_validation')})
    path = OUT / 'resume_readiness_2026-10-29.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    print(json.dumps(dict(result, P1_A_posthoc_summary={k: v for k, v in seed.items()
                        if k in ('status', 'selected_seed', 'COCO_val_summary', 'legacy_test_metrics')}), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
