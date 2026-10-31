"""Read atomic pipeline exports for a dated status report; never open databases."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/series_2026-10-23'


def inspect_export(path):
    digest = hashlib.sha256()
    rows = 0
    routes, fields = Counter(), Counter()
    ids = set()
    schema = None
    # Open once so atomic replacements do not mix two versions of the export.
    with path.open('rb') as stream:
        for raw in stream:
            digest.update(raw)
            if not raw.strip():
                continue
            value = json.loads(raw)
            rows += 1
            fields.update(value.keys())
            ids.add(str(value.get('id')))
            route = value.get('route', value.get('final_route', 'unknown'))
            routes[str(route)] += 1
            if schema is None:
                schema = {key: sorted(item.keys()) if isinstance(item, dict) else type(item).__name__
                          for key, item in value.items()}
    return dict(path=str(path), rows=rows, unique_record_ids=len(ids), SHA256=digest.hexdigest(),
                routes=dict(routes), schema=schema,
                field_counts=dict(fields), source='atomic final_pairs.jsonl export only')


def main():
    exports = {kind: inspect_export(ROOT / f'reports/{kind}_pipeline/v1/final_pairs.jsonl')
               for kind in ('wikimedia', 'visualnews')}
    artifacts = {}
    for relative in ('vn_full_finalist_decision.json', 'vn_full_seeds_posthoc.json',
                     'reference_eval_registration.json', 'reference_cache_parity.json'):
        value = json.loads((OUT / relative).read_text(encoding='utf-8'))
        artifacts[relative] = value
    human_files = [dict(path=str(p), bytes=p.stat().st_size)
                   for p in OUT.glob('annotation*/human*.jsonl')]
    assets = {}
    for name in ('wiki_v2/wiki_dev.jsonl', 'xtd10_v2/full_1k.jsonl'):
        rows = [json.loads(line) for line in (OUT / name).read_text(encoding='utf-8').splitlines() if line.strip()]
        assets[name] = dict(rows=len(rows), first_row_fields=sorted(rows[0]),
                           first_image_field={k: rows[0][k] for k in rows[0]
                                              if 'path' in k or k == 'image'})
    result = dict(inspected_utc=datetime.now(timezone.utc).isoformat(),
                  no_databases_opened=True, no_pipeline_changes=True,
                  pipeline_exports=exports, artifact_status=artifacts,
                  human_annotation_files=human_files, evaluation_assets=assets)
    path = OUT / 'resume_inspection_2026-10-29.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    display = dict(exports=exports, human_annotation_files=human_files, evaluation_assets=assets,
                   artifact_summary={name: {k: v for k, v in value.items()
                                           if k in ('status', 'selected_seed', 'seed', 'COCO_val', 'legacy_test', 'MSCOCO', 'source')}
                                     for name, value in artifacts.items()}, report=str(path))
    print(json.dumps(display, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
