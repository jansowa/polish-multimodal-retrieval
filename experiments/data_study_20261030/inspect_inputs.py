"""Read-only compact inventory; no results from session B."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/data_ood_common_20261030'))
from common import local_path, sha256


def brief(obj, depth=0):
    if isinstance(obj, dict):
        if len(obj) > 20:
            return {'keys_count': len(obj), 'sample': {k: brief(v, depth+1) for k,v in list(obj.items())[:2]}}
        return {k: brief(v, depth+1) for k,v in obj.items() if k not in ('ranks', 'history')}
    if isinstance(obj, list):
        return {'length': len(obj), 'sample': [brief(v, depth+1) for v in obj[:2]]}
    return obj


series = ROOT / 'reports/series_2026-10-23'
names = ['expanded_corpus_metadata_audit_20261029', 'expanded_corpus_preparation_decision_20261029',
         'wiki_dev_guard_enforcement', 'wiki_dev_guard_integration', 'preservation_before',
         'vn_full_seeds_posthoc', 'vn_full_seeds_training_summary', 'evaluation_scope_amendment_20261030']
for name in names:
    p = series / (name + '.json')
    value = json.loads(p.read_text())
    print(name, sha256(p), json.dumps(brief(value), ensure_ascii=False))
for p in [ROOT/'training/runs/tinyclip-visualnews-v1/context.json', ROOT/'training/runs/tinyclip-visualnews-v1/inputs.json']:
    print(str(p), json.dumps(brief(json.loads(p.read_text())), ensure_ascii=False))
folder = local_path('C:/Users/jsowa/benchmark-translations/series-2026-10-23/expanded-inputs-20261029')
for name in ['visualnews_final_pairs.jsonl','wikimedia_final_pairs.jsonl','metadata_candidates.jsonl']:
    with (folder/name).open() as stream:
        row = json.loads(next(stream))
    print('FIRST_ROW', name, json.dumps(row, ensure_ascii=False))
