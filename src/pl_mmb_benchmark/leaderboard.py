import json
import math
from pathlib import Path

from .leaderboard_view import render


def build(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    invalid = []
    derived = {}
    for p in root.glob('*/result.json'):
        result = json.loads(p.read_text())
        predictions = p.parent / 'predictions.jsonl'
        reason = None
        if result.get('model') == 'apple/MobileCLIP2-S2' and result.get('numerics') != 'mobile-fp32-unit-range-reparam-v3':
            reason = 'Superseded MobileCLIP preprocessing: requires unit-range images and reparameterization'
        if not predictions.exists():
            reason = 'Missing predictions'
        else:
            rows = [json.loads(line) for line in predictions.read_text().splitlines()]
            if len(rows) != result['examples']:
                reason = 'Prediction count mismatch'
            elif any(not math.isfinite(row['positive_score']) for row in rows):
                reason = 'Nonfinite positive similarities (historical FP16 overflow)'
            elif any(not isinstance(row.get('rank'), int) or not 1 <= row['rank'] <= row.get('candidates', 0) for row in rows):
                reason = 'Invalid prediction ranks'
        if reason:
            invalid.append({'run_id': result['run_id'], 'reason': reason})
        else:
            if 'recall@1' in result['metrics']:
                extra = {'precision@1': result['metrics']['recall@1'],
                         'ndcg@10': sum(1 / math.log2(r['rank'] + 1) if r['rank'] <= 10 else 0 for r in rows) / len(rows)}
                result['metrics'] = {**result['metrics'], **extra}
                derived[result['run_id']] = extra
            results.append(result)
    # Audit is separate: original result files are never rewritten or removed.
    (root / 'validation.json').write_text(json.dumps({'excluded': invalid}, indent=2))
    (root / 'derived_metrics.json').write_text(json.dumps(derived, indent=2, allow_nan=False))
    return render(root, results)
