"""Same pinned candidate pools, fusion, tie handling and metrics as leaderboard v2."""
import math

from pl_mmb_benchmark.evaluate import cosine_scores, rank
from .common import TASKS


def metrics(predictions):
    if not predictions:
        raise ValueError('Empty evaluation is not a benchmark')
    ranks = [p['rank'] for p in predictions]
    n = len(ranks)
    out = {f'recall@{k}': sum(r <= k for r in ranks) / n for k in (1, 5, 10)}
    out['precision@1'] = out['recall@1']
    out['mrr'] = sum(1 / r for r in ranks) / n
    out['ndcg@10'] = sum(1 / math.log2(r + 1) if r <= 10 else 0 for r in ranks) / n
    return out


def evaluate_features(examples, image_values, image_features, text_values, text_features):
    import torch
    if not examples or len({e.id for e in examples}) != len(examples):
        raise ValueError('Empty evaluation or duplicate query IDs')
    for values, features in ((image_values, image_features), (text_values, text_features)):
        if features.ndim != 2 or len(values) != len(features) or len(set(values)) != len(values):
            raise ValueError('Embedding keys/shape mismatch')
        if not torch.isfinite(features).all() or (features.norm(dim=-1) == 0).any():
            raise ValueError('Nonfinite or zero embeddings')
    if image_features.shape[1] != text_features.shape[1]:
        raise ValueError('Query and target dimensions differ')
    ii, ti = {v: i for i, v in enumerate(image_values)}, {v: i for i, v in enumerate(text_values)}
    predictions = []
    with torch.inference_mode():
        for e in examples:
            if not e.images:
                raise ValueError(f'{e.id}: empty candidate pool')
            query = text_features[ti[e.texts[0]]]
            if e.query_image:
                query = query + image_features[ii[e.query_image]]
            if query.norm() == 0:
                raise ValueError(f'{e.id}: zero composed query')
            scores = cosine_scores(text_features[ti[e.texts[0]]],
                image_features[ii[e.query_image]] if e.query_image else None,
                image_features[[ii[p] for p in e.images]])
            predictions.append(dict(id=e.id, rank=rank(scores), candidates=len(scores),
                                    positive_score=float(scores[0]), best_index=int(scores.argmax())))
    return metrics(predictions), predictions


def scores_table(results):
    scores = {k: results.get(k, {}).get('recall@1') for k in TASKS}
    scores['AVG'] = sum(scores.values()) / len(TASKS) if all(v is not None for v in scores.values()) else None
    return scores


def delta(current, previous):
    return {k: (current[k] - previous[k] if current[k] is not None and previous.get(k) is not None else None)
            for k in (*TASKS, 'AVG')}
