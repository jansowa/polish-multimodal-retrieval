"""Replay published v2 models from their original offline feature caches."""
import types

from pl_mmb_benchmark.models import Encoder, MODELS
from pl_mmb_benchmark.protocol import PROTOCOL
from .common import ROOT, TASKS, read_json, digest
from .evaluation import evaluate_features, scores_table


def values_for(examples):
    images = list(dict.fromkeys(p for e in examples for p in e.images + ([e.query_image] if e.query_image else [])))
    texts = list(dict.fromkeys(t for e in examples for t in e.texts))
    return images, texts


def cached_baseline(result, values, modality, path_only=False):
    import torch
    encoder = Encoder.__new__(Encoder)
    for key in ('model', 'revision', 'precision', 'text_max_length'):
        setattr(encoder, 'repo' if key == 'model' else key, result[key])
    encoder.components = result.get('components', {})
    encoder.sentence_clip = result['model'] in (MODELS['clip-b32-pl'], MODELS['clip-b32-pl-32768'])
    encoder.torch = types.SimpleNamespace(__version__=result['torch_version'])
    path = encoder._cache_path(values, modality)
    if not path.is_file() or path.stat().st_size == 0:
        # Older runs predate the audited shared-vision alias in Encoder._cache_path.
        # Reconstruct that precise identity, then verify every output rank below.
        identity = dict(policy='native-features-v2', repo=result['model'], revision=result['revision'],
                        components=result.get('components', {}), precision=result['precision'],
                        text_max_length=result['text_max_length'], modality=modality, values=values,
                        torch=result['torch_version'], transformers=result['libraries']['transformers'])
        legacy_path = path.parent / (digest(identity) + '.pt')
        if legacy_path.is_file() and legacy_path.stat().st_size:
            path = legacy_path
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f'Cannot reproduce {result["model"]}/{result["dataset"]}; missing cache {path}')
    if result['libraries']['transformers'] != __import__('transformers').__version__:
        raise ValueError('Baseline replay requires the recorded Transformers version')
    if path_only:
        return str(path)
    return torch.load(path, map_location='cpu', weights_only=True), str(path)


def reproduce(models, datasets, provenance):
    results = [read_json(p) for p in (ROOT / 'benchmark/results').glob('*/result.json')]
    excluded = set()
    validation = ROOT / 'benchmark/results/validation.json'
    if validation.exists():
        excluded = {r['run_id'] for r in read_json(validation).get('excluded', [])}
    baselines = []
    if len(set(models)) < 2:
        raise ValueError('E0 requires at least two distinct baseline models')
    for alias in models:
        repo = MODELS.get(alias, alias)
        benchmark_results, checks = {}, {}
        for key in TASKS:
            matches = [r for r in results if r.get('protocol') == PROTOCOL and r.get('model') == repo
                       and r.get('dataset') == key and r.get('full_dataset') and r.get('run_id') not in excluded
                       and r.get('translations_sha256') == provenance[key]['translations_sha256']]
            if not matches:
                raise ValueError(f'Missing published baseline: {repo}/{key}')
            original = max(matches, key=lambda r: r['timestamp'])
            images, texts = values_for(datasets[key])
            image_features, image_cache = cached_baseline(original, images, 'images')
            text_features, text_cache = cached_baseline(original, texts, 'texts')
            actual, predictions = evaluate_features(datasets[key], images, image_features, texts, text_features)
            differences = {m: actual[m] - original['metrics'][m] for m in actual}
            # Compare every rank, not only aggregates that can hide changed predictions.
            prior = ROOT / 'benchmark/results' / original['run_id'] / 'predictions.jsonl'
            import json
            old_predictions = [json.loads(line) for line in prior.read_text().splitlines() if line.strip()]
            rank_changes = sum((a['id'], a['rank']) != (b['id'], b['rank']) for a, b in zip(predictions, old_predictions))
            if len(predictions) != len(old_predictions) or rank_changes or any(abs(v) > 1e-9 for v in differences.values()):
                raise ValueError(f'Baseline regression for {repo}/{key}: {differences}, rank changes={rank_changes}')
            benchmark_results[key] = actual
            checks[key] = dict(original_run=original['run_id'], changed_ranks=rank_changes,
                               metric_deltas=differences, image_cache=image_cache, text_cache=text_cache)
            print(f'E0 reproduced {repo}/{key}: R@1={actual["recall@1"]:.3f}', flush=True)
        baselines.append(dict(model=repo, metrics=benchmark_results, scores=scores_table(benchmark_results), checks=checks))
    return baselines
