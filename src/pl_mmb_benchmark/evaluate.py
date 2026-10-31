import hashlib
import math
import json
import time
from importlib.metadata import version
from datetime import datetime, timezone
from pathlib import Path

from .protocol import PROTOCOL, REFERENCE_REVISION


def rank(scores):
    """Positive is position zero; ties follow original np.argmax (first position)."""
    if not bool((abs(scores) < float('inf')).all()):
        raise ValueError('Nonfinite similarities cannot be ranked')
    return 1 + int((scores[1:] > scores[0]).sum())


def cosine_scores(text, image, candidates):
    """Match MMEB get_pred(normalization=True), after adding native features."""
    import torch
    query = text if image is None else text + image
    return torch.nn.functional.normalize(candidates, dim=-1) @ torch.nn.functional.normalize(query, dim=0)


def evaluate(encoder, key, examples, translation_sha, output, limit=None):
    import torch
    if limit:
        examples = examples[:limit]
    identity = dict(protocol=PROTOCOL, model=encoder.repo, revision=encoder.revision,
                    dataset=key, translations_sha256=translation_sha, limit=limit)
    if encoder.mobile:
        identity['numerics'] = 'mobile-fp32-unit-range-reparam-v3'
    if encoder.components:
        identity['components'] = encoder.components
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    dest = Path(output) / run_id
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'result.json').exists():
        print(f'Skip completed {key}: {dest}', flush=True)
        return
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    images = list(dict.fromkeys(p for e in examples for p in e.images + ([e.query_image] if e.query_image and not encoder.native_multimodal else [])))
    texts = list(dict.fromkeys(t for e in examples for t in e.texts))
    print(f'{key}: {len(examples)} examples, {len(images)} unique images, {len(texts)} unique texts', flush=True)
    image_features = encoder.encode(images, images=True)
    text_features = encoder.encode(texts)
    normalized_images = torch.nn.functional.normalize(image_features, dim=-1)
    normalized_texts = torch.nn.functional.normalize(text_features, dim=-1)
    native_queries = None
    if encoder.native_multimodal and any(e.query_image for e in examples):
        if not all(e.query_image for e in examples):
            raise ValueError('Mixed native query modalities require an explicit adapter')
        native_queries = encoder.encode_joint([e.texts[0] for e in examples], [e.query_image for e in examples])
    image_index = {v: i for i, v in enumerate(images)}
    text_index = {v: i for i, v in enumerate(texts)}
    predictions = []
    for row_index, e in enumerate(examples):
        if key.startswith('aro_'):
            scores = normalized_texts[[text_index[t] for t in e.texts]] @ normalized_images[image_index[e.images[0]]]
        else:
            query_text = native_queries[row_index] if native_queries is not None else text_features[text_index[e.texts[0]]]
            query_image = image_features[image_index[e.query_image]] if e.query_image and native_queries is None else None
            scores = cosine_scores(query_text, query_image, image_features[[image_index[p] for p in e.images]])
        predictions.append(dict(id=e.id, rank=rank(scores), candidates=len(scores),
                                positive_score=float(scores[0]), best_index=int(scores.argmax())))
    n = len(predictions)
    metrics = {f'recall@{k}': sum(p['rank'] <= k for p in predictions) / n for k in (1, 5, 10)}
    metrics['mrr'] = sum(1 / p['rank'] for p in predictions) / n
    metrics['precision@1'] = metrics['recall@1']
    metrics['ndcg@10'] = sum(1 / math.log2(p['rank'] + 1) if p['rank'] <= 10 else 0 for p in predictions) / n
    if key.startswith('aro_'):
        metrics = {'pair_accuracy': metrics['recall@1']}
    result = dict(**identity, run_id=run_id, examples=n, full_dataset=limit is None,
                  metrics=metrics, batch_size=encoder.batch_size, precision=encoder.precision,
                  parameters=encoder.parameter_count,
                  text_max_length=encoder.text_max_length,
                  reference_revision=REFERENCE_REVISION,
                  embedding_cache_hits=encoder.cache_hits,
                  fusion='native-joint' if encoder.native_multimodal else 'raw-addition-before-final-l2', similarity='cosine', task_instructions=False,
                  device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  libraries={name: version(name) for name in ('transformers', 'open_clip_torch', 'timm', 'datasets') +
                             (('sentence-transformers',) if encoder.sentence_clip or encoder.native_multimodal else ())},
                  peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30,
                  seconds=time.monotonic() - started, timestamp=datetime.now(timezone.utc).isoformat())
    (dest / 'predictions.jsonl').write_text(''.join(json.dumps(p, allow_nan=False) + '\n' for p in predictions))
    temp = dest / 'result.tmp'
    temp.write_text(json.dumps(result, indent=2, allow_nan=False))
    temp.replace(dest / 'result.json')
    print(json.dumps(result), flush=True)
