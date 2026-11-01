"""Shared native-feature v2 retrieval and component-paired inference.

This module has no data discovery, downloads or model loading.
The caller must supply a frozen, verified gallery and explicit qrels.
"""
import hashlib
import json


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def cosine_matrix(queries, candidates):
    """The same FP32 scoring operation for full galleries and cache parity."""
    import numpy as np
    queries, candidates = np.asarray(queries, dtype=np.float32), np.asarray(candidates, dtype=np.float32)
    for value in (queries, candidates):
        if value.ndim != 2 or not np.isfinite(value).all() or (np.linalg.norm(value, axis=1) == 0).any():
            raise ValueError('Invalid native features')
    return (queries/np.linalg.norm(queries,axis=1,keepdims=True)) @ (candidates/np.linalg.norm(candidates,axis=1,keepdims=True)).T


def per_query(scores, positives):
    import numpy as np
    scores, positives = np.asarray(scores), np.asarray(positives, dtype=bool)
    if scores.ndim != 2 or scores.shape != positives.shape or not np.isfinite(scores).all():
        raise ValueError('Invalid scores/qrels')
    if not positives.any(axis=1).all():
        raise ValueError('Every query requires a positive')
    order = np.argsort(-scores, axis=1, kind='stable')
    sorted_positives = np.take_along_axis(positives, order, axis=1)
    ranks = sorted_positives.argmax(axis=1) + 1
    cutoff = min(10, scores.shape[1])
    discounts = 1 / np.log2(np.arange(cutoff) + 2)
    dcg = (sorted_positives[:, :cutoff] * discounts).sum(axis=1)
    ideal = np.array([discounts[:min(int(n), cutoff)].sum() for n in positives.sum(axis=1)])
    return {**{f'recall@{k}': (ranks <= k).astype(float) for k in (1, 5, 10)},
            'mrr': 1 / ranks, 'ndcg@10': dcg / ideal}


def evaluate_features(images, texts, image_ids, query_ids, edges, image_components, text_components,
                      dataset_id, manifest_sha256, prompt_mode='v2_no_instructions', text_candidate_features=None):
    import numpy as np
    images, texts = np.asarray(images, dtype=np.float32), np.asarray(texts, dtype=np.float32)
    if images.ndim != 2 or texts.ndim != 2 or images.shape[1] != texts.shape[1]:
        raise ValueError('Embedding dimensions differ')
    if len(images) != len(image_ids) or len(texts) != len(query_ids):
        raise ValueError('Embedding/ID length mismatch')
    if len(set(image_ids)) != len(image_ids) or len(set(query_ids)) != len(query_ids):
        raise ValueError('Duplicate IDs; canonicalize gallery and queries first')
    if len(image_components) != len(images) or len(text_components) != len(texts):
        raise ValueError('Component/ID length mismatch')
    for array in (images, texts):
        if not np.isfinite(array).all() or (np.linalg.norm(array, axis=1) == 0).any():
            raise ValueError('Nonfinite or zero features')
    positives = np.zeros((len(texts), len(images)), dtype=bool)
    edges = sorted(set(tuple(e) for e in edges))
    for ti, ii in edges:
        if not (0 <= ti < len(texts) and 0 <= ii < len(images)):
            raise ValueError('Qrel index outside gallery')
        if text_components[ti] != image_components[ii]:
            raise ValueError('Positive edges must remain within a transitive component')
        positives[ti, ii] = True
    if not positives.any(axis=0).all():
        raise ValueError('Every image needs a known text positive for i2t')
    scores=cosine_matrix(texts,images)
    candidate_texts=texts if text_candidate_features is None else np.asarray(text_candidate_features,dtype=np.float32)
    if candidate_texts.shape!=texts.shape or not np.isfinite(candidate_texts).all() or (np.linalg.norm(candidate_texts,axis=1)==0).any():
        raise ValueError('Invalid text-candidate embeddings')
    inverse_scores=cosine_matrix(images,candidate_texts)
    shared = dict(dataset_id=dataset_id, manifest_sha256=manifest_sha256, prompt_mode=prompt_mode,
                  protocol='v2-no-instructions-native-cosine-full-gallery-multi-positive',
                  qrels_sha256=identity(edges), gallery_sha256=identity(image_ids), text_gallery_sha256=identity(query_ids))
    result = dict(**shared, directions={})
    for direction, values, ids, components in (
        ('t2i', per_query(scores, positives), query_ids, text_components),
        ('i2t', per_query(inverse_scores, positives.T), image_ids, image_components)):
        result['directions'][direction] = dict(query_ids=list(ids), component_ids=list(components),
            metrics={k: float(v.mean()) for k, v in values.items()}, per_query={k: v.tolist() for k, v in values.items()})
    return result


def evaluate_encoder(encoder, dataset, registration):
    """All models, including a projector adapter, pass through this same function."""
    if registration.get('status') != 'closed_ready_for_execution':
        raise ValueError('Registration is not closed for execution')
    if dataset.get('status') != 'frozen_image_verified':
        raise ValueError('Dataset is not image-verified and frozen')
    images = encoder.encode(dataset['images'], images=True)
    texts = encoder.encode(dataset['texts'])
    candidate_texts=encoder.encode_text_candidates(dataset['texts']) if hasattr(encoder,'encode_text_candidates') else texts
    def array(value):
        return value.detach().float().cpu().numpy() if hasattr(value, 'detach') else value
    return evaluate_features(array(images), array(texts), dataset['image_ids'], dataset['query_ids'], dataset['edges'],
        dataset['image_components'], dataset['text_components'], dataset['id'], dataset['manifest_sha256'], dataset['prompt_mode'],array(candidate_texts))


def paired_bootstrap(ours, reference, direction, replicates=10000, seed=20261023):
    import numpy as np
    for field in ('protocol', 'dataset_id', 'manifest_sha256', 'prompt_mode', 'qrels_sha256', 'gallery_sha256', 'text_gallery_sha256'):
        if ours[field] != reference[field]:
            raise ValueError(f'Cannot pair different {field}')
    a, b = ours['directions'][direction], reference['directions'][direction]
    if a['query_ids'] != b['query_ids'] or a['component_ids'] != b['component_ids']:
        raise ValueError('Query IDs/order/components differ')
    if set(a['per_query']) != set(b['per_query']):
        raise ValueError('Metric sets differ')
    if type(replicates) is not int or replicates < 1:
        raise ValueError('Positive bootstrap replicate count required')
    components = sorted(set(a['component_ids']))
    if len(components) < 2:
        raise ValueError('At least two independent components required')
    index = {c: i for i, c in enumerate(components)}
    mapping = np.array([index[c] for c in a['component_ids']])
    sizes = np.bincount(mapping)
    keys = sorted(a['per_query'])
    differences = np.array([np.asarray(a['per_query'][k]) - np.asarray(b['per_query'][k]) for k in keys]).T
    if differences.shape != (len(mapping), len(keys)) or not np.isfinite(differences).all():
        raise ValueError('Invalid paired query metrics')
    totals = np.array([differences[mapping == i].sum(axis=0) for i in range(len(components))])
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, len(keys)))
    for r in range(replicates):
        sampled = rng.integers(0, len(components), len(components))
        draws[r] = totals[sampled].sum(axis=0) / sizes[sampled].sum()
    bounds = np.quantile(draws, [.025, .975], axis=0)
    return dict(direction=direction, units='percentage points, our minus reference', replicates=replicates,
        seed=seed, components=len(components), conditional_on='fixed gallery, observed qrels and leakage components',
        metrics={k: dict(delta_pp=float(100*differences[:, j].mean()),
                        ci95_pp=(100*bounds[:, j]).tolist()) for j, k in enumerate(keys)})
