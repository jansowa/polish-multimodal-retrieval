"""Bidirectional retrieval over explicit, fixed galleries and multi-positive qrels."""
from .common import digest


def make_graph(pairs, image_values, text_values, extra_images=()):
    """Pairs contain image, image_id (canonical pixels) and text; no inferred qrels."""
    representatives = {}
    for pair in [*pairs, *extra_images]:
        key, path = pair['image_id'], pair['image']
        representatives[key] = min(path, representatives.get(key, path))
    image_ids = sorted(representatives)
    texts = sorted({p['text'] for p in pairs})
    if not image_ids or not texts:
        raise ValueError('Empty retrieval gallery')
    im = {value: i for i, value in enumerate(image_ids)}
    tm = {value: i for i, value in enumerate(texts)}
    images = [representatives[key] for key in image_ids]
    ii = {value: i for i, value in enumerate(image_values)}
    ti = {value: i for i, value in enumerate(text_values)}
    graph = dict(protocol='fixed-gallery-multi-positive-v1', images=images, image_ids=image_ids, texts=texts,
                 image_indices=[ii[p] for p in images], text_indices=[ti[t] for t in texts],
                 edges=sorted({(tm[p['text']], im[p['image_id']]) for p in pairs}),
                 tie_policy='descending cosine then ascending canonical gallery index')
    graph['sha256'] = digest(graph)
    return graph


def graph_from_records(records, image_values, text_values):
    return make_graph([dict(image=r['image'], image_id=r['target_id'], text=r['text_pl']) for r in records],
                      image_values, text_values)


def graph_from_examples(examples, blacklist, image_values, text_values):
    def image(path):
        return dict(image=path, image_id=blacklist['images'][path]['pixels_sha256'])
    pairs = [dict(**image(e.images[0]), text=e.texts[0]) for e in examples]
    candidates = [image(path) for e in examples for path in e.images]
    return make_graph(pairs, image_values, text_values, candidates)


def positive_mask(graph):
    import torch
    mask = torch.zeros((len(graph['texts']), len(graph['images'])), dtype=torch.bool)
    edges = torch.tensor(graph['edges'], dtype=torch.long)
    mask[edges[:, 0], edges[:, 1]] = True
    return mask


def best_positive_ranks(scores, positives):
    import torch
    if scores.shape != positives.shape or not torch.isfinite(scores).all():
        raise ValueError('Invalid retrieval score matrix')
    if not positives.any(dim=1).all():
        raise ValueError('Every retrieval query needs a known positive')
    best = scores.masked_fill(~positives, -torch.inf).max(dim=1).values
    columns = torch.arange(scores.shape[1])[None, :]
    first = columns.expand_as(scores).masked_fill(~(positives & (scores == best[:, None])), scores.shape[1]).min(dim=1).values
    return 1 + ((scores > best[:, None]) | ((scores == best[:, None]) & (columns < first[:, None]))).sum(dim=1)


def evaluate_graph(graph, image_features, text_features, include_ranks=False):
    import torch
    import torch.nn.functional as F
    with torch.inference_mode():
        images = image_features[graph['image_indices']]
        texts = text_features[graph['text_indices']]
        if not torch.isfinite(images).all() or not torch.isfinite(texts).all() or (images.norm(dim=1) == 0).any() or (texts.norm(dim=1) == 0).any():
            raise ValueError('Invalid retrieval embeddings')
        scores = F.normalize(texts, dim=1) @ F.normalize(images, dim=1).T
        mask = positive_mask(graph)
        active_images = mask.any(dim=0)
        ranks_t = best_positive_ranks(scores, mask)
        ranks_i = best_positive_ranks(scores.T[active_images], mask.T[active_images])
        def metrics(ranks, candidates):
            return {**{f'recall@{k}': float((ranks <= k).double().mean()) for k in (1, 5, 10)},
                    'queries': len(ranks), 'candidates': candidates}
        result = dict(protocol=graph['protocol'], graph_sha256=graph['sha256'], positive_pairs=len(graph['edges']),
                      t2i=metrics(ranks_t, scores.shape[1]), i2t=metrics(ranks_i, scores.shape[0]))
        result['mean_r1'] = (result['t2i']['recall@1'] + result['i2t']['recall@1']) / 2
        if include_ranks:
            result['ranks'] = dict(t2i=ranks_t.tolist(), i2t=ranks_i.tolist(),
                                   image_query_indices=active_images.nonzero().flatten().tolist())
        return result
