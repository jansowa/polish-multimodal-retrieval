"""Read-only diagnostics on the frozen FULL_CLEAN development snapshot."""
import json
from collections import defaultdict

import torch
import torch.nn.functional as F

from . import tinyclip_adaptation as base
from .common import ROOT, file_sha, read_json, write_json
from .losses import PairLoss, sigmoid_loss
from .retrieval import best_positive_ranks, evaluate_graph, make_graph, positive_mask

OUT = ROOT / 'training/runs/tinyclip-diagnostics'


def leave_one_out_centroids(texts, positives):
    """Remove the query text from EVERY associated image prototype, not just one.

    Captions are deduplicated by the caller. This is a caption-informed diagnostic,
    not an attainable image model or a mathematical upper bound on retrieval.
    """
    q = F.normalize(texts.double(), dim=1)
    mask = positives.to(q.device)
    sums = mask.double().T @ q
    counts = mask.sum(0)
    dots = q @ sums.T
    # Squared length after removing this query from associated prototypes.
    squared = sums.square().sum(1)[None, :] - 2 * dots * mask + mask
    scores = (dots - mask.double()) / squared.clamp_min(1e-24).sqrt()
    available = counts[None, :] - mask.long() > 0
    scores = scores.masked_fill(~available, -1e10)
    known = mask & available
    eligible = known.any(1)
    ranks = best_positive_ranks(scores[eligible].cpu(), known[eligible].cpu())
    return dict(**{f'recall@{k}': float((ranks <= k).double().mean()) if len(ranks) else None for k in (1, 5, 10)},
                queries=len(ranks), excluded_queries=int((~eligible).sum()), images=mask.shape[1],
                caption_counts_quantiles=torch.quantile(counts.double(), torch.tensor([0., .5, 1.], dtype=torch.double, device=q.device)).tolist(),
                interpretation='Leave-query-out caption prototypes; NOT a retrieval upper bound; other validation captions are used.')


def ridge_fit(x, y, fraction):
    """Image-balanced affine ridge with an unpenalized intercept."""
    x, y = x.double(), y.double()
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    gram = xc.T @ xc
    alpha = fraction * gram.trace() / x.shape[1]
    if fraction == 0:
        w = torch.linalg.lstsq(xc, yc).solution
    else:
        w = torch.linalg.solve(gram + alpha * torch.eye(x.shape[1], device=x.device, dtype=x.dtype), xc.T @ yc)
    return w, ym - xm @ w, float(alpha)


def prototypes(data):
    graph = data.graph
    pol = leave_one_out_centroids(data.texts[graph['text_indices']], positive_mask(graph))
    teacher_path = base.OUT / 'teacher/features.pt'
    meta = read_json(teacher_path.parent / 'complete.json')
    assert file_sha(teacher_path) == meta['sha256']
    teacher = torch.load(teacher_path, weights_only=True)
    rows = [r for r in data.val if r['dataset'] == 'mbeir_mscoco']
    en_values = sorted({r['text_en'] for r in rows})
    en_graph = make_graph([dict(image=r['image'], image_id=r['target_id'], text=r['text_en']) for r in rows],
                          data.images, en_values)
    assert en_graph['image_ids'] == graph['image_ids']
    lookup = {r['text_en']: i for i, r in enumerate(data.val)}
    en = teacher['validation_texts'][[lookup[s] for s in en_graph['texts']]]
    sig = leave_one_out_centroids(en, positive_mask(en_graph))
    return dict(poldense_pl=pol, siglip2_en=sig, teacher_cache_sha256=meta['sha256'],
                caveat='Different language AND encoder; the comparison does not isolate translation quality.')


def ridge(data, x):
    groups = defaultdict(dict)
    ti, tt, _ = data.tensors['train']
    for index, row in enumerate(data.train):
        groups[row['target_id']].setdefault(row['text_pl'], index)
    image_ids = sorted(groups)
    indices = [list(groups[k].values()) for k in image_ids]
    xx = x[[int(ti[ii[0]]) for ii in indices]].to('cuda')
    yy = torch.stack([F.normalize(tt[ii], dim=1).mean(0) for ii in indices]).to('cuda')
    all_x = x.to('cuda')
    results = []
    for source in ('native', 'unit'):
        fit_x = xx if source == 'native' else F.normalize(xx, dim=1)
        eval_x = all_x if source == 'native' else F.normalize(all_x, dim=1)
        for fraction in (0., .01, .1, 1.):
            w, b, alpha = ridge_fit(fit_x, yy, fraction)
            pred = (eval_x.double() @ w + b).float().cpu()
            result = evaluate_graph(data.graph, pred, data.texts)
            results.append(dict(vision_input=source, ridge_fraction=fraction, ridge_alpha=alpha, validation=result))
    return dict(images=len(image_ids), weighting='one image, mean of its unique unit PL captions; native and unit vision inputs; affine ridge',
                training_only_fit=True, candidates=results,
                caveat='Different objective from sigmoid gold; a weak ridge result does NOT prove AdamW has reached the best retrieval projector.')


def scalar_diagnostic(data, x):
    folder = ROOT / 'training/runs/clean-v1/FULL_CLEAN/seed-42/checkpoints'
    a1 = torch.load(base.OUT / 'A1/best.pt', weights_only=True)
    aligned = PairLoss('siglip_sigmoid').to('cuda')
    aligned.load_state_dict(a1['loss'])
    scalar_sets = dict(initial=(1/.07, -10.), aligned=(aligned.scalars()['logit_scale'], aligned.scalars()['logit_bias']))
    model = base.projector(512, 0).to('cuda')
    learned = PairLoss('siglip_sigmoid').to('cuda')
    x = x.to('cuda')
    split_batches = {}
    gen = torch.Generator().manual_seed(1701)
    for split in ('train', 'validation'):
        ii, tt, pp = data.tensors[split]
        selection = torch.randperm(len(ii), generator=gen)[:2048] if split == 'train' else torch.arange(len(ii))
        split_batches[split] = [(ii[part].to('cuda'), tt[part].to('cuda'), pp.mask(part.tolist()).to('cuda'))
                                for part in selection.split(256)]
    rows = []
    with torch.inference_mode():
        for epoch in range(1, 21):
            path = folder / f'epoch-{epoch:02d}.pt'
            checkpoint = torch.load(path, weights_only=True, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            learned.load_state_dict(checkpoint['loss_state_dict'])
            row = dict(epoch=epoch, scalars=learned.scalars(), checkpoint_sha256=file_sha(path))
            for split, batches in split_batches.items():
                losses = defaultdict(float)
                count = 0
                for ii, tt, mask in batches:
                    pred = model(x[ii])
                    cosine = F.normalize(tt, dim=1) @ F.normalize(pred, dim=1).T
                    losses['learned'] += float(learned(pred, tt, mask)) * len(ii)
                    for name, (scale, bias) in scalar_sets.items():
                        losses[name] += float(sigmoid_loss(cosine * scale + bias, mask)) * len(ii)
                    count += len(ii)
                row[split] = {k: v/count for k, v in losses.items()}
            rows.append(row)
    return dict(fixed_scalars=scalar_sets, epochs=rows,
                train_protocol='Same fixed 2048 TRAIN rows at every checkpoint, shuffled batch membership fixed.',
                validation_protocol='All 7695 validation rows with original fixed batch order.',
                caveat='Different train/val source and caption frequencies still affect batch loss; compare trends within a split.')


def hard_negatives(data):
    from .features import encode_cached
    snapshots = ROOT / 'training/runs/e1-full/translation_snapshots'
    pairs, source_meta = [], {}
    for dataset, field in [('aro_visual_relation', 'false_caption'), ('aro_visual_attribution', 'false_caption'), ('sugarcrepe', 'negative_caption')]:
        path = snapshots / (dataset + '.jsonl')
        meta = read_json(path.with_suffix('.json'))
        assert file_sha(path) == meta['sha256']
        source_meta[dataset] = meta['sha256']
        selected = {r['translation_id']: i for i, r in enumerate(data.train) if r['dataset'] == dataset}
        seen = set()
        for line in path.open(encoding='utf-8'):
            patch = json.loads(line)
            if patch['id'] in selected:
                i = selected[patch['id']]
                positive_field = 'caption' if dataset == 'sugarcrepe' else 'true_caption'
                assert patch['translated'][positive_field] == data.train[i]['text_pl']
                pairs.append((i, patch['translated'][field], dataset))
                seen.add(patch['id'])
        assert seen == set(selected)
    negatives = sorted({s for _, s, _ in pairs})
    config = read_json(ROOT / 'training/runs/e1-full/config.json')
    lock = read_json(ROOT / 'training/runs/e1-full/cache.json')['model_lock']
    features, meta = encode_cached(negatives, 'texts', config, lock, OUT / 'negative_cache')
    # Check that freshly encoded negatives use the same numerical text pipeline.
    probe_indices = [next(i for i, r in enumerate(data.train) if r['dataset'] == ds) + j
                     for ds in sorted(source_meta) for j in range(8)]
    probe_text = [data.train[i]['text_pl'] for i in probe_indices]
    probe, _ = encode_cached(probe_text, 'texts', config, lock, OUT / 'positive_parity')
    parity = F.cosine_similarity(probe, data.tensors['train'][1][probe_indices])
    assert float(parity.min()) > .999, 'Text pipeline drift between cached gold and new negatives'
    lookup = {s: i for i, s in enumerate(negatives)}
    tt = data.tensors['train'][1]
    result = {}
    for dataset in sorted(source_meta):
        subset = [(i, s) for i, s, ds in pairs if ds == dataset]
        pos = F.normalize(tt[[i for i, _ in subset]], dim=1)
        neg = F.normalize(features[[lookup[s] for _, s in subset]], dim=1)
        cos = (pos * neg).sum(1).clamp(-1, 1)
        # Raw cosine thresholds are misleading in an anisotropic embedding space.
        random_cos = (pos * pos.roll(137, dims=0)).sum(1)
        result[dataset] = dict(pairs=len(subset), cosine_quantiles=torch.quantile(cos, torch.tensor([0., .1, .5, .9, .99, 1.])).tolist(),
                               random_caption_cosine_quantiles=torch.quantile(random_cos, torch.tensor([0., .1, .5, .9, .99, 1.])).tolist(),
                               fraction_above_095=float((cos > .95).float().mean()),
                               exact_same_string=sum(data.train[i]['text_pl'] == s for i, s in subset),
                               near_identical_vectors=int((cos > .99999).sum()))
    return dict(sources=source_meta, cache=meta, by_source=result, fresh_vs_cached_positive_min_cosine=float(parity.min()),
                caveat='High cosine restricts attainable score margin, but is not impossibility unless vectors coincide. This audit changes no loss or snapshot.')


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = base.Inputs(OUT)
    meta = read_json(base.OUT / 'cache/complete.json')
    assert file_sha(base.OUT / 'cache/features.pt') == meta['files']['features.pt']
    x = torch.load(base.OUT / 'cache/features.pt', weights_only=True)['projected']
    result = dict(inputs=data.hashes, validation_graph=data.graph['sha256'], test_used=False, xm3600_used=False)
    for name, function in [('prototypes', lambda: prototypes(data)), ('ridge', lambda: ridge(data, x)),
                           ('scalar_diagnostic', lambda: scalar_diagnostic(data, x)), ('hard_negatives', lambda: hard_negatives(data))]:
        result[name] = function()
        write_json(OUT / 'diagnostics.json', result)
        write_json(ROOT / 'reports/tinyclip_diagnostics.json', result)
        print(f'DIAGNOSTIC {name} complete', flush=True)
    data.verify()
    write_json(OUT / 'status.json', dict(status='complete', snapshot_unchanged=True, test_used=False, xm3600_used=False))

