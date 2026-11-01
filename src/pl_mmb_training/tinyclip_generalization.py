"""Matched native/augmented controls and regularized partial vision adaptation."""
import copy
import gc
import json
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import tinyclip_adaptation as base
from . import tinyclip_optimization as opt
from .common import ROOT, digest, file_sha, read_json, write_json
from .losses import PairLoss
from .retrieval import evaluate_graph

OUT = ROOT / 'training/runs/tinyclip-generalization'
CACHE = base.OUT / 'cache'


def augment(pixels, generator):
    """Small isotropic zoom and translation; no reflection of the scene or color edits.

    Every view retains at least 85.7% of the native crop's area before translation.
    Reflection padding is only a boundary fill, not a horizontal image flip.
    """
    values = torch.rand((len(pixels), 3), generator=generator)
    theta = torch.zeros((len(pixels), 2, 3), device=pixels.device)
    scale = (1 / (1 + .08 * values[:, 0])).to(pixels.device)
    theta[:, 0, 0] = scale
    theta[:, 1, 1] = scale
    # grid coordinates span [-1,1], so 0.06 corresponds to 3% image width/height.
    theta[:, :, 2] = ((values[:, 1:] * 2 - 1) * .06).to(pixels.device)
    grid = F.affine_grid(theta, pixels.shape, align_corners=False)
    return F.grid_sample(pixels, grid, mode='bilinear', padding_mode='reflection', align_corners=False)


def geometry_loss(features, original):
    # Equivalent to 1-cosine, but nonnegative with exactly zero gradient at identity.
    return .5 * (F.normalize(features, dim=1) - F.normalize(original.detach(), dim=1)).square().sum(1).mean()


def specs():
    common = dict(seed=42, blocks=0, augmentation=False, anchor_lambda=0.,
                  epochs=12, batch_size=256, projector_lr=1e-4, vision_lr=1e-5,
                  warmup_epochs=1, cosine_floor=.1, weight_decay=.01,
                  init='A1_epoch9', early_stopping=False)
    return {name: dict(common, **change) for name, change in [
        ('G0_frozen_native', {}),
        ('G1_frozen_aug', dict(augmentation=True)),
        ('G2_last2_aug', dict(blocks=2, augmentation=True)),
        ('G3_last2_aug_anchor', dict(blocks=2, augmentation=True, anchor_lambda=50.)),
    ]}


def pixels_cache(data, model, prep, stats):
    folder = OUT / 'pixels'
    folder.mkdir(exist_ok=True)
    path, metadata = folder / 'native_rgb.npy', folder / 'complete.json'
    identity = dict(images=data.images, image_sha256=[data.expected_sha[p] for p in data.images],
                    inputs=data.hashes, preprocessing=stats['preprocessing'], storage='native crop RGB uint8, no vision features cached for augmented views')
    if metadata.exists():
        meta = read_json(metadata)
        assert meta['identity'] == identity and file_sha(path) == meta['sha256']
        return np.load(path, mmap_mode='r'), meta
    required = len(data.images) * 3 * 224 * 224
    if shutil.disk_usage(OUT).free < required + 2_000_000_000:
        raise RuntimeError('Insufficient space for RGB cache plus checkpoint reserve')
    pixels = np.lib.format.open_memmap(path, mode='w+', dtype='uint8', shape=(len(data.images), 3, 224, 224))
    mean = torch.tensor(stats['preprocessing']['image_mean'])[None, :, None, None]
    std = torch.tensor(stats['preprocessing']['image_std'])[None, :, None, None]

    def load(path):
        assert file_sha(path) == data.expected_sha[path]
        with Image.open(path) as im:
            return im.convert('RGB')

    with ThreadPoolExecutor(max_workers=8) as pool, torch.inference_mode():
        for start in range(0, len(data.images), 64):
            opened = list(pool.map(load, data.images[start:start + 64]))
            try:
                native = prep(opened)
                rgb = ((native * std + mean) * 255).round().clamp(0, 255).byte()
                restored = (rgb.float() / 255 - mean) / std
                assert float((native - restored).abs().max()) < 2e-6
                pixels[start:start + len(opened)] = rgb.numpy()
            finally:
                for im in opened:
                    im.close()
            if start % 1024 == 0:
                print(f'RGB CACHE {start}/{len(data.images)}', flush=True)
    pixels.flush()
    # Verify the storage roundtrip against the established native vision features.
    reference = torch.load(CACHE / 'features.pt', weights_only=True)['projected'][:32].to('cuda')
    restored = (torch.from_numpy(np.array(pixels[:32], copy=True)).to('cuda').float() / 255 - mean.to('cuda')) / std.to('cuda')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        actual = model(pixel_values=restored).image_embeds.float()
    parity = float(F.cosine_similarity(reference, actual).min())
    assert parity > .999, 'RGB cache changed native vision features'
    meta = dict(identity=identity, sha256=file_sha(path), native_feature_min_cosine=parity)
    write_json(metadata, meta)
    return np.load(path, mmap_mode='r'), meta


def train(data, name, spec, native, prefix, rgb, vision, stats, cache_meta, rgb_meta):
    dest = OUT / name
    dest.mkdir(exist_ok=True)
    source = {p: file_sha(ROOT / p) for p in ('src/pl_mmb_training/tinyclip_generalization.py',
              'src/pl_mmb_training/tinyclip_adaptation.py', 'src/pl_mmb_training/tinyclip_optimization.py',
              'src/pl_mmb_training/losses.py', 'src/pl_mmb_training/retrieval.py')}
    identity = dict(inputs=data.identity, source=source, spec=spec,
                    feature_cache=cache_meta['files'], rgb_cache_sha256=rgb_meta['sha256'],
                    warm_start_sha256=file_sha(opt.WARM_START),
                    schedule='12 complete epochs; update-level 1-epoch warmup then cosine to 0.1; same monitoring in all controls',
                    selection=data.identity['selection'], test_used=False, xm3600_used=False)
    run_id = digest(identity)
    if (dest / 'result.json').exists():
        result = read_json(dest / 'result.json')
        assert result['run_id'] == run_id and result['checkpoint_sha256'] == file_sha(dest / 'best.pt')
        return result
    write_json(dest / 'config.json', identity)
    write_json(OUT / 'status.json', dict(status='running', experiment=name, pid=os.getpid()))
    torch.manual_seed(spec['seed'])
    model = base.projector(512, 0).to('cuda')
    criterion = PairLoss('siglip_sigmoid').to('cuda')
    initial = torch.load(opt.WARM_START, weights_only=True, map_location='cpu')
    assert initial['epoch'] == 9 and initial['vision_suffix'] is None
    model.load_state_dict(initial['projector'])
    criterion.load_state_dict(initial['loss'])
    fixed_criterion = copy.deepcopy(criterion).requires_grad_(False)
    suffix = base.Suffix(vision, 'projected', spec['blocks']).to('cuda') if spec['blocks'] else None
    groups = [dict(name='projector', params=list(model.parameters()), lr=spec['projector_lr'], weight_decay=.01),
              dict(name='loss', params=list(criterion.parameters()), lr=spec['projector_lr'], weight_decay=0.)]
    if suffix is not None:
        groups.append(dict(name='vision', params=[p for p in suffix.parameters() if p.requires_grad], lr=spec['vision_lr'], weight_decay=.01))
    optimizer = torch.optim.AdamW(groups)
    trainable = sum(p.numel() for g in optimizer.param_groups for p in g['params'])
    initial_vision = [p.detach().clone() for p in suffix.parameters() if p.requires_grad] if suffix is not None else []
    gen = torch.Generator().manual_seed(spec['seed'])
    aug_gen = torch.Generator().manual_seed(spec['seed'] + 10000)
    ti, tt, positives = data.tensors['train']
    vi, vt, vp = data.tensors['validation']
    tt, vt = tt.to('cuda'), vt.to('cuda')
    val_batches = [(p, vp.mask(p.tolist()).to('cuda')) for p in torch.arange(len(vi)).split(256)]
    mean = torch.tensor(stats['preprocessing']['image_mean'], device='cuda')[None, :, None, None]
    std = torch.tensor(stats['preprocessing']['image_std'], device='cuda')[None, :, None, None]
    steps = math.ceil(len(ti) / spec['batch_size'])
    history, best, update, first_epoch, elapsed_before = [], None, 0, 1, 0.
    resume_path = dest / 'last.pt'
    if resume_path.exists():
        saved = torch.load(resume_path, weights_only=True, map_location='cpu')
        assert saved['run_id'] == run_id
        model.load_state_dict(saved['projector'])
        criterion.load_state_dict(saved['loss'])
        if suffix is not None:
            suffix.load_state_dict(saved['vision_suffix'])
        optimizer.load_state_dict(saved['optimizer'])
        gen.set_state(saved['generator'])
        aug_gen.set_state(saved['augmentation_generator'])
        history, best, update = saved['history'], saved['best'], saved['updates']
        first_epoch = saved['epoch'] + 1
        elapsed_before = history[-1]['elapsed_seconds']
    (dest / 'epochs.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in history), encoding='utf-8')
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    captured = []
    handle = vision.vision_model.encoder.layers[-2].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach())) if suffix is not None else None

    def checkpoint(epoch):
        return dict(projector=model.state_dict(), loss=criterion.state_dict(),
                    vision_suffix=suffix.state_dict() if suffix is not None else None,
                    epoch=epoch, updates=update, run_id=run_id, spec=spec)

    def monitor(epoch, totals, count, diagnostics):
        nonlocal best
        model.eval()
        if suffix is not None:
            suffix.eval()
        with torch.inference_mode():
            current = native if suffix is None else torch.zeros_like(native)
            needed = torch.unique(vi)
            if suffix is not None:
                for part in needed.split(128):
                    tokens = torch.from_numpy(np.array(prefix[part.numpy()], copy=True)).to('cuda')
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        current[part.to('cuda')] = suffix(tokens).float()
            val, fixed = 0., 0.
            for part, mask in val_batches:
                projected = model(current[vi[part].to('cuda')])
                val += float(criterion(projected, vt[part], mask)) * len(part)
                fixed += float(fixed_criterion(projected, vt[part], mask)) * len(part)
            graph = data.graph
            projected = model(current[graph['image_indices']]).cpu()
            texts = data.texts[graph['text_indices']]
            local = dict(graph, image_indices=list(range(len(projected))), text_indices=list(range(len(texts))))
            retrieval = evaluate_graph(local, projected, texts)
            drift = float(geometry_loss(current[needed.to('cuda')], native[needed.to('cuda')]))
            param_drift = 0.
            if suffix is not None:
                params = [p for p in suffix.parameters() if p.requires_grad]
                param_drift = float(torch.stack([(p-p0).square().sum() for p,p0 in zip(params, initial_vision)]).sum().sqrt() /
                                    torch.stack([p.square().sum() for p in initial_vision]).sum().sqrt())
        row = dict(epoch=epoch, updates=update, train_loss=totals['gold']/count if count else None,
                   train_anchor_loss=totals['anchor']/count if count else None,
                   train_total_loss=totals['total']/count if count else None,
                   validation_loss=val/len(vi), validation_loss_fixed_scalars=fixed/len(vi), validation=retrieval,
                   validation_feature_cosine_drift=drift, vision_parameter_relative_drift=param_drift,
                   gradient_norm=diagnostics, trainable_params=trainable, peak_vram_bytes=torch.cuda.max_memory_allocated(),
                   elapsed_seconds=elapsed_before + time.monotonic()-started, **criterion.scalars())
        history.append(row)
        with (dest / 'epochs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(f'{name} epoch={epoch:.3f} t2i={retrieval["t2i"]["recall@1"]:.5f} i2t={retrieval["i2t"]["recall@1"]:.5f} val={row["validation_loss"]:.4f} drift={drift:.6f} seconds={row["elapsed_seconds"]:.1f}', flush=True)
        if best is None or base.checkpoint_key(row) > base.checkpoint_key(best):
            best = row
            opt.atomic_save(checkpoint(epoch), dest / 'best.pt')
        model.train()
        if suffix is not None:
            suffix.train()

    try:
        if not history:
            monitor(0., dict(gold=0., anchor=0., total=0.), 0, {})
        for epoch in range(first_epoch, spec['epochs'] + 1):
            totals, count = dict(gold=0., anchor=0., total=0.), 0
            grad_sums = {g['name']: 0. for g in optimizer.param_groups}
            grad_max = dict(grad_sums)
            checkpoints = {math.ceil(steps * f) for f in (.25, .5, .75, 1.)} if epoch <= 2 else {steps}
            permutation = torch.randperm(len(ti), generator=gen)
            for batch_no, part in enumerate(permutation.split(spec['batch_size']), 1):
                multiplier = opt.lr_factor(update, steps * spec['epochs'], steps * spec['warmup_epochs'], True, spec['cosine_floor'])
                for g in optimizer.param_groups:
                    g['lr'] = (spec['vision_lr'] if g['name'] == 'vision' else spec['projector_lr']) * multiplier
                optimizer.zero_grad(set_to_none=True)
                ii = ti[part]
                if spec['augmentation']:
                    pixels = (rgb[ii.to('cuda')].float() / 255 - mean) / std
                    pixels = augment(pixels, aug_gen)
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                        original = vision(pixel_values=pixels).image_embeds.float()
                    if suffix is not None:
                        tokens = captured.pop()
                        assert not captured and not tokens.requires_grad
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            features = suffix(tokens).float()
                        if update == 0:
                            assert float(F.cosine_similarity(features.detach(), original).min()) > .9999
                    else:
                        features = original
                else:
                    features = original = native[ii.to('cuda')]
                anchor = geometry_loss(features, original) if spec['anchor_lambda'] else features.new_zeros(())
                gold = criterion(model(features), tt[part], positives.mask(part.tolist()).to('cuda'))
                loss = gold + spec['anchor_lambda'] * anchor
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                loss.backward()
                for g in optimizer.param_groups:
                    norm = float(opt.norm(g['params'], gradient=True))
                    if not math.isfinite(norm):
                        raise RuntimeError('Non-finite gradient norm')
                    grad_sums[g['name']] += norm
                    grad_max[g['name']] = max(grad_max[g['name']], norm)
                optimizer.step()
                criterion.constrain()
                update += 1
                for k, value in [('gold', gold), ('anchor', anchor), ('total', loss)]:
                    totals[k] += float(value.detach()) * len(part)
                count += len(part)
                if batch_no % 50 == 0:
                    print(f'{name} epoch={epoch} batch={batch_no}/{steps} gold={float(gold.detach()):.4f}', flush=True)
                if batch_no in checkpoints:
                    diagnostics = {n: dict(mean=v/batch_no, maximum=grad_max[n]) for n,v in grad_sums.items()}
                    monitor(epoch - 1 + batch_no/steps, totals, count, diagnostics)
            opt.atomic_save(dict(checkpoint(epoch), optimizer=optimizer.state_dict(), history=history, best=best,
                                 generator=gen.get_state(), augmentation_generator=aug_gen.get_state()), resume_path)
        result = dict(experiment=name, run_id=run_id, spec=spec, history=history,
                      best_epoch=best['epoch'], selected_validation=best['validation'], trainable_params=trainable,
                      checkpoint_sha256=file_sha(dest / 'best.pt'), peak_vram_bytes=max(r['peak_vram_bytes'] for r in history),
                      selection_uses_test=False, test_mscoco=None, xm3600_used=False)
        assert max(history, key=base.checkpoint_key)['epoch'] == best['epoch']
        write_json(dest / 'result.json', result)
        return result
    finally:
        if handle is not None:
            handle.remove()


def export(results):
    write_json(ROOT / 'reports/tinyclip_generalization.json', dict(results=results, selection_uses_test=False, xm3600_used=False))
    lines = ['# TinyCLIP: augmentacja i ochrona geometrii', '',
             'FULL_CLEAN i walidacja bez zmian. PolDense frozen. Seed 42; wspólny start A1 epoka 9. ',
             'LR head 1e-4, vision 1e-5; warmup 1 epoka, cosine do 10%, pełne 12 epok.',
             'Checkpoint po val t2i R@1; gęstszy pomiar w pierwszych dwóch epokach, identyczny we wszystkich kontrolach.',
             'Augmentacja: zoom 1–1.08 i przesunięcie ±3%; bez flipów i zmian koloru.',
             'Anchor: 50 × (1-cos) do oryginalnych cech TinyCLIP na tym samym widoku. Siła z gradientów TRAIN.', '',
             '| Experiment | Feature source | Projector | Vision trainable blocks | Trainable params | Val t2i R@1 | Val i2t R@1 | Best epoch |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for r in results:
        v = r['selected_validation']
        lines.append(f'| {r["experiment"]} | projected TinyCLIP | Linear 512→384 | {r["spec"]["blocks"]} | {r["trainable_params"]:,} | {100*v["t2i"]["recall@1"]:.2f}% | {100*v["i2t"]["recall@1"]:.2f}% | {r["best_epoch"]:.3f} |')
    lines += ['', 'MSCOCO test i XM3600 nieuruchamiane. To scout jednego seeda; małe różnice nie są dowodem poprawy.']
    (ROOT / 'reports/tinyclip_generalization.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    import fcntl
    OUT.mkdir(exist_ok=True)
    with (OUT / 'worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        data = base.Inputs(OUT)
        cache_meta = read_json(CACHE / 'complete.json')
        assert cache_meta['identity']['inputs'] == data.identity and cache_meta['identity']['images'] == data.images
        for name, sha in cache_meta['files'].items():
            assert file_sha(CACHE / name) == sha
        vision, prep, stats = base.load_vision(data)
        assert not any(p.requires_grad for p in vision.parameters())
        pixels, rgb_meta = pixels_cache(data, vision, prep, stats)
        rgb = torch.from_numpy(np.array(pixels, copy=True)).to('cuda')
        native = torch.load(CACHE / 'features.pt', weights_only=True)['projected'].to('cuda')
        prefix = np.load(CACHE / 'prefix.npy', mmap_mode='r')
        write_json(OUT / 'plan.json', dict(experiments=specs(), pixel_cache=rgb_meta,
                   regularization_calibration=read_json(OUT / 'audit.json')['regularization_weights'],
                   test_used=False, xm3600_used=False))
        results = []
        for name, spec in specs().items():
            results.append(train(data, name, spec, native, prefix, rgb, vision, stats, cache_meta, rgb_meta))
            export(results)
            gc.collect()
            torch.cuda.empty_cache()
        data.verify()
        write_json(OUT / 'verification.json', dict(status='passed', inputs=data.hashes,
                   checkpoint_selection_verified=True, test_used=False, xm3600_used=False))
        write_json(OUT / 'status.json', dict(status='complete', experiments=list(specs())))

