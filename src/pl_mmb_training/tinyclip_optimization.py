"""Validation-only LR/schedule and linear-probe-then-finetune experiments."""
import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import tinyclip_adaptation as base
from .common import ROOT, digest, file_sha, read_json, write_json
from .losses import PairLoss
from .retrieval import evaluate_graph

OUT = ROOT / 'training/runs/tinyclip-optimization'
CACHE = base.OUT / 'cache'
WARM_START = base.OUT / 'A1/best.pt'


def lr_factor(step, total_steps, warmup_steps=0, cosine=False, floor=.1):
    """LR for the next optimizer update, using a zero-based update index."""
    if not 0 <= step < total_steps or not 0 <= warmup_steps < total_steps:
        raise ValueError('Invalid scheduler step or warmup')
    if not 0 <= floor <= 1:
        raise ValueError('Invalid cosine floor')
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if not cosine:
        return 1.
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * progress))


def specs():
    common = dict(seed=42, feature='projected', hidden=0, projector_lr=1e-3,
                  epochs=20, batch_size=256, warmup_epochs=0, cosine=False,
                  cosine_floor=.1, init='scratch', weight_decay=.01,
                  patience=6, minimum_epochs=10, gradient_clipping=None)
    values = [
        ('S0_frozen_schedule', dict(blocks=0, vision_lr=0., warmup_epochs=2, cosine=True)),
        ('S1_last1_low_constant', dict(blocks=1, vision_lr=3e-6)),
        ('S2_last1_schedule', dict(blocks=1, vision_lr=1e-5, warmup_epochs=2, cosine=True)),
        ('S3_last1_low_schedule', dict(blocks=1, vision_lr=3e-6, warmup_epochs=2, cosine=True)),
        ('S4_last2_low_schedule', dict(blocks=2, vision_lr=3e-6, warmup_epochs=2, cosine=True)),
    ]
    warm = dict(init='A1_epoch9', projector_lr=1e-4, epochs=12, warmup_epochs=1,
                cosine=True, minimum_epochs=8)
    values += [
        ('T0_frozen_aligned', dict(warm, blocks=0, vision_lr=0.)),
        ('T1_last1_aligned', dict(warm, blocks=1, vision_lr=1e-6)),
        ('T2_last2_aligned', dict(warm, blocks=2, vision_lr=1e-6)),
    ]
    return {name: dict(common, **change) for name, change in values}


def atomic_save(value, path):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def norm(parameters, gradient=False):
    tensors = [p.grad if gradient else p for p in parameters]
    values = [t.detach().float().square().sum() for t in tensors if t is not None]
    return torch.stack(values).sum().sqrt() if values else torch.tensor(0., device='cuda')


def train(data, name, spec, cache_meta):
    dest = data.output / name
    dest.mkdir(exist_ok=True)
    source = {p: file_sha(ROOT / p) for p in (
        'src/pl_mmb_training/tinyclip_optimization.py', 'src/pl_mmb_training/tinyclip_adaptation.py',
        'src/pl_mmb_training/losses.py', 'src/pl_mmb_training/retrieval.py')}
    identity = dict(inputs=data.identity, spec=spec, source=source, cache=cache_meta['files'],
                    warm_start_sha256=file_sha(WARM_START) if spec['init'] != 'scratch' else None,
                    torch_version=torch.__version__, selection_uses_test=False, xm3600_used=False)
    run_id = digest(identity)
    result_path = dest / 'result.json'
    if result_path.exists():
        result = read_json(result_path)
        assert result['run_id'] == run_id and file_sha(dest / 'best.pt') == result['checkpoint_sha256']
        return result
    write_json(dest / 'config.json', identity)
    write_json(data.output / 'status.json', dict(status='running', experiment=name, pid=os.getpid()))
    torch.manual_seed(spec['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    x = torch.load(CACHE / 'features.pt', weights_only=True)['projected'].to('cuda')
    model = base.projector(x.shape[1], 0).to('cuda')
    criterion = PairLoss('siglip_sigmoid', .07, -10.).to('cuda')
    if spec['init'] != 'scratch':
        initial = torch.load(WARM_START, weights_only=True, map_location='cpu')
        assert initial['epoch'] == 9 and initial['vision_suffix'] is None
        model.load_state_dict(initial['projector'])
        criterion.load_state_dict(initial['loss'])
    suffix = None
    prefix = None
    if spec['blocks']:
        vision, _, _ = base.load_vision(data)
        suffix = base.Suffix(vision, 'projected', spec['blocks']).to('cuda')
        del vision
        prefix = np.load(CACHE / 'prefix.npy', mmap_mode='r')
    named_groups = [('projector', list(model.parameters()), spec['projector_lr'], .01),
                    ('loss', list(criterion.parameters()), spec['projector_lr'], 0.)]
    if suffix is not None:
        named_groups.append(('vision', [p for p in suffix.parameters() if p.requires_grad], spec['vision_lr'], .01))
    optimizer = torch.optim.AdamW([dict(params=p, lr=lr, weight_decay=wd, name=n)
                                  for n, p, lr, wd in named_groups])
    parameters = [p for _, ps, _, _ in named_groups for p in ps]
    initial_weights = {n: [p.detach().clone() for p in ps] for n, ps, _, _ in named_groups}
    trainable = sum(p.numel() for p in parameters)
    gen = torch.Generator().manual_seed(spec['seed'])
    ti, tt, positives = data.tensors['train']
    vi, vt, vp = data.tensors['validation']
    tt, vt = tt.to('cuda'), vt.to('cuda')
    val_batches = [list(range(s, min(s + 256, len(vi)))) for s in range(0, len(vi), 256)]
    val_masks = [vp.mask(i).to('cuda') for i in val_batches]
    steps_per_epoch = math.ceil(len(ti) / 256)
    total_steps = steps_per_epoch * spec['epochs']
    warmup_steps = steps_per_epoch * spec['warmup_epochs']
    history, best, first_epoch, update = [], None, 1, 0
    elapsed_before, recorded_peak = 0., 0
    resume_path = dest / 'last.pt'
    if resume_path.exists():
        resume = torch.load(resume_path, weights_only=True, map_location='cpu')
        assert resume['run_id'] == run_id
        model.load_state_dict(resume['projector'])
        criterion.load_state_dict(resume['loss'])
        if suffix is not None:
            suffix.load_state_dict(resume['vision_suffix'])
        optimizer.load_state_dict(resume['optimizer'])
        gen.set_state(resume['generator'])
        torch.set_rng_state(resume['torch_rng'])
        torch.cuda.set_rng_state(resume['cuda_rng'])
        history, best = resume['history'], resume['best']
        first_epoch, update = resume['epoch'] + 1, resume['update']
        if resume['stopped']:
            first_epoch = spec['epochs'] + 1
        elapsed_before = history[-1]['elapsed_seconds']
        recorded_peak = max(r['peak_vram_bytes'] for r in history)
        (dest / 'epochs.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in history), encoding='utf-8')
    else:
        (dest / 'epochs.jsonl').write_text('', encoding='utf-8')
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    def features(indices):
        if suffix is None:
            return x[indices.to('cuda')]
        tokens = torch.from_numpy(np.array(prefix[indices.numpy()], copy=True)).to('cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return suffix(tokens).float()

    def checkpoint(epoch):
        return dict(projector=model.state_dict(), loss=criterion.state_dict(),
                    vision_suffix=suffix.state_dict() if suffix is not None else None,
                    spec=spec, epoch=epoch, run_id=run_id)

    def monitor(epoch, train_loss, diagnostics):
        model.eval()
        if suffix is not None:
            suffix.eval()
        with torch.inference_mode():
            current = x if suffix is None else torch.zeros_like(x)
            needed = torch.unique(vi)
            if suffix is not None:
                for part in needed.split(128):
                    current[part.to('cuda')] = features(part)
            val_loss = sum(float(criterion(model(current[vi[i].to('cuda')]), vt[i], mask)) * len(i)
                           for i, mask in zip(val_batches, val_masks)) / len(vi)
            graph = data.graph
            projected = model(current[graph['image_indices']]).cpu()
            texts = data.texts[graph['text_indices']]
            local = dict(graph, image_indices=list(range(len(projected))), text_indices=list(range(len(texts))))
            retrieval = evaluate_graph(local, projected, texts)
            drift = float((1 - nn.functional.cosine_similarity(current[needed.to('cuda')], x[needed.to('cuda')])).mean())
            deltas = {}
            for n, ps, _, _ in named_groups:
                numerator = torch.stack([(p - p0).square().sum() for p, p0 in zip(ps, initial_weights[n])]).sum().sqrt()
                denominator = torch.stack([p0.square().sum() for p0 in initial_weights[n]]).sum().sqrt()
                deltas[n] = float(numerator / denominator.clamp_min(1e-20))
        row = dict(epoch=epoch, train_loss=train_loss, validation_loss=val_loss, validation=retrieval,
                   diagnostics=diagnostics, parameter_relative_drift=deltas, validation_feature_cosine_drift=drift,
                   peak_vram_bytes=max(recorded_peak, torch.cuda.max_memory_allocated()), trainable_params=trainable,
                   elapsed_seconds=elapsed_before + time.monotonic() - started, updates=update, **criterion.scalars())
        history.append(row)
        with (dest / 'epochs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(f'{name} epoch={epoch} train={train_loss} val={val_loss:.4f} '
              f't2i={retrieval["t2i"]["recall@1"]:.4f} i2t={retrieval["i2t"]["recall@1"]:.4f} '
              f'drift={drift:.5f}', flush=True)
        return row

    if not history:
        best = monitor(0, None, {})
        atomic_save(checkpoint(0), dest / 'best.pt')
    for epoch in range(first_epoch, spec['epochs'] + 1):
        model.train()
        if suffix is not None:
            suffix.train()
        aggregates = {n: dict(gradient_norm_sum=0., gradient_norm_max=0., relative_step_sum=0., relative_step_max=0.,
                              lr_first=None, lr_last=None, lr_min=float('inf'), lr_max=0.) for n, _, _, _ in named_groups}
        total, batches = 0., 0
        for part in torch.randperm(len(ti), generator=gen).split(256):
            indices = part.tolist()
            multiplier = lr_factor(update, total_steps, warmup_steps, spec['cosine'], spec['cosine_floor'])
            for group, (_, _, peak_lr, _) in zip(optimizer.param_groups, named_groups):
                group['lr'] = peak_lr * multiplier
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features(ti[part])), tt[part], positives.mask(indices).to('cuda'))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            before = {}
            for group, (n, ps, _, _) in zip(optimizer.param_groups, named_groups):
                g = float(norm(ps, gradient=True))
                if not math.isfinite(g):
                    raise ValueError('Nonfinite gradient: ' + n)
                a = aggregates[n]
                a['gradient_norm_sum'] += g
                a['gradient_norm_max'] = max(a['gradient_norm_max'], g)
                a['lr_first'] = group['lr'] if a['lr_first'] is None else a['lr_first']
                a['lr_last'] = group['lr']
                a['lr_min'] = min(a['lr_min'], group['lr'])
                a['lr_max'] = max(a['lr_max'], group['lr'])
                # Sample actual AdamW update size every 25 steps; includes decay and preconditioning.
                if update % 25 == 0:
                    before[n] = [p.detach().clone() for p in ps]
            optimizer.step()
            criterion.constrain()
            for n, old in before.items():
                ps = next(ps for gn, ps, _, _ in named_groups if gn == n)
                delta = torch.stack([(p.detach() - p0).square().sum() for p, p0 in zip(ps, old)]).sum().sqrt()
                denominator = torch.stack([p0.square().sum() for p0 in old]).sum().sqrt().clamp_min(1e-20)
                relative = float(delta / denominator)
                a = aggregates[n]
                a['relative_step_sum'] += relative
                a['relative_step_max'] = max(a['relative_step_max'], relative)
                a['sampled_steps'] = a.get('sampled_steps', 0) + 1
            update += 1
            batches += 1
            total += float(loss.detach()) * len(part)
        for a in aggregates.values():
            a['gradient_norm_mean'] = a.pop('gradient_norm_sum') / batches
            a['relative_step_mean'] = a.pop('relative_step_sum') / max(1, a.get('sampled_steps', 0))
        row = monitor(epoch, total / len(ti), aggregates)
        if base.checkpoint_key(row) > base.checkpoint_key(best):
            best = row
            atomic_save(checkpoint(epoch), dest / 'best.pt')
        stopped = epoch >= spec['minimum_epochs'] and epoch - best['epoch'] >= spec['patience']
        atomic_save(dict(checkpoint(epoch), optimizer=optimizer.state_dict(), generator=gen.get_state(),
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), history=history,
                         best=best, update=update, stopped=stopped), resume_path)
        if stopped:
            break
    selected = torch.load(dest / 'best.pt', weights_only=True, map_location='cpu')
    assert selected['epoch'] == best['epoch']
    data.verify()
    result = dict(experiment=name, spec=spec, run_id=run_id, status='complete', history=history,
                  best_epoch=best['epoch'], selected_validation=best['validation'], validation_loss=best['validation_loss'],
                  trainable_params=trainable, peak_vram_bytes=max(r['peak_vram_bytes'] for r in history),
                  checkpoint_sha256=file_sha(dest / 'best.pt'), selection_uses_test=False, xm3600_used=False,
                  optimization_steps=update, starting_validation=history[0]['validation'])
    write_json(result_path, result)
    write_json(dest / 'selection.json', dict(best_epoch=best['epoch'], validation=best['validation'], test_used=False))
    del model, criterion, optimizer, suffix, x
    torch.cuda.empty_cache()
    return result


def export(output):
    results = {p.parent.name: read_json(p) for p in sorted(output.glob('*/result.json'))}
    old = {n: read_json(base.OUT / n / 'result.json') for n in ('A1', 'C1', 'C2')}
    rows, epochs = [], []
    for name, r in [*[(n, r) for n, r in old.items()], *results.items()]:
        s = r.get('spec', {})
        v = r['selected_validation']
        row = dict(experiment=name, init=s.get('init', 'scratch'), vision_blocks=s.get('blocks', r.get('vision_blocks', 0)),
                   projector_lr=s.get('projector_lr', r.get('projector_lr', .001)), vision_lr=s.get('vision_lr', r.get('vision_lr', 0.)),
                   warmup_epochs=s.get('warmup_epochs', 0), cosine=s.get('cosine', False),
                   trainable_params=r['trainable_params'], best_epoch=r.get('best_epoch'),
                   val_t2i_r1=v['t2i']['recall@1'], val_t2i_r5=v['t2i']['recall@5'], val_t2i_r10=v['t2i']['recall@10'],
                   val_i2t_r1=v['i2t']['recall@1'], peak_vram_gib=r['peak_vram_bytes'] / 2**30,
                   gain_vs_original_frozen_pp=100*(v['t2i']['recall@1']-old['A1']['selected_validation']['t2i']['recall@1']))
        rows.append(row)
        if name in old:
            continue
        for e in r['history']:
            epochs.append(dict(experiment=name, **e))
    write_json(ROOT / 'reports/tinyclip_optimization.json', dict(comparison=rows, runs=results, selection_uses_test=False, xm3600_used=False))
    if rows:
        with (ROOT / 'reports/tinyclip_optimization.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (ROOT / 'reports/tinyclip_optimization_epochs.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in epochs), encoding='utf-8')
    lines = ['# TinyCLIP: learning rate, warmup and aligned-head finetuning', '',
             'FULL_CLEAN and validation unchanged. Seed 42. No test or XM3600 evaluation.',
             'Native projected TinyCLIP-39M features, linear projector, frozen PolDense, unchanged multi-positive sigmoid.', '',
             '| experiment | init | blocks | projector LR | vision LR | warmup epochs | cosine | trainable params | val t2i R@1 | val i2t R@1 | best epoch |',
             '|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["experiment"]} | {r["init"]} | {r["vision_blocks"]} | {r["projector_lr"]:g} | '
                     f'{r["vision_lr"]:g} | {r["warmup_epochs"]} | {r["cosine"]} | {r["trainable_params"]:,} | '
                     f'{100*r["val_t2i_r1"]:.2f}% | {100*r["val_i2t_r1"]:.2f}% | {r["best_epoch"]} |')
    lines += ['', 'S: train from scratch, 20 epoch ceiling. T: same A1 epoch-9 projector/loss, fresh optimizer, 12 additional epochs.',
              'T0 matches continuation of the head without vision updates. T1/T2 start from the identical head and loss scalars.',
              'Warmup is linear per update; cosine decays to 10% of peak LR. Loss-scale/bias LR follows projector LR.',
              'Early stopping: patience 6, at least 10 epochs for S / 8 for T. Epoch 0 is eligible to preserve the starting checkpoint.',
              'Diagnostics include each group LR, gradient norms, sampled actual AdamW relative update size, parameter drift and feature drift.',
              'Peak VRAM is peak PyTorch allocated memory. Each experiment uses one seed.', '',
              'Rationale for aligned-head adaptation: https://arxiv.org/abs/2202.10054',
              'Scheduler semantics: https://docs.pytorch.org/docs/stable/optim.html', '']
    (ROOT / 'reports/tinyclip_optimization.md').write_text('\n'.join(lines), encoding='utf-8')


def verify(data, cache_meta):
    data.verify()
    checks = {}
    for name, spec in specs().items():
        result = read_json(data.output / name / 'result.json')
        chosen = max(result['history'], key=base.checkpoint_key)
        assert chosen['epoch'] == result['best_epoch']
        assert chosen['validation'] == result['selected_validation']
        assert chosen['validation']['graph_sha256'] == data.graph['sha256']
        assert result['spec'] == spec
        assert not result['selection_uses_test'] and not result['xm3600_used']
        checkpoint = data.output / name / 'best.pt'
        assert file_sha(checkpoint) == result['checkpoint_sha256']
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
        assert saved['epoch'] == chosen['epoch'] and saved['run_id'] == result['run_id']
        for e in result['history'][1:]:
            for g, d in e['diagnostics'].items():
                peak = spec['vision_lr'] if g == 'vision' else spec['projector_lr']
                assert 0 < d['lr_min'] <= d['lr_max'] <= peak * (1+1e-12)
                assert math.isfinite(d['gradient_norm_mean']) and math.isfinite(d['relative_step_mean'])
        checks[name] = dict(checkpoint_verified=True, selection_verified=True, epochs=len(result['history'])-1)
    write_json(ROOT / 'reports/tinyclip_optimization_verification.json', dict(status='passed', runs=checks,
               unchanged_inputs=data.hashes, cache=cache_meta['files'], selection_uses_test=False, xm3600_used=False))
    print('Verification passed for all eight experiments', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['all', 'scout', 'aligned', 'report', 'verify'])
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'report':
        export(args.output)
        return
    import fcntl
    with (args.output / 'worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(str(os.getpid()))
        lock.flush()
        torch.set_num_threads(4)
        data = base.Inputs(args.output)
        cache_meta = read_json(CACHE / 'complete.json')
        assert cache_meta['identity']['inputs'] == data.identity and cache_meta['identity']['images'] == data.images
        for filename, sha in cache_meta['files'].items():
            assert file_sha(CACHE / filename) == sha
        if args.stage == 'verify':
            verify(data, cache_meta)
            return
        write_json(args.output / 'plan.json', dict(experiments=specs(), original_baseline='A1',
                   selection_uses_test=False, xm3600_used=False, no_kd=True,
                   rationale='isolate lower vision LR, scheduled optimization, and aligned-head initialization with matched frozen controls'))
        for name, spec in specs().items():
            if args.stage == 'scout' and not name.startswith('S'):
                continue
            if args.stage == 'aligned' and not name.startswith('T'):
                continue
            train(data, name, spec, cache_meta)
            export(args.output)
        if args.stage == 'all':
            results = [read_json(args.output / n / 'result.json') for n in specs()]
            winner = max(results, key=base.key)
            write_json(args.output / 'selection.json', dict(winner=winner['experiment'], validation=winner['selected_validation'],
                       selection_uses_test=False, xm3600_used=False))
            verify(data, cache_meta)
        write_json(args.output / 'status.json', dict(status='complete', stage=args.stage, pid=os.getpid()))


if __name__ == '__main__':
    main()
