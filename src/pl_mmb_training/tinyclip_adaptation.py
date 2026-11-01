"""Controlled, validation-only A/B/C study. Never opens a test dataset."""
import argparse
import copy
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .common import ROOT, digest, file_sha, read_json, write_json
from .commercial_backbones import load_backbone
from .losses import PairLoss, PositivePairs
from .retrieval import evaluate_graph, graph_from_records
from .cleanroom_training import checkpoint_key

OUT = ROOT / 'training/runs/tinyclip-adaptation'
CLEAN = ROOT / 'training/runs/clean-v1'


def records(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]


class Inputs:
    def __init__(self, output):
        self.output = output
        audit = read_json(CLEAN / 'audit.json')
        assert audit['known_remaining_overlap_count'] == 0
        self.hashes = {str(CLEAN / 'FULL_CLEAN.jsonl'): audit['variants']['FULL_CLEAN']['sha256'],
                       **audit['inputs']}
        # Freeze validation against the original data snapshot, rather than trusting a new split.
        self.train = records(CLEAN / 'FULL_CLEAN.jsonl')
        self.val = records(CLEAN / 'validation.jsonl')
        original = records(ROOT / 'training/runs/e1-full/data.jsonl')
        assert self.val == [r for r in original if r['split'] == 'validation']
        assert self.train == [r for r in original if r['split'] == 'train']
        self.hashes[str(CLEAN / 'validation.jsonl')] = file_sha(CLEAN / 'validation.jsonl')
        self.verify()
        assert {r['target_id'] for r in self.train}.isdisjoint(r['target_id'] for r in self.val)
        assert {p for r in self.train for p in r['parent_ids']}.isdisjoint(p for r in self.val for p in r['parent_ids'])
        self.images = sorted({r['image'] for r in self.train + self.val})
        self.ii = {p: i for i, p in enumerate(self.images)}
        self.expected_sha = {r['image']: r['sha256'] for r in self.train + self.val}
        manifest = read_json(ROOT / 'training/runs/e1-full/cache.json')
        meta = manifest['text_features']
        assert file_sha(meta['path']) == meta['sha256']
        text = torch.load(meta['path'], weights_only=True, map_location='cpu')
        ti = {s: i for i, s in enumerate(manifest['texts'])}
        self.texts = text
        self.graph = graph_from_records([r for r in self.val if r['dataset'] == 'mbeir_mscoco'],
                                        self.images, manifest['texts'])
        self.tensors = {}
        for split, rows in [('train', self.train), ('validation', self.val)]:
            self.tensors[split] = (torch.tensor([self.ii[r['image']] for r in rows]),
                                   text[[ti[r['text_pl']] for r in rows]], PositivePairs(rows))
        self.lock = read_json(ROOT / 'training/runs/vision-commercial-five/model_lock.json')['tinyclip-39m']
        self.config = read_json(ROOT / 'training/runs/vision-four/reference/artifacts/config.json')
        self.identity = dict(inputs=self.hashes, model=self.lock, text_features=meta,
                             validation_graph=self.graph['sha256'], seed=42,
                             optimization=self.config['optimization'], precision='fp32-weights-bf16-vision-autocast',
                             selection='validation_t2i_r1_then_mean_r1_then_earliest_epoch',
                             test_used=False, xm3600_used=False)
        p = output / 'inputs.json'
        if p.exists() and read_json(p) != self.identity:
            raise ValueError('Immutable inputs changed; choose another output')
        write_json(p, self.identity)
        write_json(output / 'validation_graph.json', self.graph)

    def verify(self):
        for path, sha in self.hashes.items():
            if file_sha(path) != sha:
                raise ValueError('Frozen input changed: ' + path)


def load_vision(data):
    model, prep, _, stats = load_backbone('tinyclip-39m', data.lock)
    model.to('cuda')
    return model, prep, stats


def cache(data):
    """Cache only train/validation images; frozen prefix is exact, without quantization."""
    folder = data.output / 'cache'
    folder.mkdir(exist_ok=True)
    identity = dict(inputs=data.identity, images=data.images, prefix='input_to_last_two_blocks_fp32')
    meta_path = folder / 'complete.json'
    if meta_path.exists():
        meta = read_json(meta_path)
        assert meta['identity'] == identity
        for name, sha in meta['files'].items():
            assert file_sha(folder / name) == sha
        return
    model, prep, _ = load_vision(data)
    n = len(data.images)
    width = model.config.hidden_size
    tokens = (model.config.image_size // model.config.patch_size) ** 2 + 1
    prefix = np.lib.format.open_memmap(folder / 'prefix.npy', mode='w+', dtype='float32', shape=(n, tokens, width))
    features = {'projected': [], 'pooled': []}
    captured = []
    handle = model.vision_model.encoder.layers[-2].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))

    def open_image(path):
        from PIL import Image
        import io
        payload = Path(path).read_bytes()
        if file_sha(path) != data.expected_sha[path]:
            raise ValueError('Image changed: ' + path)
        with Image.open(io.BytesIO(payload)) as im:
            return im.convert('RGB')

    with ThreadPoolExecutor(max_workers=8) as pool, torch.inference_mode():
        for start in range(0, n, 32):
            opened = list(pool.map(open_image, data.images[start:start + 32]))
            try:
                pixels = prep(opened).to('cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result = model.vision_model(pixel_values=pixels)
                    projected = model.visual_projection(result.pooler_output)
                features['pooled'].append(result.pooler_output.float().cpu())
                features['projected'].append(projected.float().cpu())
                prefix[start:start + len(opened)] = captured.pop().float().cpu().numpy()
            finally:
                for im in opened:
                    im.close()
            if start % 800 == 0:
                print(f'CACHE {start + len(opened)}/{n}', flush=True)
    handle.remove()
    prefix.flush()
    torch.save({k: torch.cat(v) for k, v in features.items()}, folder / 'features.pt')
    # Compare A1 against the established native, projected TinyCLIP cache.
    old = read_json(ROOT / 'training/runs/vision-commercial-five/models/tinyclip-39m/feature_cache.json')
    assert file_sha(old['path']) == old['sha256']
    old_x = torch.load(old['path'], weights_only=True, map_location='cpu')
    manifest = read_json(ROOT / 'training/runs/e1-full/cache.json')
    old_ii = {p: i for i, p in enumerate(manifest['images'])}
    new = torch.cat(features['projected'])
    reference = old_x[[old_ii[p] for p in data.images]]
    cosine = torch.nn.functional.cosine_similarity(new, reference).min().item()
    if cosine < .999:
        raise ValueError(f'Native projected feature parity failed: {cosine}')
    write_json(meta_path, dict(identity=identity, files={name: file_sha(folder / name) for name in ('prefix.npy', 'features.pt')},
                              projected_native_min_cosine=cosine))
    del model
    torch.cuda.empty_cache()


class Suffix(nn.Module):
    def __init__(self, vision, feature, blocks):
        super().__init__()
        self.layers = copy.deepcopy(vision.vision_model.encoder.layers[-2:])
        self.post_layernorm = copy.deepcopy(vision.vision_model.post_layernorm)
        self.projection = copy.deepcopy(vision.visual_projection) if feature == 'projected' else nn.Identity()
        self.requires_grad_(False)
        for layer in self.layers[-blocks:]:
            layer.requires_grad_(True)

    def forward(self, tokens):
        for layer in self.layers:
            tokens = layer(tokens, attention_mask=None, causal_attention_mask=None)[0]
        return self.projection(self.post_layernorm(tokens[:, 0]))


def projector(width, hidden):
    return nn.Linear(width, 384) if hidden == 0 else nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 384))


def train(data, name, feature, hidden=0, blocks=0):
    dest = data.output / name
    dest.mkdir(exist_ok=True)
    spec = dict(experiment=name, feature=feature, hidden=hidden, vision_blocks=blocks, seed=42,
                vision_lr=1e-5 if blocks else 0., projector_lr=.001, epochs=20, batch_size=256,
                early_stopping_patience=5 if blocks else None)
    identity = dict(inputs=data.identity, spec=spec,
                    source_sha256=file_sha(__file__), cache=read_json(data.output / 'cache/complete.json')['files'])
    run_id = digest(identity)
    result_path = dest / 'result.json'
    if result_path.exists():
        result = read_json(result_path)
        assert result['run_id'] == run_id
        assert result['checkpoint_sha256'] == file_sha(dest / 'best.pt')
        return result
    write_json(dest / 'config.json', identity)
    torch.manual_seed(42)
    random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    x = torch.load(data.output / 'cache/features.pt', weights_only=True)[feature].to('cuda')
    model = projector(x.shape[1], hidden).to('cuda')
    criterion = PairLoss('siglip_sigmoid', data.config['temperature'], data.config['sigmoid_bias']).to('cuda')
    suffix = None
    prefix = None
    groups = [dict(params=model.parameters(), weight_decay=.01, lr=.001),
              dict(params=criterion.parameters(), weight_decay=0., lr=.001)]
    if blocks:
        vision, _, _ = load_vision(data)
        suffix = Suffix(vision, feature, blocks).to('cuda')
        del vision
        prefix = np.load(data.output / 'cache/prefix.npy', mmap_mode='r')
        groups.append(dict(params=[p for p in suffix.parameters() if p.requires_grad], weight_decay=.01, lr=1e-5))
    optimizer = torch.optim.AdamW(groups)
    parameters = [p for g in optimizer.param_groups for p in g['params']]
    trainable = sum(p.numel() for p in parameters)
    generator = torch.Generator().manual_seed(42)
    train_i, train_t, positives = data.tensors['train']
    val_i, val_t, val_positives = data.tensors['validation']
    train_t, val_t = train_t.to('cuda'), val_t.to('cuda')
    val_batches = [list(range(s, min(s + 256, len(val_i)))) for s in range(0, len(val_i), 256)]
    val_masks = [val_positives.mask(i).to('cuda') for i in val_batches]
    history = []
    best = None
    torch.cuda.reset_peak_memory_stats()

    def vision_features(indices):
        if suffix is None:
            return x[indices.to('cuda')]
        tokens = torch.from_numpy(np.array(prefix[indices.numpy()], copy=True)).to('cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return suffix(tokens).float()

    def monitor(epoch, loss, grad_mean, grad_max):
        model.eval()
        if suffix is not None:
            suffix.eval()
        with torch.inference_mode():
            # Recompute adapted validation embeddings each epoch. No stale feature cache.
            if suffix is None:
                current = x
            else:
                current = torch.zeros_like(x)
                needed = torch.unique(val_i)
                for part in needed.split(128):
                    current[part.to('cuda')] = vision_features(part)
            val_loss = sum(float(criterion(model(current[val_i[i].to('cuda')]), val_t[i], mask)) * len(i)
                           for i, mask in zip(val_batches, val_masks)) / len(val_i)
            graph = data.graph
            projected = model(current[graph['image_indices']]).cpu()
            texts = data.texts[graph['text_indices']]
            local = dict(graph, image_indices=list(range(len(projected))), text_indices=list(range(len(texts))))
            retrieval = evaluate_graph(local, projected, texts)
        row = dict(epoch=epoch, train_loss=loss, validation_loss=val_loss, validation=retrieval,
                   gradient_norm_mean=grad_mean, gradient_norm_max=grad_max,
                   peak_vram_bytes=torch.cuda.max_memory_allocated(), trainable_params=trainable,
                   elapsed_seconds=time.monotonic() - started, **criterion.scalars())
        history.append(row)
        with (dest / 'epochs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(f'{name} epoch={epoch} train={loss} val_loss={val_loss:.5f} '
              f't2i={retrieval["t2i"]["recall@1"]:.4f} i2t={retrieval["i2t"]["recall@1"]:.4f} '
              f'grad={grad_mean} VRAM={row["peak_vram_bytes"]/2**30:.2f}GiB', flush=True)
        return row

    started = time.monotonic()
    (dest / 'epochs.jsonl').write_text('', encoding='utf-8')
    monitor(0, None, None, None)
    for epoch in range(1, 21):
        model.train()
        if suffix is not None:
            suffix.train()
        perm = torch.randperm(len(train_i), generator=generator)
        total, norms = 0., []
        for part in perm.split(256):
            indices = part.tolist()
            optimizer.zero_grad(set_to_none=True)
            features = vision_features(train_i[part])
            loss = criterion(model(features), train_t[part], positives.mask(indices).to('cuda'))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, float('inf'))
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite gradient')
            norms.append(float(norm))
            optimizer.step()
            criterion.constrain()
            total += float(loss.detach()) * len(part)
        row = monitor(epoch, total / len(train_i), sum(norms) / len(norms), max(norms))
        if best is None or checkpoint_key(row) > checkpoint_key(best):
            best = row
            torch.save(dict(projector=model.state_dict(), vision_suffix=suffix.state_dict() if suffix is not None else None,
                            loss=criterion.state_dict(), spec=spec, epoch=epoch, run_id=run_id), dest / 'best.pt')
        if blocks and epoch - best['epoch'] >= 5:
            break
    data.verify()
    result = dict(**spec, run_id=run_id, status='complete', history=history, selected_validation=best['validation'],
                  best_epoch=best['epoch'], trainable_params=trainable,
                  projector_params=sum(p.numel() for p in model.parameters()),
                  vision_trainable_params=sum(p.numel() for p in suffix.parameters() if p.requires_grad) if suffix else 0,
                  peak_vram_bytes=max(r['peak_vram_bytes'] for r in history),
                  checkpoint_sha256=file_sha(dest / 'best.pt'), selection_uses_test=False, test_mscoco=None)
    write_json(dest / 'selection.json', dict(best_epoch=best['epoch'], validation=best['validation'], test_used=False))
    write_json(result_path, result)
    del model, optimizer, suffix, x
    torch.cuda.empty_cache()
    return result


def key(result):
    v = result['selected_validation']
    return v['t2i']['recall@1'], v['mean_r1']


def export(output):
    results = [read_json(p) for p in sorted(output.glob('*/result.json'))]
    lines = ['# TinyCLIP adaptation — FULL_CLEAN, seed 42', '',
             'Checkpoint/config selection: validation t2i R@1, then mean t2i/i2t R@1. No test or XM3600 evaluation.', '',
             '| experiment | feature source | projector | vision trainable blocks | trainable params | val t2i R@1 | val i2t R@1 | best epoch | test MSCOCO |',
             '|---|---|---|---:|---:|---:|---:|---:|---|']
    for r in results:
        v = r['selected_validation']
        p = 'Linear' if not r['hidden'] else f'Linear→{r["hidden"]}→GELU→Linear→384'
        lines.append(f'| {r["experiment"]} | {r["feature"]} | {p} | {r["vision_blocks"]} | {r["trainable_params"]} | '
                     f'{100*v["t2i"]["recall@1"]:.2f}% | {100*v["i2t"]["recall@1"]:.2f}% | {r["best_epoch"]} | — |')
    lines += ['', 'Trainable params include the two learned sigmoid scale/bias scalars.',
              'A2 uses the native post-layernorm pooled CLS feature before visual_projection.',
              'C caches only the frozen prefix; adapted blocks and validation features are recomputed.', '']
    (ROOT / 'reports/tinyclip_adaptation.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(ROOT / 'reports/tinyclip_adaptation.json', results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['all', 'cache', 'A', 'B', 'C', 'report'])
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'report':
        export(output)
        return
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    data = Inputs(output)
    cache(data)
    if args.stage == 'cache':
        return
    def run(name, feature, hidden=0, blocks=0):
        write_json(output / 'status.json', dict(status='running', experiment=name, pid=os.getpid()))
        value = train(data, name, feature, hidden, blocks)
        export(output)
        return value
    if args.stage in ['all', 'A']:
        a1 = run('A1', 'projected')
        a2 = run('A2', 'pooled')
        winner_a = max([a1, a2], key=key)
        write_json(output / 'A_selection.json', dict(winner=winner_a['experiment'], feature=winner_a['feature'], test_used=False))
    if args.stage in ['all', 'B']:
        a = read_json(output / 'A_selection.json')
        # B1 reuses the exact linear run from A; no duplicate training or new seed.
        b1 = read_json(output / a['winner'] / 'result.json')
        write_json(output / 'B1_reuse.json', dict(source=a['winner'], run_id=b1['run_id']))
        b2 = run('B2', a['feature'], 768)
        winner_b = max([b1, b2], key=key)
        write_json(output / 'B_selection.json', dict(winner=winner_b['experiment'], feature=winner_b['feature'],
                   hidden=winner_b['hidden'], test_used=False))
    if args.stage in ['all', 'C']:
        b = read_json(output / 'B_selection.json')
        c0 = read_json(output / b['winner'] / 'result.json')
        write_json(output / 'C0_reuse.json', dict(source=b['winner'], run_id=c0['run_id']))
        c1 = run('C1', b['feature'], b['hidden'], 1)
        c2 = run('C2', b['feature'], b['hidden'], 2)
        winner_c = max([c0, c1, c2], key=key)
        write_json(output / 'C_selection.json', dict(winner=winner_c['experiment'], test_used=False,
                   frozen_baseline=c0['experiment'], delta_c1=key(c1)[0]-key(c0)[0], delta_c2=key(c2)[0]-key(c0)[0]))
    write_json(output / 'status.json', dict(status='complete', stage=args.stage, pid=os.getpid()))
