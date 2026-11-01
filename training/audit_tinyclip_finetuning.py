"""Audit real CUDA forward/gradients, precision, and caption/image sampling."""
import copy
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pl_mmb_training import tinyclip_adaptation as base
from pl_mmb_training import tinyclip_followups as follow
from pl_mmb_training.common import ROOT, file_sha, read_json, write_json
from pl_mmb_training.losses import PairLoss

OUT = ROOT / 'training/runs/tinyclip-generalization'
CACHE = base.OUT / 'cache'


def vector(grads):
    return torch.cat([g.detach().float().flatten() for g in grads if g is not None])


def compare(a, b):
    return dict(cosine=float(F.cosine_similarity(a.flatten()[None], b.flatten()[None])),
                relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-20)),
                max_absolute=float((a-b).abs().max()))


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    OUT.mkdir(exist_ok=True)
    data = base.Inputs(OUT)
    meta = read_json(CACHE / 'complete.json')
    for name, sha in meta['files'].items():
        assert file_sha(CACHE / name) == sha
    prefix = np.load(CACHE / 'prefix.npy', mmap_mode='r')
    native_x = torch.load(CACHE / 'features.pt', weights_only=True)['projected'].to('cuda')
    ti, tt, positives = data.tensors['train']
    saved = torch.load(base.OUT / 'A1/best.pt', weights_only=True, map_location='cpu')
    model = base.projector(512, 0).to('cuda')
    model.load_state_dict(saved['projector'])
    criterion = PairLoss('siglip_sigmoid').to('cuda')
    criterion.load_state_dict(saved['loss'])
    vision, prep, stats = base.load_vision(data)
    for layer in vision.vision_model.encoder.layers[-2:]:
        layer.requires_grad_(True)
    suffix = base.Suffix(vision, 'projected', 2).to('cuda')
    gen = torch.Generator().manual_seed(20261020)
    indices = torch.randperm(len(ti), generator=gen)[:32]
    image_indices = ti[indices]
    opened = []
    for index in image_indices:
        with Image.open(data.images[index]) as im:
            opened.append(im.convert('RGB'))
    pixels = prep(opened).to('cuda')
    for im in opened:
        im.close()
    tokens = torch.from_numpy(np.array(prefix[image_indices.numpy()], copy=True)).to('cuda')
    mask = positives.mask(indices.tolist()).to('cuda')
    text = tt[indices].to('cuda')
    vp = [p for p in vision.parameters() if p.requires_grad]
    sp = [p for p in suffix.parameters() if p.requires_grad]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        full = vision(pixel_values=pixels).image_embeds.float()
    full_loss = criterion(model(full), text, mask)
    full_grad = vector(torch.autograd.grad(full_loss, vp))
    with torch.autocast('cuda', dtype=torch.bfloat16):
        cached = suffix(tokens).float()
    cached_loss = criterion(model(cached), text, mask)
    cached_grad = vector(torch.autograd.grad(cached_loss, sp))
    fp32 = suffix(tokens)
    fp32_loss = criterion(model(fp32), text, mask)
    fp32_grad = vector(torch.autograd.grad(fp32_loss, sp))
    parity = dict(features_full_vs_cached=compare(full.detach(), cached.detach()),
                  gradients_full_vs_cached=compare(full_grad, cached_grad),
                  features_bf16_vs_fp32=compare(cached.detach(), fp32.detach()),
                  gradients_bf16_vs_fp32=compare(cached_grad, fp32_grad),
                  loss_full=float(full_loss), loss_cached=float(cached_loss), loss_fp32=float(fp32_loss))
    if parity['gradients_full_vs_cached']['cosine'] < .995 or parity['features_full_vs_cached']['cosine'] < .999:
        raise ValueError('Real native/cache parity failed: ' + str(parity))
    # Audit one complete original permutation, including repeated canonical images.
    gen.manual_seed(42)
    batches = list(torch.randperm(len(ti), generator=gen).split(256))
    unique_counts, positive_counts, parent_collisions = [], [], []
    for batch in batches:
        ids = batch.tolist()
        unique_counts.append(len({data.train[i]['target_id'] for i in ids}))
        positive_counts.append(int(positives.mask(ids).sum()))
        image_parents = {}
        for i in ids:
            r = data.train[i]
            for p in r['parent_ids']:
                image_parents.setdefault(p, set()).add(r['target_id'])
        parent_collisions.append(sum(len(v) > 1 for v in image_parents.values()))
    frequencies = Counter(r['target_id'] for r in data.train)
    sampling = dict(rows=len(ti), images=len(frequencies), maximum_rows_per_image=max(frequencies.values()),
                    image_frequency_quantiles=np.quantile(list(frequencies.values()), [0,.5,.9,.99,1]).tolist(),
                    mean_unique_images_per_full_batch=float(np.mean(unique_counts[:-1])),
                    min_unique_images_per_full_batch=min(unique_counts[:-1]),
                    mean_positive_pairs_per_full_batch=float(np.mean(positive_counts[:-1])),
                    mean_parent_families_with_different_crops_per_batch=float(np.mean(parent_collisions[:-1])),
                    sources=dict(Counter(r['dataset'] for r in data.train)))
    # Calibrate two independent regularizer strengths on TRAIN gradients only.
    adapted = torch.load(ROOT / 'training/runs/tinyclip-optimization/T4_last2_aligned_mid/best.pt', weights_only=True)
    suffix.load_state_dict(adapted['vision_suffix'])
    model.load_state_dict(adapted['projector'])
    criterion.load_state_dict(adapted['loss'])
    teacher_meta = read_json(base.OUT / 'teacher/complete.json')
    assert file_sha(base.OUT / 'teacher/features.pt') == teacher_meta['sha256']
    teacher = {k: t.to('cuda') for k, t in torch.load(base.OUT / 'teacher/features.pt', weights_only=True).items()}
    calibration = []
    for batch in batches[:8]:
        ii = ti[batch]
        tok = torch.from_numpy(np.array(prefix[ii.numpy()], copy=True)).to('cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            features = suffix(tok).float()
        projected = model(features)
        gold = criterion(projected, tt[batch].to('cuda'), positives.mask(batch.tolist()).to('cuda'))
        anchor = (1-F.cosine_similarity(features, native_x[ii.to('cuda')])).mean()
        kd = follow.similarity_kd(projected, tt[batch].to('cuda'), teacher['images'][ii.to('cuda')], teacher['train_texts'][batch])
        gg = vector(torch.autograd.grad(gold, sp, retain_graph=True))
        ga = vector(torch.autograd.grad(anchor, sp, retain_graph=True))
        gk = vector(torch.autograd.grad(kd, sp))
        calibration.append(dict(gold=float(gold), anchor=float(anchor), kd=float(kd),
             gold_grad=float(gg.norm()), anchor_grad=float(ga.norm()), kd_grad=float(gk.norm()),
             gold_anchor_grad_cosine=float(F.cosine_similarity(gg[None],ga[None])),
             gold_kd_grad_cosine=float(F.cosine_similarity(gg[None],gk[None]))))
    # Aim for half the gold gradient norm at the calibration checkpoint. No val metric enters this choice.
    weights = dict(anchor_lambda=float(np.clip(np.median([.5*r['gold_grad']/max(r['anchor_grad'],1e-12) for r in calibration]), 1, 1000)),
                   kd_lambda=float(np.clip(np.median([.5*r['gold_grad']/max(r['kd_grad'],1e-12) for r in calibration]), .1, 10)),
                   rule='median TRAIN vision-gradient norm ratio across 8 fixed batches at previous T4 best checkpoint; regularizer target=0.5*gold; no validation tuning')
    value = dict(status='passed', inputs=data.hashes, real_cuda_parity=parity, sampling=sampling,
                 calibration=calibration, regularization_weights=weights, native_model=stats,
                 test_used=False, xm3600_used=False)
    write_json(OUT / 'audit.json', value)
    write_json(ROOT / 'reports/tinyclip_finetuning_audit.json', value)
    data.verify()
    print(json.dumps(dict(parity=parity, sampling=sampling, weights=weights), indent=2), flush=True)


if __name__ == '__main__':
    main()
