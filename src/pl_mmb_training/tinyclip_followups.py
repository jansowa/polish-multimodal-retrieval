"""Small conditional LoRA, similarity KD and TinyCLIP scaling followups."""
import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import tinyclip_adaptation as base
from .common import ROOT, digest, file_sha, read_json, write_json
from .losses import PairLoss
from .retrieval import evaluate_graph


class LoRALinear(nn.Module):
    def __init__(self, linear, rank=8, alpha=16):
        super().__init__()
        self.base = linear.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, linear.in_features))
        self.b = nn.Parameter(torch.zeros(linear.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=5 ** .5)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale


def suffix_for(data, spec):
    if not spec['blocks']:
        return None
    vision, _, _ = base.load_vision(data)
    suffix = base.Suffix(vision, spec['feature'], spec['blocks']).to('cuda')
    if spec.get('lora'):
        suffix.requires_grad_(False)
        for layer in suffix.layers:
            for name in ('q_proj', 'v_proj'):
                linear = getattr(layer.self_attn, name)
                setattr(layer.self_attn, name, LoRALinear(linear).to('cuda'))
    del vision
    return suffix


def similarity_kd(student_images, student_texts, teacher_images, teacher_texts, temperature=.07):
    """Symmetric KL between cross-modal similarity distributions; no embedding regression."""
    student = F.normalize(student_texts, dim=-1) @ F.normalize(student_images, dim=-1).T / temperature
    with torch.no_grad():
        teacher = F.normalize(teacher_texts, dim=-1) @ F.normalize(teacher_images, dim=-1).T / temperature
    return .5 * (F.kl_div(F.log_softmax(student, dim=1), F.softmax(teacher, dim=1), reduction='batchmean')
                  + F.kl_div(F.log_softmax(student.T, dim=1), F.softmax(teacher.T, dim=1), reduction='batchmean'))


def teacher_cache(data):
    """Use frozen original SigLIP2 image cache and native English text tower."""
    folder = data.output / 'teacher'
    folder.mkdir(exist_ok=True)
    meta_path = folder / 'complete.json'
    en = list(dict.fromkeys(r['text_en'] for r in data.train + data.val))
    manifest = read_json(ROOT / 'training/runs/e1-full/cache.json')
    lock = manifest['model_lock']['vision']
    identity = dict(model=lock, texts_sha256=digest(en), images_sha256=digest(data.images),
                    image_features=manifest['image_features'], pooling='native_SiglipTextModel.pooler_output',
                    max_length=64, teacher_frozen=True)
    if meta_path.exists():
        meta = read_json(meta_path)
        assert meta['identity'] == identity
        assert file_sha(folder / 'features.pt') == meta['sha256']
        return torch.load(folder / 'features.pt', weights_only=True)
    from transformers import AutoTokenizer, SiglipTextModel
    # Read the pinned local snapshot first. Fetch missing native tokenizer assets only if necessary.
    try:
        tokenizer = AutoTokenizer.from_pretrained(lock['model'], revision=lock['revision'], local_files_only=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(lock['model'], revision=lock['revision'])
    teacher = SiglipTextModel.from_pretrained(lock['model'], revision=lock['revision'],
                 local_files_only=True, attn_implementation='sdpa').eval().requires_grad_(False).to('cuda')
    chunks = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for start in range(0, len(en), 128):
            path = folder / f'text-{start:06d}.pt'
            if path.exists():
                chunks.append(torch.load(path, weights_only=True))
                continue
            encoded = tokenizer(en[start:start + 128], padding='max_length', max_length=64,
                                truncation=True, return_tensors='pt').to('cuda')
            features = teacher(**encoded).pooler_output.float().cpu()
            if not torch.isfinite(features).all():
                raise ValueError('Invalid teacher text features')
            torch.save(features, path)
            chunks.append(features)
            if start % 4096 == 0:
                print(f'TEACHER EN {start}/{len(en)}', flush=True)
    del teacher
    meta = manifest['image_features']
    assert file_sha(meta['path']) == meta['sha256']
    old = torch.load(meta['path'], weights_only=True)
    ii = {p: i for i, p in enumerate(manifest['images'])}
    ti = {s: i for i, s in enumerate(en)}
    texts = torch.cat(chunks)
    tensors = dict(images=old[[ii[p] for p in data.images]],
                   train_texts=texts[[ti[r['text_en']] for r in data.train]],
                   validation_texts=texts[[ti[r['text_en']] for r in data.val]])
    torch.save(tensors, folder / 'features.pt')
    write_json(meta_path, dict(identity=identity, sha256=file_sha(folder / 'features.pt')))
    torch.cuda.empty_cache()
    return tensors


def train(data, name, spec, x, teacher=None):
    dest = data.output / name
    dest.mkdir(exist_ok=True)
    identity = dict(inputs=data.identity, spec=spec, source_sha256=file_sha(__file__),
                    base_source_sha256=file_sha(base.__file__),
                    features_sha256=hashlib.sha256(x.cpu().contiguous().numpy().tobytes()).hexdigest(),
                    teacher=read_json(data.output / 'teacher/complete.json') if teacher is not None else None)
    run_id = digest(identity)
    path = dest / 'result.json'
    if path.exists():
        result = read_json(path)
        assert result['run_id'] == run_id
        assert result['checkpoint_sha256'] == file_sha(dest / 'best.pt')
        return result
    write_json(dest / 'config.json', identity)
    write_json(data.output / 'status.json', dict(status='running', experiment=name, pid=os.getpid()))
    torch.manual_seed(42)
    x = x.to('cuda')
    model = base.projector(x.shape[1], spec['hidden']).to('cuda')
    criterion = PairLoss('siglip_sigmoid', .07, -10.).to('cuda')
    suffix = suffix_for(data, spec)
    prefix = np.load(data.output / 'cache/prefix.npy', mmap_mode='r') if suffix is not None else None
    groups = [dict(params=model.parameters(), weight_decay=.01, lr=.001),
              dict(params=criterion.parameters(), weight_decay=0., lr=.001)]
    if suffix is not None:
        groups.append(dict(params=[p for p in suffix.parameters() if p.requires_grad], weight_decay=.01,
                           lr=1e-4 if spec.get('lora') else 1e-5))
    optimizer = torch.optim.AdamW(groups)
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    trainable = sum(p.numel() for p in parameters)
    ti, tt, positives = data.tensors['train']
    vi, vt, vp = data.tensors['validation']
    tt, vt = tt.to('cuda'), vt.to('cuda')
    if teacher is not None:
        teacher = {k: t.to('cuda') for k, t in teacher.items()}
    gen = torch.Generator().manual_seed(42)
    history, best = [], None
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    (dest / 'epochs.jsonl').write_text('', encoding='utf-8')

    def features(indices):
        if suffix is None:
            return x[indices.to('cuda')]
        tokens = torch.from_numpy(np.array(prefix[indices.numpy()], copy=True)).to('cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return suffix(tokens).float()

    def monitor(epoch, train_loss, gold_loss, kd_loss, grad_norm):
        model.eval()
        if suffix is not None:
            suffix.eval()
        with torch.inference_mode():
            current = x if suffix is None else torch.zeros_like(x)
            if suffix is not None:
                for part in torch.unique(vi).split(128):
                    current[part.to('cuda')] = features(part)
            val_loss, val_kd = 0., 0.
            for start in range(0, len(vi), 256):
                indices = list(range(start, min(start + 256, len(vi))))
                projected = model(current[vi[indices].to('cuda')])
                gold = criterion(projected, vt[indices], vp.mask(indices).to('cuda'))
                val_loss += float(gold) * len(indices)
                if teacher is not None:
                    kd = similarity_kd(projected, vt[indices], teacher['images'][vi[indices].to('cuda')],
                                       teacher['validation_texts'][indices])
                    val_kd += float(kd) * len(indices)
            graph = data.graph
            projected = model(current[graph['image_indices']]).cpu()
            texts = data.texts[graph['text_indices']]
            local = dict(graph, image_indices=list(range(len(projected))), text_indices=list(range(len(texts))))
            retrieval = evaluate_graph(local, projected, texts)
        row = dict(epoch=epoch, train_loss=train_loss, train_gold_loss=gold_loss, train_kd_loss=kd_loss,
                   validation_loss=val_loss / len(vi), validation_kd_loss=val_kd / len(vi),
                   validation=retrieval, gradient_norm_mean=grad_norm,
                   peak_vram_bytes=torch.cuda.max_memory_allocated(), trainable_params=trainable,
                   elapsed_seconds=time.monotonic() - started, **criterion.scalars())
        history.append(row)
        with (dest / 'epochs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(f'{name} epoch={epoch} train={train_loss} gold={gold_loss} KD={kd_loss} '
              f'val={row["validation_loss"]:.4f} t2i={retrieval["t2i"]["recall@1"]:.4f} '
              f'i2t={retrieval["i2t"]["recall@1"]:.4f}', flush=True)
        return row

    monitor(0, None, None, None, None)
    for epoch in range(1, 21):
        model.train()
        if suffix is not None:
            suffix.train()
        total, gold_total, kd_total, norms = 0., 0., 0., []
        for part in torch.randperm(len(ti), generator=gen).split(256):
            indices = part.tolist()
            optimizer.zero_grad(set_to_none=True)
            projected = model(features(ti[part]))
            gold = criterion(projected, tt[part], positives.mask(indices).to('cuda'))
            kd = torch.zeros((), device='cuda')
            if teacher is not None:
                kd = similarity_kd(projected, tt[part], teacher['images'][ti[part].to('cuda')], teacher['train_texts'][part])
            loss = gold + spec.get('kd_lambda', 0.) * kd
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite followup loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, float('inf'))
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite followup gradient')
            norms.append(float(norm))
            optimizer.step()
            criterion.constrain()
            total += float(loss.detach()) * len(part)
            gold_total += float(gold.detach()) * len(part)
            kd_total += float(kd.detach()) * len(part)
        row = monitor(epoch, total/len(ti), gold_total/len(ti), kd_total/len(ti), sum(norms)/len(norms))
        if best is None or base.checkpoint_key(row) > base.checkpoint_key(best):
            best = row
            torch.save(dict(projector=model.state_dict(), vision_suffix=suffix.state_dict() if suffix else None,
                            loss=criterion.state_dict(), spec=spec, epoch=epoch, run_id=run_id), dest / 'best.pt')
        if spec['blocks'] and epoch - best['epoch'] >= 5:
            break
    data.verify()
    result = dict(experiment=name, **spec, vision_blocks=spec['blocks'], run_id=run_id, status='complete',
                  history=history, selected_validation=best['validation'], best_epoch=best['epoch'],
                  trainable_params=trainable, peak_vram_bytes=max(r['peak_vram_bytes'] for r in history),
                  checkpoint_sha256=file_sha(dest / 'best.pt'), selection_uses_test=False, test_mscoco=None)
    write_json(dest / 'selection.json', dict(best_epoch=best['epoch'], validation=best['validation'], test_used=False))
    write_json(path, result)
    del model, optimizer, suffix
    torch.cuda.empty_cache()
    base.export(data.output)
    return result


def scaling_cache(data):
    """One 61M checkpoint, native projected features, no model-specific tuning."""
    folder = data.output / 'tinyclip61'
    folder.mkdir(exist_ok=True)
    lock_path = folder / 'model_lock.json'
    from huggingface_hub import HfApi, snapshot_download
    from safetensors import safe_open
    from transformers import CLIPConfig, CLIPVisionModelWithProjection, CLIPImageProcessor
    model_id = 'wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M'
    if lock_path.exists():
        lock = read_json(lock_path)
    else:
        revision = HfApi().model_info(model_id).sha
        snapshot = Path(snapshot_download(model_id, revision=revision,
                           allow_patterns=['config.json', 'preprocessor_config.json', 'model.safetensors']))
        lock = dict(model=model_id, revision=revision, snapshot=str(snapshot),
                    files={str(p): file_sha(p) for p in snapshot.iterdir() if p.is_file()})
        write_json(lock_path, lock)
    for path, sha in lock['files'].items():
        assert file_sha(path) == sha
    snapshot = Path(lock['snapshot'])
    path = folder / 'features.pt'
    meta_path = folder / 'complete.json'
    identity = dict(model=lock, images_sha256=digest(data.images), expected_content_sha256=digest(data.expected_sha))
    if meta_path.exists():
        meta = read_json(meta_path)
        assert meta['identity'] == identity and file_sha(path) == meta['sha256']
        return torch.load(path, weights_only=True)
    config = CLIPConfig.from_pretrained(snapshot, local_files_only=True)
    config.vision_config.projection_dim = config.projection_dim
    model = CLIPVisionModelWithProjection(config.vision_config)
    with safe_open(snapshot / 'model.safetensors', framework='pt', device='cpu') as checkpoint:
        weights = {k: checkpoint.get_tensor(k) for k in checkpoint.keys() if k.startswith(('vision_model.', 'visual_projection.'))}
    positions = weights.pop('vision_model.embeddings.position_ids', None)
    if positions is not None:
        assert torch.equal(positions, model.vision_model.embeddings.position_ids)
    model.load_state_dict(weights, strict=True)
    model.eval().requires_grad_(False).to('cuda')
    prep = CLIPImageProcessor.from_pretrained(snapshot, local_files_only=True)
    from PIL import Image
    chunks = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for start in range(0, len(data.images), 32):
            chunk_path = folder / f'{start:06d}.pt'
            if chunk_path.exists():
                chunks.append(torch.load(chunk_path, weights_only=True))
                continue
            opened = []
            try:
                for p in data.images[start:start + 32]:
                    assert file_sha(p) == data.expected_sha[p]
                    with Image.open(p) as im:
                        opened.append(im.convert('RGB'))
                output = model(**prep(images=opened, return_tensors='pt').to('cuda')).image_embeds.float().cpu()
                torch.save(output, chunk_path)
                chunks.append(output)
            finally:
                for im in opened:
                    im.close()
            if start % 800 == 0:
                print(f'TINYCLIP61 CACHE {start}/{len(data.images)}', flush=True)
    tensor = torch.cat(chunks)
    assert torch.isfinite(tensor).all() and (tensor.norm(dim=-1) > 0).all()
    torch.save(tensor, path)
    write_json(meta_path, dict(identity=identity, sha256=file_sha(path), vision_params=sum(p.numel() for p in model.parameters())))
    del model
    torch.cuda.empty_cache()
    return tensor


def best_recipe(result):
    return dict(feature=result['feature'], hidden=result['hidden'],
                blocks=result['vision_blocks'], lora=result.get('lora', False), kd_lambda=0.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['D', 'E', 'F', 'all'])
    parser.add_argument('--output', type=Path, default=base.OUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    data = base.Inputs(args.output)
    base.cache(data)
    c = read_json(args.output / 'C_selection.json')
    winner = read_json(args.output / c['winner'] / 'result.json')
    x = torch.load(args.output / 'cache/features.pt', weights_only=True)
    if args.stage in ('D', 'all'):
        evidence = []
        for name in ('C1', 'C2'):
            r = read_json(args.output / name / 'result.json')
            decline = base.key(r)[0] - r['history'][-1]['validation']['t2i']['recall@1']
            evidence.append(dict(experiment=name, best=base.key(r)[0], final=r['history'][-1]['validation']['t2i']['recall@1'],
                                 decline=decline, overfit=decline >= .01))
        baseline = read_json(args.output / c['frozen_baseline'] / 'result.json')
        trigger = any(e['overfit'] for e in evidence) or base.key(winner)[0] <= base.key(baseline)[0]
        write_json(args.output / 'D_decision.json', dict(run=trigger, evidence=evidence,
                   rule='>=1pp best-to-final val t2i decline, or no improvement over C0'))
        if trigger:
            spec = dict(feature=baseline['feature'], hidden=baseline['hidden'], blocks=2, lora=True,
                        lora_rank=8, lora_alpha=16, lora_targets=['q_proj', 'v_proj'], vision_lr=1e-4, kd_lambda=0.)
            d = train(data, 'D_LoRA', spec, x[spec['feature']])
            winner = max([winner, d], key=base.key)
        write_json(args.output / 'D_selection.json', dict(winner=winner['experiment'], test_used=False, ran=trigger))
    elif (args.output / 'D_selection.json').exists():
        selected = read_json(args.output / 'D_selection.json')['winner']
        winner = read_json(args.output / selected / 'result.json')
    if args.stage in ('E', 'all'):
        teacher = teacher_cache(data)
        spec = best_recipe(winner)
        e_results = []
        for lam in (.1, .3):
            e_results.append(train(data, f'E_KD_{lam}', dict(spec, kd_lambda=lam,
                                   kd='symmetric_KL_similarity', kd_temperature=.07), x[spec['feature']], teacher))
        winner = max([winner, *e_results], key=base.key)
        write_json(args.output / 'E_selection.json', dict(winner=winner['experiment'], test_used=False,
                   teacher='SigLIP2 Base/16 256, EN', student='TinyCLIP39M + frozen PolDense32M, PL'))
    if args.stage in ('F', 'all'):
        # Scaling uses the best frozen recipe from A/B, never the adapted C/D/E architecture.
        b = read_json(args.output / 'B_selection.json')
        f = train(data, 'F_TinyCLIP61', dict(feature='projected', hidden=b['hidden'], blocks=0, lora=False,
                   kd_lambda=0., backbone='wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M'), scaling_cache(data))
        write_json(args.output / 'F_selection.json', dict(experiment=f['experiment'], test_used=False))
    write_json(args.output / 'status.json', dict(status='complete', stage=args.stage, pid=os.getpid()))
