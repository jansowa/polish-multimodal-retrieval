"""Offline feature extraction. Optimizers never see either pretrained encoder."""
import gc
import os
import pickle
from importlib.metadata import version

from .common import digest, file_sha, write_json


def validate_features(values, tensor):
    import torch
    if tensor.ndim != 2 or len(values) != len(tensor) or not torch.isfinite(tensor).all() or (tensor.norm(dim=-1) == 0).any():
        raise ValueError('Invalid embedding cache (shape, NaN/Inf, or zero vector)')
    return tensor


def load_chunk(values, path):
    """A crash may leave a truncated intermediate chunk; recompute it safely."""
    import torch
    try:
        return validate_features(values, torch.load(path, weights_only=True, map_location='cpu'))
    except (EOFError, RuntimeError, pickle.UnpicklingError, ValueError):
        return None


def encode_cached(values, kind, config, lock, cache_root):
    import torch
    values = list(values)
    model_config = config['text' if kind == 'texts' else 'vision']
    identity = dict(schema=1, encoder=model_config, revision=lock['text' if kind == 'texts' else 'vision']['revision'],
                    kind=kind, torch=torch.__version__, transformers=version('transformers'),
                    sentence_transformers=version('sentence-transformers'), precision='fp32-weights-bf16-autocast',
                    pillow=version('Pillow'),
                    pooling='sentence-transformers' if kind == 'texts' else 'siglip-vision-pooler-output',
                    values=values)
    if kind == 'images':
        identity['content_sha256'] = [file_sha(p) for p in values]
    path = cache_root / 'embeddings' / (digest(identity) + '.pt')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        tensor = validate_features(values, torch.load(path, weights_only=True, map_location='cpu'))
        print(f'Offline cache hit: {len(values)} {kind}', flush=True)
        return tensor, dict(path=str(path), sha256=file_sha(path), identity_sha256=digest(identity))
    if not torch.cuda.is_available():
        raise RuntimeError('Feature extraction requires CUDA; training from existing caches can use CPU')
    if kind == 'texts':
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_config['model'], revision=lock['text']['revision'], local_files_only=True,
                    device='cuda', trust_remote_code=False, model_kwargs={'attn_implementation': 'sdpa'})
        model.max_seq_length = model_config['max_length']
        processor = None
    else:
        from transformers import SiglipVisionModel, AutoImageProcessor
        model = SiglipVisionModel.from_pretrained(model_config['model'], revision=lock['vision']['revision'],
                    local_files_only=True, attn_implementation='sdpa').to('cuda')
        processor = AutoImageProcessor.from_pretrained(model_config['model'], revision=lock['vision']['revision'],
                    local_files_only=True, use_fast=False)
    model.eval().requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    batch_size = config['encoding_batch_size']
    chunk_root = path.with_suffix('.chunks')
    chunk_root.mkdir(exist_ok=True)
    chunks = []
    from PIL import Image
    for offset in range(0, len(values), batch_size):
        batch = values[offset:offset + batch_size]
        chunk_path = chunk_root / f'{offset:08d}.pt'
        if chunk_path.exists():
            cached = load_chunk(batch, chunk_path)
            if cached is not None:
                chunks.append(cached)
                continue
            print(f'Recomputing invalid cached chunk: {offset}', flush=True)
        opened = []
        try:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                if kind == 'texts':
                    output = model.encode([model_config['prefix'] + t for t in batch], batch_size=len(batch),
                        convert_to_tensor=True, normalize_embeddings=False, show_progress_bar=False, prompt='')
                else:
                    for p in batch:
                        with Image.open(p) as im:
                            opened.append(im.convert('RGB'))
                    inputs = processor(images=opened, return_tensors='pt').to('cuda')
                    output = model(**inputs).pooler_output
                output = validate_features(batch, output.float().cpu())
            tmp = chunk_path.with_suffix(f'.{os.getpid()}.tmp')
            with tmp.open('wb') as stream:
                torch.save(output, stream)
                stream.flush()
                os.fsync(stream.fileno())
            tmp.replace(chunk_path)
            chunks.append(output)
        finally:
            for im in opened:
                im.close()
        if offset % (batch_size * 25) == 0 or offset + batch_size >= len(values):
            print(f'Cached {min(offset + batch_size, len(values))}/{len(values)} {kind}', flush=True)
    tensor = torch.cat(chunks)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    torch.save(tensor, tmp)
    tmp.replace(path)
    write_json(path.with_suffix('.json'), identity)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return tensor, dict(path=str(path), sha256=file_sha(path), identity_sha256=digest(identity))
