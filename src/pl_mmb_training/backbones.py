"""Four requested frozen encoders, with native preprocessing and audited outputs."""
import gc
import hashlib
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from types import ModuleType

import torch

from .common import digest, file_sha, read_json, write_json
from .features import validate_features, load_chunk

ORDER = ('dinov2-small', 'mobileclip2-s2', 'convnext-tiny', 'ibot-vits16')


def _file(lock, suffix):
    paths = [Path(p) for p in lock['files'] if p.endswith(suffix)]
    if len(paths) != 1:
        raise ValueError('Ambiguous model artifact: ' + suffix)
    return paths[0]


def load_backbone(key, lock):
    """Return model, batch preprocessing, forward, and exact used parameter count."""
    for path, sha in lock['files'].items():
        if file_sha(path) != sha:
            raise ValueError('Official model artifact changed: ' + path)
    stats = dict(key=key, model=lock['model'], revision=lock['revision'], checkpoint_files=lock['files'])
    if key == 'dinov2-small':
        from transformers import AutoModel, AutoImageProcessor
        snapshot = _file(lock, '/config.json').parent
        model = AutoModel.from_pretrained(snapshot, local_files_only=True, attn_implementation='sdpa')
        processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True, use_fast=False)
        preprocess = lambda images: processor(images=images, return_tensors='pt')['pixel_values']
        forward = lambda pixels: model(pixel_values=pixels).last_hidden_state[:, 0]
        stats.update(pretraining='vision-only self-supervised DINOv2', representation='final normalized CLS token',
                     feature_dim=model.config.hidden_size, input_size=224, preprocessing=processor.to_dict())
    elif key == 'mobileclip2-s2':
        import open_clip
        cfg = open_clip.get_pretrained_cfg('MobileCLIP2-S2', 'dfndr2b')
        full, _, transform = open_clip.create_model_and_transforms('MobileCLIP2-S2',
            pretrained=str(_file(lock, '/mobileclip2_s2.pt')), device='cpu',
            image_mean=cfg['mean'], image_std=cfg['std'], image_interpolation=cfg['interpolation'],
            image_resize_mode=cfg['resize_mode'])
        full.eval()
        with torch.inference_mode():
            probe = torch.zeros(1, 3, 256, 256)
            if not torch.equal(full.encode_image(probe, normalize=False), full.visual(probe)):
                raise ValueError('Standard encode_image parity check failed')
        # CLIP.encode_image(normalize=False) calls this exact module.
        model = full.visual
        del full
        preprocess = lambda images: torch.stack([transform(im) for im in images])
        forward = model
        stats.update(pretraining='image-text pretrained MobileCLIP2',
                     representation='standard encode_image output in CLIP space, including original image projection',
                     feature_dim=512, input_size=256, preprocessing=repr(transform), pretrained_cfg=cfg,
                     standard_encode_image_parity=True)
    elif key == 'convnext-tiny':
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        model = convnext_tiny(weights=None)
        model.load_state_dict(torch.load(_file(lock, '.pth'), weights_only=True), strict=True)
        full_count = sum(p.numel() for p in model.parameters())
        model.classifier[-1] = torch.nn.Identity()
        transform = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
        preprocess = lambda images: torch.stack([transform(im) for im in images])
        forward = model
        stats.update(pretraining='supervised ImageNet-1K', representation='global average pool + final LayerNorm, before classification Linear',
                     feature_dim=768, input_size=224, preprocessing=repr(transform), original_with_classifier_params=full_count)
    elif key == 'ibot-vits16':
        source = _file(lock, '/ibot_vision_transformer.py').read_text()
        # Preserve the official forward; only replace a utility import and remove
        # an unused timm registry import. Initialization is overwritten by weights.
        source = source.replace('from utils import trunc_normal_', 'from torch.nn.init import trunc_normal_')
        source = source.replace('from timm.models.registry import register_model', '')
        module = ModuleType('official_ibot_vit')
        exec(compile(source, str(_file(lock, '/ibot_vision_transformer.py')), 'exec'), module.__dict__)
        model = module.vit_small(patch_size=16, num_classes=0)
        state = torch.load(_file(lock, '/ibot_vits16_teacher.pth'), weights_only=True)
        if 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state, strict=True)
        from torchvision import transforms as T
        transform = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224),
                               T.ToTensor(), T.Normalize((.485, .456, .406), (.229, .224, .225))])
        preprocess = lambda images: torch.stack([transform(im) for im in images])
        forward = model
        stats.update(pretraining='vision-only iBOT self-supervised ImageNet-1K teacher',
                     representation='official backbone forward: final normalized CLS token',
                     feature_dim=384, input_size=224, preprocessing=repr(transform),
                     source_adaptation='trunc_normal_ utility import only; unused registry import removed',
                     adapted_source_sha256=hashlib.sha256(source.encode()).hexdigest())
    else:
        raise ValueError('Only the first four requested backbones are authorized')
    model.eval().requires_grad_(False)
    stats['vision_parameters'] = sum(p.numel() for p in model.parameters())
    stats['frozen'] = not any(p.requires_grad for p in model.parameters())
    if not stats['frozen']:
        raise ValueError('Vision encoder must remain frozen')
    return model, preprocess, forward, stats


def _save_tensor(path, tensor):
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('wb') as stream:
        torch.save(tensor, stream)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def cache_backbone(key, lock, image_paths, expected_sha, cache_root, batch_size=32, loader=None):
    model, preprocess, forward, stats = (loader or load_backbone)(key, lock)
    environment = {p: version(p) for p in ('torch', 'torchvision', 'transformers', 'timm', 'open_clip_torch', 'Pillow')}
    identity = dict(schema=1, stats=stats, environment=environment,
                    precision='fp32-weights-bf16-autocast', images=list(image_paths),
                    image_sha256=[expected_sha[p] for p in image_paths])
    folder = Path(cache_root) / key / digest(identity)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'features.pt'
    metadata = dict(path=str(path), identity_sha256=digest(identity), model_stats=stats, environment=environment)
    if path.exists():
        tensor = validate_features(image_paths, torch.load(path, weights_only=True))
        print(f'{key}: cache hit, {len(tensor)} images', flush=True)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for new frozen feature extraction')
        model.to('cuda')
        chunks = []
        started = time.monotonic()
        def open_image(p):
            from PIL import Image
            payload = Path(p).read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected_sha[p]:
                raise ValueError('Training/evaluation image changed since E1: ' + p)
            with Image.open(io.BytesIO(payload)) as im:
                return im.convert('RGB')
        with ThreadPoolExecutor(max_workers=8) as pool:
            for offset in range(0, len(image_paths), batch_size):
                batch = image_paths[offset:offset + batch_size]
                chunk = folder / f'{offset:08d}.pt'
                cached = load_chunk(batch, chunk) if chunk.exists() else None
                if cached is not None:
                    chunks.append(cached)
                    continue
                opened = list(pool.map(open_image, batch))
                try:
                    pixels = preprocess(opened).to('cuda')
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                        encoded = validate_features(batch, forward(pixels).float().cpu())
                    if encoded.shape[1] != stats['feature_dim']:
                        raise ValueError('Unexpected encoder output dimension')
                    _save_tensor(chunk, encoded)
                    chunks.append(encoded)
                finally:
                    for im in opened:
                        im.close()
                if offset % (batch_size * 25) == 0 or offset + batch_size >= len(image_paths):
                    print(f'{key}: cached {min(offset+batch_size,len(image_paths))}/{len(image_paths)} images '
                          f'elapsed={time.monotonic()-started:.1f}s', flush=True)
        tensor = validate_features(image_paths, torch.cat(chunks))
        _save_tensor(path, tensor)
        write_json(folder / 'identity.json', identity)
    metadata['sha256'] = file_sha(path)
    write_json(folder / 'metadata.json', metadata)
    del model, forward
    gc.collect()
    torch.cuda.empty_cache()
    return tensor, metadata
