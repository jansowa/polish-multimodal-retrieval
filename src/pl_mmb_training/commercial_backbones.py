"""Standard projected image representations for the five requested checkpoints."""
import ast
from collections import OrderedDict
from types import ModuleType

import torch
import torch.nn.functional as F
from torch import nn

from .backbones import _file
from .common import ROOT, file_sha, read_json

ORDER = ('tinyclip-39m', 'tinyclip-40m', 'chinese-clip-rn50', 'rn50-cc12m', 'rn50-yfcc15m')


def load_backbone(key, lock):
    if key not in ORDER:
        raise ValueError('Only the five requested candidates are authorized')
    for path, sha in lock['files'].items():
        if file_sha(path) != sha:
            raise ValueError('Checkpoint or license changed: ' + path)
    stats = dict(key=key, model=lock['model'], revision=lock['revision'], checkpoint_files=lock['files'],
                 license=lock['license'], input_size=224,
                 representation='standard unnormalized image embedding, including original image projection')
    if key.startswith('tinyclip'):
        from transformers import CLIPConfig, CLIPVisionModelWithProjection, CLIPImageProcessor
        from safetensors import safe_open
        snapshot = _file(lock, '/config.json').parent
        config = CLIPConfig.from_pretrained(snapshot, local_files_only=True)
        config.vision_config.projection_dim = config.projection_dim
        model = CLIPVisionModelWithProjection(config.vision_config)
        with safe_open(_file(lock, '/model.safetensors'), framework='pt', device='cpu') as checkpoint:
            weights = {k: checkpoint.get_tensor(k) for k in checkpoint.keys()
                       if k.startswith(('vision_model.', 'visual_projection.'))}
        # Old Transformers serialized this deterministic buffer; current versions
        # recreate it as non-persistent. Check its value before dropping it.
        legacy_positions = weights.pop('vision_model.embeddings.position_ids', None)
        if legacy_positions is not None and not torch.equal(legacy_positions, model.vision_model.embeddings.position_ids):
            raise ValueError('Legacy position_ids differ from native positions')
        model.load_state_dict(weights, strict=True)
        processor = CLIPImageProcessor.from_pretrained(snapshot, local_files_only=True)
        preprocess = lambda images: processor(images=images, return_tensors='pt')['pixel_values']
        forward = lambda pixels: model(pixel_values=pixels).image_embeds
        stats.update(feature_dim=config.projection_dim, preprocessing=processor.to_dict(),
                     pretraining='TinyCLIP image-text distillation, ' + ('YFCC-15M' if key == 'tinyclip-39m' else 'LAION-400M'))
    elif key == 'chinese-clip-rn50':
        provenance = read_json(ROOT / 'training/runs/vision-commercial-five/official/lock.json')
        for path, sha in provenance['files'].items():
            if file_sha(path) != sha:
                raise ValueError('Official Chinese-CLIP source changed')
        tree = ast.parse(_file(provenance, '/cn_clip_clip_model.py').read_text())
        # Execute only the three official image modules, without importing a text tower.
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name in ('Bottleneck', 'AttentionPool2d', 'ModifiedResNet')]
        if len(classes) != 3:
            raise ValueError('Unexpected official source structure')
        module = ModuleType('official_chinese_clip_visual')
        module.__dict__.update(torch=torch, nn=nn, F=F, OrderedDict=OrderedDict)
        exec(compile(ast.Module(body=classes, type_ignores=[]), '<pinned-official-chinese-clip>', 'exec'), module.__dict__)
        state = torch.load(_file(lock, '/clip_cn_rn50.pt'), map_location='cpu', weights_only=True)
        state = state.get('state_dict', state)
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        weights = {k.removeprefix('visual.'): v.float() for k, v in state.items() if k.startswith('visual.')}
        output_dim = weights['attnpool.c_proj.weight'].shape[0]
        model = module.ModifiedResNet((3, 4, 6, 3), output_dim=output_dim, heads=32, input_resolution=224, width=64)
        model.load_state_dict(weights, strict=True)
        from torchvision import transforms as T
        transform = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                               T.ToTensor(), T.Normalize((.48145466, .4578275, .40821073), (.26862954, .26130258, .27577711))])
        preprocess = lambda images: torch.stack([transform(im) for im in images])
        forward = model
        stats.update(feature_dim=output_dim, preprocessing=repr(transform), official_source=provenance,
                     pretraining='Chinese-CLIP image-text pretraining, Chinese 200M pairs',
                     source_adaptation='AST extraction of three unchanged official image classes; no text imports')
    else:
        from safetensors import safe_open
        import open_clip
        cfg = read_json(_file(lock, '/open_clip_config.json'))
        prep = cfg['preprocess_cfg']
        full, _, transform = open_clip.create_model_and_transforms('RN50-quickgelu', pretrained=None,
            image_mean=prep['mean'], image_std=prep['std'], image_interpolation=prep['interpolation'],
            image_resize_mode=prep['resize_mode'])
        with safe_open(_file(lock, '/open_clip_model.safetensors'), framework='pt', device='cpu') as checkpoint:
            weights = {k: checkpoint.get_tensor(k) for k in checkpoint.keys()}
        full.load_state_dict(weights, strict=True)
        full.eval()
        model = full.visual
        with torch.inference_mode():
            probe = torch.zeros(1, 3, 224, 224)
            if not torch.equal(full.encode_image(probe, normalize=False), model(probe)):
                raise ValueError('Standard encode_image parity failed')
        del full, weights
        preprocess = lambda images: torch.stack([transform(im) for im in images])
        forward = model
        stats.update(feature_dim=cfg['model_cfg']['embed_dim'], preprocessing=repr(transform), native_config=cfg,
                     standard_encode_image_parity=True,
                     pretraining='OpenCLIP image-text pretraining, ' + ('CC12M' if key == 'rn50-cc12m' else 'YFCC-15M'))
    model.eval().requires_grad_(False)
    stats.update(vision_parameters=sum(p.numel() for p in model.parameters()),
                 frozen=not any(p.requires_grad for p in model.parameters()))
    return model, preprocess, forward, stats
