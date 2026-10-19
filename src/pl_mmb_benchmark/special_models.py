"""Pinned author implementations for TIPSv2, BGE-VL and Perception Encoder."""
from pathlib import Path
import sys
import os
import urllib.request
import torch
from huggingface_hub import snapshot_download, hf_hub_download

PE_CODE_REVISION = '3e352cca660658d4b5c90f42a7808b11469e4c66'
PE_CODE_FILES = ('__init__.py', 'config.py', 'pe.py', 'rope.py', 'tokenizer.py',
                 'transforms.py', 'bpe_simple_vocab_16e6.txt.gz')


def pe_source():
    root = Path.home()/'.cache/pl_mmb_benchmark/sources/perception_models'/PE_CODE_REVISION
    (root/'core/vision_encoder').mkdir(parents=True, exist_ok=True)
    (root/'core/__init__.py').touch(exist_ok=True)
    for name in PE_CODE_FILES:
        relative = 'core/vision_encoder/' + name
        target = root/relative
        if not target.exists():
            url = 'https://raw.githubusercontent.com/facebookresearch/perception_models/' + PE_CODE_REVISION + '/' + relative
            target.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    return root


def load_special(repo, revision):
    from torchvision import transforms
    if repo.startswith('google/tipsv2-') or repo == 'BAAI/BGE-VL-base':
        if repo.startswith('google/tipsv2-'):
            os.environ.setdefault('XFORMERS_DISABLED', '1')
        from transformers import AutoModel
        local = snapshot_download(repo, revision=revision,
            allow_patterns=['*.json', '*.py', '*.model', '*.txt', 'model.safetensors'])
        model, info = AutoModel.from_pretrained(local, trust_remote_code=True, torch_dtype=torch.float32,
            low_cpu_mem_usage=False, output_loading_info=True)
        if info['missing_keys'] or info['unexpected_keys'] or info.get('mismatched_keys'):
            raise RuntimeError('Incomplete checkpoint load: ' + str(info))
        if repo.startswith('google/tipsv2-'):
            model._tokenizer = model._load_tokenizer()
            processor = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
            return model, processor, None, {'implementation': 'author-hf-remote-code',
                'image_resolution': 448, 'pixel_range': 'zero-one', 'image_pooling': 'cls-token-0',
                'text_pooling': 'masked-mean', 'tokenizer_revision': revision}, 64
        from transformers import CLIPProcessor
        processor = CLIPProcessor.from_pretrained(local)
        model.processor = processor
        return model, processor, None, {'implementation': 'author-hf-remote-code',
            'composition': 'author-raw-sum-before-final-l2', 'tokenizer_revision': revision}, 77
    if repo == 'facebook/PE-Core-B16-224':
        sys.path.insert(0, str(pe_source()))
        from core.vision_encoder import pe
        from core.vision_encoder import transforms as pe_transforms
        checkpoint = hf_hub_download(repo, 'PE-Core-B16-224.pt', revision=revision)
        model = pe.CLIP.from_config('PE-Core-B16-224', pretrained=False)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if 'state_dict' in state: state = state['state_dict']
        elif 'weights' in state: state = state['weights']
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        return model, pe_transforms.get_image_transform(model.image_size), pe_transforms.get_text_tokenizer(model.context_length), {
            'implementation': 'facebookresearch/perception_models', 'implementation_revision': PE_CODE_REVISION,
            'image_resolution': model.image_size, 'tokenizer_context': model.context_length}, model.context_length
    raise ValueError('Unknown special model: ' + repo)
