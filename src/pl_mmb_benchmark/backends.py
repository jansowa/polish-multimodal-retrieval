"""Pinned OpenCLIP and M-CLIP components, preserving raw feature magnitudes."""
import json
from pathlib import Path
import torch
from huggingface_hub import snapshot_download, hf_hub_download
from transformers import AutoTokenizer, XLMRobertaConfig, XLMRobertaModel

MCLIP_IMAGE_REPO = 'timm/vit_base_patch16_plus_clip_240.laion400m_e31'
MCLIP_IMAGE_REVISION = '6dbe024bee4f13f90ad3676d92957e68868a616d'


def open_clip_snapshot(repo, revision):
    return snapshot_download(repo, revision=revision,
        allow_patterns=['*.json', '*.model', '*.txt', 'open_clip_model.safetensors'])


class MultilingualText(torch.nn.Module):
    def __init__(self, config, state):
        super().__init__()
        with torch.device('meta'):
            self.transformer = XLMRobertaModel(config, add_pooling_layer=any(k.startswith('transformer.pooler.') for k in state))
            self.LinearTransformation = torch.nn.Linear(config.hidden_size, 640)
        # Older Transformers persisted this deterministic buffer in checkpoints.
        state = dict(state)
        positions = state.pop('transformer.embeddings.position_ids', None)
        expected = torch.arange(config.max_position_embeddings).expand((1, -1))
        if positions is not None and not torch.equal(positions, expected):
            raise ValueError('Unexpected M-CLIP position IDs')
        self.load_state_dict(state, strict=True, assign=True)
        # Nonpersistent buffers are not restored by assign=True from a meta model.
        self.transformer.embeddings.position_ids = expected
        self.transformer.embeddings.token_type_ids = torch.zeros_like(expected)

    def forward(self, tokens):
        hidden = self.transformer(**tokens).last_hidden_state
        mask = tokens['attention_mask'].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        return self.LinearTransformation(pooled)


def load_mclip(repo, revision):
    import open_clip
    info = json.loads((Path(__file__).with_name('mclip_base_config.json')).read_text())
    config = XLMRobertaConfig.from_pretrained(info['repo'], revision=info['revision'])
    checkpoint = hf_hub_download(repo, 'pytorch_model.bin', revision=revision)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = MultilingualText(config, state)
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, use_fast=True)
    local = open_clip_snapshot(MCLIP_IMAGE_REPO, MCLIP_IMAGE_REVISION)
    paired, _, processor = open_clip.create_model_and_transforms('local-dir:' + local)
    vision = paired.visual
    components = {'image_encoder': MCLIP_IMAGE_REPO, 'image_revision': MCLIP_IMAGE_REVISION,
                  'text_pipeline': 'xlm-roberta-masked-mean-linear-640',
                  'text_config_repo': info['repo'], 'text_config_revision': info['revision']}
    return model, vision, processor, tokenizer, components



def place_mclip_text(model, device='cuda'):
    # Bounded, aligned allocations avoid large WSL CUDA mappings and host pinning.
    parameters = list(model.parameters())
    groups, current, used, dtype = [], [], 0, None
    for parameter in parameters:
        alignment = max(1, 256 // parameter.element_size())
        start = ((used + alignment - 1) // alignment) * alignment
        if current and (dtype != parameter.dtype or (start + parameter.numel()) * parameter.element_size() > 256 * 1024**2):
            groups.append((current, used, dtype))
            current, used, start = [], 0, 0
        current.append((parameter, start))
        used, dtype = start + parameter.numel(), parameter.dtype
    if current: groups.append((current, used, dtype))
    with torch.no_grad():
        for group, size, dtype in groups:
            packed = torch.empty(size, dtype=dtype, device=device)
            for parameter, offset in group:
                flat = parameter.detach().reshape(-1)
                chunk_items = 8 * 1024**2 // parameter.element_size()
                for start in range(0, flat.numel(), chunk_items):
                    end = min(start + chunk_items, flat.numel())
                    packed[offset + start:offset + end].copy_(flat[start:end])
                parameter.data = packed[offset:offset + flat.numel()].view_as(parameter)
    model.to(device)
