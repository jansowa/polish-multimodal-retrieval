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
    # One GPU allocation, small host transfers: avoid WSL pinning a giant CPU block.
    parameters = list(model.parameters())
    with torch.no_grad():
        packed = torch.empty(sum(p.numel() for p in parameters), dtype=parameters[0].dtype, device=device)
        offset = 0
        for parameter in parameters:
            flat = parameter.detach().reshape(-1)
            for start in range(0, flat.numel(), 4_194_304):
                end = min(start + 4_194_304, flat.numel())
                packed[offset + start:offset + end].copy_(flat[start:end])
            offset += flat.numel()
        torch.nn.utils.vector_to_parameters(packed, parameters)
    model.to(device)
