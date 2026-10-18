from pathlib import Path
import os

MODELS = {
    'metaclip-s16': 'alphaedge-ai/metaclip-2-worldwide-s16-384-pol-16384',
    'metaclip-m16': 'alphaedge-ai/metaclip-2-worldwide-m16-384-pol-16384',
    'mobileclip2-s2': 'apple/MobileCLIP2-S2',
}
REVISIONS = {
    MODELS['metaclip-s16']: '006223472923ba925226b59b6acd1e3e266c2382',
    MODELS['metaclip-m16']: 'bc15d9f5ddabe0497898578493a6e81288057c90',
    MODELS['mobileclip2-s2']: '72424e7025436db18f15c3eff6ee8c7c15ad4481',
}
MODEL_METADATA = {
    MODELS['metaclip-s16']: {'parameters': 49722241, 'license': 'CC-BY-NC-4.0',
        'license_url': 'https://huggingface.co/alphaedge-ai/metaclip-2-worldwide-s16-384-pol-16384/blob/006223472923ba925226b59b6acd1e3e266c2382/README.md'},
    MODELS['metaclip-m16']: {'parameters': 85301761, 'license': 'CC-BY-NC-4.0',
        'license_url': 'https://huggingface.co/alphaedge-ai/metaclip-2-worldwide-m16-384-pol-16384/blob/bc15d9f5ddabe0497898578493a6e81288057c90/README.md'},
    MODELS['mobileclip2-s2']: {'parameters': 99243329, 'license': 'Apple AMLR',
        'license_url': 'https://huggingface.co/apple/MobileCLIP2-S2/blob/72424e7025436db18f15c3eff6ee8c7c15ad4481/LICENSE'},
}


def check_cache():
    # Run before importing HF libraries. Reject mounted Windows caches, including symlinks.
    default = Path.home() / '.cache/huggingface'
    paths = [os.environ.get(k, str(default)) for k in
             ('HF_HOME', 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE', 'HF_DATASETS_CACHE')]
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if str(path).startswith('/mnt/'):
            raise RuntimeError(f'Benchmark requires native Linux HF cache: {path}')
    os.environ.setdefault('HF_HOME', str(default))
    return os.environ['HF_HOME']


class Encoder:
    def __init__(self, repo, batch_size):
        import torch
        from huggingface_hub import HfApi, hf_hub_download
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required; CPU fallback disabled')
        self.torch = torch
        self.batch_size = batch_size
        self.revision = REVISIONS.get(repo) or HfApi().model_info(repo, timeout=30).sha
        self.repo = repo
        self.mobile = repo == MODELS['mobileclip2-s2']
        self.precision = 'cuda-fp32' if self.mobile else 'cuda-fp16-autocast'
        if self.mobile:
            import open_clip
            checkpoint = hf_hub_download(repo, 'mobileclip2_s2.pt', revision=self.revision)
            self.model, _, self.processor = open_clip.create_model_and_transforms(
                'MobileCLIP2-S2', pretrained=checkpoint,
                image_mean=(0., 0., 0.), image_std=(1., 1., 1.), image_interpolation='bilinear')
            from timm.utils import reparameterize_model
            self.model.eval()
            self.parameter_count = sum(p.numel() for p in self.model.parameters())
            self.model = reparameterize_model(self.model, inplace=True)
            self.tokenizer = open_clip.get_tokenizer('MobileCLIP2-S2')
        else:
            from transformers import AutoModel, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(repo, revision=self.revision, use_fast=False)
            self.model = AutoModel.from_pretrained(repo, revision=self.revision)
            self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.model.eval().to('cuda')

    def encode(self, values, images=False):
        from PIL import Image
        torch = self.torch
        chunks = []
        offset = 0
        while offset < len(values):
            batch = values[offset:offset + self.batch_size]
            opened = []
            try:
                if images:
                    for path in batch:
                        with Image.open(path) as im:
                            opened.append(im.convert('RGB'))
                # FastViT activations can overflow FP16 before final normalization.
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=not self.mobile):
                    if self.mobile:
                        inputs = torch.stack([self.processor(im) for im in opened]).cuda() if images else self.tokenizer(batch).cuda()
                        output = self.model.encode_image(inputs) if images else self.model.encode_text(inputs)
                    else:
                        inputs = self.processor(images=opened, return_tensors='pt') if images else self.processor(text=batch, padding=True, truncation=True, max_length=77, return_tensors='pt')
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                        output = self.model.get_image_features(**inputs) if images else self.model.get_text_features(**inputs)
                    output = torch.nn.functional.normalize(output.float(), dim=-1).cpu()
                    if not torch.isfinite(output).all():
                        raise RuntimeError('Nonfinite model embeddings')
                chunks.append(output)
                offset += len(batch)
            except torch.cuda.OutOfMemoryError:
                if self.batch_size == 1:
                    raise
                self.batch_size = max(1, self.batch_size // 2)
                torch.cuda.empty_cache()
                print(f'CUDA OOM: retry batch_size={self.batch_size}', flush=True)
            finally:
                for im in opened:
                    im.close()
        return torch.cat(chunks)
