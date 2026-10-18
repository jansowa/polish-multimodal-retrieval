from pathlib import Path
import os

MODELS = {
    'metaclip-s16': 'alphaedge-ai/metaclip-2-worldwide-s16-384-pol-16384',
    'metaclip-m16': 'alphaedge-ai/metaclip-2-worldwide-m16-384-pol-16384',
    'mobileclip2-s2': 'apple/MobileCLIP2-S2',
    'clip-b32-pl': 'alphaedge-ai/clip-ViT-B-32-multilingual-v1-pl-16384',
    'siglip2-pl': 'alphaedge-ai/siglip2-base-patch16-512-pol-16384',
    'siglip2-base': 'google/siglip2-base-patch16-512',
}
REVISIONS = {
    MODELS['metaclip-s16']: '006223472923ba925226b59b6acd1e3e266c2382',
    MODELS['metaclip-m16']: 'bc15d9f5ddabe0497898578493a6e81288057c90',
    MODELS['mobileclip2-s2']: '72424e7025436db18f15c3eff6ee8c7c15ad4481',
    MODELS['clip-b32-pl']: 'f889319e4605da4342341fdb62ad39386396d6e1',
    MODELS['siglip2-pl']: '002ad19c05defef76aae6bc20e048c5fe930995b',
    MODELS['siglip2-base']: 'a89f5c5093f902bf39d3cd4d81d2c09867f0724b',
}
CLIP_IMAGE_REPO = 'openai/clip-vit-base-patch32'
CLIP_IMAGE_REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
MODEL_METADATA = {
    MODELS['metaclip-s16']: {'parameters': 49722241, 'license': 'CC-BY-NC-4.0',
        'license_url': 'https://huggingface.co/alphaedge-ai/metaclip-2-worldwide-s16-384-pol-16384/blob/006223472923ba925226b59b6acd1e3e266c2382/README.md'},
    MODELS['metaclip-m16']: {'parameters': 85301761, 'license': 'CC-BY-NC-4.0',
        'license_url': 'https://huggingface.co/alphaedge-ai/metaclip-2-worldwide-m16-384-pol-16384/blob/bc15d9f5ddabe0497898578493a6e81288057c90/README.md'},
    MODELS['mobileclip2-s2']: {'parameters': 99243329, 'license': 'Apple AMLR',
        'license_url': 'https://huggingface.co/apple/MobileCLIP2-S2/blob/72424e7025436db18f15c3eff6ee8c7c15ad4481/LICENSE'},
    MODELS['clip-b32-pl']: {'parameters': 143747328, 'license': 'MIT',
        'license_url': 'https://huggingface.co/alphaedge-ai/clip-ViT-B-32-multilingual-v1-pl-16384/blob/f889319e4605da4342341fdb62ad39386396d6e1/README.md'},
    MODELS['siglip2-pl']: {'parameters': 191798786, 'license': 'Apache-2.0',
        'license_url': 'https://huggingface.co/alphaedge-ai/siglip2-base-patch16-512-pol-16384/blob/002ad19c05defef76aae6bc20e048c5fe930995b/README.md'},
    MODELS['siglip2-base']: {'parameters': 375823874, 'license': 'Apache-2.0',
        'license_url': 'https://huggingface.co/google/siglip2-base-patch16-512/blob/a89f5c5093f902bf39d3cd4d81d2c09867f0724b/README.md'},
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
        self.sentence_clip = repo == MODELS['clip-b32-pl']
        self.components = {}
        self.text_max_length = 77
        self.autocast_dtype = torch.float16
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
        elif self.sentence_clip:
            from sentence_transformers import SentenceTransformer
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
            self.model = SentenceTransformer(repo, revision=self.revision, device='cuda', trust_remote_code=False)
            self.processor = CLIPImageProcessor.from_pretrained(CLIP_IMAGE_REPO, revision=CLIP_IMAGE_REVISION)
            self.vision_model = CLIPVisionModelWithProjection.from_pretrained(CLIP_IMAGE_REPO, revision=CLIP_IMAGE_REVISION).eval().to('cuda')
            self.text_max_length = self.model.max_seq_length
            self.parameter_count = sum(p.numel() for p in self.model.parameters()) + sum(p.numel() for p in self.vision_model.parameters())
            self.components = {'image_encoder': CLIP_IMAGE_REPO, 'image_revision': CLIP_IMAGE_REVISION,
                               'text_pipeline': 'sentence-transformers-mean-pooling-dense'}
        else:
            from transformers import AutoModel, AutoProcessor
            if repo in (MODELS['siglip2-pl'], MODELS['siglip2-base']):
                from transformers import AutoImageProcessor, AutoTokenizer, SiglipProcessor
                self.processor = SiglipProcessor(
                    image_processor=AutoImageProcessor.from_pretrained(repo, revision=self.revision, use_fast=False),
                    tokenizer=AutoTokenizer.from_pretrained(repo, revision=self.revision, use_fast=True))
            else:
                self.processor = AutoProcessor.from_pretrained(repo, revision=self.revision, use_fast=False)
            self.model = AutoModel.from_pretrained(repo, revision=self.revision)
            self.parameter_count = sum(p.numel() for p in self.model.parameters())
            self.text_max_length = self.model.config.text_config.max_position_embeddings
            if self.model.config.model_type == 'siglip':
                self.autocast_dtype = torch.bfloat16
                self.precision = 'cuda-bf16-autocast'
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
                with torch.inference_mode(), torch.autocast('cuda', dtype=self.autocast_dtype, enabled=not self.mobile):
                    if self.mobile:
                        inputs = torch.stack([self.processor(im) for im in opened]).cuda() if images else self.tokenizer(batch).cuda()
                        output = self.model.encode_image(inputs) if images else self.model.encode_text(inputs)
                    elif self.sentence_clip:
                        if images:
                            inputs = {k: v.cuda() for k, v in self.processor(images=opened, return_tensors='pt').items()}
                            output = self.vision_model(**inputs).image_embeds
                        else:
                            output = self.model.encode(batch, batch_size=len(batch), convert_to_tensor=True,
                                                       normalize_embeddings=False, show_progress_bar=False)
                    else:
                        padding = 'max_length' if self.model.config.model_type == 'siglip' else True
                        inputs = self.processor(images=opened, return_tensors='pt') if images else self.processor(text=batch, padding=padding, truncation=True, max_length=self.text_max_length, return_tensors='pt')
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
