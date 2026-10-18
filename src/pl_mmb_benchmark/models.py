from pathlib import Path
import os
import hashlib
import json

MODELS = {
    'metaclip-s16': 'alphaedge-ai/metaclip-2-worldwide-s16-384-pol-16384',
    'metaclip-m16': 'alphaedge-ai/metaclip-2-worldwide-m16-384-pol-16384',
    'mobileclip2-s2': 'apple/MobileCLIP2-S2',
    'clip-b32-pl': 'alphaedge-ai/clip-ViT-B-32-multilingual-v1-pl-16384',
    'siglip2-pl': 'alphaedge-ai/siglip2-base-patch16-512-pol-16384',
    'siglip2-base': 'google/siglip2-base-patch16-512',
    'siglip2-pl-32768': 'alphaedge-ai/siglip2-base-patch16-512-pol-32768',
    'embeddinggemma-2': 'google/embeddinggemma-2',
    'siglip2-naflex': 'google/siglip2-base-patch16-naflex',
    'clip-b32-pl-32768': 'alphaedge-ai/clip-ViT-B-32-multilingual-v1-pl-32768',
}
REVISIONS = {
    MODELS['metaclip-s16']: '006223472923ba925226b59b6acd1e3e266c2382',
    MODELS['metaclip-m16']: 'bc15d9f5ddabe0497898578493a6e81288057c90',
    MODELS['mobileclip2-s2']: '72424e7025436db18f15c3eff6ee8c7c15ad4481',
    MODELS['clip-b32-pl']: 'f889319e4605da4342341fdb62ad39386396d6e1',
    MODELS['siglip2-pl']: '002ad19c05defef76aae6bc20e048c5fe930995b',
    MODELS['siglip2-base']: 'a89f5c5093f902bf39d3cd4d81d2c09867f0724b',
    MODELS['siglip2-pl-32768']: '1a0edc19692189ba5ee44da531995cfd4138dd57',
    MODELS['embeddinggemma-2']: '914f7f89142e33e77833254d9c9b90c3cef7303b',
    MODELS['siglip2-naflex']: 'b53b807d3a2d5e2b3911292f2d69e5341cdc064c',
    MODELS['clip-b32-pl-32768']: '62436cd6cd74f072dda57463b6916c95324e5b85',
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
    MODELS['siglip2-pl-32768']: {'parameters': 204381698, 'license': 'Apache-2.0', 'license_url': 'https://huggingface.co/alphaedge-ai/siglip2-base-patch16-512-pol-32768'},
    MODELS['embeddinggemma-2']: {'parameters': 438760448, 'license': 'Apache-2.0', 'license_url': 'https://huggingface.co/google/embeddinggemma-2'},
    MODELS['siglip2-naflex']: {'parameters': 375234050, 'license': 'Apache-2.0', 'license_url': 'https://huggingface.co/google/siglip2-base-patch16-naflex'},
    MODELS['clip-b32-pl-32768']: {'parameters': 156330240, 'license': 'MIT', 'license_url': 'https://huggingface.co/alphaedge-ai/clip-ViT-B-32-multilingual-v1-pl-32768'},
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
        self.sentence_clip = repo in (MODELS['clip-b32-pl'], MODELS['clip-b32-pl-32768'])
        self.native_multimodal = repo == MODELS['embeddinggemma-2']
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
        elif self.native_multimodal:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(repo, revision=self.revision, device='cuda',
                model_kwargs={'torch_dtype': torch.bfloat16}, config_kwargs={'audio_config': None},
                default_prompt_name=None, trust_remote_code=False)
            self.model.max_seq_length = 8192
            self.text_max_length = 8192
            self.autocast_dtype = torch.bfloat16
            self.precision = 'cuda-bf16'
            self.parameter_count = sum(p.numel() for p in self.model.parameters())
            if self.parameter_count >= 500_000_000 or any('audio' in n for n, _ in self.model.named_parameters()):
                raise RuntimeError('EmbeddingGemma-2 must load only text and vision, below 500M parameters')
            self.components = {'audio_encoder': False, 'vision_encoder': True, 'query_encoding': 'native-joint',
                               'pooling': 'mean', 'vision_soft_tokens': 280, 'prompt': ''}
        else:
            from transformers import AutoModel, AutoProcessor
            if repo in (MODELS['siglip2-pl'], MODELS['siglip2-base'], MODELS['siglip2-pl-32768']):
                from transformers import AutoImageProcessor, AutoTokenizer, SiglipProcessor
                self.processor = SiglipProcessor(
                    image_processor=AutoImageProcessor.from_pretrained(repo, revision=self.revision, use_fast=False),
                    tokenizer=AutoTokenizer.from_pretrained(repo, revision=self.revision, use_fast=True))
            elif repo == MODELS['siglip2-naflex']:
                self.processor = AutoProcessor.from_pretrained(repo, revision=self.revision, use_fast=True)
            else:
                from transformers import AutoImageProcessor
                self.processor = AutoProcessor.from_pretrained(repo, revision=self.revision, use_fast=True)
                self.processor.image_processor = AutoImageProcessor.from_pretrained(repo, revision=self.revision, use_fast=False)
            self.model = AutoModel.from_pretrained(repo, revision=self.revision)
            self.parameter_count = sum(p.numel() for p in self.model.parameters())
            self.text_max_length = self.model.config.text_config.max_position_embeddings
            if self.model.config.model_type in ('siglip', 'siglip2'):
                self.autocast_dtype = torch.bfloat16
                self.precision = 'cuda-bf16-autocast'
        self.model.eval().to('cuda')

    def encode(self, values, images=False):
        cache = self._cache_path(values, 'images' if images else 'texts')
        if cache.exists():
            cached = self.torch.load(cache, map_location='cpu', weights_only=True)
            if cached.ndim == 2 and len(cached) == len(values) and self.torch.isfinite(cached).all():
                print(f'Embedding cache: {len(values)} {"images" if images else "texts"}', flush=True)
                return cached
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
                    elif self.native_multimodal:
                        inputs = opened if images else batch
                        output = self.model.encode(inputs, batch_size=len(batch), convert_to_tensor=True,
                                                   normalize_embeddings=False, prompt='', show_progress_bar=False)
                    else:
                        padding = 'max_length' if self.model.config.model_type in ('siglip', 'siglip2') else True
                        inputs = self.processor(images=opened, return_tensors='pt') if images else self.processor(text=batch, padding=padding, truncation=True, max_length=self.text_max_length, return_tensors='pt')
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                        output = self.model.get_image_features(**inputs) if images else self.model.get_text_features(**inputs)
                    # Preserve native feature magnitudes until after multimodal fusion.
                    output = output.float().cpu()
                    if not torch.isfinite(output).all() or (output.norm(dim=-1) == 0).any():
                        raise RuntimeError('Nonfinite or zero model embeddings')
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
        result = torch.cat(chunks)
        self._save_cache(cache, result)
        return result

    def _cache_path(self, values, modality):
        identity = dict(policy='native-features-v2', repo=self.repo, revision=self.revision,
                        components=self.components, precision=self.precision,
                        text_max_length=self.text_max_length, modality=modality, values=values)
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        root = Path(os.environ.get('HF_HOME', str(Path.home()/'.cache/huggingface'))) / 'pl_mmb_benchmark/embeddings'
        root.mkdir(parents=True, exist_ok=True)
        return root / f'{digest}.pt'

    def _save_cache(self, path, tensor):
        temporary = path.with_suffix(f'.{os.getpid()}.tmp')
        self.torch.save(tensor, temporary)
        temporary.replace(path)

    def encode_joint(self, texts, images):
        """Native image+text query for EmbeddingGemma-2, without task prefixes."""
        torch = self.torch
        cache = self._cache_path(list(zip(texts, images)), 'native-joint')
        if cache.exists():
            cached = torch.load(cache, map_location='cpu', weights_only=True)
            if len(cached) == len(texts) and torch.isfinite(cached).all():
                return cached
        chunks = []
        offset = 0
        while offset < len(texts):
            batch = [{'text': text + ' <|image|>', 'image': image} for text, image in
                     zip(texts[offset:offset+self.batch_size], images[offset:offset+self.batch_size])]
            try:
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    output = self.model.encode(batch, batch_size=len(batch), convert_to_tensor=True,
                                               normalize_embeddings=False, prompt='', show_progress_bar=False).float().cpu()
            except torch.cuda.OutOfMemoryError:
                if self.batch_size == 1:
                    raise
                self.batch_size = max(1, self.batch_size//2)
                torch.cuda.empty_cache()
                print(f'CUDA OOM (joint): retry batch_size={self.batch_size}', flush=True)
                continue
            if not torch.isfinite(output).all():
                raise RuntimeError('Nonfinite native query embeddings')
            chunks.append(output)
            offset += len(batch)
        result = torch.cat(chunks)
        self._save_cache(cache, result)
        return result
