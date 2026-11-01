"""Native, revision-pinned encoders for the commercial <100M comparison."""
from pathlib import Path
import sys

from pl_mmb_benchmark.models import Encoder, REVISIONS


class SmallEncoder(Encoder):
    def __init__(self, job, batch_size=64):
        import torch
        from huggingface_hub import hf_hub_download
        self.family = job['family']
        REVISIONS[job['repository']] = job['revision']
        if self.family == 'tinyclip':
            super().__init__(job['repository'], batch_size)
            return
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        self.torch, self.batch_size = torch, batch_size
        self.repo, self.revision = job['repository'], job['revision']
        self.mobile = self.tips = self.bge = self.mclip = self.sentence_clip = self.native_multimodal = False
        self.pe = self.family == 'pe'
        self.open_clip = self.family == 'vitamin'
        self.autocast_dtype, self.precision = torch.bfloat16, 'cuda-bf16'
        self.cache_hits, self.components = [], {}
        self.tokenizer = None
        self.text_max_length = 77
        def download(name):
            return hf_hub_download(self.repo, name, revision=self.revision)
        if self.pe:
            from pl_mmb_benchmark.special_models import pe_source, PE_CODE_REVISION
            sys.path.insert(0, str(pe_source()))
            from core.vision_encoder import pe
            from core.vision_encoder import transforms
            import zipfile
            checkpoint = download(job['files'][0]['name'])
            with torch.device('meta'):
                self.model = pe.CLIP.from_config(self.repo.split('/')[-1], pretrained=False)
            state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=zipfile.is_zipfile(checkpoint))
            state = state.get('state_dict', state.get('weights', state))
            state = {k.removeprefix('module.'): v.bfloat16() if v.is_floating_point() else v for k,v in state.items()}
            self.model.load_state_dict(state, strict=True, assign=True)
            self.model.visual.rope.init_tensors()
            if self.model.attn_mask is not None:
                self.model.attn_mask = self.model.build_causal_mask()
            if any(t.is_meta for t in list(self.model.parameters()) + list(self.model.buffers())):
                raise RuntimeError('Unrestored PE tensor')
            self.processor = transforms.get_image_transform(self.model.image_size)
            self.tokenizer = transforms.get_text_tokenizer(self.model.context_length)
            self.text_max_length = self.model.context_length
            self.components = dict(implementation='facebookresearch/perception_models',
                implementation_revision=PE_CODE_REVISION, image_resolution=self.model.image_size)
        elif self.open_clip:
            import open_clip
            name = self.repo.split('/')[-1]
            self.model, _, self.processor = open_clip.create_model_and_transforms(
                name, pretrained=download(job['files'][0]['name']))
            self.tokenizer = open_clip.get_tokenizer(name)
            self.text_max_length = self.model.context_length
            self.components = dict(implementation='open_clip', model_config=name)
        elif self.family == 'uform':
            from uform.torch_encoders import ImageEncoder, TextEncoder
            from uform.torch_processors import ImageProcessor, TextProcessor
            config = download('config.json')
            tokenizer = download('tokenizer.json')
            self.processor = ImageProcessor(config)
            self.text_processor = TextProcessor(config, tokenizer)
            self.model = torch.nn.ModuleDict({
                'image_encoder': ImageEncoder.from_pretrained(config, download('image_encoder.pt')),
                'text_encoder': TextEncoder.from_pretrained(config, download('text_encoder.pt'))})
            self.text_max_length = self.model['text_encoder'].max_position_embeddings
            self.components = dict(implementation='uform', package_version=__import__('importlib.metadata').metadata.version('uform'),
                                   modalities=['image_encoder','text_encoder'], cross_modal_reranking=False)
        else:
            raise ValueError('Unknown native family')
        self.model.eval().bfloat16().to('cuda')
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

    def _cache_path(self, values, modality):
        return Path('/proc/self/__commercial_cache_disabled__')

    def _save_cache(self, path, tensor):
        pass

    def encode(self, values, images=False):
        if self.family != 'uform':
            return super().encode(values, images)
        from PIL import Image
        torch = self.torch
        chunks = []
        for start in range(0, len(values), self.batch_size):
            batch = values[start:start+self.batch_size]
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                if images:
                    prepared = []
                    for path in batch:
                        with Image.open(path) as im:
                            prepared.append(self.processor(im.convert('RGB'))['images'][0])
                    features = self.model['image_encoder'].encode(torch.stack(prepared).cuda())
                else:
                    tokens = {k:v.cuda() for k,v in self.text_processor(batch).items()}
                    features = self.model['text_encoder'].encode(tokens)
                if isinstance(features, tuple):
                    raise RuntimeError('Expected native embeddings, not intermediate features')
                features = features.float().cpu()
                if not torch.isfinite(features).all() or (features.norm(dim=-1)==0).any():
                    raise RuntimeError('Invalid native embeddings')
                chunks.append(features)
        return torch.cat(chunks)
