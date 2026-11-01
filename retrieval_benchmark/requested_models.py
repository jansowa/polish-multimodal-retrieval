"""Adapters for the owner's November reference evaluation; pinned by registration."""
from pathlib import Path


def require_last_token_pooling(config):
    """Sentence Transformers 5 and 6 serialize the same pooling differently."""
    modern = config.get('pooling_mode')
    if modern is not None and modern != 'lasttoken':
        raise ValueError('Author pooling is not last-token')
    if modern != 'lasttoken' and config.get('pooling_mode_lasttoken') is not True:
        raise ValueError('Author pooling is not last-token')
    if any(value is True for key,value in config.items() if key.startswith('pooling_mode_') and key!='pooling_mode_lasttoken'):
        raise ValueError('Author pooling combines multiple modes')


def last_token_pool(hidden,mask):
    import torch
    if mask.ndim!=2 or hidden.shape[:2]!=mask.shape or not mask.bool().any(1).all():
        raise ValueError('Invalid pooling attention mask')
    last=mask.shape[1]-1-mask.flip(1).argmax(1)
    return hidden[torch.arange(len(hidden),device=hidden.device),last]


def smoke_encoder(encoder,destination):
    """Tiny synthetic text/image checks in both prompt modes, not benchmark scores."""
    import json
    import time
    import torch
    from PIL import Image
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    paths=[]
    for name,color in [('red','red'),('blue','blue')]:
        path=destination/(name+'.png')
        Image.new('RGB',(224,224),color).save(path)
        paths.append(str(path))
    started=time.monotonic()
    if any(p.device.type!='cuda' for p in encoder.model.parameters()):
        raise ValueError('2B smoke requires the entire encoder on CUDA')
    checks=[]
    for mode in ['v2_no_instructions','author_recommended']:
        encoder.mode=mode
        text=encoder.encode(['A red square.','Niebieski kwadrat.'])
        vision=encoder.encode(paths,images=True)
        if text.shape!=vision.shape or len(text)!=2:
            raise ValueError('Text/image embedding dimensions differ')
        checks.append(dict(mode=mode,text_shape=list(text.shape),image_shape=list(vision.shape),finite=True))
    encoder.mode='v2_no_instructions'
    result=dict(status='passed',checks=checks,seconds=time.monotonic()-started,all_parameters_cuda=True,
                repository=encoder.repo,revision=encoder.revision,precision=encoder.precision)
    (destination/'adapter-smoke.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


class RequestedEncoder:
    def __init__(self, job, cache):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoImageProcessor
        self.torch = torch
        self.repo, self.revision = job['repository'], job['revision']
        self.batch_size = job['batch_size']
        self.kind = job['backend']
        self.mode = 'v2_no_instructions'
        self.text_max_length = 77
        self.components = dict(backend=self.kind)
        self.precision = 'cuda-bf16'
        self.cache = cache
        paths = {}
        for spec in job['components']:
            paths[spec['repository']] = snapshot_download(spec['repository'], revision=spec['revision'],
                cache_dir=str(cache), allow_patterns=spec['download_files'], max_workers=2)
        local = paths[self.repo]
        if self.kind == 'existing':
            from pl_mmb_benchmark.models import Encoder, REVISIONS
            REVISIONS.update({s['repository']: s['revision'] for s in job['components']})
            class Uncached(Encoder):
                def _cache_path(self, values, modality):
                    return Path('/proc/self/__requested_cache_disabled__')
                def _save_cache(self, path, tensor):
                    pass
            self.delegate = Uncached(self.repo, self.batch_size)
            self.__dict__.update({k: v for k, v in self.delegate.__dict__.items() if k not in ('cache',)})
        elif self.kind == 'openclip':
            import open_clip
            # Some older LAION repos use a different safetensors filename.
            alternate = Path(local)/'open_clip_pytorch_model.safetensors'
            canonical = Path(local)/'open_clip_model.safetensors'
            if alternate.exists() and not canonical.exists():
                canonical.symlink_to(alternate)
            self.model, _, self.processor = open_clip.create_model_and_transforms('local-dir:'+local)
            self.tokenizer = open_clip.get_tokenizer('local-dir:'+local)
            self.model.eval().bfloat16().to('cuda')
            self.text_max_length = self.model.context_length
        elif self.kind == 'bge':
            self.model = AutoModel.from_pretrained(local, trust_remote_code=True,
                torch_dtype=torch.bfloat16).eval().to('cuda')
            self.processor = AutoProcessor.from_pretrained(local)
            self.text_max_length = self.model.config.text_config.max_position_embeddings
        elif self.kind == 'siglip':
            from transformers import SiglipProcessor
            self.processor = SiglipProcessor(image_processor=AutoImageProcessor.from_pretrained(local, use_fast=False),
                                             tokenizer=AutoTokenizer.from_pretrained(local, use_fast=True))
            self.model = AutoModel.from_pretrained(local, torch_dtype=torch.bfloat16).eval().to('cuda')
            self.text_max_length = self.model.config.text_config.max_position_embeddings
        elif self.kind in ('qwen3', 'gme'):
            kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation='sdpa')
            # Preserve BF16 weights for 8B, spilling layers to CPU when required.
            if self.repo.endswith('-8B'):
                kwargs.update(device_map='auto', max_memory={0: '12GiB', 'cpu': '24GiB'},
                              offload_folder=str(cache/'offload'))
            self.model, loading = AutoModel.from_pretrained(local, output_loading_info=True, **kwargs)
            # GME also stores a generation head; it does not participate in embeddings.
            unexpected = [k for k in loading.get('unexpected_keys', []) if k != 'lm_head.weight']
            if loading.get('missing_keys') or unexpected or loading.get('mismatched_keys'):
                raise ValueError('Incomplete VLM checkpoint load: '+str(loading))
            self.processor = AutoProcessor.from_pretrained(local, padding_side='right',
                min_pixels=4096 if self.kind == 'qwen3' else 256*28*28,
                max_pixels=1800*32*32 if self.kind == 'qwen3' else 1280*28*28)
            self.model.eval()
            if not self.repo.endswith('-8B'):
                self.model.to('cuda')
            self.text_max_length = 8192 if self.kind == 'qwen3' else 1800
            import json
            pooling = json.loads((Path(local)/'1_Pooling/config.json').read_text())
            require_last_token_pooling(pooling)
            self.components.update(pooling='author last non-padding token',
                                   image_resize='native processor with author min/max pixels',
                                   cpu_offload=self.repo.endswith('-8B'))
        elif self.kind == 'nomic':
            code_spec = next(s for s in job['components'] if s['repository'] == 'nomic-ai/nomic-bert-2048')
            opts = dict(trust_remote_code=True, code_revision=code_spec['revision'],
                        torch_dtype=torch.bfloat16, use_flash_attn=False,
                        fused_bias_fc=False, fused_dropout_add_ln=False)
            self.vision_model = AutoModel.from_pretrained(local, **opts).eval().to('cuda')
            text = paths['nomic-ai/nomic-embed-text-v1.5']
            self.model = AutoModel.from_pretrained(text, **opts).eval().to('cuda')
            self.tokenizer = AutoTokenizer.from_pretrained(text)
            self.processor = AutoImageProcessor.from_pretrained(local)
            self.text_max_length = 8192
            self.components.update(text_repository='nomic-ai/nomic-embed-text-v1.5',
                                   vision_pooling='CLS', text_pooling='masked-mean-layer-norm')
        else:
            raise ValueError('Unknown backend '+self.kind)

    def encode(self, values, images=False):
        import torch
        from PIL import Image
        if self.kind == 'existing':
            return self.delegate.encode(values, images=images)
        chunks = []
        offset = 0
        while offset < len(values):
            batch = values[offset:offset+self.batch_size]
            opened = []
            try:
                if images:
                    for path in batch:
                        with Image.open(path) as im:
                            opened.append(im.convert('RGB'))
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    if self.kind == 'openclip':
                        inputs = torch.stack([self.processor(im) for im in opened]).cuda() if images else self.tokenizer(batch).cuda()
                        output = self.model.encode_image(inputs) if images else self.model.encode_text(inputs)
                    elif self.kind in ('siglip', 'bge'):
                        inputs = self.processor(images=opened, return_tensors='pt') if images else self.processor(
                            text=batch, padding='max_length' if self.kind == 'siglip' else True,
                            truncation=True, max_length=self.text_max_length, return_tensors='pt')
                        inputs = {k: v.cuda() for k,v in inputs.items()}
                        output = self.model.get_image_features(**inputs) if images else self.model.get_text_features(**inputs)
                        if not isinstance(output, torch.Tensor):
                            output = output.pooler_output
                    elif self.kind in ('qwen3', 'gme'):
                        instruction = ("Represent the user's input." if self.kind == 'qwen3' else 'You are a helpful assistant.')
                        if self.mode == 'author_recommended' and not images:
                            instruction = 'Find an image that matches the given text.'
                        messages = []
                        for i,value in enumerate(batch):
                            if self.kind == 'gme':
                                content = '<|vision_start|><|image_pad|><|vision_end|>' if images else value
                                messages.append(f'<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>')
                            else:
                                content = [dict(type='image', image=opened[i])] if images else [dict(type='text', text=value)]
                                conversation = [dict(role='system',content=[dict(type='text',text=instruction)]),
                                                dict(role='user',content=content)]
                                messages.append(self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False))
                        inputs = self.processor(text=messages, images=opened if images else None,
                            padding=True, truncation=True, max_length=self.text_max_length, return_tensors='pt')
                        inputs = inputs.to('cuda')
                        hidden = self.model(**inputs, use_cache=False).last_hidden_state
                        mask = inputs['attention_mask']
                        output = last_token_pool(hidden,mask)
                    else:
                        if images:
                            inputs = {k:v.cuda() for k,v in self.processor(images=opened, return_tensors='pt').items()}
                            output = self.vision_model(**inputs).last_hidden_state[:,0]
                        else:
                            prefix = 'search_query: ' if self.mode == 'author_recommended' else ''
                            inputs = self.tokenizer([prefix+t for t in batch], padding=True, truncation=True,
                                max_length=self.text_max_length, return_tensors='pt').to('cuda')
                            hidden = self.model(**inputs).last_hidden_state
                            mask = inputs['attention_mask'].unsqueeze(-1)
                            output = (hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
                            output = torch.nn.functional.layer_norm(output, (output.shape[-1],))
                    output = output.detach().float().cpu()
                    if output.ndim != 2 or len(output) != len(batch) or not torch.isfinite(output).all() or (output.norm(dim=-1)==0).any():
                        raise ValueError('Invalid embedding')
                    chunks.append(output)
                offset += len(batch)
            except torch.cuda.OutOfMemoryError:
                if self.batch_size == 1:
                    raise
                self.batch_size = max(1, self.batch_size//2)
                torch.cuda.empty_cache()
                print('OOM_RETRY_BATCH', self.batch_size, flush=True)
            finally:
                for im in opened:
                    im.close()
        return torch.cat(chunks)
