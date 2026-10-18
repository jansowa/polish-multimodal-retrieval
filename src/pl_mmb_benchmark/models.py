from pathlib import Path
import os

MODELS = {
    'metaclip-s16': 'alphaedge-ai/metaclip-2-worldwide-s16-384-pol-16384',
    'metaclip-m16': 'alphaedge-ai/metaclip-2-worldwide-m16-384-pol-16384',
    'mobileclip2-s2': 'apple/MobileCLIP2-S2',
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
        self.revision = HfApi().model_info(repo).sha
        self.repo = repo
        self.mobile = repo == MODELS['mobileclip2-s2']
        if self.mobile:
            import open_clip
            checkpoint = hf_hub_download(repo, 'mobileclip2_s2.pt', revision=self.revision)
            self.model, _, self.processor = open_clip.create_model_and_transforms('MobileCLIP2-S2', pretrained=checkpoint)
            self.tokenizer = open_clip.get_tokenizer('MobileCLIP2-S2')
        else:
            from transformers import AutoModel, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(repo, revision=self.revision)
            self.model = AutoModel.from_pretrained(repo, revision=self.revision)
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
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                    if self.mobile:
                        inputs = torch.stack([self.processor(im) for im in opened]).cuda() if images else self.tokenizer(batch).cuda()
                        output = self.model.encode_image(inputs) if images else self.model.encode_text(inputs)
                    else:
                        inputs = self.processor(images=opened, return_tensors='pt') if images else self.processor(text=batch, padding=True, truncation=True, max_length=77, return_tensors='pt')
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                        output = self.model.get_image_features(**inputs) if images else self.model.get_text_features(**inputs)
                    output = torch.nn.functional.normalize(output.float(), dim=-1).cpu()
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
