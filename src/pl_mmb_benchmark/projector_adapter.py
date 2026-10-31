"""Frozen TinyCLIP + PolDense checkpoint adapter for the shared v2 harness."""
import json
from contextlib import nullcontext
from pathlib import Path
import torch
from pl_mmb_training.common import ROOT,file_sha,read_json


class ProjectorEncoder:
    def __init__(self,checkpoint_path,batch_size=64,device='cuda',local_files_only=True):
        from sentence_transformers import SentenceTransformer
        from pl_mmb_training.commercial_backbones import load_backbone
        self.device=torch.device(device);self.batch_size=batch_size;self.checkpoint_path=Path(checkpoint_path)
        self.repo='local/'+self.checkpoint_path.parent.name;self.revision=file_sha(self.checkpoint_path)
        saved=torch.load(self.checkpoint_path,weights_only=True,map_location='cpu')
        self.projector=torch.nn.Linear(512,384).to(self.device).eval()
        self.projector.load_state_dict(saved['projector'] if 'projector' in saved else saved['state_dict'])
        self.projector.requires_grad_(False)
        lock=read_json(ROOT/'training/runs/vision-commercial-five/model_lock.json')['tinyclip-39m']
        self.vision,self.preprocess,self.vision_forward,self.vision_stats=load_backbone('tinyclip-39m',lock)
        self.vision=self.vision.to(self.device).eval().requires_grad_(False)
        self.text=SentenceTransformer('OPI-PIB/PolDense-32M',revision='ca70494a25dad90d980db7c735658dc7bf88ed25',
            local_files_only=local_files_only,trust_remote_code=False,device=str(self.device),model_kwargs={'attn_implementation':'sdpa'})
        self.text.max_seq_length=128;self.text.eval().requires_grad_(False)
        self.mobile=False;self.native_multimodal=False;self.sentence_clip=True
        self.precision='fp32-weights-bf16-autocast-cache-fp32' if self.device.type=='cuda' else 'cpu-fp32'
        self.parameter_count=sum(p.numel() for p in self.vision.parameters())+sum(p.numel() for p in self.text.parameters())+sum(p.numel() for p in self.projector.parameters())
        self.text_max_length=128;self.cache_hits=[]
        self.components=dict(vision=lock['model'],vision_revision=lock['revision'],
            text='OPI-PIB/PolDense-32M',text_revision='ca70494a25dad90d980db7c735658dc7bf88ed25',
            projector_sha256=self.revision,text_prefix='[query]: ',frozen=True)

    def _amp(self):
        return torch.autocast('cuda',dtype=torch.bfloat16) if self.device.type=='cuda' else nullcontext()

    def encode(self,values,images=False):
        from PIL import Image
        chunks=[]
        for start in range(0,len(values),self.batch_size):
            batch=values[start:start+self.batch_size]
            with torch.inference_mode():
                if images:
                    opened=[]
                    try:
                        for path in batch:
                            with Image.open(path) as image:opened.append(image.convert('RGB'))
                        pixels=self.preprocess(opened).to(self.device)
                        with self._amp():native=self.vision_forward(pixels).float()
                        output=self.projector(native)
                    finally:
                        for image in opened:image.close()
                else:
                    with self._amp():
                        output=self.text.encode(['[query]: '+v for v in batch],batch_size=len(batch),convert_to_tensor=True,
                            normalize_embeddings=False,show_progress_bar=False,prompt='')
                output=output.float().cpu()
                if not torch.isfinite(output).all() or (output.norm(dim=-1)==0).any():raise ValueError('Invalid projector-model embedding')
                chunks.append(output)
        return torch.cat(chunks)

    def encode_text_candidates(self,values):
        # [query]: is part of our trained embedding space in both directions/modes.
        return self.encode(values)


class AuthorPromptEncoder:
    """Only Gemma's documented text query/document prefixes differ in this suite."""
    def __init__(self,encoder,alias,mode):self.encoder=encoder;self.alias=alias;self.mode=mode
    def __getattr__(self,name):return getattr(self.encoder,name)
    def encode(self,values,images=False):
        if not images and self.mode=='author_recommended' and self.alias=='embeddinggemma-2':
            values=['task: search result | query: '+v for v in values]
        return self.encoder.encode(values,images=images)
    def encode_text_candidates(self,values):
        if self.mode=='author_recommended' and self.alias=='embeddinggemma-2':
            return self.encoder.encode(['title: none | text: '+v for v in values])
        return self.encoder.encode(values)
