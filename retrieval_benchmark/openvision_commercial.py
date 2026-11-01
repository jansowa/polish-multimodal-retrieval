"""Additional Apache-2.0 OpenVision references; reuse the registered serial worker."""
import argparse
import json
import os
from pathlib import Path
import ssl
import subprocess
import sys
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from retrieval_benchmark import commercial as engine
from retrieval_benchmark.core import HERE, local_path, read, sha, write
from retrieval_benchmark.ingest import configuration,sync
from retrieval_benchmark.licenses import fetch
from retrieval_benchmark.runner import selected_datasets

CANDIDATES=[('openvision-s8-224','UCSC-VLAA/openvision-vit-small-patch8-224'),
            ('openvision-s16-384','UCSC-VLAA/openvision-vit-small-patch16-384'),
            ('openvision-t8-384','UCSC-VLAA/openvision-vit-tiny-patch8-384')]


def destination():
    return local_path('C:/Users/jsowa/benchmark-translations/retrieval-benchmark-v1/commercial-openvision-20261101')


original_implementation=engine.implementation
def implementation():
    return dict(original_implementation(),openvision_commercial=sha(__file__))


def prepare():
    target=destination()/'registration.json'
    if target.exists():return target
    config=configuration(HERE/'local/config.json')
    context=ssl.create_default_context(cafile=str(ROOT/'combined-ca.pem'))
    jobs=[]
    for alias,repo in CANDIDATES:
        with urlopen('https://huggingface.co/api/models/'+repo+'?blobs=true',context=context,timeout=60) as response:
            info=json.load(response)
        revision=info['sha'];_,license_info=fetch((repo,revision))
        if license_info['id']!='apache-2.0':raise ValueError('Unexpected weight license')
        item=next(i for i in info['siblings'] if i['rfilename']=='open_clip_pytorch_model.bin')
        lfs=item['lfs'];digest=lfs.get('sha256') or lfs.get('oid','').removeprefix('sha256:')
        if len(digest)!=64:raise ValueError('Missing LFS SHA')
        files=[dict(name=item['rfilename'],bytes=item.get('size') or lfs['size'],sha256=digest)]
        jobs.append(dict(alias=alias,repository=repo,revision=revision,family='openvision',files=files,
            model_id='hf:'+repo+'@'+revision,license=license_info,certified_encoder_parameters=None))
        config['models'][alias]=dict(repository=repo,revision=revision,role='reference',representative=True,
            download_budget_bytes=sum(f['bytes'] for f in files),native_language='English',commercial_use_license='apache-2.0')
    protocols=selected_datasets(config,list(config['datasets']))
    registration=dict(schema_version=2,status='closed_before_new_scores',group='openvision',source='commercial-openvision-20261101',
        storage=str(destination()),max_encoder_parameters_exclusive=100_000_000,implementation=implementation(),
        selection='Further primary-source inventory found a newer contrastive+generative encoder family. Select Small/8-224, Small/16-384 and Tiny/8-384 before their scores; not generative-only OpenVision2/3.',
        native_language='English; unmodified Polish queries, no translation.',config=config,jobs=jobs,batch_size=64,
        simultaneous_models=1,parameter_policy='All loaded V+T including token embeddings; abort at >=100M before scoring.',
        datasets={k:dict(protocol_sha256=config['datasets'][k]['protocol_sha256'],manifest_sha256=v[0]['manifest_sha256']) for k,v in protocols.items()},
        preprocessing='Pinned repository open_clip_config and tokenizer files, native OpenCLIP local-dir loader; author image mean/std, bilinear squash resize and strip_sep_token.',
        cleanup='Only newly downloaded, registered LFS weights after complete results; preserve existing weights and all datasets.')
    write(target,registration,immutable=True);write(HERE/'local/config.json',config)
    return target


def worker(alias):
    sys.path.insert(0,str(ROOT/'src'))
    from retrieval_benchmark import commercial_models
    from pl_mmb_benchmark.models import Encoder
    import torch
    import open_clip
    from huggingface_hub import snapshot_download
    class OpenVisionEncoder(Encoder):
        def __init__(self,job,batch_size):
            if not torch.cuda.is_available():raise RuntimeError('CUDA required')
            self.torch,self.batch_size=torch,batch_size
            self.repo,self.revision=job['repository'],job['revision']
            self.mobile=self.tips=self.bge=self.mclip=self.sentence_clip=self.native_multimodal=self.pe=False
            self.open_clip=True
            self.autocast_dtype,self.precision=torch.bfloat16,'cuda-bf16'
            self.cache_hits=[]
            local=snapshot_download(self.repo,revision=self.revision,
                allow_patterns=['*.json','*.txt','*.model','open_clip_pytorch_model.bin'])
            self.model,_,self.processor=open_clip.create_model_and_transforms('local-dir:'+local)
            self.tokenizer=open_clip.get_tokenizer('local-dir:'+local)
            self.text_max_length=self.model.context_length
            self.components=dict(implementation='open_clip',image_preprocess='pinned-author-config',tokenizer_revision=self.revision)
            self.model.eval().bfloat16().cuda()
            self.parameter_count=sum(p.numel() for p in self.model.parameters())
        def _cache_path(self,values,modality):return Path('/proc/self/__openvision_cache_disabled__')
        def _save_cache(self,path,tensor):pass
    # Process-local factory injection: existing workers and frozen source files stay untouched.
    commercial_models.SmallEncoder=OpenVisionEncoder
    engine.implementation=implementation
    engine.worker(alias,destination()/'registration.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','worker']);parser.add_argument('--model')
    args=parser.parse_args()
    if args.command=='prepare':print(prepare());return
    if args.command=='worker':worker(args.model);return
    reg=read(destination()/'registration.json')
    for job in reg['jobs']:
        if (destination()/job['alias']/'cleanup_complete.json').exists():continue
        print('START',job['alias'],flush=True)
        with (destination()/(job['alias']+'.log')).open('a',encoding='utf-8') as log:
            result=subprocess.run(['/home/jsowa/.venvs/pl-mmb-legacy-cu130/bin/python','-u',str(Path(__file__).resolve()),'worker','--model',job['alias']],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Worker failed: '+job['alias'])
        print(json.dumps(sync(configuration(HERE/'local/config.json'))),flush=True)
    write(destination()/'complete.json',dict(status='complete',models=[j['alias'] for j in reg['jobs']]),immutable=True)
    print('QUEUE_COMPLETE',flush=True)


if __name__=='__main__':main()
