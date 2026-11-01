"""Evaluate the two strongest archived non-TinyCLIP project families; no training."""
import argparse
import gc
import json
import os
import ssl
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from retrieval_benchmark.core import HERE,identity,local_path,read,sha,write
from retrieval_benchmark.ingest import configuration,sync
from retrieval_benchmark.runner import DatasetEncoder,Originals,selected_datasets,tokenization
from retrieval_benchmark.completion import cleanup,destination,metadata,parameter_guard


def dest():return destination()/'own-legacy'


def prepare():
    path=dest()/'registration.json'
    if path.exists():return path
    config=configuration(HERE/'local/config.json')
    cache=read(ROOT/'training/runs/e1-full/cache.json')
    models=[]
    for kind,alias,checkpoint in [
        ('siglip2','ours-siglip2-poldense32-seed42','training/runs/p0-e2/E2b-seed-42/projector.pt'),
        ('mobileclip2','ours-mobileclip2-poldense32-seed42','training/runs/vision-four/models/mobileclip2-s2/E2b-seed-42/projector.pt')]:
        if kind=='siglip2':
            repo='google/siglip2-base-patch16-256';revision='3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab'
            info=metadata(repo,revision);features=cache['image_features'];lock=None
        else:
            lock=read(ROOT/'training/runs/vision-four/model_lock.json')['mobileclip2-s2']
            repo=lock['model'];revision=lock['revision']
            context=ssl.create_default_context(cafile=str(ROOT/'combined-ca.pem'))
            with urlopen('https://huggingface.co/api/models/'+repo+'/revision/'+revision+'?blobs=true',context=context,timeout=60) as response:
                api=json.load(response)
            f=next(f for f in api['siblings'] if f['rfilename']=='mobileclip2_s2.pt');lfs=f['lfs']
            info=dict(files=[dict(name=f['rfilename'],bytes=f['size'],sha256=lfs.get('sha256') or lfs['oid'])])
            features=read(ROOT/'training/runs/vision-four/models/mobileclip2-s2/feature_cache.json')
        cp=ROOT/checkpoint
        if not cp.is_file() or not local_path(features['path']).is_file():raise ValueError('Missing original checkpoint/features')
        models.append(dict(kind=kind,alias=alias,checkpoint=str(cp),checkpoint_sha256=sha(cp),repository=repo,
            revision=revision,files=info['files'],lock=lock,original_image_features=features,
            original_text_features=cache['text_features']))
    reg=dict(schema_version=1,status='registered_before_first_new_score',models=models,config=config,
        seed_rule='Original seed42 minimum-validation-loss checkpoint; no selection on new OOD outcomes.',
        rationale='Best archived non-TinyCLIP families in earlier controlled vision study; same original pretrained embedding and frozen PolDense32M.',
        original_cache_manifest_path=str(ROOT/'training/runs/e1-full/cache.json'),
        original_cache_manifest_sha256=sha(ROOT/'training/runs/e1-full/cache.json'),
        datasets={k:dict(protocol_sha256=v['protocol_sha256']) for k,v in config['datasets'].items()},
        worker_sha256=sha(__file__),scorer_sha256=sha(ROOT/'src/pl_mmb_benchmark/full_gallery.py'),
        max_encoder_parameters=500000000,check_control='First original 32 image/text vectors must match historical caches; maximum abs <=1e-5.',
        parent_reference_registration_sha256=sha(destination()/'registration.json'),XM3600=False,DOCCI=False)
    write(path,reg,immutable=True)
    return path


def worker(alias):
    import fcntl
    sys.path.insert(0,str(ROOT/'src'))
    import torch
    from huggingface_hub import hf_hub_download
    from sentence_transformers import SentenceTransformer
    from pl_mmb_benchmark.projector_adapter import ProjectorEncoder,AuthorPromptEncoder
    from pl_mmb_benchmark.full_gallery import evaluate_encoder
    from pl_mmb_benchmark.efficiency import parameter_counts
    reg=read(dest()/'registration.json');item=next(m for m in reg['models'] if m['alias']==alias)
    folder=dest()/alias
    if (folder/'cleanup_complete.json').exists():return
    if sha(__file__)!=reg['worker_sha256']:raise ValueError('Registered own worker changed')
    if sha(local_path(item['checkpoint']))!=item['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    config=reg['config'];torch.set_num_threads(2)
    with (ROOT/'training/runs/tinyclip-visualnews-v1/worker.lock').open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        os.environ['REQUESTS_CA_BUNDLE']=str(ROOT/'combined-ca.pem');os.environ['SSL_CERT_FILE']=str(ROOT/'combined-ca.pem')
        snapshot=Path.home()/'.cache/huggingface/hub'/('models--'+item['repository'].replace('/','--'))/'snapshots'/item['revision']
        ownership=read(folder/'ownership.json') if (folder/'ownership.json').exists() else [dict(**f,snapshot_path=str(snapshot/f['name']),preexisting=(snapshot/f['name']).is_file()) for f in item['files']]
        write(folder/'ownership.json',ownership,immutable=True)
        # Download only this model's originally pinned vision source; PolDense remains protected/local.
        for f in item['files']:
            p=hf_hub_download(item['repository'],f['name'],revision=item['revision'])
            if sha(p)!=f['sha256']:raise ValueError('Vision weight SHA mismatch')
        if item['kind']=='mobileclip2':
            hf_hub_download(item['repository'],'config.json',revision=item['revision'])
        class LegacyProjector(ProjectorEncoder):
            def __init__(self):
                self.batch_size=64;self.device=torch.device('cuda');self.checkpoint_path=local_path(item['checkpoint'])
                self.repo=alias;self.revision=item['checkpoint_sha256']
                saved=torch.load(self.checkpoint_path,weights_only=True,map_location='cpu')
                self.projector=torch.nn.Linear(saved['input_dim'],saved['output_dim']).to(self.device).eval()
                self.projector.load_state_dict(saved['state_dict']);self.projector.requires_grad_(False)
                if item['kind']=='siglip2':
                    from transformers import SiglipVisionModel,AutoImageProcessor
                    self.vision=SiglipVisionModel.from_pretrained(item['repository'],revision=item['revision'],local_files_only=True,attn_implementation='sdpa').eval().to(self.device)
                    processor=AutoImageProcessor.from_pretrained(item['repository'],revision=item['revision'],local_files_only=True,use_fast=False)
                    self.preprocess=lambda images:processor(images=images,return_tensors='pt')['pixel_values']
                    self.vision_forward=lambda pixels:self.vision(pixel_values=pixels).pooler_output
                else:
                    from pl_mmb_training.backbones import load_backbone
                    self.vision,self.preprocess,self.vision_forward,self.vision_stats=load_backbone('mobileclip2-s2',item['lock'])
                    self.vision=self.vision.eval().to(self.device)
                self.vision.requires_grad_(False)
                self.text=SentenceTransformer('OPI-PIB/PolDense-32M',revision='ca70494a25dad90d980db7c735658dc7bf88ed25',local_files_only=True,trust_remote_code=False,device='cuda',model_kwargs={'attn_implementation':'sdpa'})
                self.text.max_seq_length=128;self.text.eval().requires_grad_(False)
                self.text_max_length=128;self.precision='fp32-weights-bf16-autocast-cache-fp32'
        encoder=LegacyProjector();counts=parameter_guard(parameter_counts(encoder))
        manifest_path=local_path(reg['original_cache_manifest_path'])
        if sha(manifest_path)!=reg['original_cache_manifest_sha256']:raise ValueError('Original control manifest changed')
        cache=read(manifest_path)
        from PIL import Image
        images=[]
        try:
            for path in cache['images'][:32]:
                with Image.open(local_path(path)) as image:images.append(image.convert('RGB'))
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                image_vectors=encoder.vision_forward(encoder.preprocess(images).to('cuda')).float().cpu()
                text_vectors=encoder.text.encode(['[query]: '+t for t in cache['texts'][:32]],batch_size=32,convert_to_tensor=True,normalize_embeddings=False,prompt='',show_progress_bar=False).float().cpu()
        finally:
            for image in images:image.close()
        control={}
        for modality,actual in [('image',image_vectors),('text',text_vectors)]:
            info=item['original_'+modality+'_features'];path=local_path(info['path'])
            if sha(path)!=info['sha256']:raise ValueError('Original feature cache changed')
            expected=torch.load(path,weights_only=True,map_location='cpu')[:32]
            delta=float((actual-expected).abs().max());control[modality]=dict(max_abs=delta,exact=torch.equal(actual,expected))
            if not torch.allclose(actual,expected,rtol=0,atol=1e-5):raise ValueError('Legacy feature parity failed: '+modality+' '+str(delta))
        write(folder/'feature_parity.json',control,immutable=True);print('OWN_PARITY_OK',alias,control,flush=True)
        model_id='checkpoint:'+item['checkpoint_sha256']
        model=dict(label=alias,repository=None,revision=item['checkpoint_sha256'],parameters=counts,role='project',representative=True,seed=42,efficiency=None,
                   encoder_revisions=dict(vision=item['revision'],text='ca70494a25dad90d980db7c735658dc7bf88ed25'),trained_on_coco=True,trained_on_visualnews=False,trained_on_wiki=False)
        artifacts=[]
        for name,(protocol,manifest) in selected_datasets(config,list(config['datasets'])).items():
            features=DatasetEncoder(encoder,Originals(manifest,config))
            for mode in ('v2_no_instructions','author_recommended'):
                run_id=identity(dict(registration=sha(dest()/'registration.json'),alias=alias,dataset=name,mode=mode));out=local_path(config['output'])/run_id
                if not (out/'result.json').exists():
                    result=evaluate_encoder(AuthorPromptEncoder(features,alias,mode),dict(protocol,prompt_mode=mode),dict(status='closed_ready_for_execution'))
                    result.update(checkpoint_sha256=item['checkpoint_sha256'],precision=encoder.precision,parameters=counts,tokenization=tokenization(encoder,protocol['texts'],alias,mode))
                    write(out/'scores.json',result,immutable=True)
                    row=dict(model_id=model_id,dataset_id=name,prompt_mode=mode,profile='historical-native-v2',protocol=result['protocol'],metrics={d:result['directions'][d]['metrics'] for d in ('t2i','i2t')},query_counts={d:len(result['directions'][d]['query_ids']) for d in ('t2i','i2t')},precision=encoder.precision,exposure='ID source COCO' if name.startswith('xtd10') else 'source-OOD to projector',source='own-legacy-completion-20261031',source_sha256=sha(out/'scores.json'),registration_sha256=sha(dest()/'registration.json'),tokenization=result['tokenization'],**{k:result[k] for k in ('manifest_sha256','gallery_sha256','text_gallery_sha256','qrels_sha256')})
                    write(out/'result.json',dict(status='complete',model_id=model_id,model_metadata=model,public_result=row,execution_sha256=sha(dest()/'registration.json')),immutable=True)
                artifacts.extend(dict(path=str(out/f),sha256=sha(out/f)) for f in ('scores.json','result.json'))
                print('OWN_RESULT',alias,name,mode,flush=True)
            del features
        write(folder/'complete.json',dict(status='complete',artifacts=artifacts,parameters=counts),immutable=True)
        del encoder;gc.collect();torch.cuda.empty_cache()
        cleanup(item,ownership,folder)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','worker']);parser.add_argument('--model');args=parser.parse_args()
    if args.command=='prepare':print(prepare());return
    if args.command=='worker':worker(args.model);return
    while not (destination()/'complete.json').exists():time.sleep(5)
    reg=read(dest()/'registration.json')
    for item in reg['models']:
        if (dest()/item['alias']/'cleanup_complete.json').exists():continue
        print('OWN_START',item['alias'],flush=True)
        with (dest()/(item['alias']+'.log')).open('a',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--model',item['alias']],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('OWN_COMPLETE',item['alias'],flush=True);print(json.dumps(sync(reg['config'])),flush=True)
    write(dest()/'complete.json',dict(status='complete',models=[m['alias'] for m in reg['models']]),immutable=True)


if __name__=='__main__':main()
