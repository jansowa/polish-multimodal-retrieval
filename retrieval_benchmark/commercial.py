"""Serial, preregistered native commercial-use references below 100M V+T."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from retrieval_benchmark.core import HERE, identity, local_path, parameters, read, sha, write
from retrieval_benchmark.ingest import configuration, sync
from retrieval_benchmark.runner import DatasetEncoder, Originals, selected_datasets, tokenization
from retrieval_benchmark.completion import cleanup, PROTECTED
from retrieval_benchmark.licenses import fetch

LIMIT = 100_000_000
CANDIDATES = [
    ('pe-core-s16','facebook/PE-Core-S16-384','pe',87187712),
    ('pe-core-t16','facebook/PE-Core-T16-384','pe',69545600),
    ('vitamin-s','jienengchen/ViTamin-S','vitamin',62469184),
    ('tinyclip-40m-native','wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M','tinyclip',84205568),
    ('tinyclip-39m-native','wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M','tinyclip',83101184),
    ('tinyclip-8m-native','wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M','tinyclip',23446016),
    ('uform3-english-small','unum-cloud/uform3-image-text-english-small','uform',None),
]
EXTENDED = [
    ('tinyclip-61m-native','wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M','tinyclip',115382144),
    ('vitamin-s-ltt','jienengchen/ViTamin-S-LTT','vitamin',145822912),
]


def destination(group='primary'):
    name='commercial-under100m-20261101' if group=='primary' else 'commercial-100to150m-20261101'
    return local_path('C:/Users/jsowa/benchmark-translations/retrieval-benchmark-v1/'+name)


def implementation():
    paths = ['retrieval_benchmark/commercial.py','retrieval_benchmark/commercial_models.py',
             'src/pl_mmb_benchmark/models.py','src/pl_mmb_benchmark/special_models.py',
             'src/pl_mmb_benchmark/full_gallery.py']
    return {p:sha(ROOT/p) for p in paths}


def prepare(group='primary'):
    target=destination(group)/'registration-v2.json'
    if target.exists():return target
    config=configuration(HERE/'local/config.json')
    context=ssl.create_default_context(cafile=str(ROOT/'combined-ca.pem'))
    jobs=[]
    for alias,repo,family,certified in (CANDIDATES if group=='primary' else EXTENDED):
        with urlopen('https://huggingface.co/api/models/'+repo+'?blobs=true',context=context,timeout=60) as response:
            info=json.load(response)
        revision=info['sha']
        _,license_info=fetch((repo,revision))
        if license_info['id'] not in ('mit','apache-2.0'):
            raise ValueError('Not an approved commercial-use license: '+repo)
        filenames=([repo.split('/')[-1]+'.pt'] if family=='pe' else
                   ['pytorch_model.bin'] if family=='vitamin' else
                   ['image_encoder.pt','text_encoder.pt'] if family=='uform' else ['model.safetensors'])
        files=[]
        for name in filenames:
            item=next(i for i in info['siblings'] if i['rfilename']==name)
            lfs=item['lfs'];digest=lfs.get('sha256') or lfs.get('oid','').removeprefix('sha256:')
            if len(digest)!=64:raise ValueError('Missing LFS digest')
            files.append(dict(name=name,bytes=item.get('size') or lfs['size'],sha256=digest))
        job=dict(alias=alias,repository=repo,revision=revision,family=family,files=files,
            model_id='hf:'+repo+'@'+revision,license=license_info,certified_encoder_parameters=certified,
            missing=[(name,mode) for name in config['datasets'] for mode in ('v2_no_instructions','author_recommended')])
        jobs.append(job)
        config['models'][alias]=dict(repository=repo,revision=revision,role='reference',representative=True,
            download_budget_bytes=sum(f['bytes'] for f in files),native_language='English',commercial_use_license=license_info['id'])
        print('REGISTER',alias,revision,license_info['display'],flush=True)
    protocols=selected_datasets(config,list(config['datasets']))
    limit=LIMIT if group=='primary' else 150_000_000
    reg=dict(schema_version=2,status='closed_before_new_scores',max_encoder_parameters_exclusive=limit,
        group=group,storage=str(destination(group)),source='commercial-'+group+'-20261101',
        selection='Published, jointly aligned native image/text checkpoints with MIT or Apache-2.0 weights; selected before scores; not an exhaustive SOTA certification.',
        excluded=({'TinyCLIP-61M-Text-29M':'Full encoders include token embeddings; actual V+T exceeds 100M; evaluated separately in the extended group.',
                  'ViTamin-S-LTT':'Large text tower makes full V+T exceed 100M.',
                  'Apple MobileCLIP/2':'Model weights license permits research use only; MIT code does not relicense weights.',
                  'Polish MetaCLIP S/M/B':'CC BY-NC 4.0 weights.',
                  'CLIP ViT-B / Polish alphaedge clip-b32':'Full V+T exceeds 100M.'} if group=='primary' else {}),
        native_language='English; evaluate original Polish benchmark queries directly, without translating queries.',
        config=config,jobs=jobs,implementation=implementation(),simultaneous_models=1,batch_size=64,
        datasets={k:dict(protocol_sha256=config['datasets'][k]['protocol_sha256'],manifest_sha256=v[0]['manifest_sha256']) for k,v in protocols.items()},
        parameter_policy='Count all actual V+T parameters including token embeddings and native projections; scalar logit/head reported separately. Abort at or above the registered group limit before scoring.',
        cleanup='Only newly downloaded registered weight blobs after complete scoring; existing primary TinyCLIP39 weights retained.')
    previous=destination(group)/'registration.json'
    if previous.exists():
        reg['amendment']=dict(parent_sha256=sha(previous),reason='Native PE tokenizer rejects HF length-reporting kwargs; report unavailable rather than abort. No score artifacts were saved by the failed first worker. Candidate selection and frozen datasets unchanged.')
    write(target,reg,immutable=True)
    write(HERE/'local/config.json',config)
    return target


def worker(alias,registration):
    import fcntl
    reg=read(registration)
    if reg['implementation']!=implementation():raise ValueError('Registered implementation changed')
    job=next(j for j in reg['jobs'] if j['alias']==alias)
    dest=local_path(reg['storage'])/alias
    if (dest/'cleanup_complete.json').exists():return
    protocols=selected_datasets(reg['config'],list(reg['datasets']))
    with (ROOT/'training/runs/tinyclip-visualnews-v1/worker.lock').open('a+') as lock:
        print('WAITING_FOR_GPU_LOCK',alias,flush=True)
        fcntl.flock(lock,fcntl.LOCK_EX)
        os.environ['REQUESTS_CA_BUNDLE']=os.environ['SSL_CERT_FILE']=str(ROOT/'combined-ca.pem')
        sys.path.insert(0,str(ROOT/'src'))
        from retrieval_benchmark.commercial_models import SmallEncoder
        from pl_mmb_benchmark.models import check_cache
        from pl_mmb_benchmark.efficiency import parameter_counts
        from pl_mmb_benchmark.projector_adapter import AuthorPromptEncoder
        from pl_mmb_benchmark.full_gallery import evaluate_encoder
        import torch
        torch.set_num_threads(2)
        check_cache()
        snapshot=Path.home()/'.cache/huggingface/hub'/('models--'+job['repository'].replace('/','--'))/'snapshots'/job['revision']
        path=dest/'ownership.json'
        ownership=read(path) if path.exists() else [dict(**f,snapshot_path=str(snapshot/f['name']),preexisting=(snapshot/f['name']).is_file()) for f in job['files']]
        missing=sum(f['bytes'] for f in ownership if not Path(f['snapshot_path']).is_file())
        for disk,extra in [('/mnt/c',missing+256*2**20),('/mnt/d',0),(str(Path.home()),missing+256*2**20)]:
            if shutil.disk_usage(disk).free<4*2**30+extra:raise RuntimeError('Insufficient disk reserve: '+disk)
        write(path,ownership,immutable=True)
        if not (dest/'complete.json').exists():
            encoder=SmallEncoder(job,reg['batch_size'])
            counts=parameters(parameter_counts(encoder))
            if counts['total']>=reg['max_encoder_parameters_exclusive']:raise ValueError('Actual V+T exceeds registered group limit: '+str(counts))
            if job['certified_encoder_parameters'] and counts['total']!=job['certified_encoder_parameters']:
                raise ValueError('Certified parameter count mismatch')
            for item in ownership:
                actual=Path(item['snapshot_path']).resolve(strict=True)
                if actual.stat().st_size!=item['bytes'] or sha(actual)!=item['sha256']:
                    raise ValueError('Downloaded weights differ from registered LFS')
            metadata=dict(label=alias,repository=job['repository'],revision=job['revision'],parameters=counts,
                role='reference',representative=True,efficiency=None,license=job['license'],
                native_language='English',components=encoder.components)
            print('PARAMETERS',alias,json.dumps(counts),flush=True)
            artifacts=[]
            for name,(protocol,manifest) in protocols.items():
                features=DatasetEncoder(encoder,Originals(manifest,reg['config']))
                for mode in ('v2_no_instructions','author_recommended'):
                    run_id=identity(dict(registration=sha(registration),model=alias,dataset=name,mode=mode))
                    folder=local_path(reg['config']['output'])/run_id
                    if (folder/'result.json').exists():
                        saved=read(folder/'result.json')
                        if sha(folder/'scores.json')!=saved['public_result']['source_sha256']:raise ValueError('Completed score changed')
                    else:
                        result=evaluate_encoder(AuthorPromptEncoder(features,alias,mode),dict(protocol,prompt_mode=mode),dict(status='closed_ready_for_execution'))
                        try:
                            lengths=tokenization(encoder,protocol['texts'],alias,mode)
                        except TypeError as error:
                            if 'unexpected keyword argument' not in str(error):raise
                            lengths=dict(status='unavailable',reason='Native tokenizer does not expose untruncated HF length reporting.')
                        result.update(model=alias,revision=encoder.revision,precision=encoder.precision,parameters=counts,
                            registration_sha256=sha(registration),tokenization=lengths)
                        write(folder/'scores.json',result,immutable=True)
                        row=dict(model_id=job['model_id'],dataset_id=name,prompt_mode=mode,profile='historical-native-v2',
                            protocol=result['protocol'],metrics={d:result['directions'][d]['metrics'] for d in ('t2i','i2t')},
                            query_counts={d:len(result['directions'][d]['query_ids']) for d in ('t2i','i2t')},
                            precision=encoder.precision,exposure='unknown',source=reg['source'],
                            source_sha256=sha(folder/'scores.json'),registration_sha256=sha(registration),tokenization=result['tokenization'],
                            **{k:result[k] for k in ('manifest_sha256','gallery_sha256','text_gallery_sha256','qrels_sha256')})
                        write(folder/'result.json',dict(status='complete',model_id=job['model_id'],model_metadata=metadata,
                            public_result=row,execution_sha256=sha(registration)),immutable=True)
                    artifacts.extend(dict(path=str(folder/f),sha256=sha(folder/f)) for f in ('scores.json','result.json'))
                    print('RESULT',alias,name,mode,flush=True)
                del features
            write(dest/'complete.json',dict(status='complete',artifacts=artifacts,parameters=counts,registration_sha256=sha(registration)),immutable=True)
            del encoder
            gc.collect();torch.cuda.empty_cache()
        if job['repository'] in PROTECTED:
            write(dest/'cleanup_complete.json',dict(removed_weight_bytes=0,primary_weights_untouched=True,no_datasets_removed=True),immutable=True)
        else:cleanup(job,ownership,dest)
        print('MODEL_COMPLETE',alias,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','worker'])
    parser.add_argument('--model')
    parser.add_argument('--group',choices=['primary','extended'],default='primary')
    args=parser.parse_args()
    if args.command=='prepare':print(prepare(args.group));return
    registration=destination(args.group)/'registration-v2.json'
    if args.command=='worker':worker(args.model,registration);return
    reg=read(registration)
    if args.group=='extended':
        print('WAITING_FOR_PRIMARY_GROUP',flush=True)
        while not (destination()/'complete.json').exists():time.sleep(15)
    for job in reg['jobs']:
        if (destination(args.group)/job['alias']/'cleanup_complete.json').exists():continue
        interpreter='/home/jsowa/.venvs/pl-mmb-legacy-cu130/bin/python' if job['family']=='vitamin' else '/home/jsowa/.venvs/pl-mmb-benchmark/bin/python'
        print('START',job['alias'],flush=True)
        with (destination(args.group)/(job['alias']+'.log')).open('a',encoding='utf-8') as log:
            result=subprocess.run([interpreter,'-u',str(Path(__file__).resolve()),'worker','--model',job['alias'],'--group',args.group],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Worker failed: '+job['alias'])
        print(json.dumps(sync(configuration(HERE/'local/config.json'))),flush=True)
    write(destination(args.group)/'complete.json',dict(status='complete',models=[j['alias'] for j in reg['jobs']],registration_sha256=sha(registration)),immutable=True)
    print('QUEUE_COMPLETE',flush=True)


if __name__=='__main__':main()
