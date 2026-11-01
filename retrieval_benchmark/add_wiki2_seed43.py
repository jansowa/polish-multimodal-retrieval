"""Add the exact user-named Wiki2 R3/seed43 checkpoint, retaining old results."""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from retrieval_benchmark.core import HERE,identity,local_path,parameters,read,sha,write
from retrieval_benchmark.ingest import configuration,sync
from retrieval_benchmark.runner import plan,run_worker

ALIAS='R3_wiki2_seed43'
CHECKPOINT='C:/Users/jsowa/benchmark-translations/data-ood-20261030/A/wiki-recovery-20261031/runs/R3/seed-43/best.pt'
EXPECTED='094122a574d8f2322b1434bb0eb0a5d539b510c36b5cfc74c2eb140c0e1f7636'
MODES=['v2_no_instructions','author_recommended']
DEST=local_path('C:/Users/jsowa/benchmark-translations/retrieval-benchmark-v1/wiki2-r3-seed43-20261031')


def main():
    if sha(local_path(CHECKPOINT))!=EXPECTED:raise ValueError('The user-named checkpoint changed')
    config=configuration(HERE/'local/config.json')
    final_output=local_path(config['output'])
    spec=dict(checkpoint=CHECKPOINT,revision=EXPECTED,role='project',representative=True,seed=43,
        generation=2,trained_on_wiki=True,
        exposure=dict(VN_dev='ID / checkpoint selection',wiki_native_dev='ID / Wiki variant selection',
            flickr30k_pl_local_test='source-OOD to projector; pretrained encoder exposure unknown',
            vizwiz_pl_local_validation='source-OOD to projector; pretrained encoder exposure unknown',
            xm3600_pl='held-out Open Images; pretrained encoder exposure unknown',
            xtd10_full_1k='ID source COCO; known historical overlap',xtd10_no_overlap='ID source COCO'))
    if ALIAS in config['models'] and config['models'][ALIAS]!=spec:raise ValueError('Alias already refers to another model')
    config['models'][ALIAS]=spec
    write(HERE/'local/config.json',config)
    study=local_path(CHECKPOINT).parents[3]
    originals={name:study/'evaluation/R3/seed-43'/(name+'.json') for name in ('VN_dev','wiki_native_dev','xtd10_full_1k','xtd10_no_overlap')}
    for name,path in originals.items():
        result=read(path)
        if result['checkpoint_sha256']!=EXPECTED or result['dataset_id']!=name:raise ValueError('Original score checkpoint mismatch')
        protocol=read(local_path(config['datasets'][name]['protocol']))
        if result['manifest_sha256']!=protocol['manifest_sha256']:raise ValueError('Original score gallery mismatch')
    execution=plan(config,ALIAS,['wiki_native_dev','flickr30k_pl_local_test','vizwiz_pl_local_validation','xm3600_pl'],MODES,64)
    execution['profile']='historical-native-v2'
    registration=dict(status='registered_before_missing_scores',checkpoint_sha256=EXPECTED,alias=ALIAS,
        user_checkpoint=CHECKPOINT,execution=execution,
        reused_scores={name:dict(path=str(path),sha256=sha(path)) for name,path in originals.items()},
        parity='Require exact original Wiki per-query metrics before publishing any result',
        prompt_equivalence='Frozen PolDense intrinsic query prefix identical in both modes; same ProjectorEncoder as existing project models')
    write(DEST/'registration.json',registration,immutable=True)
    config['output']=str(DEST/'staging')
    while True:
        try:run_worker(config,execution);break
        except BlockingIOError:
            print('WAITING_FOR_SHARED_GPU_LOCK',flush=True);time.sleep(5)
    folders={}
    for name in execution['datasets']:
        for mode in MODES:
            run_id=identity(dict(execution=execution,dataset=name,prompt_mode=mode))
            folders[(name,mode)]=local_path(config['output'])/run_id
    wiki=read(folders[('wiki_native_dev','v2_no_instructions')]/'scores.json')
    previous=read(originals['wiki_native_dev'])
    for key in ('protocol','dataset_id','manifest_sha256','gallery_sha256','text_gallery_sha256','qrels_sha256','prompt_mode','directions'):
        if wiki[key]!=previous[key]:raise ValueError('Adapter differs from original training evaluation: '+key)
    witness=read(folders[('wiki_native_dev','v2_no_instructions')]/'result.json')
    model=dict(witness['model_metadata'],seed=43,generation=2,trained_on_wiki=True,trained_on_coco=True,trained_on_visualnews=True)
    artifacts=[]
    def publish(folder,target):
        for name in ('scores.json','result.json'):
            source=folder/name;destination=target/name
            payload=source.read_bytes()
            if destination.exists():
                if destination.read_bytes()!=payload:raise ValueError('Existing result differs')
            else:
                destination.parent.mkdir(parents=True,exist_ok=True)
                with destination.open('xb') as stream:stream.write(payload)
            artifacts.append(dict(path=str(destination),sha256=sha(destination)))
    for (name,mode),folder in folders.items():
        result=read(folder/'result.json')
        if sha(folder/'scores.json')!=result['public_result']['source_sha256']:raise ValueError('New score SHA mismatch')
        result['model_metadata']=model
        # Staging wrappers may be enriched before their first public admission.
        write(folder/'result.json',result)
        publish(folder,final_output/folder.name)
    for name in ('VN_dev','xtd10_full_1k','xtd10_no_overlap'):
        source=originals[name];result=read(source)
        if sha(source)!=registration['reused_scores'][name]['sha256']:raise ValueError('Original score changed')
        for mode in MODES:
            key=identity(dict(registration=sha(DEST/'registration.json'),dataset=name,mode=mode))
            folder=DEST/'reused'/key
            score=folder/'scores.json';score.parent.mkdir(parents=True,exist_ok=True)
            if score.exists():
                if sha(score)!=sha(source):raise ValueError('Reused score changed')
            else:
                with score.open('xb') as stream:stream.write(source.read_bytes())
            row=dict(model_id='checkpoint:'+EXPECTED,dataset_id=name,prompt_mode=mode,profile='historical-native-v2',
                protocol=result['protocol'],metrics={d:result['directions'][d]['metrics'] for d in ('t2i','i2t')},
                query_counts={d:len(result['directions'][d]['query_ids']) for d in ('t2i','i2t')},
                precision=witness['public_result']['precision'],exposure=spec['exposure'][name],source='Wiki2-original-evaluation',
                source_sha256=sha(score),original_score_sha256=sha(source),registration_sha256=sha(DEST/'registration.json'),
                **{k:result[k] for k in ('manifest_sha256','gallery_sha256','text_gallery_sha256','qrels_sha256')})
            write(folder/'result.json',dict(status='complete',model_id='checkpoint:'+EXPECTED,model_metadata=model,
                public_result=row,execution_sha256=sha(DEST/'registration.json')),immutable=True)
            publish(folder,final_output/key)
    write(DEST/'complete.json',dict(status='complete',checkpoint_sha256=EXPECTED,results=14,
        adapter_parity='exact_original_Wiki_all_per_query_metrics',parameters=model['parameters'],artifacts=artifacts),immutable=True)
    print(json.dumps(sync(configuration(HERE/'local/config.json'))),flush=True)
    print('CHECKPOINT_ADDED',ALIAS,EXPECTED,flush=True)


if __name__=='__main__':main()
