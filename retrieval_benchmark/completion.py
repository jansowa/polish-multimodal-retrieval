"""Registered, sequential gap filling below 500M V+T; existing artifacts stay immutable."""
import argparse
import gc
import json
import os
import shutil
import ssl
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrieval_benchmark.core import HERE, identity, local_path, parameters, public_check, read, sha, write
from retrieval_benchmark.ingest import configuration, sync
from retrieval_benchmark.runner import DatasetEncoder, Originals, selected_datasets, tokenization

LIMIT = 500_000_000
OLD = ['siglip2-pl', 'siglip2-pl-32768', 'siglip-i18n-256', 'siglip2-base', 'siglip2-b32-256']
NEW = {
    'metaclip-s16-32768': 'alphaedge-ai/metaclip-2-worldwide-s16-384-pol-32768',
    'metaclip-m16-32768': 'alphaedge-ai/metaclip-2-worldwide-m16-384-pol-32768',
    'metaclip-b16-32768': 'alphaedge-ai/metaclip-2-worldwide-b16-384-pol-32768',
    'metaclip-l14-32768': 'alphaedge-ai/metaclip-2-worldwide-l14-pol-32768',
}
PROTECTED = {'wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M', 'OPI-PIB/PolDense-32M'}


def destination():
    return local_path('C:/Users/jsowa/benchmark-translations/retrieval-benchmark-v1/completion-20261031')


def metadata(repo, revision=None):
    url = 'https://huggingface.co/api/models/' + repo + ('/revision/' + revision if revision else '') + '?blobs=true'
    context = ssl.create_default_context(cafile=str(ROOT/'combined-ca.pem'))
    with urlopen(url, context=context, timeout=60) as response:
        value = json.load(response)
    files = []
    for item in value['siblings']:
        name = item['rfilename']
        # Native AutoModel uses the safetensors checkpoint, not duplicate PyTorch weights.
        if name.endswith('.safetensors'):
            lfs = item.get('lfs') or {}
            digest = lfs.get('sha256') or lfs.get('oid', '').removeprefix('sha256:')
            if len(digest) != 64:
                raise ValueError('Missing weight LFS SHA256: ' + name)
            files.append(dict(name=name, bytes=item.get('size') or lfs['size'], sha256=digest))
    if not files:
        raise ValueError('No native safetensors checkpoint for ' + repo)
    if any(f['name']=='open_clip_model.safetensors' for f in files) and repo.startswith('timm/'):
        files=[f for f in files if f['name']=='open_clip_model.safetensors']
    count = value.get('safetensors', {}).get('total')
    if not isinstance(count, int):
        counts = value.get('safetensors', {}).get('parameters', {})
        count = sum(counts.values()) if counts else None
    return dict(repository=repo, revision=value['sha'], files=files, total_checkpoint_parameters=count,
                source_url='https://huggingface.co/' + repo + '/tree/' + value['sha'])


def prepare():
    dest = destination()
    target = dest/'registration.json'
    if target.exists():
        return target
    config = configuration(HERE/'local/config.json')
    public = public_check(read(HERE/'space/results.json'))
    registry = read(HERE/'local/source_index.json')['artifacts']
    ready = list(config['datasets'])
    jobs = []
    for alias in OLD + list(NEW):
        spec = config['models'].get(alias)
        info = metadata(NEW[alias]) if alias in NEW else metadata(spec['repository'], spec['revision'])
        if spec and info['revision'] != spec['revision']:
            raise ValueError('Historical model revision changed')
        mid = 'hf:' + info['repository'] + '@' + info['revision']
        known = public['models'].get(mid, {}).get('parameters')
        info['certified_encoder_parameters'] = known['total'] if known else info['total_checkpoint_parameters']
        if info['certified_encoder_parameters'] is None or info['certified_encoder_parameters'] > LIMIT:
            raise ValueError('Metadata cannot certify <=500M: ' + alias + ': ' + str(info['total_checkpoint_parameters']))
        modes = ['v2_no_instructions', 'author_recommended']
        missing = [(name, mode) for name in ready for mode in modes if not any(
            r['model_id'] == mid and r['dataset_id'] == name and r['prompt_mode'] == mode and
            r['profile'] == 'historical-native-v2' for r in public['results'])]
        if not missing:
            continue
        parity = None
        if spec:
            source = next(r for r in registry if r['model_id'] == mid and r['dataset_id'] == 'wiki_native_dev'
                          and r['prompt_mode'] == 'v2_no_instructions' and r['profile'] == 'historical-native-v2')
            historical = read(local_path(source['path']))
            parity = dict(path=source['path'], sha256=source['sha256'], precision=historical['precision'])
        jobs.append(dict(alias=alias, model_id=mid, **info, missing=missing, parity=parity))
        print('REGISTER', alias, info['certified_encoder_parameters'], len(missing), flush=True)
    datasets = selected_datasets(config, ready)
    registration = dict(schema_version=1, status='closed_before_new_scores', max_encoder_parameters=LIMIT,
        user_authorization='Fill strongest project models and strongest parameter-class candidates <=500M; GPU authorized; serial reference weight removal authorized earlier.',
        selection='Before new scores: existing Pareto candidates plus larger-vocabulary Polish MetaCLIP S/M/B/L; not an exhaustive global best-model claim.',
        bins=['<50M', '50-100M', '100-200M', '200-300M', '300-400M', '400-500M'],
        already_complete=['metaclip-s16', 'metaclip-m16', 'embeddinggemma-2', 'R0_seed44', 'R1_seed44', 'R3_seed44'],
        excluded={'mclip-b16plus': '677760256 V+T >500M', 'MetaCLIP huge/giant': '>500M',
                  'SigLIP2 large PL': '>500M', 'BGE/TIPS/English-only controls': 'not strongest Polish candidates in historical comparisons'},
        project_reuse='Original A R1/R3 saved results; R0_seed44 four-gallery metrics exactly reproduced historical results; intrinsic query prefix identical in both modes.',
        jobs=jobs, config=config, datasets={k: dict(protocol_sha256=config['datasets'][k]['protocol_sha256'],
            manifest_sha256=v[0]['manifest_sha256']) for k,v in datasets.items()},
        scorer_sha256=sha(ROOT/'src/pl_mmb_benchmark/full_gallery.py'),
        encoder_sha256=sha(ROOT/'src/pl_mmb_benchmark/models.py'),
        worker_sha256=sha(__file__), same_algorithm_profile='historical-native-v2',
        parity_policy='Every existing model must reproduce all per-query metrics on Wiki before computing missing tasks.',
        simultaneous_models=1, batch_size=64, XM3600=False, DOCCI=False,
        cleanup='Only newly downloaded registered LFS weights, after all results, SHA+size+exclusive references; primary weights protected; no directories/datasets removed.')
    write(target, registration, immutable=True)
    return target


def parameter_guard(counts):
    p = parameters(counts)
    if p['total'] > LIMIT:
        raise ValueError('Loaded vision+text encoders exceed 500M')
    return p


def parity_check(actual, expected):
    for key in ('protocol','dataset_id','manifest_sha256','gallery_sha256','text_gallery_sha256','qrels_sha256','prompt_mode'):
        if actual[key] != expected[key]:
            raise ValueError('Parity identity mismatch: ' + key)
    for direction in ('t2i','i2t'):
        if actual['directions'][direction] != expected['directions'][direction]:
            raise ValueError('Historical per-query parity differs: ' + direction)


def cleanup(job, ownership, dest):
    hub = (Path.home()/'.cache/huggingface/hub').resolve()
    if job['repository'] in PROTECTED:
        raise ValueError('Primary weights protected')
    complete = read(dest/'complete.json')
    if complete['status'] != 'complete':
        raise ValueError('Cannot clean incomplete worker')
    for artifact in complete['artifacts']:
        if sha(local_path(artifact['path'])) != artifact['sha256']:
            raise ValueError('Result changed before cleanup')
    repo = hub/('models--'+job['repository'].replace('/','--'))
    allowed = repo/'snapshots'/job['revision']
    candidates = []
    seen = set()
    for item in ownership:
        if item['preexisting']:
            continue
        link = Path(item['snapshot_path'])
        target = link.resolve(strict=True)
        if target in seen:
            continue
        seen.add(target)
        if not target.is_relative_to(hub) or not (target.is_relative_to((repo/'blobs').resolve()) or target.is_relative_to((hub/'blobs').resolve())):
            raise ValueError('Weight escapes exact cache blob scope')
        if target.stat().st_size != item['bytes'] or sha(target) != item['sha256']:
            raise ValueError('Registered LFS SHA/size mismatch')
        refs = []
        for other in hub.glob('models--*'):
            for reference in (other/'snapshots').rglob('*'):
                if reference.is_symlink() and reference.resolve() == target:
                    if not reference.is_relative_to(allowed):
                        raise ValueError('Weight has another snapshot owner; retain it')
                    refs.append(str(reference))
        candidates.append(dict(path=str(target), sha256=item['sha256'], bytes=item['bytes'], references=refs))
    write(dest/'cleanup_plan.json', dict(files=candidates, no_directories=True, newly_downloaded_only=True), immutable=True)
    for item in candidates:
        target = Path(item['path'])
        if sha(target) != item['sha256']:
            raise ValueError('Weight changed after cleanup plan')
        target.unlink()
    write(dest/'cleanup_complete.json', dict(removed_weight_bytes=sum(f['bytes'] for f in candidates),
                                           primary_weights_untouched=True, no_datasets_removed=True), immutable=True)


def worker(alias, registration_path):
    import fcntl
    sys.path.insert(0, str(ROOT/'src'))
    reg = read(registration_path)
    if sha(__file__) != reg['worker_sha256'] or sha(ROOT/'src/pl_mmb_benchmark/models.py') != reg['encoder_sha256'] or sha(ROOT/'src/pl_mmb_benchmark/full_gallery.py') != reg['scorer_sha256']:
        raise ValueError('Registered implementation changed')
    job = next(j for j in reg['jobs'] if j['alias'] == alias)
    dest = destination()/alias
    if (dest/'cleanup_complete.json').exists():
        print('ALREADY_COMPLETE',alias,flush=True)
        return
    config = reg['config']
    dataset_names = list(dict.fromkeys(name for name,mode in job['missing']))
    if job['parity']:
        dataset_names = list(dict.fromkeys(['wiki_native_dev']+dataset_names))
    protocols = selected_datasets(config, dataset_names)
    lock = ROOT/'training/runs/tinyclip-visualnews-v1/worker.lock'
    with lock.open('a+') as handle:
        # Other authorized training sessions use the same lock; do not interrupt them.
        fcntl.flock(handle, fcntl.LOCK_EX)
        os.environ['REQUESTS_CA_BUNDLE'] = str(ROOT/'combined-ca.pem')
        os.environ['SSL_CERT_FILE'] = str(ROOT/'combined-ca.pem')
        from pl_mmb_benchmark.models import Encoder, REVISIONS, check_cache
        from pl_mmb_benchmark.efficiency import parameter_counts
        from pl_mmb_benchmark.projector_adapter import AuthorPromptEncoder
        from pl_mmb_benchmark.full_gallery import evaluate_encoder
        import torch
        torch.set_num_threads(2)
        check_cache()
        hub = Path.home()/'.cache/huggingface/hub'
        snapshot = hub/('models--'+job['repository'].replace('/','--'))/'snapshots'/job['revision']
        ownership_path = dest/'ownership.json'
        ownership = read(ownership_path) if ownership_path.exists() else [dict(**f, snapshot_path=str(snapshot/f['name']),
            preexisting=(snapshot/f['name']).is_file()) for f in job['files']]
        missing = sum(f['bytes'] for f in ownership if not Path(f['snapshot_path']).is_file())
        for disk, extra in [('/mnt/c',missing+256*2**20),('/mnt/d',0),(str(Path.home()),missing+256*2**20)]:
            if shutil.disk_usage(disk).free < 4*2**30+extra:
                raise RuntimeError('Insufficient physical reserve: '+disk)
        write(ownership_path,ownership,immutable=True)
        if (dest/'complete.json').exists():
            cleanup(job,ownership,dest)
            return
        REVISIONS[job['repository']] = job['revision']
        class Uncached(Encoder):
            def _cache_path(self,values,modality):return Path('/proc/self/__completion_cache_disabled__')
            def _save_cache(self,path,tensor):pass
        encoder = Uncached(job['repository'],64)
        counts = parameter_guard(parameter_counts(encoder))
        if alias in OLD and counts['total'] != job['certified_encoder_parameters']:
            raise ValueError('Historical encoder count mismatch')
        if encoder.revision != job['revision']:
            raise ValueError('Loaded wrong revision')
        if job['parity'] and encoder.precision != job['parity']['precision']:
            raise ValueError('Historical precision mismatch')
        for f in ownership:
            actual = Path(f['snapshot_path']).resolve(strict=True)
            if actual.stat().st_size != f['bytes'] or sha(actual) != f['sha256']:
                raise ValueError('Downloaded weight differs from registered HF LFS')
        metadata_model = dict(label=alias,repository=job['repository'],revision=job['revision'],parameters=counts,
                              role='reference',representative=True,efficiency=None)
        artifacts = []
        for name,(protocol,manifest) in protocols.items():
            features = DatasetEncoder(encoder,Originals(manifest,config))
            if name == 'wiki_native_dev' and job['parity']:
                previous = local_path(job['parity']['path'])
                if sha(previous) != job['parity']['sha256']:raise ValueError('Parity source changed')
                actual = evaluate_encoder(AuthorPromptEncoder(features,alias,'v2_no_instructions'),
                    dict(protocol,prompt_mode='v2_no_instructions'),dict(status='closed_ready_for_execution'))
                parity_check(actual,read(previous))
                write(dest/'parity.json',dict(status='exact_all_per_query_metrics',source_sha256=sha(previous),
                    scorer_sha256=reg['scorer_sha256'],precision=encoder.precision),immutable=True)
                print('PARITY_OK',alias,flush=True)
            for dataset,mode in job['missing']:
                if dataset != name:continue
                run_id = identity(dict(registration=sha(registration_path),model=alias,dataset=name,mode=mode))
                folder = local_path(config['output'])/run_id
                if (folder/'result.json').exists():
                    saved = read(folder/'result.json')
                    if sha(folder/'scores.json') != saved['public_result']['source_sha256']:raise ValueError('Completed score changed')
                else:
                    result = evaluate_encoder(AuthorPromptEncoder(features,alias,mode),dict(protocol,prompt_mode=mode),dict(status='closed_ready_for_execution'))
                    result.update(model=alias,revision=encoder.revision,precision=encoder.precision,parameters=counts,
                        registration_sha256=sha(registration_path),tokenization=tokenization(encoder,protocol['texts'],alias,mode))
                    write(folder/'scores.json',result,immutable=True)
                    row = dict(model_id=job['model_id'],dataset_id=name,prompt_mode=mode,profile='historical-native-v2',
                        protocol=result['protocol'],metrics={d:result['directions'][d]['metrics'] for d in ('t2i','i2t')},
                        query_counts={d:len(result['directions'][d]['query_ids']) for d in ('t2i','i2t')},
                        precision=encoder.precision,exposure='unknown',source='registered-completion-20261031',
                        source_sha256=sha(folder/'scores.json'),registration_sha256=sha(registration_path),
                        tokenization=result['tokenization'],
                        **{k:result[k] for k in ('manifest_sha256','gallery_sha256','text_gallery_sha256','qrels_sha256')})
                    write(folder/'result.json',dict(status='complete',model_id=job['model_id'],model_metadata=metadata_model,
                        public_result=row,execution_sha256=sha(registration_path)),immutable=True)
                artifacts.extend(dict(path=str(folder/f),sha256=sha(folder/f)) for f in ('scores.json','result.json'))
                print('RESULT',alias,name,mode,flush=True)
            del features
        write(dest/'complete.json',dict(status='complete',artifacts=artifacts,parameters=counts,
                                       registration_sha256=sha(registration_path)),immutable=True)
        del encoder
        gc.collect();torch.cuda.empty_cache()
        cleanup(job,ownership,dest)
        print('MODEL_COMPLETE_WEIGHTS_REMOVED',alias,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','worker'])
    parser.add_argument('--model')
    args=parser.parse_args()
    if args.command=='prepare':
        print(prepare());return
    registration=destination()/'registration.json'
    if args.command=='worker':worker(args.model,registration);return
    sys.path.insert(0,str(ROOT/'src'))
    from pl_mmb_benchmark.models import worker_python
    reg=read(registration)
    for job in reg['jobs']:
        alias=job['alias']
        if (destination()/alias/'cleanup_complete.json').exists():continue
        print('START',alias,flush=True)
        log=destination()/(alias+'.log')
        with log.open('a',encoding='utf-8') as stream:
            result=subprocess.run([worker_python(job['repository'],sys.executable),'-u',str(Path(__file__).resolve()),
                'worker','--model',alias],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            print('FAILED',alias,'see',log,flush=True)
            raise RuntimeError('Worker stopped; do not substitute model after scores')
        print('COMPLETE',alias,flush=True)
        print(json.dumps(sync(reg['config'])),flush=True)
    write(destination()/'complete.json',dict(status='complete',models=[j['alias'] for j in reg['jobs']],
          max_encoder_parameters=LIMIT,registration_sha256=sha(registration),XM3600_scored=False),immutable=True)
    print('QUEUE_COMPLETE',flush=True)


if __name__=='__main__':main()
