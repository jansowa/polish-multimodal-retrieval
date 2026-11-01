"""Resumable serial reference queue with private weight cleanup and Git publication.

Run exclusively in WSL. `prepare` freezes missing cells before inference;
`start` detaches a supervisor, `status` never loads a model.
"""
import argparse
import datetime as dt
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))
from retrieval_benchmark.core import HERE, identity, local_path, parameters, public_check, read, sha, write
from retrieval_benchmark.ingest import configuration, sync
from retrieval_benchmark.runner import DatasetEncoder, Originals, selected_datasets, tokenization

REPORT = ROOT/'reports/leaderboard_requested_20261101'
RUN = ROOT/'training/runs/f_series/leaderboard-requested-20261101'
REG = REPORT/'registration.json'
CACHE = Path.home()/'.cache/pl_mmb_requested_20261101/hub'
MODES = ['v2_no_instructions', 'author_recommended']
PROFILE = 'historical-native-v2'
STAGE = 'S04-LEADERBOARD'


def completed_cells(document, model_id):
    """Any completed numerical profile satisfies the owner's no-repeat request."""
    return {(r['dataset_id'], r['prompt_mode']) for r in document['results'] if r['model_id'] == model_id}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def event(kind, **fields):
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN/'events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(time=now(), event=kind, **fields))+'\n')
    print(kind, fields, flush=True)


def component(info, *, code_only=False):
    files = info['files']
    names = [f['name'] for f in files]
    if code_only:
        weights = []
    elif 'open_clip_model.safetensors' in names and info['repository'].startswith(('timm/', 'laion/')):
        weights = ['open_clip_model.safetensors']
    elif 'open_clip_pytorch_model.safetensors' in names and info['repository'].startswith('laion/'):
        weights = ['open_clip_pytorch_model.safetensors']
    elif 'model.safetensors' in names:
        weights = ['model.safetensors']
    else:
        weights = [n for n in names if n.startswith('model-') and n.endswith('.safetensors')]
        if not weights:
            weights = ['pytorch_model.bin'] if 'pytorch_model.bin' in names else []
    if not code_only and not weights:
        raise ValueError('No supported weights for '+info['repository'])
    selected = [f for f in files if f['name'] in weights]
    for f in selected:
        if not f.get('lfs') or len(f['lfs'].get('sha256', '')) != 64:
            raise ValueError('Missing registered LFS digest')
    small = [n for n in names if n.endswith(('.json', '.py', '.txt', '.model', '.tiktoken', '.gz', '.jinja'))
             and not n.startswith(('onnx/', 'openvino/', '.'))]
    return dict(repository=info['repository'], revision=info['revision'],
                download_files=sorted(set(small+weights)), weights=selected)


def prepare():
    if REG.exists():
        return read(REG)
    from retrieval_benchmark.inspect_requested import REPOSITORIES, inspect
    repos = [r.replace('/aclip-', '/clip-') for r in REPOSITORIES]
    inventory = {i['repository']: i for i in read(REPORT/'inventory.json')}
    # Nomic's published auto_map uses this separately versioned code repository.
    extra = inspect('nomic-ai/nomic-bert-2048')
    if 'error' in extra:
        raise ValueError(extra)
    inventory[extra['repository']] = extra
    write(REPORT/'inventory.json', list(inventory.values()))
    config = configuration(HERE/'local/config.json')
    protocols = selected_datasets(config, list(config['datasets']))
    doc = public_check(read(HERE/'space/results.json'))
    source_index = read(HERE/'local/source_index.json')
    old = {v.get('repository'): (k, v) for k,v in doc['models'].items()}
    jobs = []
    for repo in repos:
        info = inventory[repo]
        if 'error' in info:
            raise ValueError(info)
        mid = 'hf:'+repo+'@'+info['revision']
        known = old.get(repo)
        if known and known[0] != mid:
            raise ValueError('Existing reference revision differs: '+repo)
        complete = completed_cells(doc, mid)
        # Existing results in any numerical profile satisfy the owner's no-repeat rule.
        missing = [(name, mode) for name in protocols for mode in MODES if (name, mode) not in complete]
        specs = [component(info)]
        if known:
            backend = 'existing'
            alias = known[1]['label']
        else:
            alias = repo.split('/')[-1]
            backend = ('qwen3' if repo.startswith('Qwen/') else 'gme' if repo.startswith('Alibaba-NLP/')
                       else 'openclip' if repo.startswith('laion/') else 'nomic' if repo.startswith('nomic-ai/')
                       else 'bge' if repo.startswith('BAAI/') else 'siglip')
        if repo.startswith('M-CLIP/'):
            specs.append(component(inventory['timm/vit_base_patch16_plus_clip_240.laion400m_e31']))
        elif '/clip-ViT-B-32' in repo:
            specs.append(component(inventory['openai/clip-vit-base-patch32']))
        elif backend == 'nomic':
            specs.extend([component(inventory['nomic-ai/nomic-embed-text-v1.5']), component(extra, code_only=True)])
        old_sources = [s for s in source_index['artifacts'] if s['model_id'] == mid]
        for source in old_sources:
            if sha(local_path(source['path'])) != source['sha256']:
                raise ValueError('Existing score integrity failed: '+source['path'])
        jobs.append(dict(repository=repo, revision=info['revision'], alias=alias, model_id=mid,
                         backend=backend, components=specs, missing=missing,
                         existing_cells=sorted(complete), existing_sources=old_sources,
                         batch_size=1 if backend in ('qwen3', 'gme') else 64 if known else 16,
                         interpreter='pl-mmb-modern' if backend == 'qwen3' else
                                     'pl-mmb-benchmark' if repo.startswith('M-CLIP/') or backend in ('gme','bge','nomic') else
                                     'pl-mmb-legacy-cu130',
                         weight_bytes=sum(f['bytes'] for s in specs for f in s['weights']),
                         license=info.get('license')))
    jobs.sort(key=lambda j: j['weight_bytes'])
    frozen = dict(schema_version=1, created=now(), stage=STAGE,
                  authorization='Owner requested these references, corrected aclip to clip, authorized serial weight removal and GitHub push; never recompute completed cells.',
                  scope='Ordinary leaderboard reference evaluation, no F-series training or selection; frozen F protocol untouched.',
                  jobs=jobs, config=config, profile=PROFILE,
                  datasets={n:dict(protocol_sha256=config['datasets'][n]['protocol_sha256'],
                                  manifest_sha256=p['manifest_sha256']) for n,(p,m) in protocols.items()},
                  implementation={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), HERE/'requested_models.py',
                      ROOT/'src/pl_mmb_benchmark/models.py', ROOT/'src/pl_mmb_benchmark/full_gallery.py']},
                  protected={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/pl_mmb_training/holdouts.py',
                      HERE/'catalog.json', HERE/'catalog-current.json', ROOT/'protocols/protocol-v3-f.json']},
                  baseline_results_sha256=sha(HERE/'space/results.json'),
                  cleanup='Only weight blobs in this queue private HF cache; never remove shared caches, datasets or result files.')
    write(REG, frozen, immutable=True)
    return frozen


def verify(reg):
    for name,digest in {**reg['implementation'], **reg['protected']}.items():
        if sha(ROOT/name) != digest:
            raise ValueError('Registered/protected code changed: '+name)
    selected_datasets(reg['config'], list(reg['datasets']))


def folder_for(job):
    return RUN/job['repository'].replace('/', '--')


def clean(job):
    """Delete only the registered blobs in a dedicated queue-owned cache."""
    if CACHE.resolve() != (Path.home()/'.cache/pl_mmb_requested_20261101/hub').resolve():
        raise ValueError('Unexpected private cache root')
    removed = []
    for spec in job['components']:
        repo = CACHE/('models--'+spec['repository'].replace('/', '--'))
        for item in spec['weights']:
            digest = item['lfs']['sha256']
            for suffix in ('', '.incomplete'):
                blob = repo/'blobs'/(digest+suffix)
                if not blob.exists():
                    continue
                target = blob.resolve()
                if not target.is_relative_to(CACHE.resolve()) or target.parent != (repo/'blobs').resolve():
                    raise ValueError('Weight target escapes owned cache')
                if not suffix and (target.stat().st_size != item['bytes'] or sha(target) != digest):
                    raise ValueError('Weight SHA/size mismatch before deletion')
                removed.append(dict(path=str(target), bytes=target.stat().st_size, sha256=digest, partial=bool(suffix)))
                target.unlink()
    write(folder_for(job)/'cleanup.json', dict(time=now(), removed=removed, shared_cache_untouched=True))
    event('WEIGHTS_REMOVED', model=job['alias'], bytes=sum(i['bytes'] for i in removed))


def worker(index):
    reg = read(REG)
    job = reg['jobs'][index]
    verify(reg)
    os.environ['HF_HUB_CACHE'] = str(CACHE)
    os.environ['HUGGINGFACE_HUB_CACHE'] = str(CACHE)
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['REQUESTS_CA_BUNDLE'] = str(ROOT/'combined-ca.pem')
    os.environ['SSL_CERT_FILE'] = str(ROOT/'combined-ca.pem')
    from experiments.data_ood_common_20261030.common import gpu_lock
    from retrieval_benchmark.requested_models import RequestedEncoder
    from pl_mmb_benchmark.efficiency import parameter_counts
    from pl_mmb_benchmark.full_gallery import evaluate_encoder
    with gpu_lock():
        verify(reg)
        current = public_check(read(HERE/'space/results.json'))
        done = completed_cells(current, job['model_id'])
        missing = [(n,m) for n,m in job['missing'] if (n,m) not in done]
        if not missing:
            write(folder_for(job)/'complete.json', dict(status='complete', no_inference=True, time=now()))
            return
        # The WSL virtual disk's host volume must also retain physical free space.
        for path in (Path.home(), Path('/mnt/c'), Path('/mnt/d')):
            # D stores only code/small logs. Weights and large scores live on C.
            required = 128*2**20 if path == Path('/mnt/d') else 4*2**30+job['weight_bytes']
            if shutil.disk_usage(path).free < required:
                raise RuntimeError('Insufficient disk reserve for '+str(path))
        event('LOADING', model=job['alias'], missing=len(missing))
        encoder = RequestedEncoder(job, CACHE)
        counts = parameters(parameter_counts(encoder))
        protocols = selected_datasets(reg['config'], list(dict.fromkeys(n for n,m in missing)))
        metadata = dict(label=job['alias'], repository=job['repository'], revision=job['revision'],
                        parameters=counts, role='reference', representative=True, efficiency=None,
                        license=dict(status='reported', spdx=job['license'], display=job['license'] or 'Brak deklaracji',
                                     url='https://huggingface.co/'+job['repository']+'/blob/'+job['revision']+'/README.md'))
        if job['model_id'] in current['models']:
            metadata = current['models'][job['model_id']]
            if counts != metadata['parameters']:
                raise ValueError('Existing parameter count mismatch')
        artifacts = []
        for name,(protocol,manifest) in protocols.items():
            cached = DatasetEncoder(encoder, Originals(manifest, reg['config']))
            for dataset,mode in missing:
                if dataset != name:
                    continue
                encoder.mode = mode
                # Different native instructions have distinct RAM caches.
                if encoder.kind in ('qwen3','gme','nomic'):
                    cached.cache.clear()
                run_id = identity(dict(registration=sha(REG), model=job['model_id'], dataset=name, mode=mode))
                folder = local_path(reg['config']['output'])/run_id
                path = folder/'result.json'
                if path.exists():
                    result = read(path)
                    if sha(folder/'scores.json') != result['public_result']['source_sha256']:
                        raise ValueError('Completed scores changed')
                    event('SKIP_COMPLETED', model=job['alias'], dataset=name, mode=mode)
                else:
                    scores_path = folder/'scores.json'
                    if scores_path.exists():
                        result = read(scores_path)
                        if result['registration_sha256'] != sha(REG) or result['manifest_sha256'] != protocol['manifest_sha256']:
                            raise ValueError('Interrupted score identity changed')
                        tok = result['tokenization']
                    else:
                        result = evaluate_encoder(cached, dict(protocol,prompt_mode=mode), dict(status='closed_ready_for_execution'))
                        if encoder.kind in ('qwen3','gme','nomic'):
                            tok = dict(status='unavailable', reason='Native chat/prefix tokenization; raw-text lengths would omit template tokens',
                                       context_limit=encoder.text_max_length)
                        else:
                            try:
                                tok = tokenization(encoder, protocol['texts'], job['alias'], mode)
                            except (TypeError, AttributeError, KeyError):
                                tok = dict(status='unavailable', reason='Native tokenizer does not expose untruncated lengths')
                        result.update(model=job['alias'], revision=job['revision'], parameters=counts,
                                      precision=encoder.precision, components=encoder.components,
                                      registration_sha256=sha(REG), tokenization=tok, actual_batch_size=encoder.batch_size)
                        write(scores_path, result, immutable=True)
                    row = dict(model_id=job['model_id'], dataset_id=name, prompt_mode=mode, profile=PROFILE,
                               protocol=result['protocol'], metrics={d:result['directions'][d]['metrics'] for d in ('t2i','i2t')},
                               query_counts={d:len(result['directions'][d]['query_ids']) for d in ('t2i','i2t')},
                               precision=encoder.precision, exposure='unknown', source='requested-reference-20261101',
                               source_sha256=sha(folder/'scores.json'), registration_sha256=sha(REG), tokenization=tok,
                               **{k:result[k] for k in ('manifest_sha256','gallery_sha256','text_gallery_sha256','qrels_sha256')})
                    write(path, dict(status='complete', model_id=job['model_id'], model_metadata=metadata,
                                     public_result=row, execution_sha256=sha(REG)), immutable=True)
                    event('RESULT', model=job['alias'], dataset=name, mode=mode)
                    write(folder_for(job)/'smoke.json', dict(status='passed', first_result=str(path), sha256=sha(path), time=now()))
                artifacts.append(dict(path=str(path), sha256=sha(path)))
            del cached
        write(folder_for(job)/'complete.json', dict(status='complete', time=now(), artifacts=artifacts))


def publish(config):
    # Append only this queue's verified artifacts. Preserve the existing export.
    doc = public_check(read(HERE/'space/results.json'))
    keys = {(r['model_id'], r['dataset_id'], r['prompt_mode'], r['profile']):r for r in doc['results']}
    additions = 0
    for job in read(REG)['jobs']:
        for path in local_path(config['output']).glob('*/result.json'):
            run = read(path)
            row = run.get('public_result', {})
            if run.get('status') != 'complete' or row.get('registration_sha256') != sha(REG) or run['model_id'] != job['model_id']:
                continue
            if sha(path.parent/'scores.json') != row['source_sha256']:
                raise ValueError('Saved source SHA mismatch')
            key = (row['model_id'],row['dataset_id'],row['prompt_mode'],row['profile'])
            if key in keys:
                if keys[key] != row:
                    raise ValueError('Conflicting completed result')
                continue
            if run['model_id'] in doc['models'] and doc['models'][run['model_id']]['parameters'] != run['model_metadata']['parameters']:
                raise ValueError('Parameter counts changed')
            doc['models'].setdefault(run['model_id'], run['model_metadata'])
            doc['results'].append(row)
            keys[key] = row
            additions += 1
    public_check(doc)
    if additions:
        write(HERE/'space/results.json',doc)
        event('LEADERBOARD_UPDATED', additions=additions)
    write(RUN/'PUBLISH_REQUEST.json',dict(time=now(), results_sha256=sha(HERE/'space/results.json')))
    event('WINDOWS_GIT_PUBLICATION_REQUESTED')


def run():
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN/'queue.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reg = read(REG)
        failures = []
        for index,job in enumerate(reg['jobs']):
            folder = folder_for(job)
            folder.mkdir(parents=True, exist_ok=True)
            if not job['missing']:
                event('SKIP_MODEL_COMPLETE', model=job['alias'])
                continue
            if not (folder/'complete.json').exists():
                state = dict(status='running', model=job['alias'], index=index, time=now(), supervisor_pid=os.getpid())
                write(RUN/'STATE.json', state)
                python = Path.home()/'.venvs'/job['interpreter']/'bin/python'
                with (folder/'worker.log').open('a') as stream:
                    process = subprocess.Popen([str(python), '-u', str(Path(__file__).resolve()), 'worker', '--index', str(index)],
                                               cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                    state['worker_pid'] = process.pid
                    write(RUN/'STATE.json',state)
                    code = process.wait()
                if code:
                    failures.append(job['alias'])
                    write(folder/'failure.json', dict(exit_code=code, time=now(), log=str(folder/'worker.log')))
                    event('MODEL_FAILED_CONTINUING', model=job['alias'], code=code)
            # Subprocess has exited, so no loaded model can still use these private blobs.
            try:
                clean(job)
            except Exception:
                event('CLEANUP_FAILED', model=job['alias'], error=traceback.format_exc())
                raise  # Never accumulate models after an unsuccessful cleanup.
            try:
                publish(reg['config'])
            except Exception:
                event('PUBLICATION_FAILED_RESULTS_RETAINED', error=traceback.format_exc())
        try:
            publish(reg['config'])
        except Exception:
            event('PUBLICATION_FAILED_RESULTS_RETAINED', error=traceback.format_exc())
        doc = read(HERE/'space/results.json')
        outstanding = {j['repository']: [(n,m) for n,m in j['missing'] if not any(
            r['model_id']==j['model_id'] and r['dataset_id']==n and r['prompt_mode']==m for r in doc['results'])]
            for j in reg['jobs']}
        outstanding = {k:v for k,v in outstanding.items() if v}
        write(RUN/'STATE.json', dict(status='partial' if outstanding else 'complete', failures=failures,
                                    outstanding=outstanding, time=now()))
        event('QUEUE_FINISHED', outstanding=list(outstanding))


def start():
    reg = read(REG)
    RUN.mkdir(parents=True, exist_ok=True)
    # Fail synchronously if an existing supervisor still owns this queue.
    with (RUN/'queue.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (RUN/'supervisor.log').open('a') as stream:
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'run'], cwd=ROOT,
                                   stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True)
    command = '~/.venvs/pl-mmb-benchmark/bin/python retrieval_benchmark/requested_queue.py start'
    ticket = dict(run_id=RUN.name, pid=process.pid, started=now(), start_command=command, resume_command=command,
                  estimated_end='Within 48 hours; 8B CPU offload may extend runtime; not a measured ETA.',
                  registration=str(REG.relative_to(ROOT)), registration_sha256=sha(REG),
                  state=str((RUN/'STATE.json').relative_to(ROOT)), log=str((RUN/'supervisor.log').relative_to(ROOT)),
                  output=reg['config']['output'], git_target='origin main',
                  user_authorized_weights_cleanup=True, simultaneous_models=1)
    write(RUN/'JOB.json',ticket)
    print(json.dumps(ticket, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','start','run','worker','status'])
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if args.command == 'prepare':
        reg = prepare()
        print(json.dumps([dict(model=j['alias'], existing=len(j['existing_cells']), missing=len(j['missing'])) for j in reg['jobs']], indent=2))
    elif args.command == 'worker':
        worker(args.index)
    elif args.command == 'start':
        start()
    elif args.command == 'run':
        run()
    else:
        print(json.dumps(read(RUN/'STATE.json') if (RUN/'STATE.json').exists() else {'status':'not_started'}, indent=2))
