"""Local one-model workers. The published Space never imports this module."""
import contextlib
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from .core import HERE, ROOT, CATALOG, MODES, PROTOCOL, identity, local_path, parameters, read, sha, write


def initialize(config):
    """Resolve local artifacts once; keep host paths outside the public Space."""
    registry = read(ROOT / 'reports/series_2026-10-23/reference_eval_registration_v3_20261029.json')
    models = {r['alias']: dict(repository=r['model'], revision=r['revision'],
                              role='english_control' if r.get('english_text_only_control') else 'reference')
              for r in registry['reference_models']}
    storage = read(ROOT / 'reports/series_2026-10-23/reference_storage_preflight_20261029.json')['models']
    for alias, model in models.items():
        aliases = {alias}
        if alias in ('clip-b32-pl', 'clip-b32-pl-32768'):
            aliases.add('shared-clip-image')
        if alias == 'mclip-b16plus':
            aliases.add('mclip-image-component')
        model['download_budget_bytes'] = sum(f['bytes'] for item in storage if item['alias'] in aliases for f in item['files'])
    for r in read(ROOT / 'reports/ood_eval_20261030/summary.json')['models']:
        models[r['id']] = dict(checkpoint=r['path'], revision=r['sha256'], role='diagnostic' if r['variant'] == 'R3' else 'project',
                              representative=r['representative'], trained_on_wiki=r['trained_on_Wiki'], seed=r['seed'])
        models[r['id']]['exposure'] = dict(flickr30k_pl_local_test='source-OOD to projector',
            vizwiz_pl_local_validation='source-OOD to projector', VN_dev='ID / development',
            wiki_native_dev='ID / development' if r['trained_on_Wiki'] else 'development / source-OOD',
            xtd10_full_1k='ID source COCO', xtd10_no_overlap='ID source COCO')
    config['models'] = models
    directory = 'reports/series_2026-10-23/'
    protocols = dict(wiki_native_dev=directory+'wiki_v2/retrieval_protocol.json',
                     VN_dev=directory+'vn_dev_protocol/retrieval_protocol.json',
                     xtd10_full_1k=directory+'xtd10_v2/full_retrieval_protocol.json',
                     xtd10_no_overlap=directory+'xtd10_v2/clean_retrieval_protocol.json')
    for key in ('flickr30k_pl_local_test', 'vizwiz_pl_local_validation'):
        protocols[key] = str(local_path(config['ood_frozen']) / key / 'retrieval_protocol.json')
    config['datasets'] = {}
    for key, value in protocols.items():
        path = local_path(value)
        protocol = read(path)
        candidates = [path.parent / name for name in ('manifest.json', 'manifest.jsonl', 'wiki_dev.jsonl', 'full_1k.jsonl', 'no_overlap.jsonl')]
        # Exact digest, not name, decides which manifest belongs to the protocol.
        manifest = next((p for p in candidates if p.is_file() and sha(p) == protocol['manifest_sha256']), None)
        config['datasets'][key] = dict(protocol=str(path), protocol_sha256=sha(path),
                                      image_manifest=str(manifest) if manifest else None)
    return config


def selected_datasets(config, names):
    catalog = read(CATALOG)['datasets']
    if names == ['core']:
        names = [k for k, v in catalog.items() if v['role'] == 'core' and v['status'] == 'ready']
    result = {}
    for name in dict.fromkeys(names):
        if name not in catalog or catalog[name]['status'] != 'ready':
            raise ValueError('Dataset is not released for execution: ' + name)
        item = config['datasets'][name]
        path = local_path(item['protocol'])
        if sha(path) != item['protocol_sha256']:
            raise ValueError('Frozen protocol changed: ' + name)
        protocol = read(path)
        if protocol.get('status') != 'frozen_image_verified' or protocol['id'] != name:
            raise ValueError('Not a frozen, image-verified protocol')
        if not item.get('image_manifest'):
            raise ValueError('Configure the original image_manifest for ' + name + ' before execution')
        manifest = local_path(item['image_manifest'])
        if sha(manifest) != protocol['manifest_sha256']:
            raise ValueError('Dataset manifest SHA mismatch')
        result[name] = (protocol, manifest)
    return result


def plan(config, model, names, modes, batch_size):
    if batch_size < 1 or not modes or any(m not in MODES for m in modes):
        raise ValueError('Invalid batch size / prompt modes')
    if model not in config.get('models', {}):
        raise ValueError('Add the model with an immutable revision to local config first')
    spec = config['models'][model]
    if not re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', spec.get('revision', '')):
        raise ValueError('Pinned 40/64-character revision required')
    if 'repository' in spec and (type(spec.get('download_budget_bytes')) is not int or spec['download_budget_bytes'] <= 0):
        raise ValueError('Set a positive download_budget_bytes for all required weights before running')
    if 'checkpoint' in spec and sha(local_path(spec['checkpoint'])) != spec['revision']:
        raise ValueError('Checkpoint SHA mismatch')
    datasets = selected_datasets(config, names)
    implementation = {name: sha(HERE / name) for name in ('runner.py', 'core.py')}
    implementation['encoder'] = sha(ROOT / 'src/pl_mmb_benchmark/models.py')
    implementation['projector'] = sha(ROOT / 'src/pl_mmb_benchmark/projector_adapter.py')
    implementation['scorer'] = sha(ROOT / 'src/pl_mmb_benchmark/full_gallery.py')
    return dict(schema_version=1, model=model, model_spec=spec, prompt_modes=modes, batch_size=batch_size,
                datasets={k: dict(protocol_sha256=config['datasets'][k]['protocol_sha256'],
                                  manifest_sha256=v[0]['manifest_sha256']) for k, v in datasets.items()},
                implementation=implementation, profile='runner-v1-' + identity(dict(code=implementation, batch_size=batch_size))[:12],
                protocol=PROTOCOL, gpu_lock='training/runs/tinyclip-visualnews-v1/worker.lock')


class Originals:
    def __init__(self, manifest, config):
        if manifest.suffix == '.jsonl':
            rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
        else:
            rows = read(manifest)
        self.records = {r['image']: r for r in rows}
        self.config = config
        self.parquet = None

    @contextlib.contextmanager
    def paths(self, values):
        descriptors, paths = [], []
        try:
            for value in values:
                record = self.records[value]
                if value.startswith('parquet-original://flickr30k/test/'):
                    if self.parquet is None:
                        import pyarrow.parquet as pq
                        self.parquet = pq.read_table(local_path(self.config['flickr_parquet']), columns=['image']).column('image').to_pylist()
                    raw = self.parquet[int(value.rsplit('/', 1)[1])]['bytes']
                    if hashlib.sha256(raw).hexdigest() != record['sha256']:
                        raise ValueError('Embedded image changed')
                    fd = os.memfd_create('retrieval-original', flags=0)
                    descriptors.append(fd)
                    with os.fdopen(os.dup(fd), 'wb') as stream:
                        stream.write(raw)
                    paths.append('/proc/self/fd/' + str(fd))
                else:
                    path = local_path(value)
                    if sha(path) != record['sha256']:
                        raise ValueError('Original image SHA changed: ' + value)
                    paths.append(str(path))
            yield paths
        finally:
            for fd in descriptors:
                os.close(fd)


class DatasetEncoder:
    """RAM feature reuse; raw HF adapter caching is disabled for temporary image FDs."""
    def __init__(self, encoder, originals):
        self.encoder, self.originals, self.cache = encoder, originals, {}

    def encode(self, values, images=False):
        import torch
        key = (images, tuple(values))
        if key not in self.cache:
            chunks = []
            for start in range(0, len(values), self.encoder.batch_size):
                batch = values[start:start + self.encoder.batch_size]
                if images:
                    with self.originals.paths(batch) as paths:
                        chunks.append(self.encoder.encode(paths, images=True))
                else:
                    chunks.append(self.encoder.encode(batch))
            self.cache[key] = torch.cat(chunks).float().cpu()
        return self.cache[key]


def tokenization(encoder, texts, alias, mode):
    tokenizer = (getattr(getattr(encoder, 'text', None), 'tokenizer', None)
                 or getattr(getattr(encoder, 'model', None), 'tokenizer', None)
                 or getattr(getattr(encoder, 'processor', None), 'tokenizer', None)
                 or getattr(encoder, 'tokenizer', None))
    if tokenizer is None:
        return dict(status='unavailable', reason='Native backend tokenizer does not expose length reporting')
    # OpenCLIP HFTokenizer exposes the untruncated Hugging Face tokenizer here;
    # its own __call__ accepts only texts/context_length and truncates eagerly.
    wrapped = getattr(tokenizer, 'tokenizer', None)
    if wrapped is not None and callable(wrapped):
        tokenizer = wrapped
    prefix = '[query]: ' if hasattr(encoder, 'projector') else ''
    prefixes = dict(query=prefix, document=prefix)
    if alias == 'embeddinggemma-2' and mode == 'author_recommended':
        prefixes = dict(query='task: search result | query: ', document='title: none | text: ')
    report = {}
    for kind, prefix in prefixes.items():
        lengths = []
        for start in range(0, len(texts), 128):
            value = tokenizer([prefix + t for t in texts[start:start+128]], truncation=False, padding=False, add_special_tokens=True)
            lengths.extend(map(len, value['input_ids']))
        limit = encoder.text_max_length
        report[kind] = dict(context_limit=limit, maximum=max(lengths), mean=sum(lengths)/len(lengths),
                            truncation_fraction=sum(n > limit for n in lengths)/len(lengths), prefix=prefix)
    return report


def run_worker(config, execution):
    if sys.platform != 'linux':
        raise RuntimeError('GPU workers require Linux/WSL; sync, validation and the Space work on Windows')
    import fcntl
    sys.path.insert(0, str(ROOT / 'src'))
    from pl_mmb_benchmark.models import Encoder, REVISIONS, check_cache, model_batch_size
    from pl_mmb_benchmark.projector_adapter import ProjectorEncoder, AuthorPromptEncoder
    from pl_mmb_benchmark.full_gallery import evaluate_encoder
    from pl_mmb_benchmark.efficiency import parameter_counts
    import torch
    spec, alias = execution['model_spec'], execution['model']
    output = local_path(config['output'])
    for root in (Path('/mnt/c'), Path('/mnt/d'), Path.home()):
        required = 4 * 2**30 + (spec.get('download_budget_bytes', 0) + 256 * 2**20 if root != Path('/mnt/d') else 0)
        if root.exists() and shutil.disk_usage(root).free < required:
            raise RuntimeError('Less than 4 GiB physical reserve: ' + str(root))
    output.mkdir(parents=True, exist_ok=True)
    lock = ROOT / execution['gpu_lock']
    with lock.open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_cache()
        if (ROOT / 'combined-ca.pem').is_file():
            os.environ.setdefault('REQUESTS_CA_BUNDLE', str(ROOT / 'combined-ca.pem'))
            os.environ.setdefault('SSL_CERT_FILE', str(ROOT / 'combined-ca.pem'))
        batch = execution['batch_size']
        if 'checkpoint' in spec:
            encoder = ProjectorEncoder(local_path(spec['checkpoint']), batch_size=batch)
            model_id = 'checkpoint:' + spec['revision']
        else:
            repo = spec['repository']
            REVISIONS[repo] = spec['revision']
            class UncachedEncoder(Encoder):
                def _cache_path(self, values, modality):
                    return Path('/proc/self/__retrieval_cache_disabled__')
                def _save_cache(self, path, tensor):
                    pass
            encoder = UncachedEncoder(repo, model_batch_size(repo, batch))
            model_id = 'hf:' + repo + '@' + spec['revision']
        if encoder.revision != spec['revision']:
            raise ValueError('Loaded model revision mismatch')
        metadata = dict(label=alias, repository=spec.get('repository'), revision=spec['revision'],
                        parameters=parameters(parameter_counts(encoder)), role=spec.get('role', 'reference'),
                        representative=spec.get('representative', True), efficiency=None)
        datasets = selected_datasets(config, list(execution['datasets']))
        for name, (protocol, manifest) in datasets.items():
            cached = DatasetEncoder(encoder, Originals(manifest, config))
            for mode in execution['prompt_modes']:
                run_id = identity(dict(execution=execution, dataset=name, prompt_mode=mode))
                folder = output / run_id
                if (folder / 'result.json').exists():
                    existing = read(folder / 'result.json')
                    if existing.get('execution_sha256') != identity(execution) or sha(folder/'scores.json') != existing['public_result']['source_sha256']:
                        raise ValueError('Completed result integrity mismatch')
                    print('SKIP', alias, name, mode, flush=True)
                    continue
                write(folder / 'execution.json', execution, immutable=True)
                wrapped = AuthorPromptEncoder(cached, alias, mode)
                result = evaluate_encoder(wrapped, dict(protocol, prompt_mode=mode), dict(status='closed_ready_for_execution'))
                result.update(model=alias, revision=spec['revision'], precision=encoder.precision,
                              parameters=metadata['parameters'], tokenization=tokenization(encoder, protocol['texts'], alias, mode))
                write(folder / 'scores.json', result, immutable=True)
                row = dict(model_id=model_id, dataset_id=name, prompt_mode=mode, profile=execution['profile'],
                           protocol=PROTOCOL, metrics={d: result['directions'][d]['metrics'] for d in ('t2i', 'i2t')},
                           query_counts={d: len(result['directions'][d]['query_ids']) for d in ('t2i', 'i2t')},
                           precision=encoder.precision, exposure=spec.get('exposure', {}).get(name, 'unknown'),
                           source='local-runner-v1', source_sha256=sha(folder / 'scores.json'), tokenization=result['tokenization'],
                           **{k: result[k] for k in ('manifest_sha256', 'gallery_sha256', 'text_gallery_sha256', 'qrels_sha256')})
                write(folder / 'result.json', dict(status='complete', model_id=model_id, model_metadata=metadata,
                      execution_sha256=identity(execution), public_result=row), immutable=True)
                print('COMPLETE', alias, name, mode, flush=True)
