"""Bounded, CPU-only session B preparation. No torch, image or weight downloads."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import ssl
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pl_mmb_training.series_20261023 import local_path
from pl_mmb_benchmark.report_parameters import encoder_counts

COORD = ROOT / 'reports/data_ood_coordination_20261030'
OUT = ROOT / 'reports/ood_eval_20261030'
RESERVE = 4 * 2**30
LIMIT = 512 * 1024
ROLES = {
    'COMMON_READY': {'common_helper', 'helper_usage', 'helper_verification'},
    'A_UNIVERSE_FROZEN': {'training_universe', 'protected_index', 'image_audit', 'study_registration'},
    'A_FINALISTS_FROZEN': {'training_report', 'selection_decision'},
    'A_GPU_DONE': set(),
    'B_EVAL_FROZEN': {'evaluation_registration'},
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def volumes():
    return {drive: shutil.disk_usage(local_path(drive)).free for drive in ('C:/', 'D:/')}


def write(path, value):
    """Only small B metadata, atomically replaced; never a readiness publisher."""
    path = Path(path)
    allowed = (OUT.resolve(), (COORD / 'B_STATUS.json').resolve())
    resolved = path.resolve()
    if not (resolved.is_relative_to(allowed[0]) or resolved == allowed[1]):
        raise ValueError('Not a B-owned metadata path')
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    if len(payload) > 4 * 1024**2:
        raise ValueError('Small-metadata budget exceeded')
    free = volumes()
    if min(free.values()) < RESERVE or free['D:/'] - 2 * len(payload) < RESERVE:
        raise RuntimeError('Physical volume reserve below 4 GiB; no write')
    if (COORD / 'COMMON_READY.json').exists():
        common = load_common()
        common.reserve('B', 'metadata', {'D': 2 * len(payload)},
                       committed_bytes=committed_bytes(), temporary_bytes=len(payload))
        common.atomic_json(path, value)
        common.reserve('B', 'metadata', {'D': 0}, committed_bytes=committed_bytes())
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def committed_bytes():
    roots = [OUT, ROOT / 'experiments/ood_eval_20261030',
             local_path('C:/Users/jsowa/benchmark-translations/data-ood-20261030/B/')]
    return sum(p.stat().st_size for root in roots if root.exists() for p in root.rglob('*') if p.is_file()) + sum(
        p.stat().st_size for p in COORD.glob('B_*.json'))


def load_common():
    if os.name == 'nt':
        raise RuntimeError('COMMON_READY now exists: run B preparation in WSL to use the shared POSIX helper')
    marker = verify_marker('COMMON_READY')
    path = local_path(next(a['path'] for a in marker['artifacts'] if a['role'] == 'common_helper'))
    spec = importlib.util.spec_from_file_location('data_ood_shared_common', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_marker(COORD / 'COMMON_READY.json')
    return module


def verify_marker(name):
    """Hash all declared dependencies. Existence alone grants no permission."""
    path = COORD / (name + '.json')
    value = read(path)
    expected = dict(schema_version=1, study_id='data-ood-20261030', generation=1,
                    owner='B' if name.startswith('B_') else 'A', phase=name, status='complete',
                    contract_sha256=sha(COORD / 'COORDINATION.md'))
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise ValueError(f'{name}: invalid {key}')
    artifacts = value.get('artifacts')
    if not isinstance(artifacts, list):
        raise ValueError(f'{name}: missing artifacts')
    roles = {a['role'] for a in artifacts}
    if len(roles) != len(artifacts) or not ROLES[name] <= roles:
        raise ValueError(f'{name}: missing or repeated artifact roles')
    if name == 'B_EVAL_FROZEN':
        manifest_ids = {r.split(':', 1)[1] for r in roles if r.startswith('manifest:')}
        protocol_ids = {r.split(':', 1)[1] for r in roles if r.startswith('protocol:')}
        if not manifest_ids or manifest_ids != protocol_ids:
            raise ValueError('B_EVAL_FROZEN: manifest/protocol roles do not match')
    dependency_artifacts = artifacts + value.get('normalized_text_guard_artifacts', [])
    for artifact in dependency_artifacts:
        raw = artifact['path']
        # Contract paths are absolute Windows paths, even when consumed in WSL.
        if len(raw) < 4 or raw[1:3] != ':/' or not raw[0].isalpha():
            raise ValueError(f'{name}: artifact path is not absolute Windows format')
        target = local_path(raw)
        if target.stat().st_size != artifact['bytes'] or sha(target) != artifact['sha256']:
            raise ValueError(f'{name}: artifact differs: {artifact["role"]}')
    return value


def marker_status():
    result = {}
    for name in ROLES:
        try:
            verify_marker(name)
            result[name] = dict(status='verified', sha256=sha(COORD / (name + '.json')))
        except (OSError, ValueError, KeyError, TypeError) as error:
            result[name] = dict(status='missing' if not (COORD / (name + '.json')).exists() else 'invalid',
                                reason=str(error))
    return result


def fetch(url):
    """Bound every response; snapshot metadata, never follow image URLs."""
    context = ssl.create_default_context(cafile=str(ROOT / 'combined-ca.pem'))
    request = urllib.request.Request(url, headers={'User-Agent': 'OOD-B-metadata-audit/1.0'})
    try:
        with urllib.request.urlopen(request, timeout=25, context=context) as response:
            content_type = response.headers.get('Content-Type', '')
            if not any(kind in content_type.lower() for kind in ('json', 'text/', 'xml')):
                raise ValueError(f'Not text metadata: {content_type}')
            payload = response.read(LIMIT + 1)
            if len(payload) > LIMIT:
                raise ValueError('Metadata response exceeds 512 KiB cap')
            return dict(url=url, resolved_url=response.geturl(), status='fetched', bytes=len(payload),
                        sha256=hashlib.sha256(payload).hexdigest(), content_type=content_type,
                        retrieved_at=datetime.now(timezone.utc).isoformat(), text=payload.decode('utf-8'))
    except (OSError, ValueError, urllib.error.URLError) as error:
        return dict(url=url, status='unavailable', reason=str(error))


def hf_metadata(repo):
    api = fetch('https://huggingface.co/api/datasets/' + repo)
    results = [api]
    if api['status'] == 'fetched':
        revision = json.loads(api['text'])['sha']
        results.extend(fetch(url) for url in (
            f'https://huggingface.co/datasets/{repo}/raw/{revision}/README.md',
            f'https://huggingface.co/api/datasets/{repo}/tree/{revision}',
            f'https://huggingface.co/api/datasets/{repo}/tree/{revision}/data',
        ))
    return results


def research():
    sources = {
        'flickr30k': lambda: hf_metadata('aplominski/flickr30k-multilingual'),
        'docci': lambda: hf_metadata('google/docci'),
        'vizwiz': lambda: [fetch('https://vizwiz.org/tasks-and-datasets/image-captioning/')],
        'docci_official': lambda: [fetch('https://google.github.io/docci/')],
        'aide': lambda: [fetch('https://zil.ipipan.waw.pl/Scwad/AIDe'), fetch('https://aclanthology.org/L18-1337/')],
        'flickr30k_original': lambda: [fetch('https://shannon.cs.illinois.edu/DenotationGraph/')],
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {key: pool.submit(task) for key, task in sources.items()}
        results = {key: future.result() for key, future in futures.items()}
    write(OUT / 'source_metadata.json', dict(workers=2, per_response_limit_bytes=LIMIT,
          original_images_downloaded=0, model_weights_downloaded=0, sources=results))
    return results


def register():
    original = ROOT / 'reports/series_2026-10-23/reference_eval_registration_v3_20261029.json'
    prior = read(original)
    aliases = ('metaclip-s16', 'metaclip-m16', 'embeddinggemma-2')
    models = [next(m for m in prior['reference_models'] if m['alias'] == alias) for alias in aliases]
    result = dict(schema_version=1, study_id='data-ood-20261030', generation=1, owner='B',
        status='preregistered_pending_data_and_A_not_executable',
        prior_registration=dict(path=original.as_posix(), sha256=sha(original)), reference_models=models,
        rationale='Two Polish references in nearby size classes and one larger quality reference; fixed before scores, avoid rerunning all 20.',
        our_models='All nine A-frozen checkpoints: three seeds each R0/R1/selected Wiki; three A-designated representatives, no OOD selection.',
        model_add_remove_after_scores=False, prompt_modes=['v2_no_instructions', 'author_recommended'],
        metrics=['recall@1', 'recall@5', 'recall@10', 'mrr', 'ndcg@10'], directions=['t2i', 'i2t'],
        bootstrap=dict(replicates=10000, seed=20261023, unit='transitive leakage component',
                       estimator='query weighted, fixed gallery, shared resamples', ci='percentile 2.5/97.5',
                       separate_from='mean and sample SD across three training seeds'),
        parameters=dict(encoders=encoder_counts(dict(vision=38587392, text=31883136)), head_separate=196992),
        selection=dict(target_images=1000, minimum_images=500, seed=20261023,
                       order='sha256 of seed and canonical component identity; whole components, report actual N',
                       use_all_existing_accepted_captions=True, never_select_by_model_scores=True),
        qrels='Original caption-image associations; canonical exact images; merge normalized identical EN/PL query positives; pHash never adds positives.',
        translation=dict(input='caption text only', device='cpu', torch_threads=2, workers=2,
                         new_manual_annotations=0, image_or_target_identifiers_in_prompt=False,
                         qa=['complete', 'language', 'repetitions', 'digits', 'negations', 'empty/error'],
                         pending='Pin available translator revision, parameters, QA thresholds and CPU feasibility before translating or freezing.'),
        storage=dict(B_persistent_limit_bytes=3*2**30, shared_temporary_limit_bytes=4*2**30,
                     physical_reserve_each_bytes=RESERVE, root='C:/Users/jsowa/benchmark-translations/data-ood-20261030/B/',
                     reservations='A common helper only after verified COMMON_READY', delete_datasets=False),
        forbidden=['XM3600 scores or probes', 'new manual labels', 'image caption generation', 'retrieval training', 'CUDA before handoffs and common lock'])
    write(OUT / 'evaluation_preregistration.json', result)
    return result


def status():
    markers = marker_status()
    waiting = [key for key, value in markers.items() if value['status'] != 'verified']
    now = datetime.now(timezone.utc).isoformat()
    value = dict(schema_version=1, study_id='data-ood-20261030', generation=1, owner='B', phase='CPU_PREPARATION',
        status='waiting_for_A' if any(key.startswith(('A_', 'COMMON')) for key in waiting) else 'preparing_data',
        contract_sha256=sha(COORD / 'COORDINATION.md'), updated_at=now,
        waiting_for=waiting, markers=markers, physical_free_bytes=volumes(),
        cpu_workers=2, torch_threads_limit=2, CUDA_operations=0, models_evaluated=0,
        manual_annotations_requested=0, large_downloads_started=(OUT / 'flickr30k_download.json').exists(),
        report_path=(OUT / 'REPORT.md').as_posix())
    write(COORD / 'B_STATUS.json', value)
    write(OUT / 'preparation_status.json', value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', action='store_true', help='Only registration and handoff inspection')
    args = parser.parse_args()
    register()
    if not args.offline:
        research()
    result = status()
    print(json.dumps(dict(status=result['status'], waiting_for=result['waiting_for'], physical_free_bytes=result['physical_free_bytes'])))


if __name__ == '__main__':
    main()
