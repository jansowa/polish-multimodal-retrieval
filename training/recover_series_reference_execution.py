"""Resume the registered queue without changing its scoring workers.

HF Hub's newer cache can store a repository's LFS blob in hub/blobs. Only
the completed model's exact, SHA-verified and otherwise unreferenced blob
can be removed there. The original per-repository cleanup stays in use.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import run_series_reference_evaluation as original
from pl_mmb_training.series_20261023 import ROOT, OUT, read, freeze, file_sha, require_space

REG, DEST = original.REG, original.DEST
PROTECTED = {'wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M', 'OPI-PIB/PolDense-32M'}


def exclusive_references(hub, target, allowed_repos):
    found = []
    for repo in hub.glob('models--*'):
        for link in (repo / 'snapshots').rglob('*'):
            if link.is_symlink() and link.resolve() == target:
                if repo.resolve() not in allowed_repos:
                    raise ValueError('Shared weight is referenced by another repository: ' + str(link))
                found.append(str(link))
    return sorted(found)


def cleanup_global(alias, registration):
    if not registration['cache_and_storage'].get('automatic_deletion'):
        raise ValueError('Cleanup is not authorized')
    complete = read(DEST / alias / 'complete.json')
    if complete['status'] != 'complete' or complete['registration_sha256'] != file_sha(REG):
        raise ValueError('Worker completion identity differs')
    for dataset in registration['datasets']:
        for mode in ('v2_no_instructions', 'author_recommended'):
            result = read(DEST / alias / dataset['name'] / (mode + '.json'))
            if result['registration_sha256'] != file_sha(REG):
                raise ValueError('Result identity differs')
    efficiency = read(DEST / alias / 'efficiency.json')
    if efficiency['registration_sha256'] != file_sha(REG):
        raise ValueError('Efficiency identity differs')
    hub = (Path.home() / '.cache/huggingface/hub').resolve()
    shared = (hub / 'blobs').resolve()
    if not shared.is_relative_to(hub):
        raise ValueError('Shared cache escapes the named HF root')
    items = original.model_weight_items(alias)
    if any(item['repo'] in PROTECTED for item in items):
        raise ValueError('Primary model is protected')
    repos = {(hub / ('models--' + item['repo'].replace('/', '--'))).resolve() for item in items}
    if any(not repo.is_relative_to(hub) for repo in repos):
        raise ValueError('Repository escapes the named cache')
    proof = DEST / alias / 'weight_cleanup_plan_v2_20261030.json'
    if proof.exists():
        plan = read(proof)
        if plan['registration_sha256'] != file_sha(REG) or plan['model_alias'] != alias:
            raise ValueError('Previously registered cleanup identity differs')
    else:
        files = []
        seen = set()
        for item in items:
            repo = (hub / ('models--' + item['repo'].replace('/', '--'))).resolve()
            for entry in item['files']:
                relative = Path(entry['name'])
                if relative.is_absolute() or '..' in relative.parts or relative.suffix not in ('.safetensors', '.pt', '.bin'):
                    raise ValueError('Not an authorized weight filename')
                link = repo / 'snapshots' / item['revision'] / relative
                if not link.is_file():
                    continue
                target = link.resolve(strict=True)
                if not target.is_relative_to(shared):
                    raise ValueError('Recovery accepts only the verified shared-cache case')
                if target in seen:
                    continue
                seen.add(target)
                if not link.is_symlink():
                    raise ValueError('Shared blob must have a repository snapshot link')
                first_hop = Path(os.path.normpath(str(link.parent / link.readlink())))
                expected = first_hop.name
                if first_hop.parent != repo / 'blobs' or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
                    raise ValueError('Snapshot does not identify a repository LFS SHA256 blob')
                digest = file_sha(target)
                if digest != expected or target.stat().st_size != entry['bytes']:
                    raise ValueError('LFS SHA256 or registered weight size differs')
                refs = exclusive_references(hub, target, repos)
                if str(link) not in refs:
                    raise ValueError('Snapshot reference disappeared')
                files.append(dict(path=str(target), snapshot_path=str(link), sha256=digest,
                    bytes=entry['bytes'], expected_lfs_sha256=expected, exclusive_snapshot_references=refs))
        if not files:
            raise ValueError('No recoverable shared weight files')
        plan = dict(status='registered_before_weight_file_removal', model_alias=alias,
            registration_sha256=file_sha(REG), files=files, repository_scope_verified=True,
            shared_cache_root=str(shared), no_foreign_snapshot_references=True,
            user_authorization=registration['cache_and_storage']['user_authorization'],
            recovery_decision_sha256=file_sha(OUT / 'reference_execution_recovery_decision_20261030.json'))
        freeze(proof, plan)
    for entry in plan['files']:
        target = Path(entry['path'])
        if target.resolve() != target or not target.is_relative_to(shared):
            raise ValueError('Registered deletion target changed or escapes shared cache')
        exclusive_references(hub, target, repos)
        if target.exists():
            if target.stat().st_size != entry['bytes'] or file_sha(target) != entry['sha256']:
                raise ValueError('Weight changed after registration')
            target.unlink()
    freeze(DEST / alias / 'weight_cleanup_complete.json', dict(status='complete',
        removed_weight_bytes=sum(p['bytes'] for p in plan['files']), plan_sha256=file_sha(proof),
        configs_and_embeddings_retained=True, primary_TinyCLIP_and_PolDense_untouched=True,
        physical_C_free_bytes_after=shutil.disk_usage('/mnt/c').free))
    print('REFERENCE_WEIGHTS_REMOVED ' + alias + ' checked shared-cache blob', flush=True)


def main():
    from pl_mmb_benchmark.models import worker_python
    registration = read(REG)
    previous = OUT / 'reference_execution_sequential_decision_20261029.json'
    old = read(previous)
    if old['worker_sha256'] != file_sha(original.__file__) or old['registration_sha256'] != file_sha(REG):
        raise ValueError('Original worker or registration changed')
    decision = dict(status='registered_before_resume', original_decision_sha256=file_sha(previous),
        registration_sha256=file_sha(REG), original_worker_sha256=file_sha(original.__file__),
        recovery_driver_sha256=file_sha(__file__), models=old['models'],
        amendment='Storage recovery only: SHA-checked HF shared blob without foreign repository references',
        scoring_and_timings_unchanged=True, completed_results_reused=True,
        simultaneous_reference_models=1, primary_weights_protected=True)
    freeze(OUT / 'reference_execution_recovery_decision_20261030.json', decision)
    require_space(['/mnt/c', '/mnt/d', '/'])
    for item in registration['reference_models']:
        alias = item['alias']
        if (DEST / alias / 'weight_cleanup_complete.json').exists():
            continue
        if not (DEST / alias / 'complete.json').exists():
            require_space(['/mnt/c', '/mnt/d', '/'])
            require_space(['/mnt/c'], expected_bytes=original.remaining_weights(alias) + 1024**3)
            print('REFERENCE_WORKER_START ' + alias, flush=True)
            result = subprocess.run([worker_python(item['model'], sys.executable), str(Path(original.__file__).resolve()), '--model', alias], cwd=ROOT)
            if result.returncode:
                raise RuntimeError('Worker failed; no model substitution: ' + alias)
        try:
            original.remove_completed_weights(alias, registration)
        except ValueError as error:
            if str(error) != 'Weight target escapes its exact repository':
                raise
            cleanup_global(alias, registration)
    freeze(OUT / 'reference_execution_complete_20261029.json', dict(status='complete',
        registration_sha256=file_sha(REG), models=[m['alias'] for m in registration['reference_models']],
        output=str(DEST), completed_reference_weights_removed=True, primary_weights_untouched=True))


if __name__ == '__main__':
    main()
