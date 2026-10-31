"""Strict A model/GPU handoff checks, with no torch import or CUDA side effects."""
from collections import Counter, defaultdict
from pathlib import Path
from prepare import COORD, OUT, ROOT, read, write, verify_marker, local_path, sha, load_common


def verify_A():
    common = load_common()
    finalists = verify_marker('A_FINALISTS_FROZEN')
    done = verify_marker('A_GPU_DONE')
    common.verify_marker(COORD/'A_FINALISTS_FROZEN.json')
    common.verify_marker(COORD/'A_GPU_DONE.json')
    if done.get('further_A_cuda_planned') is not False:
        raise ValueError('A still plans CUDA')
    if done.get('A_FINALISTS_FROZEN_sha256') != sha(COORD/'A_FINALISTS_FROZEN.json'):
        raise ValueError('GPU-done handoff does not bind current finalists')
    if finalists.get('universe_sha256') != sha(COORD/'A_UNIVERSE_FROZEN.json'):
        raise ValueError('Finalists depend on another A universe')
    models = finalists['models']
    if len(models) != 9 or len({m['id'] for m in models}) != 9:
        raise ValueError('Expected exactly nine distinct preregistered checkpoints')
    groups = defaultdict(list)
    checked_manifests = set()
    for model in models:
        required = ('id','variant','seed','path','sha256','training_manifest_sha256','trained_on_Wiki',
                    'representative','encoder_revisions','text_prefix')
        if any(key not in model for key in required) or model['text_prefix'] != '[query]: ':
            raise ValueError('Incomplete model identity or different text prefix')
        if sha(local_path(model['path'])) != model['sha256']:
            raise ValueError('Checkpoint SHA differs: '+model['id'])
        manifest = model['training_manifest_path']
        if manifest not in checked_manifests:
            if sha(local_path(manifest)) != model['training_manifest_sha256']:
                raise ValueError('Training manifest differs')
            checked_manifests.add(manifest)
        groups[model['variant']].append(model)
    if set(groups) != {'R0','R1','R3'} or any({m['seed'] for m in group}!={42,43,44} for group in groups.values()):
        raise ValueError('Expected registered R0/R1/selected R3 at all three seeds')
    representatives = set(finalists['representative_model_ids'])
    if representatives != {m['id'] for m in models if m['representative']} or len(representatives) != 3:
        raise ValueError('Representative identities differ')
    if any(sum(m['representative'] for m in group)!=1 for group in groups.values()):
        raise ValueError('Need one predetermined representative per variant')
    if any(m['trained_on_Wiki'] != (m['variant']=='R3') for m in models):
        raise ValueError('Training exposure flags differ')
    return finalists, done


def main():
    finalists, done = verify_A()
    common = load_common()
    prior = read(OUT/'evaluation_preregistration.json')
    storage = read(ROOT/'reports/series_2026-10-23/reference_storage_preflight_20261029.json')
    storage_models = {item['alias']:item for item in storage['models']}
    references = []
    for model in prior['reference_models']:
        item = dict(model)
        budget = storage_models[model['alias']]
        if budget['revision'] != model['revision'] or budget['repo'] != model['model']:
            raise ValueError('Reference storage revision differs')
        item['weight_storage'] = budget
        references.append(item)
    selection = read(local_path(next(a['path'] for a in finalists['artifacts'] if a['role']=='selection_decision')))
    result = dict(status='models_verified_waiting_for_frozen_B_data', generation=1, models=finalists['models'],
        representative_model_ids=finalists['representative_model_ids'], reference_models=references,
        A_FINALISTS_FROZEN_sha256=sha(COORD/'A_FINALISTS_FROZEN.json'), A_GPU_DONE_sha256=sha(COORD/'A_GPU_DONE.json'),
        A_UNIVERSE_FROZEN_sha256=sha(COORD/'A_UNIVERSE_FROZEN.json'), selection_decision=selection,
        selected_Wiki_variant='R3', guardrail_interpretation='A reports VN guardrail failed; R3 diagnostic, not a replacement winner.',
        all_nine_scored_for_seed_mean_SD=True, representative_only_paired_CI=True,
        parameter_columns=prior['parameters'], no_model_add_remove_after_scores=True,
        shared_scorer_sha256=sha(ROOT/'src/pl_mmb_benchmark/full_gallery.py'),
        adapter_sha256=sha(ROOT/'src/pl_mmb_benchmark/projector_adapter.py'),
        A_adapter_parity=common.artifact('adapter_parity',ROOT/'reports/data_study_20261030/adapter_cache_parity.json'))
    write(OUT/'model_registration.json',result)
    print('A_MODELS_VERIFIED',len(result['models']), 'REFERENCES',[(r['alias'],r['weight_storage']['weight_bytes']) for r in references])


if __name__=='__main__':
    main()
