"""Frozen vision scouting: five candidates, then replication, never RN101/61M."""
import gc
import shutil

import torch

from .ablation import train_loss
from .backbones import cache_backbone
from .commercial_backbones import ORDER, load_backbone
from .common import ROOT, file_sha, read_json, write_json
from .runner import write_records
from .vision_study import VisionContext, check_reference, reference_counts


def validate_study(config):
    if tuple(config['backbones']) != ORDER or not config['stop_after_five']:
        raise ValueError('Only the first five requested models, in exact order')
    if (config['scout_seed'], config['repeat_seeds'], config['repeat_threshold']) != (42, [42, 43, 44], .47):
        raise ValueError('Keep the requested seeds and quality threshold')
    if config['loss'] != 'siglip_sigmoid' or config['projector'] != 'linear' or config['checkpoint_selection'] != 'minimum_validation_loss':
        raise ValueError('Keep sigmoid, linear projector and validation-only checkpoint selection')
    if config['size_class_upper_bounds'] != [40000000, 50000000, 60000000]:
        raise ValueError('Freeze size class definitions before scouting')


def size_class(params, config):
    return next((bound for bound in config['size_class_upper_bounds'] if params <= bound), None)


def replication_decisions(models, config):
    # Existing vision-only permissive checkpoints are also competitors in <=40M.
    competitors = [(22056576, .258), (27820128, .202), (21665664, .233)]
    competitors += [(group['encoder']['vision_parameters'], group['runs'][0]['benchmark']['scores']['mscoco_t2i'])
                    for group in models.values()
                    if group['encoder']['license']['declared_weights_license'] in ('mit', 'apache-2.0')]
    decisions = {}
    for key, group in models.items():
        count = group['encoder']['vision_parameters']
        r1 = group['runs'][0]['benchmark']['scores']['mscoco_t2i']
        license_ok = group['encoder']['license']['declared_weights_license'] in ('mit', 'apache-2.0')
        bucket = size_class(count, config)
        same_class = [score for params, score in competitors if size_class(params, config) == bucket]
        reasons = []
        if r1 >= config['repeat_threshold'] - 1e-12:
            reasons.append('MSCOCO_t2i_R1_at_least_47_percent')
        if license_ok and bucket is not None and same_class and r1 >= max(same_class) - 1e-12:
            reasons.append('best_declared_permissive_candidate_in_size_class')
        if (r1 >= config['mobileclip_reference_r1'] - config['mobileclip_max_gap'] - 1e-12
                and count <= config['mobileclip_reference_parameters'] * config['substantially_smaller_fraction']):
            reasons.append('within_2pp_of_MobileCLIP_with_at_least_20_percent_fewer_parameters')
        decisions[key] = dict(scout_r1=r1, vision_parameters=count, size_class_upper_bound=bucket,
                              reasons=reasons, seeds=config['repeat_seeds'] if reasons else [config['scout_seed']])
    return decisions


class CommercialContext(VisionContext):
    def activate(self, key, images, metadata):
        super().activate(key, images, metadata)
        for name in ('training/commercial_vision_study.py', 'training/prepare_commercial_vision.py',
                     'training/prepare_chinese_source.py'):
            self.identity['source'][name] = file_sha(ROOT / name)


def smoke(root):
    from PIL import Image
    lock = read_json(root / 'model_lock.json')
    results = {}
    for key in ORDER:
        model, prep, forward, stats = load_backbone(key, lock[key])
        with torch.inference_mode():
            features = forward(prep([Image.new('RGB', (320, 240), 'red'), Image.new('RGB', (240, 320), 'blue')]))
        if features.shape != (2, stats['feature_dim']) or not torch.isfinite(features).all() or (features.norm(dim=1) == 0).any():
            raise ValueError('Invalid standard image embedding: ' + key)
        results[key] = dict(stats=stats, shape=list(features.shape), frozen=stats['frozen'], finite=True)
        print(f'SMOKE {key}: {list(features.shape)} parameters={stats["vision_parameters"]}', flush=True)
        del model, forward
        gc.collect()
    write_json(root / 'smoke.json', results)


def run(config, root, cache_root):
    validate_study(config)
    old_root = ROOT / 'training/runs/vision-four'
    reference = check_reference(old_root)
    counts = reference_counts(old_root)
    (root / 'common').mkdir(parents=True, exist_ok=True)
    compact = root / 'common/evaluation_compact.json'
    if not compact.exists():
        shutil.copyfile(old_root / 'common/evaluation_compact.json', compact)
    ctx = CommercialContext(config, root, reference)
    lock = read_json(root / 'model_lock.json')
    references = read_json(old_root / 'summary.json')
    models = {}

    def activate(key):
        write_json(root / 'status.json', dict(status='running', phase='cache', backbone=key))
        images, metadata = cache_backbone(key, lock[key], ctx.manifest['images'], ctx.expected_sha,
                                          cache_root, config['encoding_batch_size'], loader=load_backbone)
        ctx.activate(key, images, metadata)
        return metadata

    def train_and_evaluate(key, seed):
        write_json(root / 'status.json', dict(status='running', phase='training', backbone=key, seed=seed))
        result = train_loss(ctx, 'E2b', seed)
        dest = ctx.output / f'E2b-seed-{seed}'
        if result['benchmark'] is None:
            benchmarks, predictions = ctx.benchmarks(ctx.projector(dest / 'projector.pt'))
            result.update(benchmark=benchmarks, benchmark_status='explicit_user_requested_backbone_comparison')
            for task, rows in predictions.items():
                write_records(dest / f'{task}.predictions.jsonl', rows)
            write_json(dest / 'result.json', result)
        print(f'RESULT {key} seed={seed}: MSCOCO_t2i_R1={result["benchmark"]["scores"]["mscoco_t2i"]:.4f}', flush=True)
        return result

    def save(complete=False):
        summary = dict(status='complete' if complete else 'running', config=config, models=models,
                       reference=references['reference'], previous_models=references['models'],
                       text_parameters=counts['text_parameters'], tested_order=list(models), stop_after_five=True)
        write_json(root / 'summary.json', summary)
        return summary

    # Complete all five scouts before deciding which size-class winners to replicate.
    for key in ORDER:
        metadata = activate(key)
        models[key] = dict(encoder=metadata['model_stats'], runs=[train_and_evaluate(key, config['scout_seed'])])
        save()
    decisions = replication_decisions(models, config)
    write_json(root / 'replication_decisions.json', decisions)
    print('REPLICATION ' + str(decisions), flush=True)
    for key in ORDER:
        seeds = decisions[key]['seeds'][1:]
        if seeds:
            activate(key)
            for seed in seeds:
                models[key]['runs'].append(train_and_evaluate(key, seed))
                save()
    summary = save(complete=True)
    write_json(root / 'status.json', dict(status='complete', stopped_after=list(ORDER)))
    return summary
