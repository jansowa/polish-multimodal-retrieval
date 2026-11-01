"""Frozen backbone comparison, stopping after exactly the first four models."""
import copy
import json
import shutil
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import torch

from .ablation import Context, train_loss
from .backbones import ORDER, cache_backbone, load_backbone
from .common import ROOT, TASKS, read_json, write_json, file_sha, digest
from .evaluation import delta
from .runner import write_records


def validate_study(config):
    if tuple(config['backbones']) != ORDER:
        raise ValueError('Stop after the four explicitly requested backbones, in order')
    if config['scout_seed'] != 42 or config['dinov2_repeat_seeds'] != [42, 43, 44] or config['dinov2_repeat_threshold'] != .38:
        raise ValueError('Keep the requested scouting and DINOv2 replication rule')


def dino_seeds(first_r1, config):
    return config['dinov2_repeat_seeds'] if first_r1 >= config['dinov2_repeat_threshold'] - 1e-12 else [config['scout_seed']]


def check_reference(root):
    reference = root / 'reference'
    lock = read_json(reference / 'lock.json')
    for name, sha in lock['archive'].items():
        if file_sha(reference / name) != sha:
            raise ValueError('Frozen sigmoid reference archive changed')
    for name, sha in lock['files'].items():
        if file_sha(Path(lock['baseline']) / name) != sha:
            raise ValueError('Original sigmoid reference changed')
    return reference


def reference_counts(root):
    path = root / 'reference_parameters.json'
    if path.exists():
        return read_json(path)
    from sentence_transformers import SentenceTransformer
    from transformers import SiglipVisionModel
    lock = read_json(ROOT / 'training/runs/e1-full/model_lock.json')
    text = SentenceTransformer(lock['text']['model'], revision=lock['text']['revision'], device='cpu', local_files_only=True)
    text_params = sum(p.numel() for p in text.parameters())
    del text
    vision = SiglipVisionModel.from_pretrained(lock['vision']['model'], revision=lock['vision']['revision'], local_files_only=True)
    vision_params = sum(p.numel() for p in vision.parameters())
    del vision
    result = dict(text_parameters=text_params, vision_parameters=vision_params,
                  text_model=lock['text'], vision_model=lock['vision'], feature_dim=768, input_size=256,
                  representation='original SigLIP2 vision pooler_output',
                  pretraining='image-text pretrained SigLIP2')
    write_json(path, result)
    return result


class VisionContext(Context):
    def __init__(self, config, root, reference):
        common = root / 'common'
        if not (common / 'baseline').exists():
            shutil.copytree(ROOT / 'training/runs/p0-e2/baseline', common / 'baseline')
        super().__init__(read_json(reference / 'artifacts/config.json'), common)
        self.study_config, self.root, self.reference = config, root, reference
        self.original_identity = copy.deepcopy(self.identity)
        self.original_base = copy.deepcopy(self.base)
        self.original_manifest = copy.deepcopy(self.manifest)
        self.expected_sha = {p: info['sha256'] for p, info in read_json(self.baseline / 'blacklist.json')['images'].items()}
        self.expected_sha.update({r['image']: r['sha256'] for r in self.records})
        if any(p not in self.expected_sha for p in self.manifest['images']):
            raise ValueError('Missing image content provenance')

    def activate(self, key, images, metadata):
        self.key, self.encoder_stats = key, metadata['model_stats']
        self.output = self.root / 'models' / key
        self.output.mkdir(parents=True, exist_ok=True)
        self.images = images
        self.base = copy.deepcopy(self.original_base)
        self.base['vision'] = dict(model=self.encoder_stats['model'], revision=self.encoder_stats['revision'], frozen=True)
        self.manifest = copy.deepcopy(self.original_manifest)
        self.manifest['image_features'] = metadata
        self.splits = {s: (self.images[[self.ii[r['image']] for r in rows]], self.texts[[self.ti[r['text_pl']] for r in rows]])
                       for s, rows in self.rows.items()}
        self.identity = dict(self.original_identity, stage='frozen_vision_comparison', backbone=key,
                             vision_study_config=self.study_config, frozen_vision_features=metadata,
                             reference_lock_sha256=file_sha(self.reference / 'lock.json'),
                             torch_training_version=torch.__version__)
        for name in ('training/vision_study.py', 'training/prepare_vision_models.py', 'training/freeze_vision_reference.py'):
            self.identity['source'][name] = file_sha(ROOT / name)
        write_json(self.output / 'feature_cache.json', metadata)

    def load_test(self):
        if self.datasets is None:
            from .evaluation_snapshot import compact_snapshot
            source_sha = read_json(self.reference / 'lock.json')['files']['evaluation_examples.json']
            self.datasets = compact_snapshot(self.reference / 'artifacts/evaluation_examples.json',
                self.root / 'common/evaluation_compact.json', self.manifest['images'], self.manifest['texts'],
                source_sha, self.original_identity['feature_manifest_sha256'],
                {key: info['examples'] for key, info in self.e0['evaluation'].items()})
            self.test_graph = read_json(self.reference / 'artifacts/benchmark_graph.json')

    def metadata(self, seed, experiment, model, criterion=None):
        result = super().metadata(seed, experiment, model, criterion)
        result.update(backbone_ablation=self.key, vision_model=self.base['vision'],
                      encoder_stats=self.encoder_stats, extraction='offline_frozen_native_backbone_features',
                      text_cache_sha256=self.manifest['text_features']['sha256'],
                      vision_cache_sha256=self.manifest['image_features']['sha256'])
        return result


def smoke(root):
    from PIL import Image
    lock = read_json(root / 'model_lock.json')
    results = {}
    for key in ORDER:
        model, prep, forward, stats = load_backbone(key, lock[key])
        with torch.inference_mode():
            encoded = forward(prep([Image.new('RGB', (320, 240), 'red'), Image.new('RGB', (240, 320), 'blue')]))
        if encoded.shape != (2, stats['feature_dim']) or not torch.isfinite(encoded).all() or (encoded.norm(dim=1) == 0).any():
            raise ValueError('Native representation smoke test failed: ' + key)
        results[key] = dict(stats=stats, shape=list(encoded.shape), finite=True, frozen=stats['frozen'])
        print(f'SMOKE {key}: {list(encoded.shape)}, vision_parameters={stats["vision_parameters"]}', flush=True)
        del model, forward
    write_json(root / 'smoke.json', results)
    return results


def run(config, root, cache_root):
    validate_study(config)
    reference = check_reference(root)
    counts = reference_counts(root)
    ctx = VisionContext(config, root, reference)
    lock = read_json(root / 'model_lock.json')
    models = {}
    reference_runs = [read_json(reference / f'artifacts/E2b-seed-{seed}/result.json') for seed in (42, 43, 44)]
    for key in ORDER:
        write_json(root / 'status.json', dict(status='running', backbone=key, phase='cache'))
        images, metadata = cache_backbone(key, lock[key], ctx.manifest['images'], ctx.expected_sha, cache_root,
                                          batch_size=config['encoding_batch_size'])
        ctx.activate(key, images, metadata)
        def train_and_evaluate(seed):
            write_json(root / 'status.json', dict(status='running', backbone=key, phase='training', seed=seed))
            result = train_loss(ctx, 'E2b', seed)
            dest = ctx.output / f'E2b-seed-{seed}'
            if result['benchmark'] is None:
                benchmarks, predictions = ctx.benchmarks(ctx.projector(dest / 'projector.pt'))
                result.update(benchmark=benchmarks, benchmark_status='requested_backbone_comparison',
                              delta_vs_e1=delta(benchmarks['scores'], read_json(ctx.baseline / 'summary.json')['scores']))
                for task, rows in predictions.items():
                    write_records(dest / f'{task}.predictions.jsonl', rows)
                write_json(dest / 'result.json', result)
            print(f'RESULT {key} seed={seed}: MSCOCO_t2i_R1={result["benchmark"]["scores"]["mscoco_t2i"]:.4f}', flush=True)
            return result
        runs = [train_and_evaluate(config['scout_seed'])]
        seeds = [config['scout_seed']]
        if key == 'dinov2-small':
            first_r1 = runs[0]['benchmark']['scores']['mscoco_t2i']
            seeds = dino_seeds(first_r1, config)
            decision = dict(first_seed=config['scout_seed'], first_test_r1=first_r1,
                            threshold=config['dinov2_repeat_threshold'], seeds=seeds,
                            rule='User-requested test threshold for seed replication only; no hyperparameter tuning',
                            strong_candidate=first_r1 >= .40, reaches_previous_E1=first_r1 >= .417)
            write_json(root / 'dinov2_replication.json', decision)
            print('DINOv2 replication: ' + json.dumps(decision), flush=True)
        for seed in seeds:
            if seed != config['scout_seed']:
                runs.append(train_and_evaluate(seed))
        models[key] = dict(encoder=metadata['model_stats'], runs=runs)
        summary = dict(status='complete' if len(models) == 4 else 'running', config=config, models=models,
                       reference=dict(encoder=counts, runs=reference_runs), text_parameters=counts['text_parameters'],
                       stop_after_four=True, tested_order=list(models))
        write_json(root / 'summary.json', summary)
    write_json(root / 'status.json', dict(status='complete', stopped_after=list(ORDER)))
    return summary
