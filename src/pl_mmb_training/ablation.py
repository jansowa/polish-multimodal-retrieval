"""P0 and loss-only P1 experiments referencing the immutable full E1 baseline."""
import json
import math
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

from .common import ROOT, TASKS, read_json, write_json, digest, file_sha
from .runner import check_e0, source_snapshot, regression_loss, write_records
from .data import load_evaluation
from .features import validate_features
from .evaluation import evaluate_features, scores_table, delta
from .retrieval import graph_from_records, graph_from_examples, evaluate_graph
from .losses import PairLoss, PositivePairs


def validate_plan(config, base):
    if config['optimization'] != base['optimization'] or config['scheduler'] != 'none':
        raise ValueError('Loss ablation must retain the exact E1 optimization budget')
    if config['scout_seed'] not in base['seeds'] or config['repeat_seeds'] != base['seeds']:
        raise ValueError('Use E1 seeds for a matched comparison')
    if config['experiments'] != {'E2a': 'symmetric_infonce', 'E2b': 'siglip_sigmoid'}:
        raise ValueError('This runner implements P0 and E2a/E2b only')
    if config['checkpoint_selection'] != 'minimum_validation_loss':
        raise ValueError('Keep the E1 checkpoint selection rule')
    if config['selection_dataset'] != 'mbeir_mscoco' or config['repeat_gain'] < .01:
        raise ValueError('Choose replications using held-out COCO retrieval, never benchmark test')


class Context:
    def __init__(self, config, output):
        self.config, self.output = config, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        bound = self.output / 'config.json'
        if bound.exists() and read_json(bound) != config:
            raise ValueError('Output belongs to another ablation config')
        write_json(bound, config)
        self.baseline = ROOT / config['baseline']
        self.lock_path = self.output / 'baseline/baseline_lock.json'
        lock = read_json(self.lock_path)
        if str(self.baseline.resolve()) != lock['baseline']:
            raise ValueError('Wrong E1 baseline lock')
        for path, sha in lock['files'].items():
            if file_sha(self.baseline / path) != sha:
                raise ValueError(f'Frozen E1 artifact changed: {path}')
        for path, sha in lock['archived'].items():
            if file_sha(self.lock_path.parent / path) != sha:
                raise ValueError('Frozen E1 archive changed')
        self.base = read_json(self.baseline / 'config.json')
        validate_plan(config, self.base)
        self.e0, self.records = check_e0(self.base, self.baseline)
        self.manifest = read_json(self.baseline / 'cache.json')
        if self.manifest['config_sha256'] != digest(self.base) or self.manifest['e0_sha256'] != file_sha(self.baseline / 'e0.json'):
            raise ValueError('Wrong E1 feature manifest')
        tensors = []
        for name, keys in (('image', 'images'), ('text', 'texts')):
            meta = self.manifest[name + '_features']
            if file_sha(meta['path']) != meta['sha256']:
                raise ValueError('Frozen encoder features changed')
            tensors.append(validate_features(self.manifest[keys], torch.load(meta['path'], weights_only=True)))
        self.images, self.texts = tensors
        self.ii = {p: i for i, p in enumerate(self.manifest['images'])}
        self.ti = {t: i for i, t in enumerate(self.manifest['texts'])}
        self.rows = {s: [r for r in self.records if r['split'] == s] for s in ('train', 'validation')}
        train_ids = {p for r in self.rows['train'] for p in r['parent_ids']}
        val_ids = {p for r in self.rows['validation'] for p in r['parent_ids']}
        if train_ids & val_ids or {r['target_id'] for r in self.rows['train']} & {r['target_id'] for r in self.rows['validation']}:
            raise ValueError('Train/validation share images or parents')
        self.splits = {s: (self.images[[self.ii[r['image']] for r in rows]],
                           self.texts[[self.ti[r['text_pl']] for r in rows]]) for s, rows in self.rows.items()}
        self.positives = {s: PositivePairs(rows) for s, rows in self.rows.items()}
        val = [r for r in self.rows['validation'] if r['dataset'] == config['selection_dataset']]
        self.validation = graph_from_records(val, self.manifest['images'], self.manifest['texts'])
        write_json(self.output / 'validation_graph.json', self.validation)
        self.datasets = None
        code = source_snapshot()
        for filename in ('training/loss_ablation.py', 'training/freeze_e1.py'):
            code[filename] = file_sha(ROOT / filename)
        self.identity = dict(config=config, source=code, baseline_lock_sha256=file_sha(self.lock_path),
                             feature_manifest_sha256=file_sha(self.baseline / 'cache.json'),
                             validation_graph_sha256=self.validation['sha256'])

    def load_test(self):
        if self.datasets is not None:
            return
        path = self.output / 'evaluation_examples.json'
        if path.exists():
            value = read_json(path)
            if value['provenance'] != self.e0['evaluation'] or value['sha256'] != digest(value['datasets']):
                raise ValueError('Evaluation snapshot changed')
            self.datasets = {key: [SimpleNamespace(**row) for row in rows] for key, rows in value['datasets'].items()}
        else:
            self.datasets, provenance, _ = load_evaluation()
            if provenance != self.e0['evaluation']:
                raise ValueError('Benchmark translations changed')
            datasets = {key: [vars(e) for e in rows] for key, rows in self.datasets.items()}
            write_json(path, dict(datasets=datasets, provenance=provenance, sha256=digest(datasets)))
        blacklist = read_json(self.baseline / 'blacklist.json')
        self.test_graph = graph_from_examples(self.datasets['mscoco_t2i'], blacklist,
                                              self.manifest['images'], self.manifest['texts'])
        write_json(self.output / 'benchmark_graph.json', self.test_graph)

    def projector(self, checkpoint=None):
        model = torch.nn.Linear(self.images.shape[1], self.texts.shape[1])
        if checkpoint:
            value = torch.load(checkpoint, weights_only=True, map_location='cpu')
            model.load_state_dict(value['state_dict'])
        return model

    def validation_metrics(self, model):
        with torch.inference_mode():
            images = self.images[self.validation['image_indices']]
            texts = self.texts[self.validation['text_indices']]
            local_graph = dict(self.validation, image_indices=list(range(len(images))), text_indices=list(range(len(texts))))
            return evaluate_graph(local_graph, model(images), texts)

    def benchmarks(self, model):
        self.load_test()
        with torch.inference_mode():
            projected = model(self.images)
        metrics, predictions = {}, {}
        for key in TASKS:
            metrics[key], predictions[key] = evaluate_features(self.datasets[key], self.manifest['images'], projected,
                                                               self.manifest['texts'], self.texts)
        retrieval = evaluate_graph(self.test_graph, projected, self.texts, include_ranks=True)
        return dict(local=metrics, global_mscoco=retrieval, scores=scores_table(metrics)), predictions

    def metadata(self, seed, experiment, model, criterion=None):
        opt = self.config['optimization']
        train = self.rows['train']
        return dict(experiment=experiment, seed=seed, dataset_size=len(train), validation_size=len(self.rows['validation']),
                    number_of_unique_examples_seen=len({(r['text_pl'], r['target_id']) for r in train}),
                    number_of_unique_source_records_seen=len({r['id'] for r in train}),
                    unique_target_images=len({r['target_id'] for r in train}),
                    example_presentations=opt['epochs'] * len(train),
                    optimization_steps=opt['epochs'] * math.ceil(len(train) / opt['batch_size']),
                    text_model=self.base['text'], vision_model=self.base['vision'], projector_type='linear',
                    loss=criterion.kind if criterion else self.base['loss'], batch_size=opt['batch_size'],
                    learning_rate=opt['learning_rate'], weight_decay=opt['weight_decay'], scheduler='none',
                    trainable_parameter_count=sum(p.numel() for p in model.parameters())
                                              + (sum(p.numel() for p in criterion.parameters()) if criterion else 0),
                    projector_parameter_count=sum(p.numel() for p in model.parameters()),
                    temperature_logit_scale=criterion.scalars() if criterion else dict(temperature=None, logit_scale=None, logit_bias=None),
                    frozen_text=True, frozen_vision=True, extraction='original_E1_offline_features',
                    checkpoint_selection='minimum_validation_loss', selection_dataset=self.config['selection_dataset'],
                    data_manifest_sha256=file_sha(self.baseline / 'data.jsonl'))


def aggregate(values):
    """Aggregate the same metric tree; preserve protocol/counts separately."""
    return dict(mean=statistics.mean(values), std=statistics.stdev(values) if len(values) > 1 else None, seeds=len(values))


def run_p0(ctx):
    path = ctx.output / 'p0.json'
    identity = dict(ctx.identity, stage='P0')
    if path.exists():
        value = read_json(path)
        if value['run_id'] != digest(identity):
            raise ValueError('P0 source/config changed; choose a new output')
        return value
    runs = []
    for seed in ctx.base['seeds']:
        model = ctx.projector(ctx.baseline / f'E1-seed-{seed}/projector.pt')
        metrics, predictions = ctx.benchmarks(model)
        original = read_json(ctx.baseline / f'E1-seed-{seed}/result.json')
        for key in TASKS:
            if metrics['local'][key] != original['metrics'][key]:
                raise ValueError('Expanded evaluator changed the frozen E1 metrics')
            before = [json.loads(line) for line in (ctx.baseline / f'E1-seed-{seed}/{key}.predictions.jsonl').read_text().splitlines()]
            if [p['rank'] for p in before] != [p['rank'] for p in predictions[key]]:
                raise ValueError('Expanded evaluator changed frozen E1 ranks')
        validation = ctx.validation_metrics(model)
        run = dict(run_id=original['run_id'], metadata=ctx.metadata(seed, 'E1', model),
                   validation=validation, best_validation_score=None,
                   best_validation_score_note='Retrieval was not measured per epoch in original E1; only the frozen selected checkpoint is evaluated.',
                   best_validation_loss=original['validation_loss'],
                   selected_validation_score=validation['mean_r1'], benchmark=metrics)
        write_json(ctx.output / f'P0-E1-seed-{seed}.json', run)
        runs.append(run)
        print(f'P0 E1 seed={seed}: local_t2i_R1={metrics["scores"]["mscoco_t2i"]:.4f} '
              f'val_t2i_R1={validation["t2i"]["recall@1"]:.4f} val_i2t_R1={validation["i2t"]["recall@1"]:.4f}', flush=True)
    result = dict(stage='P0', status='complete', run_id=digest(identity), identity=identity, runs=runs,
                  e1_local_t2i_r1=aggregate([r['benchmark']['scores']['mscoco_t2i'] for r in runs]),
                  selection_uses_test=False, text_space_unchanged=True,
                  text_quality_note='Original frozen PolDense embeddings, preprocessing and weights are reused; no text mapping is trained.')
    write_json(path, result)
    return result


def train_loss(ctx, experiment, seed):
    kind = ctx.config['experiments'][experiment]
    dest = ctx.output / f'{experiment}-seed-{seed}'
    dest.mkdir(exist_ok=True)
    identity = dict(ctx.identity, experiment=experiment, seed=seed)
    run_id = digest(identity)
    result_path = dest / 'result.json'
    if result_path.exists():
        result = read_json(result_path)
        if result['run_id'] != run_id:
            raise ValueError('Completed experiment belongs to another source/config')
        if file_sha(dest / 'projector.pt') != result['checkpoint_sha256']:
            raise ValueError('Completed checkpoint changed')
        return result
    write_json(dest / 'config.json', identity)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    model = ctx.projector()
    criterion = PairLoss(kind, ctx.config['temperature'], ctx.config['sigmoid_bias'])
    opt = ctx.config['optimization']
    optimizer = torch.optim.AdamW([dict(params=model.parameters(), weight_decay=opt['weight_decay']),
                                  dict(params=criterion.parameters(), weight_decay=0.)], lr=opt['learning_rate'])
    generator = torch.Generator().manual_seed(seed)
    train_x, train_t = ctx.splits['train']
    val_x, val_t = ctx.splits['validation']
    validation_batches = [list(range(start, min(start + opt['batch_size'], len(val_x))))
                          for start in range(0, len(val_x), opt['batch_size'])]
    validation_masks = [ctx.positives['validation'].mask(indices) for indices in validation_batches]
    def validation_loss():
        with torch.inference_mode():
            return sum(float(criterion(model(val_x[indices]), val_t[indices], mask)) * len(indices)
                       for indices, mask in zip(validation_batches, validation_masks)) / len(val_x)
    def monitor(epoch, train_loss):
        val_loss = validation_loss()
        retrieval = ctx.validation_metrics(model)
        record = dict(epoch=epoch, train_loss=train_loss, validation_loss=val_loss,
                      validation=retrieval, **criterion.scalars())
        with (dest / 'epochs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        train_label = f'{train_loss:.5f}' if train_loss is not None else 'not_applicable'
        print(f'{experiment} seed={seed} epoch={epoch}/{opt["epochs"]} train={train_label} val={val_loss:.5f} '
              f'val_t2i_R1={retrieval["t2i"]["recall@1"]:.4f} val_i2t_R1={retrieval["i2t"]["recall@1"]:.4f} '
              f'temperature={record["temperature"]:.5f}', flush=True)
        return record
    (dest / 'epochs.jsonl').write_text('', encoding='utf-8')
    history = [monitor(0, None)]
    best_loss, best_epoch = float('inf'), None
    for epoch in range(1, opt['epochs'] + 1):
        model.train()
        permutation = torch.randperm(len(train_x), generator=generator)
        total = 0.
        for start in range(0, len(permutation), opt['batch_size']):
            indices = permutation[start:start + opt['batch_size']].tolist()
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train_x[indices]), train_t[indices], ctx.positives['train'].mask(indices))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            criterion.constrain()
            total += float(loss.detach()) * len(indices)
        model.eval()
        record = monitor(epoch, total / len(train_x))
        history.append(record)
        if record['validation_loss'] < best_loss:
            best_loss, best_epoch = record['validation_loss'], epoch
            torch.save(dict(state_dict=model.state_dict(), loss_state_dict=criterion.state_dict(),
                            input_dim=train_x.shape[1], output_dim=train_t.shape[1], run_id=run_id), dest / 'projector.pt')
    checkpoint = torch.load(dest / 'projector.pt', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    criterion.load_state_dict(checkpoint['loss_state_dict'])
    selected = history[best_epoch]
    result = dict(stage=experiment, seed=seed, run_id=run_id, metadata=ctx.metadata(seed, experiment, model, criterion),
                  history=history, checkpoint_epoch=best_epoch, validation_loss=best_loss,
                  selected_validation=selected['validation'], selected_validation_score=selected['validation']['mean_r1'],
                  best_validation_score=max(r['validation']['mean_r1'] for r in history[1:]),
                  checkpoint_sha256=file_sha(dest / 'projector.pt'), benchmark=None,
                  benchmark_status='not_selected', timestamp=datetime.now(timezone.utc).isoformat())
    write_json(result_path, result)
    return result


def decide(scouts, p0, config):
    baseline = next(r for r in p0['runs'] if r['metadata']['seed'] == config['scout_seed'])['validation']
    repeat = [name for name, run in scouts.items()
              if run['selected_validation']['t2i']['recall@1'] - baseline['t2i']['recall@1'] >= config['repeat_gain'] - 1e-12]
    return dict(scout_repeat=repeat, baseline_validation=baseline,
                repeat_rule='held-out MSCOCO t2i R@1 gain >= 0.01; benchmark test is not read',
                selection_rule='mean held-out MSCOCO t2i/i2t R@1 at minimum-validation-loss checkpoints')


def run_suite(ctx):
    p0 = run_p0(ctx)
    scouts = {name: train_loss(ctx, name, ctx.config['scout_seed']) for name in ctx.config['experiments']}
    decision = decide(scouts, p0, ctx.config)
    runs = {name: [run] for name, run in scouts.items()}
    # Commit the replication decision before any E2 benchmark evaluation.
    write_json(ctx.output / 'replication_decision.json', decision)
    for name in decision['scout_repeat']:
        for seed in ctx.config['repeat_seeds']:
            if seed != ctx.config['scout_seed']:
                runs[name].append(train_loss(ctx, name, seed))
    mean_validation = {name: statistics.mean(r['selected_validation_score'] for r in values) for name, values in runs.items()}
    winner = max(sorted(mean_validation), key=mean_validation.get)
    decision.update(selected_loss=winner, mean_validation=mean_validation,
                    selected_run_ids=[r['run_id'] for r in runs[winner]],
                    recommended_over_E1=mean_validation[winner] > statistics.mean(r['selected_validation_score'] for r in p0['runs']))
    write_json(ctx.output / 'selection.json', decision)
    for run in runs[winner]:
        dest = ctx.output / f'{winner}-seed-{run["seed"]}'
        if run['benchmark'] is None:
            metrics, predictions = ctx.benchmarks(ctx.projector(dest / 'projector.pt'))
            run.update(benchmark=metrics, benchmark_status='selected_by_validation',
                       delta_vs_e1=delta(metrics['scores'], read_json(ctx.baseline / 'summary.json')['scores']))
            for key, values in predictions.items():
                write_records(dest / f'{key}.predictions.jsonl', values)
            write_json(dest / 'result.json', run)
    result = dict(stage='P0+E2', status='complete', selection=decision, runs=runs,
                  baseline=p0, all_comparisons_reference='full E1 41.70 ± 0.20 local MSCOCO_t2i R@1',
                  selection_uses_test=False)
    write_json(ctx.output / 'summary.json', result)
    print(json.dumps(dict(selected_loss=winner, mean_validation=mean_validation,
                          benchmark={name: [r['benchmark']['scores'] if r['benchmark'] else None for r in values]
                                     for name, values in runs.items()}), indent=2), flush=True)
    return result
