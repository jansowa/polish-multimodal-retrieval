"""Matched TinyCLIP data ablations; neither frozen encoder is optimized."""
import json
import math
import random
import shutil
from pathlib import Path

import torch

from .cleanroom import VARIANTS, SOURCES, counts, load_inputs, write_csv
from .common import ROOT, TASKS, digest, file_sha, read_json, write_json
from .features import validate_features
from .losses import PairLoss, PositivePairs
from .runner import records_at, source_snapshot, write_records
from .vision_study import VisionContext, check_reference


def checkpoint_key(row):
    v = row['validation']
    return (v['t2i']['recall@1'], v['mean_r1'], -row['epoch'])


def context(output):
    output = Path(output)
    reference = check_reference(ROOT / 'training/runs/vision-four')
    common = output / 'common'
    common.mkdir(parents=True, exist_ok=True)
    target = common / 'evaluation_compact.json'
    if not target.exists():
        shutil.copyfile(ROOT / 'training/runs/vision-four/common/evaluation_compact.json', target)
    ctx = VisionContext(dict(protocol='CLEAN-v1', architecture='frozen-tinyclip39m-frozen-poldense-linear'), output, reference)
    meta = read_json(ROOT / 'training/runs/vision-commercial-five/models/tinyclip-39m/feature_cache.json')
    if file_sha(meta['path']) != meta['sha256']:
        raise ValueError('Frozen TinyCLIP feature checksum mismatch')
    images = validate_features(ctx.manifest['images'], torch.load(meta['path'], weights_only=True, map_location='cpu'))
    ctx.activate('tinyclip-39m', images, meta)
    return ctx


def train(ctx, output, name, audit):
    dest = output / name / 'seed-42'
    dest.mkdir(parents=True, exist_ok=True)
    info = audit['variants'][name]
    if file_sha(info['path']) != info['sha256']:
        raise ValueError('Training variant manifest changed')
    rows = [json.loads(line) for line in Path(info['path']).read_text(encoding='utf-8').splitlines()]
    if any(r['split'] != 'train' for r in rows):
        raise ValueError('Variant contains non-training examples')
    # Fixed, original validation gallery for all variants, including NON_COCO.
    val_rows = ctx.rows['validation']
    train_x = ctx.images[[ctx.ii[r['image']] for r in rows]]
    train_t = ctx.texts[[ctx.ti[r['text_pl']] for r in rows]]
    positives = PositivePairs(rows)
    val_x, val_t = ctx.splits['validation']
    opt = ctx.config['optimization']
    identity = dict(protocol='CLEAN-v1', variant=name, seed=42, epochs=opt['epochs'], optimization=opt,
                    inputs=ctx.identity, manifest_sha256=info['sha256'],
                    source=source_snapshot(), entrypoint_sha256=file_sha(ROOT / 'training/cleanroom_study.py'),
                    validation_manifest_sha256=file_sha(output / 'validation.jsonl'),
                    checkpoint_selection='validation_t2i_r1_then_mean_r1_then_earliest_epoch',
                    near_duplicate_filter_applied=False)
    run_id = digest(identity)
    result_path = dest / 'result.json'
    if result_path.exists():
        result = read_json(result_path)
        if result['run_id'] != run_id or result['checkpoint_sha256'] != file_sha(dest / 'projector.pt'):
            raise ValueError('Completed run identity or checkpoint changed')
        return result
    write_json(dest / 'config.json', identity)
    torch.manual_seed(42)
    random.seed(42)
    torch.use_deterministic_algorithms(True)
    model = ctx.projector()
    criterion = PairLoss('siglip_sigmoid', ctx.config['temperature'], ctx.config['sigmoid_bias'])
    optimizer = torch.optim.AdamW([dict(params=model.parameters(), weight_decay=opt['weight_decay']),
                                  dict(params=criterion.parameters(), weight_decay=0.)], lr=opt['learning_rate'])
    generator = torch.Generator().manual_seed(42)
    batches = [list(range(start, min(start + opt['batch_size'], len(val_x))))
               for start in range(0, len(val_x), opt['batch_size'])]
    masks = [ctx.positives['validation'].mask(indices) for indices in batches]
    (dest / 'epochs.jsonl').write_text('', encoding='utf-8')
    history = []
    best = None

    def monitor(epoch, train_loss):
        with torch.inference_mode():
            val_loss = sum(float(criterion(model(val_x[i]), val_t[i], mask)) * len(i)
                           for i, mask in zip(batches, masks)) / len(val_x)
        row = dict(epoch=epoch, train_loss=train_loss, validation_loss=val_loss,
                   validation=ctx.validation_metrics(model), **criterion.scalars())
        history.append(row)
        with (dest / 'epochs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(f"CLEAN-v1 {name} epoch={epoch}/{opt['epochs']} val_loss={val_loss:.5f} "
              f"val_t2i={row['validation']['t2i']['recall@1']:.4f} "
              f"val_i2t={row['validation']['i2t']['recall@1']:.4f}", flush=True)
        return row

    monitor(0, None)
    (dest / 'checkpoints').mkdir(exist_ok=True)
    for epoch in range(1, opt['epochs'] + 1):
        model.train()
        permutation = torch.randperm(len(rows), generator=generator)
        total = 0.
        for start in range(0, len(rows), opt['batch_size']):
            indices = permutation[start:start + opt['batch_size']].tolist()
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train_x[indices]), train_t[indices], positives.mask(indices))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite ablation loss')
            loss.backward()
            optimizer.step()
            criterion.constrain()
            total += float(loss.detach()) * len(indices)
        model.eval()
        row = monitor(epoch, total / len(rows))
        checkpoint = dict(state_dict=model.state_dict(), loss_state_dict=criterion.state_dict(),
                          input_dim=train_x.shape[1], output_dim=train_t.shape[1], run_id=run_id, epoch=epoch)
        torch.save(checkpoint, dest / f'checkpoints/epoch-{epoch:02d}.pt')
        if best is None or checkpoint_key(row) > checkpoint_key(best):
            best = row
            torch.save(checkpoint, dest / 'projector.pt')
    saved = torch.load(dest / 'projector.pt', weights_only=True, map_location='cpu')
    model.load_state_dict(saved['state_dict'])
    criterion.load_state_dict(saved['loss_state_dict'])
    # Selection is committed before the legacy benchmark is opened.
    write_json(dest / 'selection.json', dict(checkpoint_epoch=best['epoch'], validation=best['validation'],
                                           selection_uses_test=False, rule=identity['checkpoint_selection']))
    benchmark, predictions = ctx.benchmarks(model)
    for task, values in predictions.items():
        write_records(dest / f'{task}.predictions.jsonl', values)
    result = dict(protocol='CLEAN-v1', status='complete', seed=42, variant=name, run_id=run_id,
                  train=counts(rows), validation_data=counts(val_rows), history=history,
                  checkpoint_epoch=best['epoch'], checkpoint_selection=identity['checkpoint_selection'],
                  selected_validation=best['validation'], validation_loss=best['validation_loss'],
                  minimum_validation_loss_epoch=min(history[1:], key=lambda r: r['validation_loss'])['epoch'],
                  checkpoint_sha256=file_sha(dest / 'projector.pt'), benchmark=benchmark,
                  optimization_steps=opt['epochs'] * math.ceil(len(rows) / opt['batch_size']),
                  examples_seen=opt['epochs'] * len(rows), optimization=opt,
                  frozen_vision=True, frozen_text=True, projector='linear', loss='multi_positive_siglip_sigmoid',
                  feature_cache_sha256=ctx.manifest['image_features']['sha256'],
                  text_cache_sha256=ctx.manifest['text_features']['sha256'],
                  near_duplicate_filter_applied=False, selection_uses_test=False,
                  budget_note='Same 20 epochs; natural dataset sizes, steps and examples differ by design.')
    write_json(result_path, result)
    return result


def run(output, variant=None):
    output = Path(output)
    torch.set_num_threads(4)
    audit = read_json(output / 'audit.json')
    if audit['known_remaining_overlap_count']:
        raise ValueError('Known overlap must be resolved before CLEAN-v1')
    for path, sha in audit['inputs'].items():
        if file_sha(path) != sha:
            raise ValueError('Audited input changed: ' + path)
    ctx = context(output)
    names = [variant] if variant else VARIANTS
    results = {}
    for name in names:
        write_json(output / 'training_progress.json', dict(status='running', variant=name, seed=42))
        results[name] = train(ctx, output, name, audit)
        export(output)
    write_json(output / 'training_progress.json', dict(status='complete', variants=list(results)))
    return {k: dict(train=v['train'], selected_validation=v['selected_validation'], scores=v['benchmark']['scores'])
            for k, v in results.items()}


def export(output):
    output = Path(output)
    audit = read_json(output / 'audit.json')
    near_path = output / 'near_duplicate_summary.json'
    near = read_json(near_path) if near_path.exists() else None
    runs, rows = {}, []
    for name in VARIANTS:
        path = output / name / 'seed-42/result.json'
        if not path.exists():
            continue
        value = read_json(path)
        runs[name] = value
        coco = value['benchmark']['local']['mscoco_t2i']
        rows.append(dict(variant=name, **{k: value['train'][k] for k in ('records', 'unique_image_text_pairs', 'unique_images')},
                         **{f'mscoco_r{k}': coco[f'recall@{k}'] for k in (1, 5, 10)},
                         val_t2i_r1=value['selected_validation']['t2i']['recall@1'],
                         val_i2t_r1=value['selected_validation']['i2t']['recall@1'],
                         **{k: value['benchmark']['scores'][k] for k in ('cirr', 'fashioniq', 'refcoco')},
                         checkpoint_epoch=value['checkpoint_epoch'], steps=value['optimization_steps'],
                         examples_seen=value['examples_seen']))
    report = dict(protocol='CLEAN-v1', status='complete' if len(runs) == 3 else 'in_progress',
                  audit=audit, near_duplicates=near, comparison=rows,
                  recommendation='Keep MSCOCO as IN-DOMAIN / LEGACY. Add independent XM3600-PL; do not treat composition/grounding as substitutes for independent t2i.',
                  caveats=['Single seed, natural dataset sizes and unequal examples seen.',
                           'CLEAN-v1 uses validation t2i checkpoint selection; historical runs used minimum validation loss.',
                           'Near-duplicate candidates are not automatically filtered or confirmed.',
                           audit['non_coco_note']])
    write_json(ROOT / 'reports/clean_v1.json', report)
    write_csv(ROOT / 'reports/clean_v1_ablation.csv', rows)
    shutil.copyfile(output / 'leakage_by_source_benchmark.csv', ROOT / 'reports/clean_v1_leakage.csv')
    shutil.copyfile(output / 'known_remaining_overlaps.json', ROOT / 'reports/clean_v1_known_overlaps.json')
    lines = ['# CLEAN-v1: leakage guard audit and COCO exposure ablation', '',
             'Historical results remain LEGACY / pre-audit. No historical checkpoint or score was changed.', '',
             f"Known retained source/parent/hash overlaps: **{audit['known_remaining_overlap_count']}**.", '',
             '## Existing guard exclusions (train + validation preparation)', '',
             '| Source | Rejected | Parent | Content hash | Text | Unknown lineage | Retained |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for s in audit['sources']:
        lines.append(f"| {s['source']} | {s['rejected_records']} | {s['eval_parent_image']} | {s['eval_image_content']} | "
                     f"{s['eval_text']} | {s['unknown_parent_lineage']} | {s['retained']['records']} |")
    lines += ['', audit['attribution_note'], '',
              'See clean_v1_leakage.csv for attribution to each protected benchmark. Reasons can overlap.', '',
              '## Natural-size data ablation, seed 42', '',
              '| Variant | Records | Unique pairs | Unique images | COCO R@1 | R@5 | R@10 | Val t2i | Val i2t | CIRR | FashionIQ | RefCOCO |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        keys = ('mscoco_r1', 'mscoco_r5', 'mscoco_r10', 'val_t2i_r1', 'val_i2t_r1', 'cirr', 'fashioniq', 'refcoco')
        lines.append(f"| {row['variant']} | {row['records']} | {row['unique_image_text_pairs']} | {row['unique_images']} | "
                     + ' | '.join(f'{100*row[k]:.2f}' for k in keys) + ' |')
    lines += ['', 'Metrics are percentages. Same native TinyCLIP features, frozen PolDense, linear projector, sigmoid loss, batch 256, AdamW and 20 epochs.',
              'Validation gallery is fixed across variants; checkpoints use val t2i R@1, mean t2i/i2t R@1 breaks ties. Test is read only after selection.', '',
              '## Near duplicates', '']
    if near:
        lines += [f"Exhaustive pHash comparisons: {near['pairs_exhaustively_compared']}; pairs at distance <= {near['threshold']}: {near['threshold_pairs']}.",
                  'Candidates are a review queue, not confirmed leakage. No perceptual filtering was applied.', near['limitation']]
    else:
        lines += ['Pending.']
    if len(rows) == 3:
        full, no_m, non = (r['mscoco_r1'] for r in rows)
        lines += ['', '## Interpretation', '',
                  f'Removing M-BEIR changes MSCOCO R@1 by {100*(no_m-full):+.2f} pp; removing all known COCO lineage changes it by {100*(non-full):+.2f} pp.',
                  'These changes measure data/domain dependence together with reduced training size and budget. They do not establish instance leakage.',
                  'A matched-steps/examples control is needed to isolate domain exposure from the smaller dataset if these gaps are substantial.']
    lines += ['', report['recommendation'], '', '## Limits', '', *['- ' + c for c in report['caveats']], '',
              'M-BEIR source metadata: ' + audit['mbeir_source_split'], '',
              'Instance leakage, COCO domain exposure, and repeated test inspection are separate issues.']
    (ROOT / 'reports/clean_v1.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return dict(status=report['status'], comparison=rows)
