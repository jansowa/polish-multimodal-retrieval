"""Conditional control: equalize steps AND examples after the natural-size ablation."""
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from pl_mmb_training.cleanroom import VARIANTS, counts, write_csv
from pl_mmb_training.cleanroom_training import checkpoint_key, context
from pl_mmb_training.common import digest, file_sha, read_json, write_json
from pl_mmb_training.losses import PairLoss, PositivePairs
from pl_mmb_training.runner import write_records, source_snapshot


class RepeatedShuffle:
    """Consume full permutations before reshuffling; batch boundaries do not drop examples."""
    def __init__(self, size, seed):
        if size < 1:
            raise ValueError('Empty training set')
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = []
        self.cursor = 0

    def take(self, n):
        result = []
        while len(result) < n:
            if self.cursor == len(self.order):
                self.order = torch.randperm(self.size, generator=self.generator).tolist()
                self.cursor = 0
            length = min(n - len(result), len(self.order) - self.cursor)
            result.extend(self.order[self.cursor:self.cursor + length])
            self.cursor += length
        return result


def run_variant(ctx, root, name, reference):
    dest = root / 'matched-budget' / name / 'seed-42'
    dest.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(s) for s in (root / f'{name}.jsonl').read_text(encoding='utf-8').splitlines()]
    x = ctx.images[[ctx.ii[r['image']] for r in rows]]
    t = ctx.texts[[ctx.ti[r['text_pl']] for r in rows]]
    positives = PositivePairs(rows)
    opt = reference['optimization']
    size = reference['train']['records']
    batch_sizes = [min(opt['batch_size'], size - i) for i in range(0, size, opt['batch_size'])]
    identity = dict(protocol='CLEAN-v1-matched-budget', variant=name, seed=42,
                    code=source_snapshot(), script_sha256=file_sha(__file__),
                    training_manifest_sha256=file_sha(root / f'{name}.jsonl'),
                    baseline_run_id=reference['run_id'], optimization=opt, reference_batch_sizes=batch_sizes,
                    features=ctx.manifest['image_features'], text_features=ctx.manifest['text_features'])
    run_id = digest(identity)
    if (dest / 'result.json').exists():
        result = read_json(dest / 'result.json')
        if result['run_id'] != run_id or file_sha(dest / 'projector.pt') != result['checkpoint_sha256']:
            raise ValueError('Matched-budget run changed')
        return result
    write_json(dest / 'config.json', identity)
    torch.manual_seed(42)
    random.seed(42)
    torch.use_deterministic_algorithms(True)
    model = ctx.projector()
    criterion = PairLoss('siglip_sigmoid', ctx.config['temperature'], ctx.config['sigmoid_bias'])
    optimizer = torch.optim.AdamW([dict(params=model.parameters(), weight_decay=opt['weight_decay']),
                                  dict(params=criterion.parameters(), weight_decay=0.)], lr=opt['learning_rate'])
    sampler = RepeatedShuffle(len(rows), 42)
    val_x, val_t = ctx.splits['validation']
    vb = [list(range(i, min(i + opt['batch_size'], len(val_x)))) for i in range(0, len(val_x), opt['batch_size'])]
    masks = [ctx.positives['validation'].mask(i) for i in vb]
    history = []
    best = None
    steps, seen = 0, 0
    (dest / 'epochs.jsonl').write_text('', encoding='utf-8')
    for virtual_epoch in range(1, opt['epochs'] + 1):
        model.train()
        total = 0.
        for n in batch_sizes:
            indices = sampler.take(n)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x[indices]), t[indices], positives.mask(indices))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite matched-budget loss')
            loss.backward()
            optimizer.step()
            criterion.constrain()
            total += float(loss.detach()) * n
            steps += 1
            seen += n
        model.eval()
        with torch.inference_mode():
            val_loss = sum(float(criterion(model(val_x[i]), val_t[i], mask)) * len(i)
                           for i, mask in zip(vb, masks)) / len(val_x)
        row = dict(epoch=virtual_epoch, train_loss=total/size, validation_loss=val_loss,
                   validation=ctx.validation_metrics(model), optimization_steps=steps,
                   examples_seen=seen, **criterion.scalars())
        history.append(row)
        with (dest / 'epochs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(f"MATCHED {name} virtual_epoch={virtual_epoch} steps={steps} examples={seen} "
              f"val_t2i={row['validation']['t2i']['recall@1']:.4f}", flush=True)
        if best is None or checkpoint_key(row) > checkpoint_key(best):
            best = row
            torch.save(dict(state_dict=model.state_dict(), loss_state_dict=criterion.state_dict(),
                            input_dim=x.shape[1], output_dim=t.shape[1], run_id=run_id), dest / 'projector.pt')
    if steps != reference['optimization_steps'] or seen != reference['examples_seen']:
        raise ValueError('Failed to match both optimization budgets exactly')
    saved = torch.load(dest / 'projector.pt', weights_only=True, map_location='cpu')
    model.load_state_dict(saved['state_dict'])
    write_json(dest / 'selection.json', dict(checkpoint_epoch=best['epoch'], validation=best['validation'], selection_uses_test=False))
    benchmark, predictions = ctx.benchmarks(model)
    for task, values in predictions.items():
        write_records(dest / f'{task}.predictions.jsonl', values)
    result = dict(protocol='CLEAN-v1-matched-budget', variant=name, seed=42, run_id=run_id,
                  train=counts(rows), history=history, checkpoint_epoch=best['epoch'],
                  selected_validation=best['validation'], benchmark=benchmark,
                  optimization_steps=steps, examples_seen=seen, checkpoint_sha256=file_sha(dest / 'projector.pt'),
                  checkpoint_selection='validation_t2i_r1_then_mean_r1_then_earliest_epoch',
                  selection_uses_test=False, frozen_vision=True, frozen_text=True,
                  note='Repeated shuffled passes through smaller datasets; same full/tail batch-size schedule as FULL_CLEAN.')
    write_json(dest / 'result.json', result)
    return result


def export(root, results):
    rows = []
    reference = results['FULL_CLEAN']
    per_epoch_steps = math.ceil(reference['train']['records'] / reference['optimization']['batch_size'])
    for name in VARIANTS:
        r = results[name]
        rows.append(dict(variant=name, records=r['train']['records'], unique_images=r['train']['unique_images'],
                         steps=r['optimization_steps'], examples_seen=r['examples_seen'],
                         selected_checkpoint_steps=r['checkpoint_epoch'] * per_epoch_steps,
                         selected_checkpoint_examples=r['checkpoint_epoch'] * reference['train']['records'],
                         mscoco_r1=r['benchmark']['scores']['mscoco_t2i'],
                         mscoco_r5=r['benchmark']['local']['mscoco_t2i']['recall@5'],
                         mscoco_r10=r['benchmark']['local']['mscoco_t2i']['recall@10'],
                         val_t2i_r1=r['selected_validation']['t2i']['recall@1'],
                         val_i2t_r1=r['selected_validation']['i2t']['recall@1'],
                         **{k:r['benchmark']['scores'][k] for k in ('cirr','fashioniq','refcoco')}))
    write_json(ROOT / 'reports/clean_v1_matched_budget.json', dict(protocol='CLEAN-v1-matched-budget', comparison=rows,
                    full_reference_reused=True, selection_uses_test=False,
                    limitation='Equal maximum budgets do not equalize distinct images, source diversity, or the steps consumed by validation-selected checkpoints; one seed.'))
    write_csv(ROOT / 'reports/clean_v1_matched_budget.csv', rows)
    lines = ['# CLEAN-v1: matched steps and examples', '',
             'Conditional control after large natural-size ablation gaps. Same model, loss, optimizer, batch-size schedule and fixed validation.',
             'FULL_CLEAN is reused from the first ablation; smaller datasets are repeatedly shuffled.', '',
             '| Variant | Records | Unique images | Steps | Examples seen | MSCOCO R@1 | Val t2i R@1 | Val i2t R@1 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['variant']} | {r['records']} | {r['unique_images']} | {r['steps']} | {r['examples_seen']} | "
                     f"{r['mscoco_r1']*100:.2f} | {r['val_t2i_r1']*100:.2f} | {r['val_i2t_r1']*100:.2f} |")
    full, no_m, non = (r['mscoco_r1'] for r in rows)
    lines += ['', f'MSCOCO deltas vs FULL_CLEAN: NO_MBEIR {100*(no_m-full):+.2f} pp; NON_COCO {100*(non-full):+.2f} pp.',
              'The gap persists under the same available optimization budget. Distinct image count and domain/source diversity still differ.',
              'Reported steps/examples are completed training caps. Checkpoints are selected by validation and can consume different prefixes of those caps; CSV/JSON record selected_checkpoint_steps/examples separately.',
              'These are domain/data-dependence results, not evidence of instance leakage. One seed; no uncertainty estimate.',
              '', 'Recommendation: retain MSCOCO as in-domain/legacy and add XM3600-PL for independent cross-dataset t2i.']
    (ROOT / 'reports/clean_v1_matched_budget.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('all', 'report'), nargs='?', default='all')
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    torch.set_num_threads(4)
    root = ROOT / 'training/runs/clean-v1'
    audit = read_json(root / 'audit.json')
    if audit['known_remaining_overlap_count']:
        raise ValueError('Known overlap prevents matched-budget control')
    for path, sha in audit['inputs'].items():
        if file_sha(path) != sha:
            raise ValueError('Audited input changed: ' + path)
    for name, info in audit['variants'].items():
        if file_sha(root / f'{name}.jsonl') != info['sha256']:
            raise ValueError('Audited variant changed: ' + name)
    reference = read_json(root / 'FULL_CLEAN/seed-42/result.json')
    if args.command == 'report':
        results = {'FULL_CLEAN': reference}
        results.update({name: read_json(root / 'matched-budget' / name / 'seed-42/result.json') for name in VARIANTS[1:]})
        print(json.dumps(export(root, results), indent=2), flush=True)
        return
    ctx = context(root)
    results = {'FULL_CLEAN': reference}
    for name in VARIANTS[1:]:
        results[name] = run_variant(ctx, root, name, reference)
    print(json.dumps(export(root, results), indent=2), flush=True)


if __name__ == '__main__':
    main()
