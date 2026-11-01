"""Export the small, reviewable E0/E1 report; leave large caches/runs untracked."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pl_mmb_training.common import read_json, write_json, file_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'training/runs/e1-full')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/training_e0_e1_full.md')
    args = parser.parse_args()
    e0 = read_json(args.run_dir / 'e0.json')
    summary = read_json(args.run_dir / 'summary.json')
    audit = read_json(args.run_dir / 'data_audit.json')
    config = read_json(args.run_dir / 'config.json')
    runs = [read_json(args.run_dir / f'E1-seed-{seed}' / 'result.json') for seed in summary['seeds']]
    if e0['status'] != 'complete' or audit['post_filter_overlap'] or any(not r['full_dataset'] for r in runs):
        raise ValueError('Only complete, clean experiments can be exported')
    report = (args.run_dir / 'comparison.md').read_text(encoding='utf-8')
    report = report.replace('\n\nScores:', '\n\nPrimary task: Polish text query → image retrieval. '
                            'MSCOCO_t2i is the primary metric; it uses the pinned local MMEB candidate pools, '
                            'not a global gallery. CIRR/FashionIQ/RefCOCO and AVG are diagnostic until E4.\n\nScores:', 1)
    report += '\nReproduce with `train.py` and `training/configs/e1.json`; see [training instructions](../training/README.md) '
    report += f'and [E2–E11 plan](../training/EXPERIMENTS.md). Full local artifacts: `{args.run_dir}`.\n'
    report += f'\nE0 cross-dataset eval parent groups: {e0["cross_dataset_parent_groups"]}; '
    report += f'pixel groups: {e0["cross_dataset_pixel_groups"]}. Post-filter train/eval overlap: {audit["post_filter_overlap"]}.\n'
    report += '\n## Training losses\n\n'
    report += '| Seed | Validation before training | Validation epoch 1 | Train loss epoch 1 | Train loss final epoch | Best validation loss | Selected epoch |\n'
    report += '|---|---:|---:|---:|---:|---:|---:|\n'
    for run in runs:
        initial = run.get('initial_loss', {}).get('validation_loss')
        initial_label = f'{initial:.5f}' if initial is not None else 'not recorded'
        report += (f'| {run["seed"]} | {initial_label} | {run["history"][0]["validation_loss"]:.5f} | '
                   f'{run["history"][0]["train_loss"]:.5f} | '
                   f'{run["history"][-1]["train_loss"]:.5f} | {run["validation_loss"]:.5f} | {run["checkpoint_epoch"]} |\n')
    report += '\nCheckpoints were selected using validation loss, not benchmark scores. '
    report += f'Full epoch history is exported to `{args.output.stem}_loss.csv` and the JSON report.\n'
    if audit.get('sources'):
        report += (f'\nTranslated source records: {audit["translated_records"]}; retained pairs: {audit["retained"]}; '
                   f'distinct target images by decoded pixels: {audit["unique_targets"]}; '
                   f'pair limit: {audit["selection_limit"]}.\n')
        report += '\n## Source coverage\n\n| Source | Translated records | Usable before selection | Excluded |\n|---|---:|---:|---:|\n'
        for source in audit['sources']:
            report += f'| {source["dataset"]} | {source.get("translated_records", "n/a")} | {source.get("retained_before_selection", "n/a")} | {source.get("excluded", "n/a")} |\n'
        report += '\nExclusion reasons: `' + json.dumps(audit['reasons'], sort_keys=True) + '`.\n'
        report += ('\nA translated record can map to several positive images; usable/excluded counts refer to pairs '
                   '(or records without any verified mapping). Multiple exclusion reasons can apply to one entry.\n')
    attention_path = args.run_dir / 'attention_check.json'
    attention = read_json(attention_path) if attention_path.exists() else None
    if attention:
        report += '\nThe generic Sentence Transformers warning about `is_causal=False` was checked against '
        report += 'the installed ModernBERT SDPA implementation and an actual future-token perturbation test. '
        report += f'The first-token representation changed (L2={attention["first_token_difference_norm"]:.5f}), '
        report += 'confirming bidirectional attention for the pinned model. No encoder or library was changed.\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding='utf-8')
    small = dict(config=config, e0_sha256=file_sha(args.run_dir / 'e0.json'),
                 model_lock=read_json(args.run_dir / 'model_lock.json'), summary=summary,
                 attention_check=attention,
                 e0=dict(baselines=[dict(model=b['model'], scores=b['scores']) for b in e0['baselines']],
                         train=e0['train'], validation=e0['validation'], removed=e0['removed'],
                         leakage_reasons=audit['reasons'], post_filter_overlap=audit['post_filter_overlap'],
                         translated_records=audit.get('translated_records'), sources=audit.get('sources'),
                         retained=audit['retained'], selection_limit=audit.get('selection_limit'),
                         unique_targets=audit.get('unique_targets'),
                         blacklist_sha256=e0['blacklist_sha256']),
                 runs=[{k: r[k] for k in ('seed', 'run_id', 'scores', 'delta', 'checkpoint_epoch', 'validation_loss',
                                         'trainable_parameters', 'benchmark_protocol', 'timestamp', 'history',
                                         'initial_loss') if k in r} for r in runs])
    write_json(args.output.with_suffix('.json'), small)
    with args.output.with_name(args.output.stem + '_loss.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['seed', 'epoch', 'train_loss', 'validation_loss', 'selected_checkpoint'])
        for run in runs:
            for epoch in ([run['initial_loss']] if 'initial_loss' in run else []) + run['history']:
                writer.writerow([run['seed'], epoch['epoch'], epoch['train_loss'], epoch['validation_loss'],
                                 epoch['epoch'] == run['checkpoint_epoch']])
    print(args.output)


if __name__ == '__main__':
    main()
