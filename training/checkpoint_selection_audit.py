"""Audit existing epoch logs without changing checkpoints or historical scores."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pl_mmb_training.common import file_sha, read_json, write_json


def selection_key(row):
    val = row['validation']
    return (val['t2i']['recall@1'], val['mean_r1'], -row['epoch'])


def audit(root=ROOT):
    runs, epochs = [], []
    paths = [*(root / 'training/runs/p0-e2').glob('E2b-seed-*/result.json'),
             *(root / 'training/runs/vision-commercial-five/models/tinyclip-39m').glob('E2b-seed-*/result.json')]
    for path in sorted(paths):
        value = read_json(path)
        history = value['history']
        trained = [r for r in history if r['epoch'] > 0]
        loss_best = min(trained, key=lambda r: (r['validation_loss'], r['epoch']))
        retrieval_best = max(trained, key=selection_key)
        chosen = next(r for r in history if r['epoch'] == value['checkpoint_epoch'])
        for row in history:
            v = row['validation']
            epochs.append(dict(run=str(path.parent.relative_to(root)), epoch=row['epoch'],
                               validation_loss=row['validation_loss'],
                               t2i_r1=v['t2i']['recall@1'], t2i_r5=v['t2i']['recall@5'],
                               t2i_r10=v['t2i']['recall@10'], i2t_r1=v['i2t']['recall@1']))
        runs.append(dict(run=str(path.parent.relative_to(root)), source_sha256=file_sha(path),
                         checkpoint_selection='legacy_minimum_validation_loss',
                         historical_checkpoint_epoch=value['checkpoint_epoch'],
                         minimum_loss_epoch=loss_best['epoch'], retrieval_best_epoch=retrieval_best['epoch'],
                         same_epoch=loss_best['epoch'] == retrieval_best['epoch'],
                         selected_t2i_r1=chosen['validation']['t2i']['recall@1'],
                         best_t2i_r1=retrieval_best['validation']['t2i']['recall@1'],
                         selected_i2t_r1=chosen['validation']['i2t']['recall@1'],
                         best_i2t_r1=retrieval_best['validation']['i2t']['recall@1'],
                         per_epoch_checkpoints_available=(path.parent / 'checkpoints').is_dir()))
    if not runs:
        raise FileNotFoundError('No historical sigmoid runs found')
    report = dict(status='complete', historical_results_modified=False,
                  primary='validation.t2i.recall@1', secondary='validation.mean_r1', runs=runs,
                  note='Epoch metrics are observed. Missing historical epoch weights cannot be recovered from logs.')
    write_json(root / 'reports/checkpoint_selection_audit.json', report)
    with (root / 'reports/checkpoint_selection_epochs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(epochs[0]))
        writer.writeheader()
        writer.writerows(epochs)
    lines = ['# EXP0: checkpoint selection audit', '',
             'Historical runs retain legacy minimum-validation-loss selection and unchanged test results.', '',
             'New selection: validation t2i R@1, then mean t2i/i2t R@1; earliest epoch breaks ties.', '',
             '| Run | Loss epoch | Retrieval epoch | Selected val t2i R@1 | Best val t2i R@1 |',
             '|---|---:|---:|---:|---:|']
    for r in runs:
        lines.append(f"| {r['run']} | {r['minimum_loss_epoch']} | {r['retrieval_best_epoch']} | "
                     f"{100*r['selected_t2i_r1']:.2f}% | {100*r['best_t2i_r1']:.2f}% |")
    lines += ['', report['note'], '', 'See checkpoint_selection_epochs.csv for loss and both retrieval directions per epoch.']
    (root / 'reports/checkpoint_selection_audit.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(audit(), indent=2))
