"""Summarize only the prespecified validation comparisons after the suite."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pl_mmb_training import tinyclip_adaptation as base
from pl_mmb_training import tinyclip_optimization as study
from pl_mmb_training.common import ROOT, read_json, write_json


def main():
    study.export(study.OUT)
    runs = {p.parent.name: read_json(p) for p in sorted(study.OUT.glob('*/result.json'))}
    original = read_json(base.OUT / 'A1/result.json')
    chosen = read_json(study.OUT / 'selection.json')
    summary = dict(winner=chosen['winner'], original_frozen_val=base.key(original)[0],
                   phases={}, selection_uses_test=False, xm3600_used=False, seed=42)
    lines = ['', '## Interpretation of the matched comparisons', '']
    for phase, control in [('S', 'S0_frozen_schedule'), ('T', 'T0_frozen_aligned')]:
        candidates = [r for n, r in runs.items() if n.startswith(phase) and r['spec']['blocks']]
        best = max(candidates, key=base.key)
        delta = 100*(base.key(best)[0]-base.key(runs[control])[0])
        summary['phases'][phase] = dict(best_adapted=best['experiment'], matched_control=control,
                 adapted_t2i_r1=base.key(best)[0], frozen_t2i_r1=base.key(runs[control])[0],
                 gain_vs_matched_control_pp=delta,
                 gain_vs_original_frozen_pp=100*(base.key(best)[0]-base.key(original)[0]))
        lines.append(f'- {phase}: best adapted **{best["experiment"]}**, val t2i R@1 '
                     f'**{100*base.key(best)[0]:.2f}%**, versus matched frozen **{100*base.key(runs[control])[0]:.2f}%** '
                     f'({delta:+.2f} pp).')
    winner = runs[chosen['winner']]
    summary['selected_validation'] = winner['selected_validation']
    summary['winner_best_epoch'] = winner['best_epoch']
    lines += ['', f'Validation winner: **{chosen["winner"]}**, epoch {winner["best_epoch"]}.',
              'One seed; these results describe the tested recipes and do not establish that all other finetuning recipes fail.',
              'Warmup and cosine are tested together. Scheduled runs may stop before the planned cosine endpoint under the declared early-stopping rule.',
              'No test score participates in the comparisons or winner selection.', '']
    diag = ROOT / 'reports/tinyclip_optimization_validation_diagnostics.json'
    if diag.exists():
        d = read_json(diag)
        summary['post_selection_validation_diagnostics'] = d
        values = d['vs_matched_control']
        lo, hi = values['conditional_95_interval_pp']
        lines += [f'Post-selection paired validation diagnostic vs matched control: '
                  f'{values["delta_t2i_r1_pp"]:+.2f} pp, conditional bootstrap interval [{lo:+.2f}, {hi:+.2f}] pp.',
                  'Bootstrap groups captions by shared positive-image components, keeping the gallery fixed. '
                  'It does not account for model selection or between-seed variation.', '']
    if (ROOT / 'reports/tinyclip_optimization_curves.png').exists():
        lines += ['![Validation, LR and feature drift](tinyclip_optimization_curves.png)', '']
    with (ROOT / 'reports/tinyclip_optimization.md').open('a', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))
    write_json(ROOT / 'reports/tinyclip_optimization_summary.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
