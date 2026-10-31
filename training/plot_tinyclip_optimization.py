"""Export validation, LR and representation-drift diagnostics from completed runs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pl_mmb_training.common import ROOT, read_json
from pl_mmb_training.tinyclip_optimization import OUT


def main():
    results = [read_json(p) for p in sorted(OUT.glob('*/result.json'))]
    baseline = read_json(ROOT / 'training/runs/tinyclip-adaptation/A1/result.json')
    reference = 100 * baseline['selected_validation']['t2i']['recall@1']
    labels = dict(S0_frozen_schedule='Frozen, warmup + cosine', S1_last1_low_constant='Last 1, 3e-6 constant',
                  S2_last1_schedule='Last 1, 1e-5 scheduled', S3_last1_low_schedule='Last 1, 3e-6 scheduled',
                  S4_last2_low_schedule='Last 2, 3e-6 scheduled', T0_frozen_aligned='Aligned head, frozen',
                  T1_last1_aligned='Aligned head, last 1, 1e-6', T2_last2_aligned='Aligned head, last 2, 1e-6',
                  T3_last1_aligned_mid='Aligned head, last 1, 3e-6', T4_last2_aligned_mid='Aligned head, last 2, 3e-6')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for r in results:
        h = r['history']
        n = r['experiment']
        ax = axes[0, 0] if n.startswith('S') else axes[0, 1]
        values = [row for row in h if row['epoch'] > 0 or n.startswith('T')]
        ax.plot([e['epoch'] for e in values], [100*e['validation']['t2i']['recall@1'] for e in values],
                marker='.', label=labels[n])
        ax.scatter([r['best_epoch']], [100*r['selected_validation']['t2i']['recall@1']], marker='*', s=100)
        if r['spec']['blocks']:
            logged = [e for e in h if e['epoch'] > 0]
            axes[1, 0].plot([e['epoch'] for e in logged],
                            [e['diagnostics']['vision']['lr_last'] for e in logged], label=labels[n])
            aligned = [e for e in h if e['epoch'] >= 2 or n.startswith('T')]
            axes[1, 1].plot([e['validation_feature_cosine_drift'] for e in aligned],
                            [100*e['validation']['t2i']['recall@1'] for e in aligned], marker='.', label=labels[n])
    for ax, title in zip(axes[0], ['Random head initialization', 'Continue from the same aligned head']):
        ax.axhline(reference, color='black', linestyle='--', linewidth=1, label=f'Original A1: {reference:.2f}%')
        ax.set(xlabel='Epoch (T: additional epochs)', ylabel='Validation t2i R@1 (%)', title=title)
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)
    axes[1, 0].set(xlabel='Epoch', ylabel='Vision LR (last update in epoch)', yscale='log', title='Actual logged learning rate')
    axes[1, 1].set(xlabel='Mean 1 − cosine(native, adapted) on validation images',
                   ylabel='Validation t2i R@1 (%)', title='Representation drift (S: after epoch 2; T: all epochs)')
    for ax in axes[1]:
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.suptitle('TinyCLIP-39M + frozen PolDense — FULL_CLEAN, seed 42; validation only')
    path = ROOT / 'reports/tinyclip_optimization_curves.png'
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(path)


if __name__ == '__main__':
    main()
