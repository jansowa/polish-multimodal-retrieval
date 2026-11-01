"""Re-evaluate selected checkpoints on validation and plot measured curves."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from pl_mmb_training import tinyclip_adaptation as base
from pl_mmb_training import tinyclip_generalization as study
from pl_mmb_training.common import ROOT, digest, file_sha, read_json, write_json
from pl_mmb_training.losses import PairLoss
from pl_mmb_training.retrieval import evaluate_graph


def components(data):
    graph = data.graph
    parent = list(range(len(graph['image_ids'])))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[find(a)] = find(b)

    im = {k: i for i, k in enumerate(graph['image_ids'])}
    seen = {}
    for row in data.val:
        if row['target_id'] in im:
            i = im[row['target_id']]
            for p in row['parent_ids']:
                if p in seen:
                    union(i, seen[p])
                seen[p] = i
    by_text = {}
    for q, image in graph['edges']:
        if q in by_text:
            union(image, by_text[q])
        by_text[q] = image
    keys = sorted({find(i) for i in range(len(parent))})
    lookup = {k: i for i, k in enumerate(keys)}
    return np.array([lookup[find(by_text[q])] for q in range(len(graph['texts']))])


def bootstrap(a, b, groups):
    delta = (np.array(a) == 1).astype(float) - (np.array(b) == 1)
    ng = int(groups.max()) + 1
    sums = np.bincount(groups, weights=delta, minlength=ng)
    counts = np.bincount(groups, minlength=ng)
    gen = np.random.default_rng(20261021)
    indices = gen.integers(ng, size=(10000, ng))
    sampled = sums[indices].sum(1) / counts[indices].sum(1)
    return dict(delta_percentage_points=float(delta.mean()*100),
                interval_95_percentage_points=(np.quantile(sampled, [.025, .975])*100).tolist(),
                improved=int((delta > 0).sum()), regressed=int((delta < 0).sum()),
                bootstrap_components=ng, queries=len(delta), repetitions=10000,
                caveat='Conditional paired interval on this development set. Does not include configuration/checkpoint selection or seed variance.')


def plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for result in results:
        rows = result['history']
        name = result['experiment']
        epochs = [r['epoch'] for r in rows]
        axes[0, 0].plot(epochs, [100*r['validation']['t2i']['recall@1'] for r in rows], '.-', label=name)
        axes[0, 1].plot(epochs[1:], [r['train_loss'] for r in rows[1:]], label=name)
        axes[1, 0].plot(epochs, [r['validation_loss_fixed_scalars'] for r in rows], label=name)
        axes[1, 1].plot(epochs, [r['validation_feature_cosine_drift'] for r in rows], label=name)
    for ax, title in zip(axes.flat, ['Val text→image R@1 (%)', 'Train gold loss', 'Val loss, fixed initial scalars', 'Val vision drift (1−cos)']):
        ax.set_title(title)
        ax.set_xlabel('Fine-tuning epoch (after A1 epoch 9)')
        ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('TinyCLIP validation-only controls; one seed; identical monitoring')
    fig.tight_layout()
    fig.savefig(ROOT / 'reports/tinyclip_generalization_curves.png', dpi=170)
    plt.close(fig)


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = base.Inputs(study.OUT)
    results = [read_json(study.OUT / name / 'result.json') for name in study.specs()]
    x = torch.load(study.CACHE / 'features.pt', weights_only=True)['projected'].to('cuda')
    prefix = np.load(study.CACHE / 'prefix.npy', mmap_mode='r')
    vision, _, _ = base.load_vision(data)
    model = base.projector(512, 0).to('cuda')
    criterion = PairLoss('siglip_sigmoid').to('cuda')
    vi, vt, vp = data.tensors['validation']
    vt = vt.to('cuda')
    ranks, checks = {}, {}
    with torch.inference_mode():
        for result in results:
            name = result['experiment']
            folder = study.OUT / name
            config = read_json(folder / 'config.json')
            assert digest(config) == result['run_id']
            assert config['spec'] == study.specs()[name] == result['spec']
            for path, sha in config['source'].items():
                assert file_sha(ROOT / path) == sha
            assert file_sha(folder / 'best.pt') == result['checkpoint_sha256']
            saved = torch.load(folder / 'best.pt', weights_only=True, map_location='cpu')
            assert saved['run_id'] == result['run_id'] and saved['epoch'] == result['best_epoch']
            chosen = max(result['history'], key=base.checkpoint_key)
            assert chosen['epoch'] == saved['epoch'] and chosen['validation'] == result['selected_validation']
            assert result['history'][-1]['updates'] == 12 * 275
            assert not result['selection_uses_test'] and not result['xm3600_used'] and result['test_mscoco'] is None
            model.load_state_dict(saved['projector'])
            criterion.load_state_dict(saved['loss'])
            current = x
            if saved['vision_suffix'] is not None:
                suffix = base.Suffix(vision, 'projected', result['spec']['blocks']).to('cuda').eval()
                suffix.load_state_dict(saved['vision_suffix'])
                current = torch.zeros_like(x)
                for part in torch.unique(vi).split(128):
                    tok = torch.from_numpy(np.array(prefix[part.numpy()], copy=True)).to('cuda')
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        current[part.to('cuda')] = suffix(tok).float()
                del suffix
            graph = data.graph
            pred = model(current[graph['image_indices']]).cpu()
            text = data.texts[graph['text_indices']]
            local = dict(graph, image_indices=list(range(len(pred))), text_indices=list(range(len(text))))
            actual = evaluate_graph(local, pred, text, include_ranks=True)
            for direction in ('t2i', 'i2t'):
                for k in (1, 5, 10):
                    assert actual[direction][f'recall@{k}'] == result['selected_validation'][direction][f'recall@{k}']
            val_loss = 0.
            for part in torch.arange(len(vi)).split(256):
                val_loss += float(criterion(model(current[vi[part].to('cuda')]), vt[part], vp.mask(part.tolist()).to('cuda'))) * len(part)
            assert abs(val_loss/len(vi) - chosen['validation_loss']) < 2e-5
            ranks[name] = actual['ranks']['t2i']
            checks[name] = dict(checkpoint_verified=True, source_verified=True, retrieved_scores_reproduced=True,
                                val_loss_reproduced=True, full_3300_updates_verified=True)
    groups = components(data)
    comparisons = {}
    for a, b in [('G1_frozen_aug', 'G0_frozen_native'), ('G2_last2_aug', 'G1_frozen_aug'),
                 ('G3_last2_aug_anchor', 'G1_frozen_aug'), ('G3_last2_aug_anchor', 'G2_last2_aug')]:
        comparisons[a + '_minus_' + b] = bootstrap(ranks[a], ranks[b], groups)
    data.verify()
    value = dict(status='passed', checks=checks, comparisons=comparisons, inputs=data.hashes,
                 test_used=False, xm3600_used=False)
    write_json(ROOT / 'reports/tinyclip_generalization_verification.json', value)
    study.export(results)
    plot(results)
    with (ROOT / 'reports/tinyclip_generalization.md').open('a', encoding='utf-8') as stream:
        stream.write('\nCheckpointy ponownie odtworzono z dysku: R@1/5/10 obu kierunków i val loss zgodne.\n\n')
        stream.write('| Porównanie | Δ t2i R@1 (p.p.) | 95% paired interval (p.p.) |\n|---|---:|---|\n')
        for name, c in comparisons.items():
            lo, hi = c['interval_95_percentage_points']
            stream.write(f'| {name} | {c["delta_percentage_points"]:+.3f} | [{lo:+.3f}, {hi:+.3f}] |\n')
        stream.write('\nBootstrap po komponentach obrazów/parentów i wspólnych captionów; bez korekty selekcji konfiguracji i wariancji seedów.\n\n![Krzywe treningu](tinyclip_generalization_curves.png)\n')
    print(json.dumps(value, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
