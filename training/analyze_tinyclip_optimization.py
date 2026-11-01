"""Post-selection validation diagnostics; no benchmark test or external data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from pl_mmb_training import tinyclip_adaptation as base
from pl_mmb_training import tinyclip_optimization as study
from pl_mmb_training.common import ROOT, file_sha, read_json, write_json
from pl_mmb_training.retrieval import evaluate_graph


def ranks(data, checkpoint, blocks):
    saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
    x = torch.load(study.CACHE / 'features.pt', weights_only=True)['projected'].to('cuda')
    model = base.projector(x.shape[1], 0).to('cuda')
    model.load_state_dict(saved['projector'])
    model.eval()
    suffix = None
    if blocks:
        vision, _, _ = base.load_vision(data)
        suffix = base.Suffix(vision, 'projected', blocks).to('cuda')
        suffix.load_state_dict(saved['vision_suffix'])
        suffix.eval().requires_grad_(False)
        del vision
    graph = data.graph
    with torch.inference_mode():
        selected = x[graph['image_indices']]
        if suffix is not None:
            prefix = np.load(study.CACHE / 'prefix.npy', mmap_mode='r')
            current = torch.zeros_like(x)
            for indices in torch.unique(data.tensors['validation'][0]).split(128):
                tokens = torch.from_numpy(np.array(prefix[indices.numpy()], copy=True)).to('cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    current[indices.to('cuda')] = suffix(tokens).float()
            selected = current[graph['image_indices']]
        projected = model(selected).cpu()
    texts = data.texts[graph['text_indices']]
    local = dict(graph, image_indices=list(range(len(projected))), text_indices=list(range(len(texts))))
    metrics = evaluate_graph(local, projected, texts, include_ranks=True)
    del model, suffix, x
    torch.cuda.empty_cache()
    return metrics


def bootstrap(graph, reference, candidate):
    # Group all caption queries linked to the same positive image component.
    parent = list(range(len(graph['images'])))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    positives = [[] for _ in graph['texts']]
    for text, image in graph['edges']:
        positives[text].append(image)
    for images in positives:
        for image in images[1:]:
            parent[find(image)] = find(images[0])
    group = [find(images[0]) for images in positives]
    mapping = {k: i for i, k in enumerate(sorted(set(group)))}
    indices = np.array([mapping[k] for k in group])
    delta = (np.array(candidate['ranks']['t2i']) <= 1).astype(float) - (np.array(reference['ranks']['t2i']) <= 1)
    totals = np.bincount(indices, weights=delta)
    sizes = np.bincount(indices)
    rng = np.random.default_rng(20261020)
    samples = rng.integers(0, len(totals), size=(10000, len(totals)))
    values = totals[samples].sum(axis=1) / sizes[samples].sum(axis=1)
    return dict(delta_t2i_r1_pp=float(100*delta.mean()),
                conditional_95_interval_pp=(100*np.quantile(values, [.025, .975])).tolist(),
                improved_queries=int((delta > 0).sum()), regressed_queries=int((delta < 0).sum()),
                clusters=len(totals), bootstrap_samples=10000,
                caveat='Paired bootstrap over positive-image components, fixed validation gallery. Does not account for configuration selection or between-seed variance.')


def main():
    torch.set_num_threads(4)
    selection = read_json(study.OUT / 'selection.json')
    data = base.Inputs(study.OUT)
    name = selection['winner']
    result = read_json(study.OUT / name / 'result.json')
    checkpoint = study.OUT / name / 'best.pt'
    assert file_sha(checkpoint) == result['checkpoint_sha256']
    candidate = ranks(data, checkpoint, result['spec']['blocks'])
    assert {k: v for k, v in candidate.items() if k != 'ranks'} == result['selected_validation']
    original = ranks(data, study.WARM_START, 0)
    control_name = 'T0_frozen_aligned' if name.startswith('T') else 'S0_frozen_schedule'
    control = ranks(data, study.OUT / control_name / 'best.pt', 0)
    value = dict(winner=name, matched_control=control_name,
                 vs_original_frozen=bootstrap(data.graph, original, candidate),
                 vs_matched_control=bootstrap(data.graph, control, candidate),
                 selection_uses_test=False, xm3600_used=False,
                 bootstrap_does_not_establish_robustness_across_seeds=True)
    write_json(ROOT / 'reports/tinyclip_optimization_validation_diagnostics.json', value)
    write_json(study.OUT / 'selected_validation_ranks.json', dict(candidate=candidate, control=control, original=original))
    data.verify()
    print(value, flush=True)


if __name__ == '__main__':
    main()
