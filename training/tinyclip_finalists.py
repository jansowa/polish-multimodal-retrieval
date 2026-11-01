"""Commit the validation winner, then evaluate MSCOCO only for that finalist."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from pl_mmb_training import tinyclip_adaptation as base
from pl_mmb_training import tinyclip_followups as follow
from pl_mmb_training.backbones import cache_backbone
from pl_mmb_training.cleanroom_training import context
from pl_mmb_training.common import ROOT, file_sha, read_json, write_json
from pl_mmb_training.evaluation import evaluate_features
from pl_mmb_training.retrieval import evaluate_graph


def main():
    torch.set_num_threads(4)
    output = base.OUT
    for stage in ('A', 'B', 'C', 'D', 'E', 'F'):
        if not (output / f'{stage}_selection.json').exists():
            raise ValueError(f'Stage {stage} has not finished')
    results = [read_json(p) for p in sorted(output.glob('*/result.json')) if p.parent.name != 'finalist_evaluation']
    # Source order supplies a deterministic tie break; no test metric participates.
    winner = max(results, key=base.key)
    name = winner['experiment']
    decision = dict(winner=name, validation=winner['selected_validation'],
                    checkpoint_sha256=winner['checkpoint_sha256'],
                    candidate_run_ids={r['experiment']: r['run_id'] for r in results},
                    selection_uses_test=False, xm3600_used=False,
                    rule='val_t2i_r1_then_mean_r1; exact ties retain first candidate in name order')
    selection = output / 'final_selection.json'
    if selection.exists() and read_json(selection) != decision:
        raise ValueError('Committed finalist selection changed')
    write_json(selection, decision)
    dest = output / 'finalist_evaluation'
    dest.mkdir(exist_ok=True)
    if (dest / 'evaluation.json').exists():
        result = read_json(dest / 'evaluation.json')
        assert result['selection'] == decision
        print(result, flush=True)
        return
    data = base.Inputs(output)
    data.verify()
    checkpoint = output / name / 'best.pt'
    assert file_sha(checkpoint) == winner['checkpoint_sha256']
    ctx = context(dest)
    saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
    if name == 'F_TinyCLIP61':
        # Same pinned weights as training, new test-only feature cache after selection.
        import shutil
        data.output = dest / 'scaling'
        (data.output / 'tinyclip61').mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output / 'tinyclip61/model_lock.json', data.output / 'tinyclip61/model_lock.json')
        data.images, data.expected_sha = ctx.manifest['images'], ctx.expected_sha
        images = follow.scaling_cache(data)
    elif winner['vision_blocks']:
        def loader(key, lock):
            vision, prep, stats = base.load_vision(data)
            spec = follow.best_recipe(winner)
            suffix = follow.suffix_for(data, spec)
            suffix.load_state_dict(saved['vision_suffix'], strict=True)
            vision.vision_model.encoder.layers[-2:] = suffix.layers
            vision.vision_model.post_layernorm = suffix.post_layernorm
            vision.visual_projection = suffix.projection
            vision.eval().requires_grad_(False)
            stats.update(adapted_checkpoint_sha256=file_sha(checkpoint), feature_dim=512)
            return vision, prep, lambda pixels: vision(pixel_values=pixels).image_embeds, stats
        images, _ = cache_backbone('tinyclip-finalist', data.lock, ctx.manifest['images'], ctx.expected_sha,
                                    dest / 'feature_cache', loader=loader)
    else:
        images = ctx.images
    model = base.projector(images.shape[1], winner['hidden'])
    model.load_state_dict(saved['projector'], strict=True)
    model.eval()
    ctx.load_test()
    with torch.inference_mode():
        projected = model(images)
    local, predictions = evaluate_features(ctx.datasets['mscoco_t2i'], ctx.manifest['images'], projected,
                                           ctx.manifest['texts'], ctx.texts)
    global_metrics = evaluate_graph(ctx.test_graph, projected, ctx.texts)
    result = dict(selection=decision, local_mscoco=local, global_mscoco=global_metrics,
                  test_used_after_selection=True, xm3600_used=False,
                  historical_frozen_reference=read_json(ROOT / 'training/runs/clean-v1/FULL_CLEAN/seed-42/result.json')['benchmark']['local']['mscoco_t2i'])
    write_json(dest / 'evaluation.json', result)
    from pl_mmb_training.runner import write_records
    write_records(dest / 'mscoco_t2i.predictions.jsonl', predictions)
    write_json(ROOT / 'reports/tinyclip_adaptation_finalist.json', result)
    data.verify()
    print(result, flush=True)


if __name__ == '__main__':
    main()
