"""Verify completed study provenance and checkpoint choices; never evaluate a dataset."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from pl_mmb_training import tinyclip_adaptation as base
from pl_mmb_training.common import ROOT, file_sha, read_json, write_json


def main():
    data = base.Inputs(base.OUT)
    data.verify()
    meta = read_json(base.OUT / 'cache/complete.json')
    for filename, sha in meta['files'].items():
        assert file_sha(base.OUT / 'cache' / filename) == sha
    completed = {}
    for p in sorted(base.OUT.glob('*/result.json')):
        if p.parent.name == 'finalist_evaluation':
            continue
        r = read_json(p)
        assert r['status'] == 'complete'
        assert not r['selection_uses_test'] and r['test_mscoco'] is None
        chosen = max(r['history'][1:], key=base.checkpoint_key)
        assert chosen['epoch'] == r['best_epoch']
        assert chosen['validation'] == r['selected_validation']
        assert chosen['validation']['graph_sha256'] == data.graph['sha256']
        checkpoint = p.parent / 'best.pt'
        assert file_sha(checkpoint) == r['checkpoint_sha256']
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
        assert saved['run_id'] == r['run_id'] and saved['epoch'] == r['best_epoch']
        assert r['trainable_params'] > 0 and r['peak_vram_bytes'] > 0
        for row in r['history'][1:]:
            assert all(math.isfinite(row[k]) for k in ('train_loss', 'validation_loss', 'gradient_norm_mean'))
        completed[r['experiment']] = dict(best_epoch=r['best_epoch'], checkpoint_verified=True,
                                          validation_selection_verified=True, logged_epochs=len(r['history'])-1)
    assert set(completed) == {'A1', 'A2', 'B2', 'C1', 'C2', 'D_LoRA', 'E_KD_0.1', 'E_KD_0.3', 'F_TinyCLIP61'}
    for name in ('C1', 'C2'):
        config = read_json(base.OUT / name / 'config.json')
        assert config['spec']['vision_lr'] == 1e-5
    teacher = read_json(base.OUT / 'teacher/complete.json')
    assert file_sha(base.OUT / 'teacher/features.pt') == teacher['sha256']
    assert teacher['identity']['model']['model'] == 'google/siglip2-base-patch16-256'
    finalist = read_json(base.OUT / 'finalist_evaluation/evaluation.json')
    assert not finalist['selection']['selection_uses_test'] and not finalist['xm3600_used']
    results = [read_json(base.OUT / name / 'result.json') for name in sorted(completed)]
    assert finalist['selection']['winner'] == max(results, key=base.key)['experiment']
    result = dict(status='passed', runs=completed, unchanged_inputs=data.hashes,
                  train_rows=len(data.train), validation_rows=len(data.val),
                  validation_graph_sha256=data.graph['sha256'],
                  finalist=finalist['selection']['winner'], xm3600_used=False)
    write_json(ROOT / 'reports/tinyclip_adaptation_verification.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
