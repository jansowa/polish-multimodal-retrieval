"""Verify completed runs against their immutable configs and sigmoid reference."""
import sys
import shutil
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pl_mmb_training.common import TASKS, digest, read_json, file_sha, write_json
from pl_mmb_training.commercial_backbones import ORDER
from pl_mmb_training.commercial_study import replication_decisions

if __name__ == '__main__':
    root = ROOT / 'training/runs/vision-commercial-five'
    summary = read_json(root / 'summary.json')
    assert summary['status'] == 'complete' and tuple(summary['tested_order']) == ORDER
    decisions = read_json(root / 'replication_decisions.json')
    assert decisions == replication_decisions(summary['models'], summary['config'])
    reference = summary['reference']['runs'][0]['metadata']
    checks = []
    original_features = read_json(ROOT / 'training/runs/e1-full/cache.json')
    for key, group in summary['models'].items():
        features = read_json(root / 'models' / key / 'feature_cache.json')
        assert file_sha(features['path']) == features['sha256']
        assert [run['seed'] for run in group['runs']] == decisions[key]['seeds']
        for run in group['runs']:
            dest = root / 'models' / key / f'E2b-seed-{run["seed"]}'
            identity = read_json(dest / 'config.json')
            assert digest(identity) == run['run_id']
            for name, sha in identity['source'].items():
                assert file_sha(ROOT / name) == sha, ('Executed source changed', name)
                archive = root / 'source' / name
                archive.parent.mkdir(parents=True, exist_ok=True)
                if not archive.exists():
                    shutil.copyfile(ROOT / name, archive)
                assert file_sha(archive) == sha
            assert file_sha(dest / 'projector.pt') == run['checkpoint_sha256']
            metadata = run['metadata']
            assert metadata['vision_cache_sha256'] == features['sha256']
            assert metadata['text_cache_sha256'] == original_features['text_features']['sha256']
            for field in ('dataset_size', 'validation_size', 'number_of_unique_examples_seen',
                          'number_of_unique_source_records_seen', 'unique_target_images', 'example_presentations',
                          'optimization_steps', 'text_model', 'projector_type', 'loss', 'batch_size',
                          'learning_rate', 'weight_decay', 'scheduler', 'frozen_text', 'frozen_vision',
                          'checkpoint_selection', 'data_manifest_sha256'):
                assert metadata[field] == reference[field], (key, run['seed'], field)
            assert len(run['history']) == 21
            assert [epoch['epoch'] for epoch in run['history']] == list(range(21))
            selected = min(run['history'], key=lambda h: h['validation_loss'])
            assert selected['epoch'] == run['checkpoint_epoch']
            assert selected['validation_loss'] == run['validation_loss']
            assert set(run['benchmark']['scores']) == {*TASKS, 'AVG'}
            scores = run['benchmark']['scores']
            assert abs(scores['AVG'] - sum(scores[t] for t in TASKS)/4) < 1e-12
            state = torch.load(dest / 'projector.pt', map_location='cpu', weights_only=True)['state_dict']
            assert state['weight'].shape == (384, group['encoder']['feature_dim'])
            assert sum(t.numel() for t in state.values()) == metadata['projector_parameter_count']
            assert metadata['trainable_parameter_count'] == metadata['projector_parameter_count'] + 2
            checks.append(dict(backbone=key, seed=run['seed'], run_id=run['run_id'], verified=True))
    result = dict(status='passed', checks=checks, exactly_five_backbones=True,
                  unchanged_training_recipe=True, minimum_validation_checkpoint=True,
                  replication_decisions_verified=True)
    # Additional runtime wrappers affect process lifetime/cache loading only.
    # Keep their exact source alongside the source identities in each run.
    result['runtime_sources'] = {}
    for name in ('training/observe_commercial_study.py', 'training/cache_commercial_backbone.py'):
        archive = root / 'source' / name
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, archive)
        result['runtime_sources'][name] = file_sha(archive)
    write_json(root / 'verification.json', result)
    write_json(ROOT / 'reports/training_vision_commercial_verification.json', result)
    print(f'PASS: {len(checks)} runs, exactly five backbones, fixed data/splits/loss/budget, checkpoints and metrics')
