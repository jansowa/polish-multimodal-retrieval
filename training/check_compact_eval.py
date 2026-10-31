"""Prove the compact snapshot reproduces all existing sigmoid/SigLIP2 ranks."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pl_mmb_training.common import read_json, write_json, file_sha
from pl_mmb_training.evaluation_snapshot import compact_snapshot
from pl_mmb_training.evaluation import evaluate_features


if __name__ == '__main__':
    import torch
    torch.set_num_threads(4)
    root = ROOT / 'training/runs/vision-four'
    reference = root / 'reference'
    lock = read_json(reference / 'lock.json')
    source = reference / 'artifacts/evaluation_examples.json'
    assert file_sha(source) == lock['files']['evaluation_examples.json']
    baseline = ROOT / 'training/runs/e1-full'
    manifest = read_json(baseline / 'cache.json')
    e0 = read_json(baseline / 'e0.json')
    compact = root / 'common/evaluation_compact.json'
    examples = compact_snapshot(source, compact, manifest['images'], manifest['texts'],
                                lock['files']['evaluation_examples.json'], file_sha(baseline / 'cache.json'),
                                {k: info['examples'] for k, info in e0['evaluation'].items()})
    print(f'Compact snapshot: {source.stat().st_size} -> {compact.stat().st_size} bytes', flush=True)
    images = torch.load(manifest['image_features']['path'], weights_only=True)
    texts = torch.load(manifest['text_features']['path'], weights_only=True)
    checkpoint = torch.load(reference / 'artifacts/E2b-seed-42/projector.pt', weights_only=True)
    model = torch.nn.Linear(images.shape[1], texts.shape[1])
    model.load_state_dict(checkpoint['state_dict'])
    with torch.inference_mode():
        projected = model(images)
    original = read_json(reference / 'artifacts/E2b-seed-42/result.json')
    verified = {}
    for key, rows in examples.items():
        metrics, predictions = evaluate_features(rows, manifest['images'], projected, manifest['texts'], texts)
        before = [json.loads(line) for line in (reference / f'artifacts/E2b-seed-42/{key}.predictions.jsonl').read_text().splitlines()]
        assert [p['rank'] for p in predictions] == [p['rank'] for p in before]
        assert metrics == original['benchmark']['local'][key]
        verified[key] = dict(queries=len(rows), every_rank_identical=True, every_metric_identical=True)
        print(key + ': all ranks and metrics identical', flush=True)
    write_json(root / 'compact_evaluation_verification.json', dict(source_bytes=source.stat().st_size,
               compact_bytes=compact.stat().st_size, verified=verified))
