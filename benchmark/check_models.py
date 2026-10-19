"""Small paired EN/PL control; output stays separate from the leaderboard."""
import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pl_mmb_benchmark.models import Encoder, MODELS, check_cache, worker_python
from pl_mmb_benchmark.data import ROOT, REVISION, translation_path
from pl_mmb_benchmark.protocol import PROTOCOL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--internal-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    check_cache()
    if not args.internal_worker:
        for alias in args.models:
            repo = MODELS.get(alias, alias)
            executable = worker_python(repo, sys.executable)
            subprocess.run([executable, '-u', str(Path(__file__).resolve()), '--internal-worker', '--models', repo], check=True)
        return
    import torch
    from datasets import Dataset
    from huggingface_hub import hf_hub_download
    torch.set_num_threads(4)
    parquet = hf_hub_download('TIGER-Lab/MMEB-eval', 'MSCOCO_t2i/test-00000-of-00001.parquet', repo_type='dataset', revision=REVISION)
    source = Dataset.from_parquet(parquet)
    indices = list(range(0, 1000, 100))
    patches = {r['metadata']['row_index']: r for r in map(json.loads, translation_path('mscoco_t2i').read_text().splitlines())}
    images = [str(ROOT / '.cache/huggingface/pl_mmb_images/mmeb/MSCOCO_t2i' / source[i]['tgt_img_path'][0]) for i in indices]
    for alias in args.models:
        repo = MODELS.get(alias, alias)
        encoder = Encoder(repo, 16)
        visual = encoder.encode(images, images=True)
        result = dict(model=repo, parameters=encoder.parameter_count, dimension=visual.shape[1],
                      text_max_length=encoder.text_max_length, precision=encoder.precision, indices=indices)
        for language in ('en', 'pl'):
            texts = [source[i]['qry_text'] if language == 'en' else patches[i]['translated']['qry_text'] for i in indices]
            features = encoder.encode(texts)
            assert features.shape == visual.shape
            similarities = torch.nn.functional.normalize(features, dim=-1) @ torch.nn.functional.normalize(visual, dim=-1).T
            result[language] = dict(accuracy_10_candidates=float((similarities.argmax(1) == torch.arange(10)).float().mean()),
                                    diagonal_scores=similarities.diag().tolist())
        if encoder.native_multimodal:
            joint = encoder.encode_joint([patches[i]['translated']['qry_text'] for i in indices[:2]], images[:2])
            assert joint.shape == (2, visual.shape[1]) and torch.isfinite(joint).all()
            result['native_joint_checked'] = True
        print(json.dumps(result, allow_nan=False), flush=True)
        target = ROOT / 'benchmark/results/checks' / PROTOCOL / encoder.revision
        target.mkdir(parents=True, exist_ok=True)
        (target / 'paired_control.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        del encoder
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
