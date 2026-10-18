"""English control for MobileCLIP loading, separate from Polish leaderboard."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pl_mmb_benchmark.models import Encoder, MODELS, check_cache
from pl_mmb_benchmark.data import ROOT, REVISION, translation_path


def main():
    check_cache()
    import torch
    from datasets import Dataset
    from huggingface_hub import hf_hub_download
    torch.set_num_threads(2)
    parquet = hf_hub_download('TIGER-Lab/MMEB-eval', 'MSCOCO_t2i/test-00000-of-00001.parquet',
                              repo_type='dataset', revision=REVISION)
    source = Dataset.from_parquet(parquet)
    indices = list(range(0, 1000, 100))
    patches = {r['metadata']['row_index']: r for r in
               map(json.loads, translation_path('mscoco_t2i').read_text().splitlines())}
    encoder = Encoder(MODELS['mobileclip2-s2'], 16)
    images = [str(ROOT / '.cache/huggingface/pl_mmb_images/mmeb/MSCOCO_t2i' /
                  source[i]['tgt_img_path'][0]) for i in indices]
    image_features = encoder.encode(images, images=True)
    result = {'numerics': 'mobile-fp32-unit-range-reparam-v3', 'indices': indices}
    for language in ('en', 'pl'):
        texts = [source[i]['qry_text'] if language == 'en' else patches[i]['translated']['qry_text'] for i in indices]
        scores = encoder.encode(texts) @ image_features.T
        result[language] = {'accuracy_10_candidates': float((scores.argmax(1) == torch.arange(10)).float().mean()),
                            'diagonal_scores': scores.diag().tolist()}
    print(json.dumps(result, indent=2, allow_nan=False))
    assert result['en']['accuracy_10_candidates'] >= .5, 'English control failed: inspect model loading'


if __name__ == '__main__':
    main()
