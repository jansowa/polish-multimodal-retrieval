"""Fetch only the five requested candidates and pin checkpoint/license provenance."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pl_mmb_training.common import write_json, read_json, file_sha

MODELS = {
    'tinyclip-39m': ('wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M', '07a4b0bc751cb64fecd2b661c048c1dd98d69444', ['config.json', 'preprocessor_config.json', 'model.safetensors']),
    'tinyclip-40m': ('wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M', '95ec8197b3f2fe7f747865c61ca556cf0768b2f7', ['config.json', 'preprocessor_config.json', 'model.safetensors']),
    'chinese-clip-rn50': ('OFA-Sys/chinese-clip-rn50', '717ba215769231e53b9b7c6b9d329b9cc5944418', ['clip_cn_rn50.pt']),
    'rn50-cc12m': ('timm/resnet50_clip.cc12m', 'd72a2fdb0278498f216ca970010d730b95249beb', ['open_clip_config.json', 'open_clip_model.safetensors']),
    'rn50-yfcc15m': ('timm/resnet50_clip.yfcc15m', 'c94a9e3bde942b58c1cd13bb0a03d3c1ff4cb056', ['open_clip_config.json', 'open_clip_model.safetensors']),
}

def main():
    from huggingface_hub import hf_hub_download, HfApi
    root = ROOT / 'training/runs/vision-commercial-five'
    lock_path = root / 'model_lock.json'
    if lock_path.exists():
        for entry in read_json(lock_path).values():
            for path, sha in entry['files'].items():
                if file_sha(path) != sha:
                    raise ValueError('Checkpoint/license artifact changed: ' + path)
        print('All five immutable checkpoint locks verified', flush=True)
        return
    lock = {}
    for key, (repo, revision, filenames) in MODELS.items():
        info = HfApi().model_info(repo, revision=revision)
        paths = [Path(hf_hub_download(repo, name, revision=revision)) for name in ['README.md', *filenames]]
        license_name = info.card_data.to_dict()['license']
        if license_name not in ('mit', 'apache-2.0'):
            raise ValueError('Review unexpected weight license before commercial eligibility')
        lock[key] = dict(model=repo, revision=revision, files={str(p): file_sha(p) for p in paths},
                         license=dict(declared_weights_license=license_name,
                                      source=f'https://huggingface.co/{repo}/blob/{revision}/README.md',
                                      commercial_screen='permissive-model-card-declaration',
                                      scope='Model-card declaration for weights; not a clearance of pretraining data or product rights'))
        write_json(root / 'download_progress.json', lock)
        print('FETCHED ' + key + ' license=' + license_name, flush=True)
    write_json(lock_path, lock)

if __name__ == '__main__':
    main()
