"""Download the four requested official checkpoints, recording immutable hashes."""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pl_mmb_training.common import read_json, write_json, file_sha

CACHE = Path.home() / '.cache/huggingface/pl_mmb_training/vision_four'
REVISIONS = {'dinov2-small': ('facebook/dinov2-small', 'ed25f3a31f01632728cabb09d1542f84ab7b0056'),
             'mobileclip2-s2': ('apple/MobileCLIP2-S2', '72424e7025436db18f15c3eff6ee8c7c15ad4481')}


def acquire(url, path):
    import requests
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size:
        return path
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                tmp = path.with_suffix('.tmp')
                with tmp.open('wb') as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
                tmp.replace(path)
            return path
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2)


def main():
    from huggingface_hub import hf_hub_download
    CACHE.mkdir(parents=True, exist_ok=True)
    lock_path = ROOT / 'training/runs/vision-four/model_lock.json'
    if lock_path.exists():
        lock = read_json(lock_path)
        for model in lock.values():
            for path, sha in model['files'].items():
                if file_sha(path) != sha:
                    raise ValueError('Checkpoint changed: ' + path)
        print('Four official checkpoint locks verified', flush=True)
        return
    lock = {}
    for key, filenames in (('dinov2-small', ['config.json', 'preprocessor_config.json', 'model.safetensors']),
                            ('mobileclip2-s2', ['config.json', 'mobileclip2_s2.pt'])):
        repo, revision = REVISIONS[key]
        paths = [Path(hf_hub_download(repo, f, revision=revision)) for f in filenames]
        lock[key] = dict(model=repo, revision=revision, files={str(p): file_sha(p) for p in paths})
        print('Fetched ' + key, flush=True)
    path = acquire('https://download.pytorch.org/models/convnext_tiny-983f1562.pth', CACHE / 'convnext_tiny-983f1562.pth')
    if not file_sha(path).startswith('983f1562'):
        raise ValueError('Official ConvNeXt checkpoint checksum mismatch')
    lock['convnext-tiny'] = dict(model='torchvision/convnext_tiny', revision='IMAGENET1K_V1',
                                 files={str(path): file_sha(path)})
    print('Fetched ConvNeXt-Tiny', flush=True)
    revision = 'da316d82636a7a7356835ef224b13d5f3ace0489'
    weights = acquire('https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vits_16/checkpoint_teacher.pth',
                      CACHE / 'ibot_vits16_teacher.pth')
    source = acquire(f'https://raw.githubusercontent.com/bytedance/ibot/{revision}/models/vision_transformer.py', CACHE / 'ibot_vision_transformer.py')
    license_path = acquire(f'https://raw.githubusercontent.com/bytedance/ibot/{revision}/LICENSE', CACHE / 'ibot_LICENSE')
    lock['ibot-vits16'] = dict(model='bytedance/ibot-vits16-teacher', revision=revision,
                              files={str(p): file_sha(p) for p in (weights, source, license_path)})
    write_json(lock_path, lock)
    print('All four official checkpoints downloaded and locked', flush=True)


if __name__ == '__main__':
    main()
