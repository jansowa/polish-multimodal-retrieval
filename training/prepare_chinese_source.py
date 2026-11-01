"""Pin the official Chinese-CLIP image forward and preprocessing source."""
import sys
from pathlib import Path
import requests
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pl_mmb_training.common import write_json, read_json, file_sha

if __name__ == '__main__':
    root = ROOT / 'training/runs/vision-commercial-five/official'
    root.mkdir(parents=True, exist_ok=True)
    target = root / 'lock.json'
    if target.exists():
        for path, sha in read_json(target)['files'].items():
            assert file_sha(path) == sha
    else:
        response = requests.get('https://api.github.com/repos/OFA-Sys/Chinese-CLIP/commits/master', timeout=30)
        response.raise_for_status()
        revision = response.json()['sha']
        files = {}
        for name in ['cn_clip/clip/model.py', 'cn_clip/clip/utils.py', 'MIT-LICENSE.txt']:
            response = requests.get(f'https://raw.githubusercontent.com/OFA-Sys/Chinese-CLIP/{revision}/{name}', timeout=30)
            response.raise_for_status()
            path = root / name.replace('/', '_')
            path.write_bytes(response.content)
            files[str(path)] = file_sha(path)
        write_json(target, dict(repository='OFA-Sys/Chinese-CLIP', revision=revision, files=files))
    print('Official Chinese-CLIP source and license pinned', flush=True)
