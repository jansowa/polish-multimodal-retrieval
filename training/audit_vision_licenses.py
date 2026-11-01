"""Snapshot primary license evidence without changing any training configuration."""
import sys
from pathlib import Path
import requests
from huggingface_hub import hf_hub_download, HfApi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pl_mmb_training.common import file_sha, read_json, write_json

if __name__ == '__main__':
    root = ROOT / 'training/runs/vision-commercial-five/licenses'
    root.mkdir(parents=True, exist_ok=True)
    target = root / 'audit.json'
    if target.exists():
        print('License snapshot already present')
        sys.exit(0)
    entries = {}
    base = read_json(ROOT / 'training/runs/e1-full/model_lock.json')
    for key, repo, revision in [('PolDense', base['text']['model'], base['text']['revision']),
                                 ('SigLIP2', base['vision']['model'], base['vision']['revision']),
                                 ('MobileCLIP2', 'apple/MobileCLIP2-S2', '72424e7025436db18f15c3eff6ee8c7c15ad4481')]:
        info = HfApi().model_info(repo, revision=revision)
        card = Path(hf_hub_download(repo, 'README.md', revision=revision))
        entries[key] = dict(repo=repo, revision=revision, license=info.card_data.to_dict().get('license'),
                            source=f'https://huggingface.co/{repo}/blob/{revision}/README.md',
                            path=str(card), sha256=file_sha(card))
    for key, repo, name in [('TinyCLIP-code', 'microsoft/Cream', 'TinyCLIP/LICENSE'),
                             ('MobileCLIP-models', 'apple-aiml-research/ml-mobileclip', 'LICENSE_MODELS')]:
        response = requests.get(f'https://api.github.com/repos/{repo}/commits/main', timeout=30)
        response.raise_for_status()
        revision = response.json()['sha']
        url = f'https://raw.githubusercontent.com/{repo}/{revision}/{name}'
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        path = root / f'{key}.txt'
        path.write_bytes(response.content)
        entries[key] = dict(repo=repo, revision=revision, source=url, path=str(path), sha256=file_sha(path))
    url = 'https://ai.google.dev/gemma/terms'
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    path = root / 'gemma-terms.html'
    path.write_bytes(response.content)
    entries['Gemma-terms'] = dict(source=url, path=str(path), sha256=file_sha(path),
                                note='Terms retrieved during study; PolDense declares Gemma license at pinned revision')
    write_json(target, entries)
    write_json(ROOT / 'reports/training_vision_license_sources.json', entries)
    print('Primary license evidence archived', flush=True)
