"""Read-only metadata inventory for the explicitly requested reference models."""
import concurrent.futures
import importlib.util
import json
import ssl
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrieval_benchmark.core import read, write, sha

REPOSITORIES = [
    'Qwen/Qwen3-VL-Embedding-2B', 'timm/ViT-B-16-SigLIP-i18n-256',
    'alphaedge-ai/siglip2-base-patch16-512-pol-32768', 'google/siglip2-large-patch16-512',
    'google/siglip2-so400m-patch16-512', 'Alibaba-NLP/gme-Qwen2-VL-2B-Instruct',
    'alphaedge-ai/siglip2-base-patch16-512-pol-16384', 'BAAI/BGE-VL-large',
    'nomic-ai/nomic-embed-vision-v1.5', 'Qwen/Qwen3-VL-Embedding-8B',
    'alphaedge-ai/aclip-ViT-B-32-multilingual-v1-pl-16384',
    'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus',
    'laion/CLIP-ViT-H-14-laion2B-s32B-b79K', 'laion/CLIP-ViT-g-14-laion2B-s12B-b42K',
    'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
]
DEST = ROOT/'reports/leaderboard_requested_20261101'
CONTEXT = ssl.create_default_context(cafile=str(ROOT/'combined-ca.pem'))


def fetch(url):
    with urllib.request.urlopen(url, context=CONTEXT, timeout=60) as response:
        return response.read()


def inspect(repo):
    try:
        info = json.loads(fetch('https://huggingface.co/api/models/'+repo+'?blobs=true'))
        revision = info['sha']
        folder = DEST/'sources'/repo.replace('/', '--')
        folder.mkdir(parents=True, exist_ok=True)
        write(folder/'metadata.json', info)
        for filename in ('README.md', 'config.json', 'modules.json'):
            try:
                (folder/filename).write_bytes(fetch('https://huggingface.co/'+repo+'/resolve/'+revision+'/'+filename))
            except Exception:
                pass
        return dict(repository=repo, revision=revision, metadata_sha256=sha(folder/'metadata.json'),
                    files=[dict(name=f['rfilename'], bytes=f.get('size', 0), lfs=f.get('lfs'))
                           for f in info.get('siblings', [])],
                    license=(info.get('cardData') or {}).get('license'))
    except Exception as error:
        return dict(repository=repo, error=str(error))


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        jobs = list(pool.map(inspect, REPOSITORIES + ['nomic-ai/nomic-embed-text-v1.5']))
    write(DEST/'inventory.json', jobs)
    import torch, transformers
    print('ENV', transformers.__version__, torch.__version__,
          {x: bool(importlib.util.find_spec(x)) for x in ['qwen_vl_utils', 'sentence_transformers', 'open_clip']})
    doc = read(ROOT/'retrieval_benchmark/space/results.json')
    for job in jobs:
        repo = job['repository']
        known = [k for k,v in doc['models'].items() if v.get('repository') == repo]
        results = [r for r in doc['results'] if r['model_id'] in known]
        print(repo, job.get('revision', job.get('error')), 'results', len(results),
              'weights', [(f['name'], f['bytes']) for f in job.get('files', []) if f['name'].endswith(('.safetensors', '.bin'))])
