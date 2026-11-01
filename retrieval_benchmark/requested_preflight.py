"""Small metadata and interpreter probes; never loads model weights."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrieval_benchmark.inspect_requested import DEST, fetch, inspect
from retrieval_benchmark.core import read, write

if __name__ == '__main__':
    inventory = read(DEST/'inventory.json')
    inventory = [j for j in inventory if '/aclip-' not in j['repository']]
    for repo in ['alphaedge-ai/clip-ViT-B-32-multilingual-v1-pl-16384',
                 'openai/clip-vit-base-patch32', 'timm/vit_base_patch16_plus_clip_240.laion400m_e31']:
        inventory.append(inspect(repo))
    write(DEST/'inventory.json', inventory)
    for job in inventory:
        if 'error' in job:
            continue
        folder = DEST/'sources'/job['repository'].replace('/', '--')
        for item in job['files']:
            name = item['name']
            if name.endswith('.py') and '/' not in name:
                (folder/name).write_bytes(fetch('https://huggingface.co/'+job['repository']+'/resolve/'+job['revision']+'/'+name))
    # The author's Qwen implementation documents chat formatting and pooling.
    source = json.loads(fetch('https://api.github.com/repos/QwenLM/Qwen3-VL-Embedding/commits/main'))['sha']
    code = fetch('https://raw.githubusercontent.com/QwenLM/Qwen3-VL-Embedding/'+source+'/src/models/qwen3_vl_embedding.py')
    (DEST/'qwen3_author.py').write_bytes(code)
    write(DEST/'qwen3_author_source.json', dict(repository='QwenLM/Qwen3-VL-Embedding', revision=source))
    print('QWEN_SOURCE', source)
    for env in ['pl-mmb-benchmark', 'pl-mmb-legacy-cu130', 'pl-mmb-modern']:
        subprocess.run([str(Path.home()/'.venvs'/env/'bin/python'), '-c',
            'import transformers,sentence_transformers; print(transformers.__version__,sentence_transformers.__version__)'], check=True)
    print('PREFLIGHT_METADATA_COMPLETE')
