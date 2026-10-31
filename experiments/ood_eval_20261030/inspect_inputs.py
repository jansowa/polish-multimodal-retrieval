"""Read-only inspect B metadata, local CPU packages and cache directory names."""
import importlib.util
import json
from pathlib import Path
from prepare import OUT, read

print('CPU_PACKAGES', {m: bool(importlib.util.find_spec(m)) for m in ('pyarrow', 'PIL', 'numpy', 'torch', 'transformers')})
for path in (Path.home() / '.cache/huggingface/hub', Path('/mnt/c/Users/jsowa/.cache/huggingface/hub')):
    if path.exists():
        print('CACHE_NAMES', str(path), [p.name for p in path.iterdir() if p.is_dir() and any(s in p.name.lower() for s in ('nllb', 'marian', 'helsinki', 'qwen', 't5', 'm2m'))])
extra = OUT / 'additional_metadata.json'
if extra.exists():
    for r in read(extra)['sources']:
        print('EXTRA', r['url'], r['status'])
        if r['status'] == 'fetched':
            if 'docci.py' in r['url']:
                print(r['text'])
            else:
                print([(item.get('name'), item.get('size'), item.get('download_url')) for item in json.loads(r['text'])])
print('MARKERS', sorted(p.name for p in (OUT.parent / 'data_ood_coordination_20261030').glob('*.json')))
