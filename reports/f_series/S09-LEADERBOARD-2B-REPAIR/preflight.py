"""Inspect pinned pooling configs and original failures without loading weights."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from retrieval_benchmark.core import read,write,sha
from retrieval_benchmark.inspect_requested import fetch

folder=Path(__file__).parent
old=ROOT/'reports/leaderboard_requested_20261101/registration.json'
reg=read(old)
jobs=[j for j in reg['jobs'] if j['repository'] in ('Qwen/Qwen3-VL-Embedding-2B','Alibaba-NLP/gme-Qwen2-VL-2B-Instruct')]
evidence=[]
for job in jobs:
    repo,revision=job['repository'],job['revision']
    url='https://huggingface.co/'+repo+'/resolve/'+revision+'/1_Pooling/config.json'
    raw=fetch(url)
    config=json.loads(raw)
    target=folder/(job['backend']+'-pooling.json')
    target.write_bytes(raw)
    evidence.append(dict(repository=repo,revision=revision,pooling=config,pooling_sha256=sha(target),source=url))
write(folder/'preflight.json',dict(previous_registration_sha256=sha(old),models=evidence))
for item in evidence: print(item['repository'],json.dumps(item['pooling']))
