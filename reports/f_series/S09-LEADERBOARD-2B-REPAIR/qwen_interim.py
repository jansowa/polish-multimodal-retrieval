"""Compare saved Qwen cells with peers on exactly the same completed tasks."""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from retrieval_benchmark.core import identity,local_path,read,sha,write,public_check
from retrieval_benchmark.space.view import default_tasks,overview

folder=Path(__file__).parent
registration=folder/'registration.json'; reg=read(registration)
raw=(ROOT/'retrieval_benchmark/space/results.json').read_bytes(); doc=json.loads(raw)
job=next(j for j in reg['jobs'] if j['repository']=='Qwen/Qwen3-VL-Embedding-2B')
keys={(r['model_id'],r['dataset_id'],r['prompt_mode'],r['profile']) for r in doc['results']}
sources=[]
for name,mode in job['missing']:
    rid=identity(dict(registration=sha(registration),model=job['model_id'],dataset=name,mode=mode))
    path=local_path(reg['config']['output'])/rid/'result.json'
    if not path.exists(): continue
    value=read(path); row=value['public_result']
    assert sha(path.parent/'scores.json')==row['source_sha256']
    sources.append(dict(dataset=name,mode=mode,path=str(path),sha256=sha(path),scores_sha256=row['source_sha256']))
    doc['models'].setdefault(job['model_id'],value['model_metadata'])
    key=(row['model_id'],row['dataset_id'],row['prompt_mode'],row['profile'])
    if key not in keys: doc['results'].append(row); keys.add(key)
public_check(doc)
reports=[]
prompt_modes=('v2_no_instructions','author_recommended')
completed_by_mode=[{r['dataset_id'] for r in doc['results'] if r['model_id']==job['model_id'] and r['prompt_mode']==mode}
                   for mode in prompt_modes]
shared=set.intersection(*completed_by_mode)
for mode in prompt_modes:
    tasks=[n for n in default_tasks(doc) if n in shared]
    if not tasks: continue
    columns=['total','license','average']+['task:'+n for n in tasks]
    table=overview(doc,mode=mode,average_tasks=tasks,visible_columns=columns,show_controls=True)
    rows=[dict(model=r[0],**dict(zip(columns,r[1:]))) for r in table['rows'] if r[3] is not None]
    reports.append(dict(mode=mode,tasks=tasks,ranking=rows))
    print('MODE',mode,'TASKS',len(tasks))
    for rank,row in enumerate(rows[:10],1):
        print(rank,row['model'],round(row['total'],2),row['license'],round(row['average'],2))
    qwen=next(r for r in rows if r['model']==job['repository'])
    print('QWEN_TASKS',json.dumps(qwen,ensure_ascii=False))
write(folder/'qwen-interim.json',dict(public_results_sha256=hashlib.sha256(raw).hexdigest(),
      registration_sha256=sha(registration),saved_sources=sources,comparisons=reports,
      metric='recall@1',aggregation='mean of both directions and the identical completed tasks; full XTD10 excluded'))
