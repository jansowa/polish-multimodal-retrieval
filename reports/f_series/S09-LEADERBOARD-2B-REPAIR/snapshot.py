"""Small receipt of the running repair; never reads complete manifests or logs."""
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from retrieval_benchmark.core import identity,local_path,read,sha,write

folder=Path(__file__).parent
registration=folder/'registration.json'
reg=read(registration)
run=ROOT/'training/runs/f_series/leaderboard-2b-repair-20261101'
artifacts=[]
for job in reg['jobs']:
    for name,mode in job['missing']:
        rid=identity(dict(registration=sha(registration),model=job['model_id'],dataset=name,mode=mode))
        path=local_path(reg['config']['output'])/rid/'result.json'
        if path.exists():
            value=read(path)
            assert sha(path.parent/'scores.json')==value['public_result']['source_sha256']
            artifacts.append(dict(repository=job['repository'],dataset=name,mode=mode,path=str(path),sha256=sha(path)))
smokes=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),result=read(p)) for p in sorted(run.glob('*/adapter-smoke.json'))]
write(folder/'launch-receipt.json',dict(registration_sha256=sha(registration),ticket=read(run/'JOB.json'),
    state=read(run/'STATE.json'),adapter_smokes=smokes,completed_cells=artifacts,
    protected_sha256={p:sha(ROOT/p) for p in reg['protected']},
    publisher=read(run/'PUBLISH_STATUS.json') if (run/'PUBLISH_STATUS.json').exists() else dict(status='waiting_for_first_complete_model')))
print('REPAIR_RECEIPT',len(smokes),'passed adapters;',len(artifacts),'completed cells')
