"""Snapshot license classifications and verify the actual localhost leaderboard."""
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from retrieval_benchmark.core import read,write,sha
from retrieval_benchmark.space.view import default_columns,default_tasks,license_category,organizations,overview,pareto_data

folder=Path(__file__).parent
raw=(ROOT/'retrieval_benchmark/space/results.json').read_bytes()
doc=json.loads(raw)
if '--live' in sys.argv:
    from gradio_client import Client
    for attempt in range(20):
        try:
            with urllib.request.urlopen('http://127.0.0.1:7865/config',timeout=5) as stream: config=json.load(stream)
            break
        except urllib.error.URLError:
            if attempt==19: raise
            time.sleep(.5)
    assert any(c['props'].get('label')=='Licencje' for c in config['components'])
    args=['recall@1','v2_no_instructions','historical-native-v2',0,False,False,
          default_columns(doc),default_tasks(doc),0,organizations(doc),[],'commercial',False,True]
    client=Client('http://127.0.0.1:7865',verbose=False)
    checks=[]
    for category in ['commercial','noncommercial','unknown']:
        args[11]=category
        result=client.predict(*args,api_name='/leaderboard')
        table=result[0].get('value',result[0])
        expected=overview(doc,license_filter=category)['model_data']
        assert len(table['data'])==len(expected)==result[2].count('data-model-id=')
        for point in expected: assert 'data-model-id="'+point['id']+'"' in result[2]
        checks.append(dict(category=category,visible_models=len(expected),graph_matches_table=True))
    write(folder/'filter-live-validation.json',dict(url='http://localhost:7865/',status='passed',checks=checks))
    print('LIVE_LICENSE_FILTER_OK')
else:
    checks=[]
    for category in ['all','commercial','noncommercial','unknown']:
        table=overview(doc,license_filter=category)
        graph=pareto_data(doc,license_filter=category)
        assert [p['id'] for p in graph['points']]==[p['id'] for p in table['model_data']]
        checks.append(dict(category=category,model_ids=[p['id'] for p in graph['points']]))
    run=ROOT/'training/runs/f_series/leaderboard-2b-repair-20261101'
    smokes=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),result=read(p)) for p in sorted(run.glob('*/adapter-smoke.json'))]
    assert len(smokes)==2 and all(s['result']['status']=='passed' for s in smokes)
    write(folder/'validation.json',dict(results_sha256=hashlib.sha256(raw).hexdigest(),
        registration_sha256=sha(folder/'registration.json'),checks=checks,adapter_smokes=smokes,
        state=read(run/'STATE.json'),ticket=read(run/'JOB.json'),
        code_sha256={p:sha(ROOT/p) for p in ['retrieval_benchmark/requested_models.py','retrieval_benchmark/requested_queue.py',
            'retrieval_benchmark/requested_2b_retry.py','retrieval_benchmark/space/view.py','retrieval_benchmark/space/app.py']}))
    print('BOTH_2B_GPU_SMOKES_AND_LICENSES_OK')
