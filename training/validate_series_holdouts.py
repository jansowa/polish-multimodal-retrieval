"""Validate production holdout enforcement and both human-task collections."""
import importlib.util
import io
import json
import subprocess
import sys
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pl_mmb_training.series_20261023 import ROOT,OUT,read,rows,freeze,file_sha
from pl_mmb_training.series_holdout import workspace_guard


def main():
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_series_20261023.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    subprocess.run(['node','--check',str(ROOT/'experiments/series_annotation/app.js')],check=True)
    guard=workspace_guard();dev=rows(OUT/'wiki_v2/wiki_dev.jsonl');train=rows(OUT/'wiki_v2/pilot_wiki_train_disjoint.jsonl')
    if not all(guard.reasons(r) for r in dev):raise ValueError('Dev image escaped guard')
    rejected=[r['id'] for r in train if guard.reasons(r)]
    if rejected:raise ValueError('Disjoint pilot train is rejected by new guard: '+str(rejected[:3]))
    for name,sha in read(OUT/'preservation_before.json').items():
        if file_sha(ROOT/name)!=sha:raise ValueError('Protected snapshot changed')
    spec=importlib.util.spec_from_file_location('human_holdout_smoke',ROOT/'experiments/series_annotation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    collection=module.Collection();server=ThreadingHTTPServer(('127.0.0.1',0),module.handler(collection))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}';checks={}
    try:
        for mode,count in [('query',300),('audit',200)]:
            with urllib.request.urlopen(base+f'/api/tasks?mode={mode}') as response:tasks=json.load(response)
            if len(tasks)!=count:raise ValueError('Wrong task count')
            if mode=='query' and any('caption' in r for r in tasks):raise ValueError('Query caption disclosed')
            checks[mode]=count
        for path in ('/asset/query/0','/asset/query/150','/asset/audit/0','/asset/audit/100'):
            with urllib.request.urlopen(base+path,timeout=60) as response:payload=response.read()
            from PIL import Image
            with Image.open(io.BytesIO(payload)) as image:checks[path]=list(image.size)
    finally:
        server.shutdown();server.server_close();thread.join()
    proof=dict(status='passed',tests_passed=result.testsRun,Wiki_dev_blocked=len(dev),
        disjoint_pilot_train_accepted=len(train),http_smoke=checks,production_annotations_created=0,
        protected_snapshot_SHA_unchanged=True,guard_sha256=file_sha(OUT/'wiki_dev_guard_enforcement.json'),
        source_sha256={p:file_sha(ROOT/p) for p in ('src/pl_mmb_training/leakage.py','src/pl_mmb_training/series_holdout.py',
            'src/pl_mmb_training/visualnews_data.py','src/pl_mmb_training/wiki_pilot_data.py')},
        applies_to=['general training builder/runner','future VN components','future Wiki components','captionless image training'],
        new_Wiki_training_still_blocked=True,browser_visual_QA=False)
    freeze(OUT/'wiki_dev_guard_integration.json',proof)
    print(json.dumps(proof,indent=2))


if __name__=='__main__':main()
