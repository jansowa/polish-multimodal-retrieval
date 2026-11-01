"""Verify the user's actual local server, without starting another application."""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'retrieval_benchmark/space'))
from gradio_client import Client
from view import load, default_columns, default_tasks

url = 'http://127.0.0.1:7865'
with urllib.request.urlopen(url+'/config', timeout=15) as response:
    config = json.load(response)
charts = [c for c in config['components'] if c['type']=='html' and 'data-frontier' in (c['props'].get('value') or '')]
assert len(charts)==1
labels = {c['props'].get('label') for c in config['components']}
expected = ['Minimum V+T [M]', 'Maksimum V+T [M]; 0 = bez limitu', 'Organizacje']
assert all(label in labels for label in expected)
doc = load()
client = Client(url, verbose=False)
result = client.predict('recall@1','v2_no_instructions','historical-native-v2',100,False,False,
                        default_columns(doc),default_tasks(doc),0,['jansowa'],False,api_name='/leaderboard')
table = result[0].get('value',result[0])
assert table['data']
assert result[2].count('data-model-id=')==len(table['data'])
assert result[2].count('data-model-id="checkpoint:')==len(table['data'])
evidence = dict(url='http://localhost:7865/', status='passed', pareto=True, filters=expected,
                visible_own_models=len(table['data']), graph_matches_live_table=True)
Path(__file__).with_name('live-validation.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
print('LIVE_UI_AND_API_OK: Pareto, min/max, organizations, filtered table identity')
