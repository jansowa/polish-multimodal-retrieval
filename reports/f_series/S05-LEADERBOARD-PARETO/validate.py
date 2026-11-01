"""Snapshot presentation checks and a standalone SVG preview; no ML operations."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from retrieval_benchmark.core import public_check, read, sha, write
from retrieval_benchmark.space.view import organizations, overview, pareto_data, pareto_html

if __name__ == '__main__':
    folder = Path(__file__).parent
    raw = (ROOT/'retrieval_benchmark/space/results.json').read_bytes()
    doc = public_check(json.loads(raw))
    data = pareto_data(doc)
    assert len(data['points']) == len(overview(doc)['rows'])
    assert len({p['id'] for p in data['points']}) == len(data['points'])
    filter_cases = []
    for settings in [dict(max_parameters=100),dict(min_parameters=50,max_parameters=100),
                     dict(selected_organizations=['jansowa']),dict(selected_organizations=[])]:
        table = overview(doc,**settings)
        chart = pareto_data(doc,**settings)
        assert [p['id'] for p in chart['points']] == [p['id'] for p in table['model_data']]
        filter_cases.append(dict(settings=settings, model_ids=[p['id'] for p in chart['points']]))
    reg = read(ROOT/'reports/leaderboard_requested_20261101/registration.json')
    protected = {name:sha(ROOT/name) for name in reg['protected']}
    assert protected == reg['protected']
    markup = '<!doctype html><html lang="pl"><meta charset="utf-8"><title>Pareto preview</title>'
    markup += '<body style="font:15px system-ui;color:#334155;background:#fff;padding:24px;max-width:1200px;margin:auto">'
    markup += '<h1>Jakość a rozmiar modelu</h1>'+pareto_html(doc)
    markup += '<h2>Skala logarytmiczna</h2>'+pareto_html(doc,log_scale=True)+'</body></html>'
    (folder/'preview.html').write_text(markup,encoding='utf-8')
    evidence = dict(results_sha256=hashlib.sha256(raw).hexdigest(), models=len(data['points']),
                    complete=sum(p['score'] is not None for p in data['points']),
                    missing=sum(p['score'] is None for p in data['points']),
                    all_frontier=data['all_frontier'], commercial_frontier=data['commercial_frontier'],
                    organizations=organizations(doc), filter_cases=filter_cases,
                    protected_sha256=protected,
                    code_sha256={name:sha(ROOT/name) for name in ['retrieval_benchmark/space/view.py',
                        'retrieval_benchmark/space/app.py','retrieval_benchmark/tests/test_pareto.py']})
    write(folder/'validation.json',evidence)
    print('PARETO_VALIDATED',evidence['models'],'models;',evidence['complete'],'complete;',evidence['missing'],'pending')
