import csv
import html
import json

from .models import MODEL_METADATA


def primary(result):
    metrics = result['metrics']
    return metrics['pair_accuracy'] if 'pair_accuracy' in metrics else metrics['recall@1']


def matrix(results):
    """Latest full run per model and dataset. Incomplete rows have no mean."""
    latest = {}
    for result in results:
        if not result['full_dataset']:
            continue
        key = result['model'], result['dataset']
        if key not in latest or result.get('timestamp', '') > latest[key].get('timestamp', ''):
            latest[key] = result
    datasets = sorted({d for _, d in latest})
    models = []
    for model in sorted({m for m, _ in latest}):
        meta = MODEL_METADATA.get(model, {})
        cells = {d: latest[(model, d)] for d in datasets if (model, d) in latest}
        parameters = meta.get('parameters') or next((r.get('parameters') for r in cells.values() if r.get('parameters')), None)
        mean = sum(primary(r) for r in cells.values()) / len(datasets) if datasets and len(cells) == len(datasets) else None
        models.append(dict(model=model, display_name=meta.get('display_name', model), parameters_m=parameters / 1e6 if parameters else None,
                           license=meta.get('license', 'Nie podano'), license_url=meta.get('license_url'), cells=cells, mean=mean))
    models.sort(key=lambda r: (r['mean'] is None, -(r['mean'] or 0), r['model']))
    return datasets, models


def render(root, results):
    datasets, models = matrix(results)
    with (root / 'leaderboard.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'parameters_m', 'license', 'mean', *datasets])
        for m in models:
            writer.writerow([m['display_name'], m['parameters_m'], m['license'], m['mean'],
                             *[primary(m['cells'][d]) if d in m['cells'] else '' for d in datasets]])
    with (root / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'dataset', 'metric', 'score', 'run_id'])
        for m in models:
            for d, r in m['cells'].items():
                for metric, score in r['metrics'].items():
                    writer.writerow([m['model'], d, metric, score, r['run_id']])
    headers = ['Model', 'Parametry [M]', 'Licencja', 'Średnia', *datasets]
    head = ''.join(f'<th scope="col" aria-sort="{"descending" if i == 3 else "none"}"><button data-column="{i}">{html.escape(label)} <span class="arrow">{"▼" if i == 3 else "↕"}</span></button></th>' for i, label in enumerate(headers))
    rows = []
    for i, m in enumerate(models):
        license_label = html.escape(m['license'])
        if m['license_url']:
            license_label = f'<a href="{html.escape(m["license_url"], quote=True)}" target="_blank" rel="noopener">{license_label}</a>'
        size = f'{m["parameters_m"]:.2f}' if m['parameters_m'] is not None else '—'
        mean = f'{m["mean"] * 100:.2f}' if m['mean'] is not None else '—'
        cells = [f'<th scope="row" title="{html.escape(m["model"], quote=True)}">{html.escape(m["display_name"])}</th>',
                 f'<td>{size}</td>', f'<td>{license_label}</td>', f'<td class="mean score" data-score="mean">{mean}</td>']
        for d in datasets:
            r = m['cells'].get(d)
            value = f'{primary(r) * 100:.2f}' if r else '—'
            link = f'<a href="{html.escape(r["run_id"], quote=True)}/result.json">{value}</a>' if r else value
            cells.append(f'<td class="score" data-score="{html.escape(d, quote=True)}">{link}</td>')
        rows.append(f'<tr data-row="{i}">' + ''.join(cells) + '</tr>')
    # Only presentation data goes to the browser: no timing or batch information.
    payload = json.dumps({'datasets': datasets, 'models': [
        {**m, 'cells': {d: {'metrics': r['metrics']} for d, r in m['cells'].items()}} for m in models
    ]}, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    page = PAGE.replace('__HEADERS__', head).replace('__ROWS__', ''.join(rows)).replace('__DATA__', payload)
    (root / 'index.html').write_text(page, encoding='utf-8')
    return root / 'index.html'


PAGE = '''<!doctype html><html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="60"><title>Polski benchmark multimodalny</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{font:15px system-ui;background:#101827;color:#e7edf7;margin:0;padding:40px}
h1{font-size:28px;margin:0 0 16px}p{color:#b2bfd2;line-height:1.6;max-width:1100px}a{color:#7dd3fc;text-decoration:none}a:hover{text-decoration:underline}
.controls{display:flex;align-items:center;gap:20px;flex-wrap:wrap;margin:24px 0}input,select{font:inherit;background:#192538;color:#e7edf7;border:1px solid #43536a;border-radius:7px;padding:10px}
.table-wrap{overflow-x:auto;border:1px solid #334155;border-radius:10px}table{border-collapse:collapse;width:100%;white-space:nowrap}
td,th{padding:17px 20px;border-bottom:1px solid #334155;text-align:right}thead th{background:#192538}thead button{border:0;background:none;color:inherit;font:inherit;font-weight:600;cursor:pointer;padding:0}
th:first-child,td:first-child{text-align:left}th:nth-child(3),td:nth-child(3){text-align:left}tbody th{font-weight:500}tbody tr:hover{background:#192538}tbody tr:last-child>*{border-bottom:0}
.mean{font-weight:750;background:#153147;color:#a5e4ff}.score{font-variant-numeric:tabular-nums}.arrow{font-size:11px;color:#7dd3fc}.note{font-size:13px}
@media(max-width:700px){body{padding:20px}h1{font-size:23px}}
</style></head><body><h1>Porównanie modeli — polskie dane</h1>
<p>MMEB v1: bez instrukcji zadania, suma surowych embeddingów CLIP i normalizacja końcowa (cosine).
EmbeddingGemma-2 używa natywnego wspólnego kodowania obrazu i tekstu; audio jest wyłączone.
Jeden wiersz na model, wyniki w skali 0–100. Średnia arytmetyczna nadaje każdemu datasetowi taką samą wagę;
pojawia się po ukończeniu wszystkich widocznych datasetów. Uwzględniane są najnowsze prawidłowe pełne przebiegi.</p>
<div class="controls"><input id="filter" type="search" placeholder="Filtruj modele" aria-label="Filtruj modele">
<label>Metryka <select id="metric"><option value="official">Oficjalna dla benchmarku</option><option value="ndcg@10">NDCG@10</option><option value="recall@5">Recall@5</option><option value="recall@10">Recall@10</option><option value="mrr">MRR</option></select></label>
<a href="leaderboard.csv">Tabela CSV</a><a href="metrics.csv">Wszystkie metryki CSV</a></div>
<p id="metric-note" class="note">MMEB: Precision@1 (równe Recall@1 przy jednej poprawnej odpowiedzi). ARO: pair accuracy.</p>
<div class="table-wrap"><table><thead><tr>__HEADERS__</tr></thead><tbody>__ROWS__</tbody></table></div>
<p class="note">Kliknij nagłówek, aby zmienić sortowanie. Parametry obejmują enkoder obrazu i tekstu, przed reparametryzacją.
<a href="validation.json">Raport wykluczonych przebiegów</a> · <a href="index.protocol-v1.html">Archiwum wcześniejszego protokołu</a></p>
<script id="leaderboard-data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('leaderboard-data').textContent);
const body=document.querySelector('tbody'),metric=document.getElementById('metric');
let sortColumn=3,descending=true;
function score(result){if(!result)return null;const m=result.metrics;return metric.value==='official'?(m.pair_accuracy??m['precision@1']??m['recall@1']):(m[metric.value]??null)}
function mean(model){const values=data.datasets.map(d=>score(model.cells[d]));return values.length&&values.every(v=>v!==null)?values.reduce((a,b)=>a+b,0)/values.length:null}
function value(model,column){if(column===0)return model.display_name;if(column===1)return model.parameters_m;if(column===2)return model.license;if(column===3)return mean(model);return score(model.cells[data.datasets[column-4]])}
function sortRows(){const rows=Array.from(body.rows);rows.sort((a,b)=>{const x=value(data.models[a.dataset.row],sortColumn),y=value(data.models[b.dataset.row],sortColumn);if(x===null||y===null)return x===y?0:x===null?1:-1;const cmp=typeof x==='string'?x.localeCompare(y):x-y;return descending?-cmp:cmp});rows.forEach(r=>body.appendChild(r));document.querySelectorAll('thead th').forEach((th,i)=>{th.setAttribute('aria-sort',i===sortColumn?(descending?'descending':'ascending'):'none');th.querySelector('.arrow').textContent=i===sortColumn?(descending?'▼':'▲'):'↕'})}
function render(){Array.from(body.rows).forEach(row=>{const model=data.models[row.dataset.row];row.querySelectorAll('[data-score]').forEach(cell=>{const key=cell.dataset.score,v=key==='mean'?mean(model):score(model.cells[key]);const target=cell.querySelector('a')||cell;target.textContent=v===null?'—':(v*100).toFixed(2)})});document.getElementById('metric-note').textContent=metric.value==='official'?'MMEB: Precision@1 (równe Recall@1 przy jednej poprawnej odpowiedzi). ARO: pair accuracy.':'Metryka dodatkowa: '+metric.options[metric.selectedIndex].text+'. Brak metryki dla zadania oznaczono kreską.';sortRows()}
document.querySelectorAll('thead button').forEach(button=>button.addEventListener('click',()=>{const column=Number(button.dataset.column);descending=column===sortColumn?!descending:![0,2].includes(column);sortColumn=column;sortRows()}));
metric.addEventListener('change',render);document.getElementById('filter').addEventListener('input',event=>{const query=event.target.value.toLocaleLowerCase('pl');Array.from(body.rows).forEach(row=>row.hidden=!data.models[row.dataset.row].display_name.toLocaleLowerCase('pl').includes(query))});render();
</script></body></html>'''
