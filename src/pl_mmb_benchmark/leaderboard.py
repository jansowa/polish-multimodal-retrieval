import csv
import html
import json
import math
from pathlib import Path


def build(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    invalid = []
    for p in root.glob('*/result.json'):
        result = json.loads(p.read_text())
        predictions = p.parent / 'predictions.jsonl'
        reason = None
        if result.get('model') == 'apple/MobileCLIP2-S2' and result.get('numerics') != 'mobile-fp32-unit-range-reparam-v3':
            reason = 'Superseded MobileCLIP preprocessing: requires unit-range images and reparameterization'
        if not predictions.exists():
            reason = 'Missing predictions'
        else:
            rows = [json.loads(line) for line in predictions.read_text().splitlines()]
            if len(rows) != result['examples']:
                reason = 'Prediction count mismatch'
            elif any(not math.isfinite(row['positive_score']) for row in rows):
                reason = 'Nonfinite positive similarities (historical FP16 overflow)'
        if reason:
            invalid.append({'run_id': result['run_id'], 'reason': reason})
        else:
            results.append(result)
    # Audit is separate: original result files are never rewritten or removed.
    (root / 'validation.json').write_text(json.dumps({'excluded': invalid}, indent=2))
    results.sort(key=lambda r: (r['dataset'], -next(iter(r['metrics'].values()))))
    columns = ['dataset', 'model', 'examples', 'full_dataset', 'batch_size', 'seconds', 'peak_vram_gib', 'run_id']
    with (root / 'leaderboard.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns + ['metric', 'score'])
        writer.writeheader()
        for r in results:
            for metric, score in r['metrics'].items():
                writer.writerow({**{c: r[c] for c in columns}, 'metric': metric, 'score': score})
    rows = []
    for r in results:
        metric, score = next(iter(r['metrics'].items()))
        scope = 'pełny' if r['full_dataset'] else 'PILOT'
        pilot = '1' if not r['full_dataset'] else '0'
        rows.append(f'<tr data-pilot="{pilot}"' + (' hidden' if pilot == '1' else '') + f'><td>{html.escape(r["dataset"])}</td><td>{html.escape(r["model"])}</td>'
                    f'<td>{r["examples"]} ({scope})</td><td>{html.escape(metric)}</td>'
                    f'<td><div class="bar" style="width:{score*100:.2f}%"></div>{score*100:.2f}%</td>'
                    f'<td>{r["batch_size"]}</td><td>{r["seconds"]:.1f}s</td>'
                    f'<td><a href="{r["run_id"]}/result.json">JSON</a></td></tr>')
    page = '''<!doctype html><html lang="pl"><meta charset="utf-8"><meta http-equiv="refresh" content="60">
<title>Polski benchmark multimodalny</title><style>
body{font:16px system-ui;background:#101827;color:#e7edf7;margin:40px}table{border-collapse:collapse;width:100%}
td,th{text-align:left;padding:14px;border-bottom:1px solid #334155}td:nth-child(5){min-width:180px}
.bar{height:8px;background:#38bdf8;margin-bottom:6px}input{padding:12px;width:400px}a{color:#38bdf8}
</style><h1>Porównanie modeli — polskie dane</h1>
<p>Wyniki porównuj w obrębie tego samego datasetu, zakresu i wersji tłumaczeń.
MMEB: lokalna pula kandydatów, poprawny kandydat na pozycji 0; zapytania obraz + tekst: suma znormalizowanych embeddingów.
ARO: ścisłe porównanie podpisu prawdziwego z fałszywym. Remisy są błędami. Brak wspólnej średniej między zadaniami.</p>
<p><a href="leaderboard.csv">Pobierz CSV ze wszystkimi metrykami</a></p>
<script>function filterRows(){document.querySelectorAll('tbody tr').forEach(r=>r.hidden=(!document.getElementById('pilots').checked&&r.dataset.pilot==='1')||!r.textContent.toLowerCase().includes(document.getElementById('filter').value.toLowerCase()))}</script>
<input id="filter" placeholder="Filtruj model lub dataset" oninput="filterRows()">
<label><input style="width:auto" type="checkbox" id="pilots" onchange="filterRows()"> Pokaż piloty</label>
<p>Przebiegi z błędami numerycznymi wyłączono; oryginały zachowano. <a href="validation.json">Raport walidacji</a>.</p>
<table><thead><tr><th>Dataset</th><th>Model</th><th>Przykłady</th><th>Metryka</th><th>Wynik</th><th>Batch</th><th>Czas</th><th>Szczegóły</th></tr></thead><tbody>'''
    (root / 'index.html').write_text(page + ''.join(rows) + '</tbody></table></html>', encoding='utf-8')
    return root / 'index.html'
