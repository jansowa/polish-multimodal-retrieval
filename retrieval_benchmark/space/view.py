"""Pure presentation functions; this directory runs without the parent repository."""
import csv
import html
import io
import json
import math
from pathlib import Path
from urllib.parse import quote

METRICS = ['recall@1', 'recall@5', 'recall@10', 'mrr', 'ndcg@10']
TASK_ORDER = ['flickr30k_pl_local_test', 'vizwiz_pl_local_validation', 'xm3600_pl', 'wiki_native_dev',
              'VN_dev', 'xtd10_no_overlap', 'xtd10_full_1k']
TASK_LABELS = dict(flickr30k_pl_local_test='Flickr30k PL', vizwiz_pl_local_validation='VizWiz PL',
                   wiki_native_dev='Wiki PL', VN_dev='VisualNews PL',
                   xtd10_no_overlap='XTD10 PL · bez nakładań', xtd10_full_1k='XTD10 PL · pełne')
BASE_COLUMNS = dict(average='Średni wynik [%]', total='V+T [M]', vision='Wizja [M]', text='Tekst [M]',
                    license='Licencja', head='Head [M]', t2i='T2I [%]', i2t='I2T [%]', organic='Organiczne [%]',
                    synthetic='Syntetyczne [%]', mt='Tłumaczenia maszynowe [%]', coverage='Pokrycie średniej')


def load(path=None):
    return json.loads(Path(path or Path(__file__).with_name('results.json')).read_text(encoding='utf-8'))


def tasks(document):
    ready = [k for k, d in document['datasets'].items() if d['status'] == 'ready']
    return [k for k in TASK_ORDER if k in ready] + sorted(set(ready) - set(TASK_ORDER))


def default_tasks(document):
    result = tasks(document)
    if 'xtd10_no_overlap' in result and 'xtd10_full_1k' in result:
        result.remove('xtd10_full_1k')
    return result


def columns(document):
    return {**BASE_COLUMNS, **{'task:' + k: TASK_LABELS.get(k, document['datasets'][k].get('title', k)) + ' [%]'
                             for k in tasks(document)}}


def default_columns(document):
    return ['average', 'total', 'vision', 'text', 'license'] + ['task:' + k for k in tasks(document)]


def group(origin):
    # Mutually exclusive groups: MT is not counted as synthetic.
    if origin == 'machine_translation':
        return 'mt'
    if origin in ('native_human', 'human_translation'):
        return 'organic'
    if origin in ('generated_recycled_translated', 'synthetic', 'generated'):
        return 'synthetic'
    return None


def model_reference(model_id, model):
    repository = model.get('repository')
    if not repository and model_id.startswith('hf:'):
        repository = model_id[3:].rsplit('@', 1)[0]
    if repository:
        return repository, 'https://huggingface.co/' + quote(repository, safe='/._-')
    return model['label'], None


def overview(document, metric='recall@1', mode='v2_no_instructions', profile='historical-native-v2',
             max_parameters=0, show_seeds=False, show_controls=False, visible_columns=None, average_tasks=None):
    if metric not in METRICS:
        raise ValueError('Unsupported metric')
    available = tasks(document)
    selected = default_tasks(document) if average_tasks is None else list(dict.fromkeys(average_tasks))
    if any(k not in available for k in selected):
        raise ValueError('Only ready tasks may contribute to the average')
    specs = columns(document)
    visible = default_columns(document) if visible_columns is None else list(dict.fromkeys(visible_columns))
    if any(k not in specs for k in visible):
        raise ValueError('Unknown display column')
    results = {}
    for r in document['results']:
        if r['dataset_id'] in available and (r['prompt_mode'], r['profile']) == (mode, profile):
            results.setdefault(r['model_id'], {})[r['dataset_id']] = r
    entries = []
    for model_id, found in results.items():
        m = document['models'][model_id]
        if not show_seeds and not m.get('representative', True):
            continue
        if not show_controls and m.get('role') == 'english_control':
            continue
        p = m['parameters']
        if max_parameters and p['total'] >= max_parameters * 1e6:
            continue
        def mean(names, direction=None):
            if not names or any(k not in found for k in names):
                return None
            directions = [direction] if direction else ['t2i', 'i2t']
            return 100 * sum(found[k]['metrics'][d][metric] for k in names for d in directions) / (len(names) * len(directions))
        license_info = m.get('license')
        license_text = license_info.get('display', 'Brak deklaracji') if isinstance(license_info, dict) else license_info or 'Brak deklaracji'
        values = dict(average=mean(selected), license=license_text, t2i=mean(selected, 't2i'), i2t=mean(selected, 'i2t'),
                      coverage=f'{sum(k in found for k in selected)}/{len(selected)}',
                      **{k: p[k] / 1e6 for k in ('total', 'vision', 'text', 'head')})
        values.update({'task:' + k: mean([k]) for k in available})
        for name in ('organic', 'synthetic', 'mt'):
            values[name] = mean([k for k in selected if group(document['datasets'][k].get('caption_origin')) == name])
        name, url = model_reference(model_id, m)
        entries.append((name, values, url))
    # Rank only complete averages. A partial submission cannot win by omitting hard tasks.
    entries.sort(key=lambda e: (e[1]['average'] is None, -(e[1]['average'] or 0), e[0]))
    return dict(headers=['Model'] + [specs[k] for k in visible],
                rows=[[label] + [v[k] for k in visible] for label, v, _ in entries],
                model_urls=[url for _, _, url in entries],
                score_columns=[i + 1 for i, k in enumerate(visible) if k not in ('total', 'vision', 'text', 'head', 'coverage', 'license')],
                complete=sum(v['average'] is not None for _, v, _ in entries), total=len(entries), average_tasks=selected)


def cell_styles(view):
    styles = [[''] * len(view['headers']) for _ in view['rows']]
    for col in view['score_columns']:
        values = [r[col] for r in view['rows'] if isinstance(r[col], (float, int)) and math.isfinite(r[col])]
        if not values:
            continue
        low, high = min(values), max(values)
        for i, row in enumerate(view['rows']):
            value = row[col]
            if value is None:
                continue
            t = (value - low) / (high - low) if high > low else .5
            red, yellow, green = (255, 205, 210), (255, 244, 187), (165, 214, 167)
            a, b, weight = (red, yellow, t * 2) if t < .5 else (yellow, green, (t - .5) * 2)
            color = tuple(round(x + (y - x) * weight) for x, y in zip(a, b))
            styles[i][col] = f'background-color: rgb{color}; color: #17212b'
    return styles


def styled(view):
    import pandas as pd
    frame = pd.DataFrame(view['rows'], columns=view['headers'])
    urls = view.get('model_urls', [None] * len(view['rows']))
    for index, row in enumerate(view['rows']):
        name = html.escape(str(row[0]), quote=True)
        url = urls[index]
        frame.iat[index, 0] = (f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{name}</a>'
                               if url else name)
    styles = pd.DataFrame(cell_styles(view), columns=view['headers'], index=frame.index)
    formats = {header: '{:.2f}' for i, header in enumerate(view['headers']) if i in view['score_columns']}
    formats.update({h: '{:.3f}' for h in view['headers'] if h.endswith('[M]')})
    return frame.style.apply(lambda _: styles, axis=None).format(formats, na_rep='—')


def dataset_note(document, name):
    d = document['datasets'][name]
    counts = f"{d.get('images', '—')} obrazów · {d.get('texts', '—')} tekstów"
    return f"### {d['title']}\n{counts} · **{d['caption_origin']}** · {d['status']}\n\n{d['notes']}"


def csv_bytes(view):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, delimiter=';')
    writer.writerow(view['headers'])
    for row in view['rows']:
        writer.writerow([("'" + v) if isinstance(v, str) and v.startswith(('=', '+', '-', '@')) else v for v in row])
    return stream.getvalue().encode('utf-8-sig')
