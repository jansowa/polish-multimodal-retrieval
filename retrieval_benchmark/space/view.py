"""Pure presentation functions; this directory runs without the parent repository."""
import csv
import html
import io
import json
import math
import textwrap
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


def own_model(model_id, model):
    return (model_id.startswith('checkpoint:') or model.get('role') == 'project'
            or (model.get('repository') or '').split('/')[0].casefold() == 'jansowa')


def model_organization(model_id, model):
    if own_model(model_id, model):
        return 'jansowa'
    repository = model.get('repository')
    if not repository and model_id.startswith('hf:'):
        repository = model_id[3:].rsplit('@', 1)[0]
    return repository.split('/')[0] if repository and '/' in repository else 'Pozostałe'


def organizations(document):
    return sorted({model_organization(mid, model) for mid, model in document['models'].items()}, key=str.casefold)


def model_choices(document):
    choices = []
    for mid,model in document['models'].items():
        name,_ = model_reference(mid,model)
        if own_model(mid,model) and '/' not in name:
            name = 'jansowa/'+name
        choices.append((name,mid))
    return sorted(choices,key=lambda item:(item[0].casefold(),item[1]))


def overview(document, metric='recall@1', mode='v2_no_instructions', profile='historical-native-v2',
             max_parameters=0, show_seeds=False, show_controls=False, visible_columns=None, average_tasks=None,
             min_parameters=0, selected_organizations=None, selected_models=None, license_filter='all'):
    if metric not in METRICS:
        raise ValueError('Unsupported metric')
    if license_filter not in ('all','commercial','noncommercial','unknown'):
        raise ValueError('Unsupported license filter')
    min_parameters, max_parameters = min_parameters or 0, max_parameters or 0
    if any(not isinstance(v, (int,float)) or not math.isfinite(v) or v < 0 for v in (min_parameters,max_parameters)):
        raise ValueError('Parameter bounds must be finite and nonnegative')
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
        if license_filter!='all' and license_category(model_id,m)!=license_filter:
            continue
        if selected_models is not None and model_id not in selected_models:
            continue
        if not show_seeds and not m.get('representative', True):
            continue
        if not show_controls and m.get('role') == 'english_control':
            continue
        p = m['parameters']
        if p['total'] < min_parameters*1e6 or (max_parameters and p['total'] > max_parameters*1e6):
            continue
        if selected_organizations is not None and model_organization(model_id,m) not in selected_organizations:
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
        entries.append((name, values, url, model_id))
    # Rank only complete averages. A partial submission cannot win by omitting hard tasks.
    entries.sort(key=lambda e: (e[1]['average'] is None, -(e[1]['average'] or 0), e[0]))
    return dict(headers=['Model'] + [specs[k] for k in visible],
                rows=[[label] + [v[k] for k in visible] for label, v, _, _ in entries],
                model_urls=[url for _, _, url, _ in entries],
                model_data=[dict(id=mid, name=name, url=url, score=v['average'], size=v['total'],
                                 license=v['license']) for name, v, url, mid in entries],
                score_columns=[i + 1 for i, k in enumerate(visible) if k not in ('total', 'vision', 'text', 'head', 'coverage', 'license')],
                complete=sum(v['average'] is not None for _, v, _, _ in entries), total=len(entries), average_tasks=selected)


def commercial_use(model_id, model):
    """Published declarations plus the owner's explicit local-checkpoint override."""
    if own_model(model_id, model):
        return True
    info = model.get('license') or {}
    if isinstance(info, dict):
        declared = info.get('id') or info.get('spdx') or info.get('display')
    else:
        declared = info
    if not isinstance(declared, str):
        return False
    return declared.strip().casefold() in {
        'mit', 'apache-2.0', 'bsd-2-clause', 'bsd-3-clause',
        'isc', 'cc-by-4.0', 'cc by 4.0', 'cc0-1.0', 'unlicense',
    }


def license_category(model_id,model):
    if commercial_use(model_id,model):
        return 'commercial'
    info=model.get('license') or {}
    declared=(info.get('id') or info.get('spdx') or info.get('display') or '') if isinstance(info,dict) else info
    if not isinstance(declared,str):
        return 'unknown'
    normalized=declared.casefold().replace(' ','-')
    if normalized.startswith('cc-by-nc') or normalized in ('apple-amlr','apple-amlr-(research-only)'):
        return 'noncommercial'
    return 'unknown'


def frontier(points):
    """Strict score records by increasing size; ties keep the smallest model."""
    best = -math.inf
    result = []
    for point in sorted((p for p in points if p['score'] is not None),
                        key=lambda p: (p['size'], -p['score'], p['id'])):
        if point['score'] > best:
            result.append(point)
            best = point['score']
    return result


def pareto_data(document, metric='recall@1', mode='v2_no_instructions',
                profile='historical-native-v2', max_parameters=0, average_tasks=None,
                show_seeds=False, show_controls=False, min_parameters=0, selected_organizations=None,
                filtered_view=None, selected_models=None, license_filter='all'):
    view = filtered_view if filtered_view is not None else overview(
        document, metric, mode, profile, max_parameters, show_seeds, show_controls,
        average_tasks=average_tasks, min_parameters=min_parameters, selected_organizations=selected_organizations,
        selected_models=selected_models,license_filter=license_filter)
    points = [dict(point) for point in view['model_data']]
    for point in points:
        point['commercial'] = commercial_use(point['id'], document['models'][point['id']])
    return dict(points=points, all_frontier=frontier(points),
                commercial_frontier=frontier([p for p in points if p['commercial']]),
                tasks=view['average_tasks'])


def label_rows(points):
    """Variable-width full names; short local model names need very little space."""
    rows = []
    for point in points:
        lines = textwrap.wrap(point['name'],width=32,break_long_words=True,break_on_hyphens=False) or ['']
        rows.append(dict(id=point['id'],lines=lines,height=16*len(lines)+8,
                         width=max(36,max(map(len,lines))*8+10),
                         priority=(point.get('score') is None,-(point.get('score') or 0))))
    return rows


def place_point_labels(rows, coordinates, bounds, forbidden=()):
    """Try the closest free position around each dot, expanding only in crowded areas."""
    left,top,right,bottom = bounds
    result = {}
    def valid(row,xx,yy):
        w,h = row['width'],row['height']
        if xx<left or yy<top or xx+w>right or yy+h>bottom:
            return False
        if any(xx < rx+rw and xx+w > rx and yy < ry+rh and yy+h > ry for rx,ry,rw,rh in forbidden):
            return False
        if any(xx < a['x']+a['width']+5 and xx+w+5 > a['x'] and
               yy < a['y']+a['height']+5 and yy+h+5 > a['y'] for a in result.values()):
            return False
        return not any(xx-7 < px < xx+w+7 and yy-7 < py < yy+h+7 for px,py in coordinates.values())
    def distance(row,xx,yy,px,py):
        return math.hypot(max(xx-px,0,px-xx-row['width']),max(yy-py,0,py-yy-row['height']))
    # Complete, high-scoring points get local space before incomplete measurements.
    ordered = sorted(rows,key=lambda row:(row['priority'],-row['width']*row['height'],coordinates[row['id']][1],row['id']))
    for original in ordered:
        row = dict(original)
        px,py = coordinates[row['id']]
        w,h = row['width'],row['height']
        candidates = [(px+11,py-h-8),(px+11,py+8),(px-w-11,py-h-8),(px-w-11,py+8),
                      (px-w/2,py-h-11),(px-w/2,py+11),(px+11,py-h/2),(px-w-11,py-h/2)]
        available = [(xx,yy) for xx,yy in candidates if valid(row,xx,yy)]
        if not available:
            for radius in range(24,int(math.hypot(right-left,bottom-top))+48,24):
                candidates = [(px+radius*math.cos(i*math.pi/12)-w/2,
                               py+radius*math.sin(i*math.pi/12)-h/2) for i in range(24)]
                available = [(xx,yy) for xx,yy in candidates if valid(row,xx,yy)]
                if available:
                    break
        if not available:
            return None
        xx,yy = min(available,key=lambda p:(distance(row,*p,px,py),math.hypot(p[0]+w/2-px,p[1]+h/2-py)))
        row.update(x=xx,y=yy)
        result[row['id']] = row
    return result


def pareto_html(document, metric='recall@1', mode='v2_no_instructions',
                profile='historical-native-v2', max_parameters=0, average_tasks=None, log_scale=False,
                show_seeds=False, show_controls=False, min_parameters=0, selected_organizations=None,
                filtered_view=None, show_labels=True, selected_models=None, license_filter='all'):
    """Responsive SVG: no extra dependencies, external scripts or network calls."""
    data = pareto_data(document, metric, mode, profile, max_parameters, average_tasks,
                       show_seeds, show_controls, min_parameters, selected_organizations, filtered_view, selected_models,license_filter)
    points = data['points']
    if not points:
        return '<p>Brak modeli dla wybranych filtrów.</p>'
    escape = lambda value: html.escape(str(value), quote=True)
    complete = [p for p in points if p['score'] is not None]
    missing = [p for p in points if p['score'] is None]
    rows = label_rows(points) if show_labels else []
    left,right,top = 85,1235,35
    bottom = max(455,355+len(points)*8) if show_labels else 355
    canvas_width = 1280
    middle = (left+right)/2
    positive = [p['size'] for p in points if p['size'] > 0]
    high = max(positive, default=1)*1.08
    low = min(positive, default=1)/1.15 if log_scale else 0
    if log_scale:
        high = max(high, low*1.5)
    def x(size):
        if log_scale:
            value = math.log10(max(size, low))
            fraction = (value-math.log10(low))/(math.log10(high)-math.log10(low))
        else:
            fraction = size/high
        return left+(right-left)*fraction
    def y(score):
        return bottom-(bottom-top)*score/100
    for attempt in range(8):
        coordinates = {p['id']:(x(p['size']),y(p['score']) if p['score'] is not None else bottom+93) for p in points}
        placed = place_point_labels(rows,coordinates,(left+3,top+3,right-3,bottom+205 if missing else bottom-3),
                                    forbidden=[(left,bottom+1,right-left,67)]) if show_labels else {}
        if placed is not None:
            break
        bottom = int(bottom*1.35)
    if placed is None:
        raise ValueError('Cannot place labels; reduce the selected model set')
    canvas_height = bottom+(235 if show_labels and missing else 135)
    blue, orange = '#2563eb', '#d97706'
    parts = ['<div class="pareto-chart" style="overflow-x:auto">',
             '<div style="display:flex;gap:24px;flex-wrap:wrap;margin-bottom:8px">',
             f'<span><b style="color:{blue}">━━</b> Pareto: wszystkie modele</span>',
             f'<span><b style="color:{orange}">┅┅</b> Pareto: komercyjne + własne</span></div>',
             f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {canvas_width} {canvas_height}" '
             'role="img" aria-label="Średni wynik modeli względem rozmiaru, dwie granice Pareto" '
             f'style="width:100%;height:auto;min-width:{1280 if show_labels else 420}px;color:var(--body-text-color,#334155)">',
             '<style>.pareto-point:focus circle,.pareto-point:hover circle{stroke:currentColor;stroke-width:3}'
             '.pareto-point:hover .model-label,.pareto-point:focus .model-label{text-decoration:underline}'
             '.pareto-point:hover .label-leader,.pareto-point:focus .label-leader{stroke-opacity:1;stroke-width:2}</style>']
    for tick in range(0, 101, 20):
        yy = y(tick)
        parts.append(f'<path d="M{left} {yy:.2f}H{right}" stroke="var(--border-color-primary,#d7dde5)"/>')
        parts.append(f'<text x="{left-12}" y="{yy+4:.2f}" text-anchor="end" fill="currentColor" font-size="12">{tick}</text>')
    if log_scale:
        ticks = [base*10**power for power in range(math.floor(math.log10(low)), math.ceil(math.log10(high))+1)
                 for base in (1,2,5) if low <= base*10**power <= high]
    else:
        ticks = [high*i/5 for i in range(6)]
    for tick in ticks:
        xx = x(tick)
        parts.append(f'<path d="M{xx:.2f} {top}V{bottom}" stroke="var(--border-color-primary,#d7dde5)" stroke-dasharray="3 5"/>')
        label = f'{tick:,.0f}'.replace(',', ' ') if tick >= 10 else f'{tick:g}'
        parts.append(f'<text x="{xx:.2f}" y="{bottom+22}" text-anchor="middle" fill="currentColor" font-size="12">{label}</text>')
    parts.append(f'<path d="M{left} {top}V{bottom}H{right}" fill="none" stroke="currentColor"/>')
    ylabel_x, ylabel_y = left-55,(top+bottom)/2
    parts.append(f'<text x="{ylabel_x}" y="{ylabel_y}" transform="rotate(-90 {ylabel_x} {ylabel_y})" text-anchor="middle" fill="currentColor" font-size="13">Średni wynik modelu [%]</text>')
    xlabel = 'Rozmiar enkoderów V+T [mln parametrów]'+(' · skala logarytmiczna' if log_scale else '')
    parts.append(f'<text x="{middle}" y="{bottom+51}" text-anchor="middle" fill="currentColor" font-size="13">{xlabel}</text>')
    for name, color, dashed in [('all_frontier',blue,False), ('commercial_frontier',orange,True)]:
        line = data[name]
        line_coordinates = ' '.join(f'{x(p["size"]):.2f},{y(p["score"]):.2f}' for p in line)
        dash = ' stroke-dasharray="8 5"' if dashed else ''
        parts.append(f'<polyline data-frontier="{name}" points="{line_coordinates}" fill="none" stroke="{color}" stroke-width="3"{dash}/>')
    if missing:
        parts.append(f'<rect x="{left}" y="{bottom+75}" width="{right-left}" height="35" rx="6" fill="var(--background-fill-secondary,#f1f5f9)"/>')
        parts.append(f'<text x="{left}" y="{canvas_height-7}" fill="currentColor" font-size="11">Brak pełnej średniej — rozmiar bez wartości na osi Y.</text>')
    frontier_ids = {p['id'] for p in data['all_frontier']+data['commercial_frontier']}
    for point in sorted(points, key=lambda p:p['id']):
        xx, yy = coordinates[point['id']]
        score = f'{point["score"]:.2f}%' if point['score'] is not None else 'brak pełnej średniej'
        eligibility = 'komercyjny / własny' if point['commercial'] else 'poza grupą komercyjną'
        tooltip = f'{point["name"]}\n{point["size"]:.2f} mln parametrów · {score}\n{point["license"]} · {eligibility}'
        color = orange if point['commercial'] else '#64748b'
        radius = 6.5 if point['id'] in frontier_ids else 4.5
        link = f' href="{escape(point["url"])}" target="_blank" rel="noopener noreferrer"' if point['url'] else ''
        parts.append(f'<a class="pareto-point" data-model-id="{escape(point["id"])}"{link} tabindex="0" aria-label="{escape(tooltip)}">'
                     f'<title>{escape(tooltip)}</title>')
        if show_labels:
            row = placed[point['id']]
            anchor_x = max(row['x'],min(xx,row['x']+row['width']))
            anchor_y = max(row['y'],min(yy,row['y']+row['height']))
            parts.append(f'<path class="label-leader" d="M{xx:.2f} {yy:.2f}L{anchor_x:.2f} {anchor_y:.2f}" '
                         f'fill="none" stroke="{color}" stroke-opacity="0.3" stroke-width="0.8"/>')
            parts.append(f'<g class="model-label" data-label-for="{escape(point["id"])}">'
                         f'<rect x="{row["x"]}" y="{row["y"]:.2f}" width="{row["width"]}" height="{row["height"]}" '
                         'rx="3" fill="var(--body-background-fill,white)"/>'
                         f'<text fill="currentColor" font-family="monospace" font-size="12">')
            for i,line in enumerate(row['lines']):
                parts.append(f'<tspan x="{row["x"]+4}" y="{row["y"]+16+i*16:.2f}">{escape(line)}</tspan>')
            parts.append('</text></g>')
        parts.append(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="{radius}" '
                     f'fill="{color}" fill-opacity="0.8" stroke="var(--body-background-fill,white)" stroke-width="1.3"/></a>')
    parts.append('</svg>')
    parts.append(f'<p style="font-size:13px">{len(complete)}/{len(points)} modeli z pełną średnią. '
                 'Najedź na punkt, aby zobaczyć model, rozmiar i wynik. Wykres i obie granice Pareto '
                 'obejmują dokładnie modele widoczne w tabeli i reagują na wszystkie jej filtry.</p>')
    parts.append('<details><summary>Modele na granicach Pareto i kwalifikacja licencji</summary>')
    for name,title in [('all_frontier','Wszystkie modele'),('commercial_frontier','Komercyjne + własne')]:
        labels = [f'{escape(p["name"])} ({p["size"]:.1f}M; {p["score"]:.2f}%)' for p in data[name]]
        parts.append(f'<p><b>{title}:</b> '+(' → '.join(labels) or 'brak modeli')+'</p>')
    parts.append('<p>Grupa komercyjna: deklaracje MIT, Apache-2.0, BSD, ISC, CC BY 4.0, CC0, Unlicense '
                 'oraz własne checkpointy zgodnie z deklaracją właściciela. Brak deklaracji i inne licencje '
                 'nie kwalifikują automatycznie do tej linii. Deklaracje źródłowe w tabeli pozostają widoczne.</p></details></div>')
    return ''.join(parts)


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
