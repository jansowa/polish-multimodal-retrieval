"""Read-only aggregation of completed benchmarks for the selected finalist."""
import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pl_mmb_training.series_20261023 import ROOT, OUT, read, file_sha, freeze
from pl_mmb_benchmark.report_parameters import encoder_counts, parameter_cells

ALIASES = ['our_finalist', 'metaclip-s16', 'metaclip-m16', 'mobileclip2-s2']
LABELS = ['VN_FULL seed44', 'MetaCLIP-S16 PL', 'MetaCLIP-M16 PL', 'MobileCLIP2-S2 (EN)']
DATASETS = [('wiki_native_dev', 'Wiki dev'), ('VN_dev', 'VN dev'),
            ('xtd10_full_1k', 'XTD10 pełne 1000'), ('xtd10_no_overlap', 'XTD10 bez znanego overlapu 897')]
METRICS = ['recall@1', 'recall@5', 'recall@10', 'mrr', 'ndcg@10']
DOMAIN = {
    'Wiki dev': ('OOD', 'OOD — Wiki'),
    'VN dev': ('ID', 'ID — VN'),
    'XTD10 pełne 1000': ('ID-image-source', 'ID — obrazy COCO; inne podpisy; 47 powiązanych z train'),
    'XTD10 bez znanego overlapu 897': ('ID-image-source', 'ID — obrazy COCO; bez znanego overlapu'),
    'Lokalny test MSCOCO': ('ID', 'ID — COCO, odłożony test'),
    'COCO val': ('ID', 'ID — COCO, odłożona walidacja'),
    'COCO test — fixed gallery': ('ID', 'ID — COCO, odłożony test'),
}


def write_once(path, payload):
    payload = payload.encode('utf-8')
    if path.exists():
        if path.read_bytes() != payload:
            previous = path.with_name(path.stem + '_previous_' + file_sha(path)[:12] + path.suffix)
            if not previous.exists():
                with previous.open('xb') as stream:
                    stream.write(path.read_bytes())
            path.write_bytes(payload)
    else:
        with path.open('xb') as stream:
            stream.write(payload)


def main():
    sources = {}
    def load(path):
        sources[str(path.relative_to(ROOT))] = file_sha(path)
        return read(path)
    comparison = load(OUT / 'reference_new_gallery_comparisons_20261029.json')
    own = load(OUT / 'our_new_gallery_comparisons_20261029.json')
    reg = load(OUT / 'reference_eval_registration_v3_20261029.json')
    posthoc = load(OUT / 'vn_full_seeds_posthoc.json')
    coco = load(OUT / 'vn_full_COCO_val_per_seed.json')['44']
    xtd_audit = load(OUT / 'xtd10_v2/preparation.json')
    xtd_readme = OUT / 'xtd10_source/README.md'
    sources[str(xtd_readme.relative_to(ROOT))] = file_sha(xtd_readme)
    if xtd_audit['overlap_images']['FULL_CLEAN']['either'] != 47 or xtd_audit['clean_images'] != 897:
        raise ValueError('Recheck the XTD10 overlap labels against the frozen audit')
    if posthoc['selected_seed'] != 44 or reg['selected_finalist'] != 'VN_FULL_seed44':
        raise ValueError('Previously selected checkpoint differs')
    params = {alias: encoder_counts(comparison['efficiency'][alias]['parameters']) for alias in ALIASES}
    if any(p['total'] >= 100_000_000 for p in params.values()):
        raise ValueError('A selected encoder pair is not below 100M')
    small = {m['model']: m for m in reg['reference_models'] if m['alias'] in ALIASES}
    legacy = {}
    for path in (ROOT / 'benchmark/results').rglob('result.json'):
        result = read(path)
        model = small.get(result.get('model'))
        if not model or result.get('revision') != model['revision'] or not result.get('full_dataset'):
            continue
        if result.get('protocol') != 'mmeb-v1-no-instructions-raw-sum-cosine-v2':
            continue
        if result['dataset'] != 'mscoco_t2i':
            continue
        if model['alias'] == 'mobileclip2-s2' and result.get('numerics') != 'mobile-fp32-unit-range-reparam-v3':
            continue
        key = (model['alias'], result['dataset'])
        if key in legacy:
            raise ValueError('Ambiguous legacy result selection: ' + str(key))
        legacy[key] = result
        sources[str(path.relative_to(ROOT))] = file_sha(path)
    if len(legacy) != 3:
        raise ValueError('Expected all three small models on legacy MSCOCO')

    overview = []
    for key, title in [('total', 'Parametry enkoderów V+T [mln]'), ('vision', 'Enkoder wizyjny [mln]'), ('text', 'Enkoder tekstowy [mln]')]:
        overview.append([title, *[f"{params[a][key]/1e6:.2f}" for a in ALIASES]])
    details = []
    def add_details(dataset, protocol, direction, data, paired=None):
        for alias in ALIASES:
            metrics = data.get(alias)
            details.append(dict(dataset=dataset, protocol=protocol, direction=direction, model=alias,
                encoder_parameters=params[alias], metrics=metrics,
                domain_status=DOMAIN[dataset][0] if alias == 'our_finalist' else 'not_audited',
                domain_note=DOMAIN[dataset][1] if alias == 'our_finalist' else 'Nie ustalono pełnej ekspozycji treningowej konkurenta',
                selected_finalist_minus_model_R1=(paired or {}).get(alias)))
    def pair(data):
        return '—' if data is None else f"{100*data['t2i']['recall@1']:.2f} / {100*data['i2t']['recall@1']:.2f}"

    for key, title in DATASETS:
        entries = {e['model']['alias']: e for e in comparison['datasets'][key]['v2_no_instructions']}
        data = {'our_finalist': own['datasets'][key]['models']['VN_FULL_seed44']}
        data.update({a: entries[a]['metrics'] for a in ALIASES[1:]})
        for alias in ALIASES[1:]:
            author = next(e for e in comparison['datasets'][key]['author_recommended'] if e['model']['alias'] == alias)
            if author['metrics'] != data[alias]:
                raise ValueError('Author mode differs; report it separately')
        overview.append([title + ' R@1 t→i / i→t [%]', *[pair(data[a]) for a in ALIASES]])
        for direction in ('t2i', 'i2t'):
            add_details(title, 'v2 full gallery', direction, {a: v[direction] for a, v in data.items()},
                {a: entries[a]['paired'][direction]['metrics']['recall@1'] for a in ALIASES[1:]})
    for key, title in [('mscoco_t2i', 'Lokalny test MSCOCO')]:
        data = {a: legacy[(a, key)]['metrics'] for a in ALIASES[1:]}
        if key == 'mscoco_t2i':
            data['our_finalist'] = posthoc['selected_legacy_test']
        overview.append([title + ' P@1 [%]', *[f"{100*data[a]['precision@1']:.2f}" if a in data else '—' for a in ALIASES]])
        add_details(title, 'local MMEB candidate protocol v2', 'task-specific', data)
    for key, title, data in [('COCO_val', 'COCO val', coco),
                             ('COCO_fixed', 'COCO test — fixed gallery', posthoc['selected_fixed_gallery'])]:
        overview.append([title + ' R@1 t→i / i→t [%]', pair(data), '—', '—', '—'])
        for direction in ('t2i', 'i2t'):
            add_details(title, data['protocol'], direction, {'our_finalist': {m: data[direction][m] for m in METRICS if m in data[direction]}})
    overview.append(['Wymiar embeddingu', *[str(comparison['efficiency'][a]['dimension']) for a in ALIASES]])
    for key, title in [('cpu_query', 'CPU query [ms]'), ('cpu_image', 'CPU obraz [ms]'),
                       ('cuda_query', 'GPU query [ms/element]'), ('cuda_image', 'GPU obraz [ms/element]')]:
        overview.append([title, *[f"{comparison['efficiency'][a]['measurements'][key]['median_ms_per_item']:.3f}" for a in ALIASES]])
    for row in overview:
        status = next((note for name, (_, note) in DOMAIN.items()
                       if row[0].startswith(name + ' R@1') or row[0].startswith(name + ' P@1')), '—')
        row.insert(1, status)

    note = ('Model własny to **VN_FULL seed44**, finalista wybrany wcześniej medianą VN dev; '
            'nie dokonywano nowej selekcji po wynikach. Parametry obejmują tylko enkodery wizji i tekstu; '
            'osobny projektor ma 0,197 mln. MobileCLIP2-S2 jest kontrolą z tekstem EN. '
            'W komórkach z parą wyników kolejność to **tekst→obraz / obraz→tekst**. '
            '„—” oznacza brak odczytu dla tego checkpointu/protokołu. '
            'Nie liczymy średniej między różnymi galeriami.')
    domain_note = ('**ID / in-domain**: źródło danych obecne w uczeniu naszego projektora. '
                   '**OOD / out-of-domain**: źródło nieobecne w tym uczeniu. '
                   'Oznaczenia dotyczą VN_FULL seed44, nie całego pretreningu zamrożonych enkoderów. '
                   'Pełnej ekspozycji treningowej konkurencji nie ustalono; nie przypisujemy jej etykiety OOD. '
                   'Znana domena nie oznacza automatycznie użycia konkretnych przykładów testowych w treningu.')
    table = ['| Benchmark / parametr | Domena dla VN_FULL | ' + ' | '.join(LABELS) + ' |', '|---|---|---:|---:|---:|---:|']
    table += ['| ' + ' | '.join(row) + ' |' for row in overview]
    lines = ['# Nasz finalista i modele <100 mln — 30.10.2026', '', note, '', domain_note, '', *table, '',
        'Zakres: wyłącznie retrieval tekst↔obraz. Testy z zapytaniem obraz+tekst i grounding są poza tym porównaniem. '
        'COCO val i test fixed gallery nie mają jeszcze odpowiedników konkurencji w tych samych galeriach.', '',
        'Wiki: 397 obrazów / 331 tekstów; VN: 2000 / 2003; XTD10: 1000 / 1000 i 897 / 897. '
        'VN jest in-domain z podpisami modelowymi; Wiki to odczyt przed treningiem Wiki. '
        'COCO jest znaną domeną: FULL_CLEAN zawiera 17 869 par M-BEIR/MSCOCO train i 6103 przykłady SugarCrepe opartego na COCO. '
        'XTD10 jest oparty na MSCOCO2014: in-domain pod względem źródła obrazów, z innymi podpisami. '
        'Pełny XTD10 ma 47 obrazów powiązanych z FULL_CLEAN train; wariant 897 wyklucza znane powiązania z train, val i wcześniejszym eval. '
        'Brak znanego overlapu nie zmienia domeny obrazów na OOD.',
        'Lokalny MSCOCO ma 1000 zapytań i pule po 1000 kandydatów; nie jest oficjalnym COCO5k. '
        'COCO val: 419 obrazów / 2090 tekstów. COCO test fixed gallery: 1002 obrazy / 1000 tekstów, '
        'i2t obejmuje 201 uprawnionych zapytań obrazowych. Są to osobne protokoły.',
        'Czasy: mediana; CPU FP32, batch 1, 2 wątki; GPU BF16 autocast, batch 64, RTX 5070 Ti. '
        '10 rozgrzewek i 100 pomiarów; GPU podano na element batcha.', '', '## Wszystkie metryki', '',
        'R@1/5/10 i NDCG@10 poniżej w procentach; MRR w skali 0–1. W lokalnym MMEB P@1=R@1 (jedna pozytywna odpowiedź). '
        'CI to zapisane wcześniej sparowane 95% CI różnicy R@1: finalista minus referencja; nie liczono nowych przedziałów.', '']
    for dataset, protocol, direction in dict.fromkeys((d['dataset'], d['protocol'], d['direction']) for d in details):
        lines += ['### ' + dataset + ' / ' + direction, '', 'Protokół: ' + protocol + '.', '',
            '| Model | Domena | V+T [M] | Wizyjny [M] | Tekstowy [M] | R@1 | R@5 | R@10 | MRR | NDCG@10 | Δ R@1 [95% CI], p.p. |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in (d for d in details if (d['dataset'], d['protocol'], d['direction']) == (dataset, protocol, direction)):
            alias, metrics = row['model'], row['metrics'] or {}
            values = [('—' if key not in metrics else (f"{metrics[key]:.4f}" if key == 'mrr' else f"{100*metrics[key]:.2f}")) for key in METRICS]
            delta = row['selected_finalist_minus_model_R1']
            interval = '—' if delta is None else f"{delta['delta_pp']:+.2f} [{delta['ci95_pp'][0]:+.2f}, {delta['ci95_pp'][1]:+.2f}]"
            status = DOMAIN[dataset][1] if alias == 'our_finalist' else 'Nieustalona'
            lines.append('| ' + LABELS[ALIASES.index(alias)] + ' | ' + status + ' | ' + parameter_cells(comparison['efficiency'][alias]['parameters']) + ' | ' + ' | '.join(values) + ' | ' + interval + ' |')
        lines += ['']
    lines += ['## Tożsamości modeli', '',
        '- Nasz: `wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M` + `OPI-PIB/PolDense-32M` + Linear 512→384; seed44, SHA `' + posthoc['checkpoint_sha256'] + '`.']
    lines.extend('- `' + model['model'] + '`, revision `' + model['revision'] + '`.' for model in reg['reference_models'] if model['alias'] in ALIASES)
    payload = dict(status='complete', selected_checkpoint='VN_FULL_seed44', checkpoint_sha256=posthoc['checkpoint_sha256'],
        model_order=ALIASES, encoder_parameters=params, overview=overview, details=details,
        source_sha256=sources, scope='text-image retrieval only; no composed image+text queries or grounding',
        domain_scope='source exposure during our projector training; frozen-encoder pretraining and competitors not audited',
        domain_by_dataset={name: dict(status=status, note=note) for name, (status, note) in DOMAIN.items()},
        no_model_or_scoring_reruns=True, script_sha256=file_sha(__file__))
    stem = OUT / 'SMALL_MODELS_COMPARISON_20261030'
    write_once(stem.with_suffix('.md'), '\n'.join(lines) + '\n')
    buffer = io.StringIO(newline=''); writer = csv.writer(buffer, delimiter=';')
    writer.writerow(['Benchmark / parametr', 'Domena dla VN_FULL', *LABELS]); writer.writerows(overview)
    write_once(stem.with_suffix('.csv'), '\ufeff' + buffer.getvalue())
    write_once(stem.with_suffix('.json'), json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n')
    print('\n'.join(table))
    print('DETAIL_ROWS', len(details), 'SOURCE_FILES', len(sources))


if __name__ == '__main__':
    main()
