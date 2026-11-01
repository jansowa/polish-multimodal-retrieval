"""Export quality, parameter counts, license provenance and complete loss history."""
import argparse
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pl_mmb_training.common import TASKS, read_json, write_json

LABELS = {'siglip2': 'SigLIP2 Base/16 256', 'mobileclip2-s2': 'MobileCLIP2-S2',
          'tinyclip-39m': 'TinyCLIP 39M/16 YFCC', 'tinyclip-40m': 'TinyCLIP 40M/32 LAION',
          'chinese-clip-rn50': 'Chinese-CLIP RN50', 'rn50-cc12m': 'RN50 CC12M', 'rn50-yfcc15m': 'RN50 YFCC'}

def mean_std(runs, path):
    values = []
    for run in runs:
        value = run
        for key in path:
            value = value[key]
        values.append(value)
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None

def percent(runs, path):
    mean, std = mean_std(runs, path)
    return f'{100*mean:.2f}' + (f' ± {100*std:.2f}' if std is not None else '')

def export(root, output):
    summary = read_json(root / 'summary.json')
    if summary['status'] != 'complete' or len(summary['models']) != 5 or not summary['stop_after_five']:
        raise ValueError('Export only the completed first five, including qualified replications')
    decisions = read_json(root / 'replication_decisions.json')
    groups = {'siglip2': summary['reference'], 'mobileclip2-s2': summary['previous_models']['mobileclip2-s2'], **summary['models']}
    reference_score = {task: mean_std(summary['reference']['runs'], ['benchmark', 'scores', task])[0] for task in (*TASKS, 'AVG')}
    text = ('# Pięć frozen vision backbones — sigmoid / PolDense-32M\n\n'
            'Zakończono dokładnie pierwszą piątkę. RN101 i TinyCLIP 61M nie uruchomiono. '
            'Wszystkie nowe modele: seed 42; dodatkowo 43/44 zgodnie z zapisaną decyzją po pięciu scoutach.\n\n'
            'Stałe: 70 289 train / 7695 validation, 67 445 unikalnych par train, frozen PolDense i vision, '
            'liniowy projector, multi-positive sigmoid z uczoną skalą/biasem, 20 epok, batch 256, '
            'AdamW lr=.001, wd=.01 (parametry lossu wd=0), bez schedulera. '
            'Checkpoint: minimum validation loss. Ten sam manifest, blacklisty i cache tekstów EN/PL. '
            'Natywny preprocessing, standardowy image embedding z oryginalną projekcją, offline cache.\n\n'
            'Metryki R@k w %. ± to odchylenie standardowe między seedami. '
            'Jest to dotychczasowy lokalny benchmark MSCOCO MMEB, nie oficjalny COCO 5k.\n\n')
    text += '| Model | Vision M | Projector M | Cały embedder M | Seedy | t2i R@1 | Δ vs SigLIP2 p.p. | Licencja wag |\n|---|---:|---:|---:|---|---:|---:|---|\n'
    rows = []
    for key, group in groups.items():
        encoder, runs = group['encoder'], group['runs']
        projector = runs[0]['metadata']['projector_parameter_count']
        r1, std = mean_std(runs, ['benchmark', 'scores', 'mscoco_t2i'])
        license_info = encoder.get('license', {})
        license_name = license_info.get('declared_weights_license', 'Apple custom (reference)' if key == 'mobileclip2-s2' else 'Apache-2.0 (reference)')
        license_source = license_info.get('source', 'https://huggingface.co/apple/MobileCLIP2-S2' if key == 'mobileclip2-s2' else 'https://huggingface.co/google/siglip2-base-patch16-256')
        row = dict(model=key, vision_parameters=encoder['vision_parameters'], projector_parameters=projector,
                   text_parameters=summary['text_parameters'], full_parameters=encoder['vision_parameters']+projector+summary['text_parameters'],
                   seeds=[r['seed'] for r in runs], mscoco_r1=r1, std=std, delta_siglip2=r1-reference_score['mscoco_t2i'],
                   license=license_name, license_source=license_source)
        rows.append(row)
        text += (f'| {LABELS[key]} | {row["vision_parameters"]/1e6:.3f} | {projector/1e6:.3f} | {row["full_parameters"]/1e6:.3f} | '
                 f'{row["seeds"]} | {percent(runs,["benchmark","scores","mscoco_t2i"])} | {100*row["delta_siglip2"]:+.2f} | [{license_name}]({license_source}) |\n')
    new_rows = [row for row in rows if row['model'] in summary['models']]
    best = max(new_rows, key=lambda row: row['mscoco_r1'])
    mobile = next(row for row in rows if row['model'] == 'mobileclip2-s2')
    text += (f'\nNajlepszy z nowych kandydatów według średniej testowej: **{LABELS[best["model"]]}**, '
             f'**{percent(groups[best["model"]]["runs"], ["benchmark","scores","mscoco_t2i"])}% R@1** '
             f'przy **{best["vision_parameters"]/1e6:.3f}M vision parameters**. '
             f'Różnica do MobileCLIP2-S2: {100*(best["mscoco_r1"]-mobile["mscoco_r1"]):+.2f} p.p.; '
             f'do SigLIP2: {100*best["delta_siglip2"]:+.2f} p.p.\n\n')
    for threshold, limit in ((.45,40000000),(.47,50000000),(.49,60000000)):
        qualified = [LABELS[row['model']] for row in new_rows if row['mscoco_r1'] >= threshold-1e-12 and row['vision_parameters'] <= limit]
        text += f'- Cel ≥{100*threshold:g}% przy ≤{limit/1e6:g}M: ' + (', '.join(qualified) if qualified else 'brak w pierwszej piątce') + '.\n'
    text += ('\nCały embedder = używany image encoder + PolDense (31 883 136 parametrów) + nowy projector. '
             'Oryginalne text towers i dwa skalary lossu pominięte w inference. '
             'Deklaracje MIT/Apache-2.0 w kartach modeli umożliwiają rozpatrywanie użycia komercyjnego '
             'z zachowaniem warunków licencji; nie są audytem praw do danych pretrainingowych ani całego produktu. '
             'Karty modeli i kod mają zapisane rewizje i SHA. Chinese-CLIP: karta wag Apache-2.0, repozytorium kodu MIT. '
             '[MobileCLIP2](https://huggingface.co/apple/MobileCLIP2-S2) ma apple-amlr; '
             '[licencja modeli Apple](https://github.com/apple-aiml-research/ml-mobileclip/blob/main/LICENSE_MODELS) '
             'ogranicza użycie do badań niekomercyjnych i wyłącza rozwój produktów. '
             'Nie traktujemy go jako kandydata komercyjnego na tej licencji. '
             '[PolDense-32M](https://huggingface.co/OPI-PIB/PolDense-32M) deklaruje Gemma, '
             'więc cały embedder nie jest MIT/Apache-only; obejmują go również '
             '[warunki Gemma](https://ai.google.dev/gemma/terms), w tym obowiązki dystrybucji i ograniczenia użycia.\n\n')
    text += '## Wyniki scoutów i replikacja\n\n| Model | Scout seed 42 R@1 | Klasa vision ≤M | Seedy | Powody |\n|---|---:|---:|---|---|\n'
    for key, decision in decisions.items():
        text += f'| {LABELS[key]} | {100*decision["scout_r1"]:.2f} | {decision["size_class_upper_bound"]/1e6:g} | {decision["seeds"]} | {", ".join(decision["reasons"]) or "brak spełnionego progu"} |\n'
    text += ('\nKlasy przed testem: ≤40M, (40M,50M], (50M,60M]. „Istotnie mniej” = ≥20% mniej niż MobileCLIP2-S2. '
             'Kryterium najlepszego modelu w klasie obejmuje nowe modele z deklaracją MIT/Apache-2.0 i wcześniejsze '
             'DINOv2/ConvNeXt/iBOT; nie obejmuje Apple custom. To jawna selekcja dalszych seedów z testu na zlecenie użytkownika. '
             'Nie zmieniano LR/lossu/danych ani nie wybierano epok według testu.\n\n')
    text += '## MSCOCO lokalne pule\n\n| Model | t2i R@1 | R@5 | R@10 |\n|---|---:|---:|---:|\n'
    for key, group in groups.items():
        text += '| ' + LABELS[key] + ' | ' + ' | '.join(percent(group['runs'], ['benchmark','local','mscoco_t2i',f'recall@{k}']) for k in (1,5,10)) + ' |\n'
    text += '\n## Osobna stała galeria multi-positive\n\n1002 obrazy / 1000 captionów, 1000 zapytań t2i, 201 zapytań i2t.\n\n| Model | t2i R@1 | R@5 | R@10 | i2t R@1 | R@5 | R@10 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for key, group in groups.items():
        text += '| ' + LABELS[key] + ' | ' + ' | '.join(percent(group['runs'], ['benchmark','global_mscoco',d,f'recall@{k}']) for d in ('t2i','i2t') for k in (1,5,10)) + ' |\n'
    text += '\n## Cztery benchmarki\n\nCIRR/FashionIQ/RefCOCO/AVG diagnostycznie. Δ do SigLIP2 E2b i do poprzedniej pozycji w zleconej serii (średnie).\n\n| Model | CIRR | FashionIQ | MSCOCO_t2i | RefCOCO | AVG |\n|---|---:|---:|---:|---:|---:|\n'
    previous = reference_score
    for key, group in groups.items():
        scores = {task: mean_std(group['runs'], ['benchmark','scores',task])[0] for task in (*TASKS,'AVG')}
        text += '| ' + LABELS[key] + ' | ' + ' | '.join(percent(group['runs'], ['benchmark','scores',task]) for task in (*TASKS,'AVG')) + ' |\n'
        if key in summary['models']:
            text += '| Δ vs SigLIP2 p.p. | ' + ' | '.join(f'{100*(scores[t]-reference_score[t]):+.2f}' for t in (*TASKS,'AVG')) + ' |\n'
            text += '| Δ vs poprzedni test p.p. | ' + ' | '.join(f'{100*(scores[t]-previous[t]):+.2f}' for t in (*TASKS,'AVG')) + ' |\n'
            previous = scores
    text += '\n## Historia treningu\n\n| Model | Seed | Val loss początek | Val loss minimum | Epoka | Train loss koniec |\n|---|---:|---:|---:|---:|---:|\n'
    for key, group in summary['models'].items():
        for run in group['runs']:
            text += f'| {LABELS[key]} | {run["seed"]} | {run["history"][0]["validation_loss"]:.5f} | {run["validation_loss"]:.5f} | {run["checkpoint_epoch"]} | {run["history"][-1]["train_loss"]:.5f} |\n'
    text += ('\nTinyCLIP 39M vs 40M zmienia zarówno patch, jak i dane pretrainingowe. '
             'Para RN50 ma identyczną architekturę i preprocessing; to porównanie gotowych checkpointów, '
             'bez gwarancji identycznych pozostałych ustawień ich pretrainingu. '
             'MobileCLIP2 ma jeden seed referencyjny; małe różnice nie dowodzą przewagi.\n')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding='utf-8')
    write_json(output.with_suffix('.json'), dict(comparison=rows, replication=decisions, experiment=summary))
    with output.with_suffix('.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with output.with_name(output.stem + '_loss.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model','seed','epoch','train_loss','validation_loss','validation_t2i_r1','validation_i2t_r1','selected'])
        for key, group in summary['models'].items():
            for run in group['runs']:
                for epoch in run['history']:
                    writer.writerow([key,run['seed'],epoch['epoch'],epoch['train_loss'],epoch['validation_loss'],epoch['validation']['t2i']['recall@1'],epoch['validation']['i2t']['recall@1'],epoch['epoch']==run['checkpoint_epoch']])
    print(output, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'training/runs/vision-commercial-five')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/training_vision_commercial_five.md')
    args = parser.parse_args()
    export(args.run_dir, args.output)
