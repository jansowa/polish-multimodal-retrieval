"""Export P0/E2 comparison, validation decisions and complete epoch logs."""
import argparse
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pl_mmb_training.common import read_json, write_json, file_sha, TASKS


def formatted(values, with_std=True):
    if not values:
        return '—'
    label = f'{statistics.mean(values) * 100:.2f}'
    if with_std and len(values) > 1:
        label += f' ± {statistics.stdev(values) * 100:.2f}'
    return label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'training/runs/p0-e2')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/training_p0_e2.md')
    args = parser.parse_args()
    summary = read_json(args.run_dir / 'summary.json')
    if summary['status'] != 'complete' or summary['selection_uses_test']:
        raise ValueError('Export only complete, validation-selected experiments')
    baseline = summary['baseline']['runs']
    groups = {'E1 frozen': baseline, **summary['runs']}
    val_graph = read_json(args.run_dir / 'validation_graph.json')
    test_graph = read_json(args.run_dir / 'benchmark_graph.json')
    config = read_json(args.run_dir / 'config.json')
    first = baseline[0]['metadata']
    selected = summary['selection']['selected_loss']
    text = '# P0 + E2a/E2b\n\n'
    text += ('Pełny E1 pozostaje zamrożonym baseline: lokalne MSCOCO MMEB t2i R@1 '
             '**41,70% ± 0,20 p.p.** (3 seedy). PolDense i vision są frozen. '
             'Zmiana E2 dotyczy wyłącznie lossu i jego temperatury/biasu.\n\n')
    text += (f'Train: **{first["dataset_size"]} rekordów**, {first["number_of_unique_examples_seen"]} '
             f'unikalnych par PL–obraz, {first["unique_target_images"]} obrazów. Validation: '
             f'{first["validation_size"]} rekordów; retrieval wyboru lossu: '
             f'{len(val_graph["texts"])} unikalnych captionów / {len(val_graph["images"])} obrazów '
             'z 2093 rekordów M-BEIR MSCOCO w istniejącym validation splicie.\n\n')
    text += (f'Budżet każdego treningu: {config["optimization"]["epochs"]} epok, '
             f'{first["optimization_steps"]} kroków, {first["example_presentations"]} ekspozycji rekordów, '
             'batch=256, AdamW lr=0.001, wd=0.01 dla projectora, scheduler=none. '
             'Te same inicjalizacje i permutacje batchy dla odpowiadających sobie seedów.\n\n')
    text += ('Checkpoint każdego E2 wybiera minimum jego validation loss, tak jak w E1. '
             'Losy i dodatkowe seedy wybiera wyłącznie validation retrieval. Wartości różnych '
             'funkcji loss nie są bezpośrednio porównywalne.\n\n')
    text += f'Wybrany nowy loss: **{selected}**. Replikacje po progu ≥1 p.p. na validation t2i R@1: '
    text += ', '.join(summary['selection']['scout_repeat']) or 'brak'
    text += '. Decyzje zapisano przed ewaluacją testu w `replication_decision.json` i `selection.json`.\n\n'
    text += '## Walidacja retrieval — wszystkie konfiguracje\n\n'
    text += '| Model | Seedy | t2i R@1 | t2i R@5 | t2i R@10 | i2t R@1 | i2t R@5 | i2t R@10 |\n'
    text += '|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for name, runs in groups.items():
        values = [r['validation'] if name == 'E1 frozen' else r['selected_validation'] for r in runs]
        cells = [formatted([v[direction][f'recall@{k}'] for v in values]) for direction in ('t2i', 'i2t') for k in (1, 5, 10)]
        text += '| ' + name + ' | ' + str(len(runs)) + ' | ' + ' | '.join(cells) + ' |\n'
    text += '\n## Benchmark lokalny MMEB — tylko konfiguracja wybrana na val\n\n'
    text += '| Model | Seedy testu | CIRR | FashionIQ | MSCOCO_t2i | RefCOCO | AVG |\n'
    text += '|---|---:|---:|---:|---:|---:|---:|\n'
    base_scores = {key: statistics.mean(r['benchmark']['scores'][key] for r in baseline) for key in (*TASKS, 'AVG')}
    for name, runs in groups.items():
        tested = [r for r in runs if r['benchmark'] is not None]
        text += '| ' + name + ' | ' + str(len(tested)) + ' | ' + ' | '.join(
            formatted([r['benchmark']['scores'][key] for r in tested]) for key in (*TASKS, 'AVG')) + ' |\n'
        if tested and name != 'E1 frozen':
            text += '| Δ vs pełny E1 (p.p.) | — | ' + ' | '.join(
                f'{(statistics.mean(r["benchmark"]["scores"][key] for r in tested) - base_scores[key])*100:+.2f}'
                for key in (*TASKS, 'AVG')) + ' |\n'
    text += '\nNie wybrany loss ma wyniki validation i status `not_selected`; jego test nie został uruchomiony.\n\n'
    text += '| MSCOCO lokalny | t2i R@1 | t2i R@5 | t2i R@10 |\n|---|---:|---:|---:|\n'
    for name, runs in groups.items():
        values = [r['benchmark']['local']['mscoco_t2i'] for r in runs if r['benchmark'] is not None]
        text += '| ' + name + ' | ' + ' | '.join(formatted([v[f'recall@{k}'] for v in values]) for k in (1, 5, 10)) + ' |\n'
    text += '\n'
    text += '## Rozszerzony benchmark dwukierunkowy — osobny protokół\n\n'
    counts = baseline[0]['benchmark']['global_mscoco']
    text += (f'`fixed-gallery-multi-positive-v1`: galeria {len(test_graph["images"])} obrazów '
             f'i {len(test_graph["texts"])} unikalnych captionów z przykładów MSCOCO MMEB. '
             f'Liczba zapytań: {counts["t2i"]["queries"]} tekstowych oraz '
             f'**{counts["i2t"]["queries"]} obrazowych**. '
             'i2t pyta wyłącznie o obrazy ze znanym pozytywnym captionem; obrazy bez qrels pozostają '
             'negatywnymi kandydatami t2i. To osobny protokół, **nie oficjalne COCO 5k** '
             'i nie odwrócenie lokalnych pul. Multi-positive i tie policy są zapisane w grafie.\n\n')
    text += '| Model | t2i R@1 | t2i R@5 | t2i R@10 | i2t R@1 | i2t R@5 | i2t R@10 |\n'
    text += '|---|---:|---:|---:|---:|---:|---:|\n'
    for name, runs in groups.items():
        values = [r['benchmark']['global_mscoco'] for r in runs if r['benchmark'] is not None]
        text += '| ' + name + ' | ' + ' | '.join(formatted([v[d][f'recall@{k}'] for v in values])
                                                for d in ('t2i', 'i2t') for k in (1, 5, 10)) + ' |\n'
    text += '\n## Loss i temperatura\n\n'
    text += '| Model | Seed | Val loss przed treningiem | Najlepsza val loss | Epoka | Train loss final | Wybrana temperatura |\n'
    text += '|---|---:|---:|---:|---:|---:|---:|\n'
    for name, runs in summary['runs'].items():
        for run in runs:
            text += (f'| {name} | {run["seed"]} | {run["history"][0]["validation_loss"]:.5f} | '
                     f'{run["validation_loss"]:.5f} | {run["checkpoint_epoch"]} | '
                     f'{run["history"][-1]["train_loss"]:.5f} | '
                     f'{run["metadata"]["temperature_logit_scale"]["temperature"]:.5f} |\n')
    text += ('\nPełne run_id, rewizje modeli, trainable params, best/selected validation scores, '
             'temperatura i bias oraz wszystkie R@1/5/10 są w JSON. Oryginalny E1 nie mierzył '
             'retrieval co epokę; P0 mierzy zachowane checkpointy i nie dopisuje fikcyjnej historii.\n\n')
    text += ('Dalsze etapy pozostają planem: [P0–P11](../training/PLAN.md). '
             'Oryginalny wynik E1: [raport](training_e0_e1_full.md). '
             'Reprodukcja: `training/freeze_e1.py`, `training/loss_ablation.py all`, '
             '`training/export_ablation.py`.\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding='utf-8')
    write_json(args.output.with_suffix('.json'), dict(config=config, result=summary,
               baseline_lock_sha256=file_sha(args.run_dir / 'baseline/baseline_lock.json'),
               validation_graph_sha256=val_graph['sha256'], benchmark_graph_sha256=test_graph['sha256']))
    with args.output.with_name(args.output.stem + '_loss.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['experiment', 'seed', 'epoch', 'train_loss', 'validation_loss', 'val_t2i_r1',
                         'val_i2t_r1', 'val_mean_r1', 'temperature', 'logit_scale', 'logit_bias', 'selected_checkpoint'])
        for name, runs in summary['runs'].items():
            for run in runs:
                for epoch in run['history']:
                    writer.writerow([name, run['seed'], epoch['epoch'], epoch['train_loss'], epoch['validation_loss'],
                                     epoch['validation']['t2i']['recall@1'], epoch['validation']['i2t']['recall@1'],
                                     epoch['validation']['mean_r1'], epoch['temperature'], epoch['logit_scale'],
                                     epoch['logit_bias'], epoch['epoch'] == run['checkpoint_epoch']])
    print(args.output)


if __name__ == '__main__':
    main()
