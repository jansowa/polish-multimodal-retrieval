"""Compare quality and actual inference parameter counts for the first four backbones."""
import argparse
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pl_mmb_training.common import TASKS, read_json, write_json

LABELS = {'siglip2-reference': 'SigLIP2 Base/16 256 (E2b)', 'dinov2-small': 'DINOv2-Small',
          'mobileclip2-s2': 'MobileCLIP2-S2', 'convnext-tiny': 'ConvNeXt-Tiny', 'ibot-vits16': 'iBOT ViT-S/16'}


def metric(runs, path):
    values = []
    for run in runs:
        value = run
        for key in path:
            value = value[key]
        values.append(value)
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None


def percent(runs, path):
    mean, std = metric(runs, path)
    return f'{mean*100:.2f}' + (f' ± {std*100:.2f}' if std is not None else '')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'training/runs/vision-four')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/training_vision_four.md')
    args = parser.parse_args()
    summary = read_json(args.run_dir / 'summary.json')
    if summary['status'] != 'complete' or len(summary['models']) != 4 or not summary['stop_after_four']:
        raise ValueError('Export only the completed first four requested experiments')
    groups = {'siglip2-reference': summary['reference'], **summary['models']}
    text_params = summary['text_parameters']
    reference_r1 = metric(summary['reference']['runs'], ['benchmark', 'scores', 'mscoco_t2i'])[0]
    rows = []
    for key, group in groups.items():
        encoder, runs = group['encoder'], group['runs']
        projector_params = runs[0]['metadata']['projector_parameter_count']
        r1, std = metric(runs, ['benchmark', 'scores', 'mscoco_t2i'])
        rows.append(dict(model=key, label=LABELS[key], vision_parameters=encoder['vision_parameters'],
                         text_parameters=text_params, projector_parameters=projector_params,
                         full_embedder_parameters=encoder['vision_parameters']+text_params+projector_params,
                         trainable_parameters=runs[0]['metadata']['trainable_parameter_count'],
                         pretraining=encoder['pretraining'], feature_dim=encoder['feature_dim'], input_size=encoder['input_size'],
                         seeds=[r['seed'] for r in runs], mscoco_t2i_r1=r1, std=std,
                         delta_vs_sigmoid_siglip2=r1-reference_r1, delta_vs_full_e1=r1-.417))
    text = '# Porównanie czterech frozen vision encoders\n\n'
    text += ('Wykonano wyłącznie pierwsze cztery testy w zleconej kolejności; dalszych modeli nie uruchomiono. '
             'Punkt odniesienia: trzy zachowane seedy SigLIP2 + sigmoid loss (E2b).\n\n')
    text += ('Stałe: frozen PolDense-32M, sigmoid loss z uczoną temperaturą i biasem, liniowy projector, '
             '70 289 train / 7 695 validation (67 445 unikalnych par w train), identyczne splity '
             'i kolejność batchy dla seeda 42, 20 epok, batch 256, AdamW lr=0.001, wd=0.01, scheduler=none. '
             'Checkpoint wybiera minimum własnej validation loss. Wszystkie frozen features cache offline.\n\n')
    text += ('Projector jest liniowy dla każdego modelu; liczba jego wejść wynika z wymiaru backbone’u. '
             'Każdy encoder zachowuje swój oficjalny preprocessing i rozdzielczość. '
             'Wartości jakości: R@1 w %, średnia ± std tylko gdy jest więcej niż jeden seed.\n\n')
    text += '## Jakość i liczba parametrów\n\n'
    text += '| Model | Vision M | Nowy projector M | Cały embedder M | Seedy | MSCOCO t2i R@1 | Δ vs SigLIP2 E2b (p.p.) |\n'
    text += '|---|---:|---:|---:|---:|---:|---:|\n'
    for row in rows:
        text += (f'| {row["label"]} | {row["vision_parameters"]/1e6:.3f} | {row["projector_parameters"]/1e6:.3f} | '
                 f'{row["full_embedder_parameters"]/1e6:.3f} | {len(row["seeds"])} | '
                 f'{percent(groups[row["model"]]["runs"], ["benchmark","scores","mscoco_t2i"])} | '
                 f'{row["delta_vs_sigmoid_siglip2"]*100:+.2f} |\n')
    text += (f'\nPolDense ma **{text_params:,} parametrów**. Cały embedder = używana część vision '
             '+ PolDense + nowy liniowy projector. Uczone skala/bias lossu nie są potrzebne w inference. '
             'Oryginalna text tower MobileCLIP2 nie jest używana ani liczona. Jego oryginalna '
             'image projection jest zachowana i wliczona w vision. ConvNeXt pomija końcowy classifier '
             'Linear (pełny klasyfikator ImageNet ma 28 589 128 parametrów).\n\n')
    text += '## Reprezentacje i pretraining\n\n| Model | Pretraining | Input px | Feature dim | Reprezentacja |\n|---|---|---:|---:|---|\n'
    for row in rows:
        stats = groups[row['model']]['encoder']
        text += (f'| {row["label"]} | {row["pretraining"]} | {row["input_size"]} | {row["feature_dim"]} | '
                 f'{stats["representation"]} |\n')
    decision = read_json(args.run_dir / 'dinov2_replication.json')
    text += (f'\nDINOv2 seed 42: **{decision["first_test_r1"]*100:.2f}% R@1**. '
             f'Próg ≥38% {"osiągnięty" if decision["first_test_r1"] >= .38 else "nieosiągnięty"}; '
             f'użyto seedów {decision["seeds"]}. '
             'Jest to jawna decyzja o replikacji na podstawie testu; nie strojono na nim LR, '
             'temperatury początkowej, architektury ani wyboru epoki. Progi 40 i 41,7% '
             'odnoszą się do wcześniejszego E1; aktualny sigmoid/SigLIP2 ma 49,70%.\n\n')
    matched = next(r for r in summary['reference']['runs'] if r['seed'] == 42)['benchmark']['scores']['mscoco_t2i']
    mobile = groups['mobileclip2-s2']['runs'][0]['benchmark']['scores']['mscoco_t2i']
    text += (f'Porównanie tego samego seeda 42: SigLIP2 **{matched*100:.2f}%**, MobileCLIP2 **{mobile*100:.2f}%**. '
             'Referencyjne 49,70% jest średnią trzech seedów; jeden seed MobileCLIP2 nie wystarcza '
             'do rozstrzygnięcia niewielkiej różnicy między tymi modelami.\n\n')
    text += '## MSCOCO text→image — dotychczasowe lokalne pule MMEB\n\n'
    text += '| Model | R@1 | R@5 | R@10 |\n|---|---:|---:|---:|\n'
    for key, group in groups.items():
        text += '| ' + LABELS[key] + ' | ' + ' | '.join(percent(group['runs'], ['benchmark','local','mscoco_t2i',f'recall@{k}']) for k in (1,5,10)) + ' |\n'
    text += '\n## Retrieval dwukierunkowy — osobny fixed-gallery protocol\n\n'
    text += ('Ta sama galeria 1002 obrazów / 1000 captionów, 1000 zapytań t2i i 201 zapytań i2t, '
             'multi-positive qrels. To rozszerzenie MSCOCO MMEB, nie oficjalny test COCO 5k.\n\n')
    text += '| Model | t2i R@1 | t2i R@5 | t2i R@10 | i2t R@1 | i2t R@5 | i2t R@10 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for key, group in groups.items():
        text += '| ' + LABELS[key] + ' | ' + ' | '.join(percent(group['runs'], ['benchmark','global_mscoco',d,f'recall@{k}']) for d in ('t2i','i2t') for k in (1,5,10)) + ' |\n'
    text += '\n## Cztery benchmarki — R@1\n\n| Model | CIRR | FashionIQ | MSCOCO_t2i | RefCOCO | AVG |\n|---|---:|---:|---:|---:|---:|\n'
    for key, group in groups.items():
        text += '| ' + LABELS[key] + ' | ' + ' | '.join(percent(group['runs'], ['benchmark','scores',task]) for task in (*TASKS,'AVG')) + ' |\n'
        if key != 'siglip2-reference':
            text += '| Δ vs SigLIP2 E2b (p.p.) | ' + ' | '.join(f'{(metric(group["runs"],["benchmark","scores",task])[0]-metric(summary["reference"]["runs"],["benchmark","scores",task])[0])*100:+.2f}' for task in (*TASKS,'AVG')) + ' |\n'
    text += '\nCIRR/FashionIQ/RefCOCO i AVG pozostają diagnostyczne; nie wybierano pod nie architektury.\n\n'
    text += '## Historia treningu\n\n| Model | Seed | Initial val loss | Best val loss | Epoka | Train loss final |\n|---|---:|---:|---:|---:|---:|\n'
    for key, group in summary['models'].items():
        for run in group['runs']:
            text += (f'| {LABELS[key]} | {run["seed"]} | {run["history"][0]["validation_loss"]:.5f} | '
                     f'{run["validation_loss"]:.5f} | {run["checkpoint_epoch"]} | {run["history"][-1]["train_loss"]:.5f} |\n')
    text += ('\nDokładne liczby parametrów, rewizje i SHA checkpointów, preprocessing, feature cache, '
             'run_id i pełne metryki są w JSON. Modele różnią się też danymi pretrainingu i native '
             'preprocessingiem: porównanie bada gotowe reprezentacje, nie izoluje samej metody pretrainingu.\n\n')
    text += ('Źródła oficjalne: [DINOv2](https://huggingface.co/facebook/dinov2-small), '
             '[MobileCLIP2](https://huggingface.co/apple/MobileCLIP2-S2), '
             '[ConvNeXt-Tiny](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html), '
             '[iBOT](https://github.com/bytedance/ibot).\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding='utf-8')
    write_json(args.output.with_suffix('.json'), dict(comparison=rows, experiment=summary, dinov2_replication=decision))
    with args.output.with_suffix('.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with args.output.with_name(args.output.stem + '_loss.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model','seed','epoch','train_loss','validation_loss','val_t2i_r1','val_i2t_r1','temperature','selected_checkpoint'])
        for key, group in summary['models'].items():
            for run in group['runs']:
                for epoch in run['history']:
                    writer.writerow([key,run['seed'],epoch['epoch'],epoch['train_loss'],epoch['validation_loss'],
                                     epoch['validation']['t2i']['recall@1'],epoch['validation']['i2t']['recall@1'],
                                     epoch['temperature'],epoch['epoch']==run['checkpoint_epoch']])
    print(args.output)


if __name__ == '__main__':
    main()
