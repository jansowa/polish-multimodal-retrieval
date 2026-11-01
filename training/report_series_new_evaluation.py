"""Separate per-gallery tables and paired component intervals from frozen results."""
import argparse
import json
import statistics
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pl_mmb_benchmark.full_gallery import paired_bootstrap
from pl_mmb_benchmark.report_parameters import parameter_cells
from pl_mmb_training.series_20261023 import ROOT,OUT,read,freeze,file_sha

BASE=Path('/mnt/c/Users/jsowa/benchmark-translations/series-2026-10-23/evaluation')


def own_report():
    registration=read(OUT/'reference_eval_registration_v2_20261029.json')
    counts=read(OUT/'our_efficiency_20261029.json')['parameters']
    parameter_note='Suma V+T obejmuje wyłącznie enkodery wizyjny i tekstowy. Nasz projektor (0,197 mln) jest poza tą sumą.'
    results={}
    for dataset in registration['datasets']:
        name=dataset['name']
        models={model:read(BASE/model/name/'v2_no_instructions.json') for model in registration['our_checkpoints']}
        means={}
        for direction in ('t2i','i2t'):
            means[direction]={}
            for metric in ('recall@1','recall@5','recall@10','mrr','ndcg@10'):
                values=[models['VN_FULL_seed'+str(seed)]['directions'][direction]['metrics'][metric] for seed in (42,43,44)]
                means[direction][metric]=dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values))
        comparisons={model:{d:paired_bootstrap(models[registration['selected_finalist']],models[model],d)
                            for d in ('t2i','i2t')} for model in ('VN_only','historical_A1')}
        results[name]=dict(gallery=dataset,models={m:{d:r['metrics'] for d,r in v['directions'].items()} for m,v in models.items()},
                          VN_FULL_three_seed_summary=means,selected_seed44_minus_control=comparisons)
        print('OUR_PAIRED_CI_COMPLETE '+name,flush=True)
    freeze(OUT/'our_new_gallery_comparisons_20261029.json',dict(status='complete',datasets=results,
        baseline_sha256=file_sha(OUT/'wiki_dev_zero_shot_pre_wiki_training.json'),
        no_new_checkpoint_selection=True,CI='conditional paired component bootstrap; not seed or pretraining uncertainty'))
    lines=['# Nasze modele na nowych galeriach — 29.10.2026','',
        'Checkpoint finalisty: seed 44, wskazany wcześniej przez medianę VN dev. Nie zmieniano go po nowych wynikach.',
        'Wiki jest zero-shot dla tych checkpointów; VN dev jest in-domain z etykietami modelowymi. XTD10 jest kontrolą, nie niezależnym holdoutem.',
        'Każda tabela ma własną galerię. Nie porównywać wyników między tabelami jako rankingu.',parameter_note,'']
    for name,value in results.items():
        lines += ['## '+name,'',f"Obrazy: {value['gallery']['images']}; unikalne query tekstowe: {value['gallery']['texts']}.",'',
                  '| Model | Enkodery V+T [M] | Wizyjny [M] | Tekstowy [M] | t2i R@1 | i2t R@1 | t2i MRR | i2t MRR |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for model,metrics in value['models'].items():
            t,i=metrics['t2i'],metrics['i2t']
            lines.append(f"| {model} | {parameter_cells(counts)} | {100*t['recall@1']:.2f}% | {100*i['recall@1']:.2f}% | {t['mrr']:.4f} | {i['mrr']:.4f} |")
        means=value['VN_FULL_three_seed_summary']
        lines += ['',f"VN_FULL, 3 seedy, t2i R@1: {100*means['t2i']['recall@1']['mean']:.2f}% ± {100*means['t2i']['recall@1']['sample_sd']:.2f} p.p. SD.",'',
                  '| Seed 44 minus kontrola | Enkodery V+T [M] | Wizyjny [M] | Tekstowy [M] | t2i Δ R@1 [95% CI], p.p. | i2t Δ R@1 [95% CI], p.p. |','|---|---:|---:|---:|---:|---:|']
        for model,comparison in value['selected_seed44_minus_control'].items():
            cells=[]
            for direction in ('t2i','i2t'):
                metric=comparison[direction]['metrics']['recall@1'];lo,hi=metric['ci95_pp']
                cells.append(f"{metric['delta_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}]")
            lines.append('| '+model+' | '+parameter_cells(counts)+' | '+' | '.join(cells)+' |')
        lines += ['']
    lines += ['Wszystkie R@5/10, NDCG@10 oraz przedziały pozostałych metryk są w JSON. Bootstrap: 10 000 replik, seed 20261023, jednostka: komponent.',
              'Tryb autorów jest identyczny dla naszych modeli: `[query]: ` stanowi część wytrenowanej przestrzeni. Nie powtarzano encoderów dla tego trybu.',
              'Kontrola stylu XTD10 human-vs-Qwen niedostępna: nie znaleziono jednoznacznych istniejących tłumaczeń dokładnie tych samych EN. Nie generowano zastępczych danych.','']
    with (OUT/'OUR_NEW_GALLERIES_20261029.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines))


def reference_report():
    registration=read(OUT/'reference_eval_registration_v3_20261029.json')
    if not (OUT/'reference_execution_complete_20261029.json').exists():raise ValueError('Complete the entire preregistered list before final comparison tables')
    efficiencies={'our_finalist':read(OUT/'our_efficiency_20261029.json')}
    efficiencies.update({m['alias']:read(BASE/'references'/m['alias']/'efficiency.json') for m in registration['reference_models']})
    output={};lines=['# Porównanie z konkurencją — serie zarejestrowane 29.10.2026','',
        'Paired Δ to wynik wcześniej wybranego VN_FULL seed44 minus referencja. Dodatnia wartość oznacza przewagę finalisty.',
        'Przedziały warunkowe po komponentach: 10 000 replik, seed 20261023. Nie obejmują zmienności treningu ani pretreningu.',
        'VN dev jest dla naszego modelu in-domain, z podpisami modelowymi. Wiki dev to odczyt przed jakimkolwiek nowym treningiem Wiki.',
        'XTD10 jest kontrolą: pełny zbiór ma znany overlap; wariant 897 wyklucza znane powiązania. Nie ustalono braku overlapu z pretreningiem konkurencji.',
        'Tryb v2 nie dodaje instrukcji; intrinsic `[query]: ` PolDense pozostaje częścią naszego modelu. Tryb autorów jest odrębnym odczytem wrażliwości.',
        'To osobne galerie. Nie tworzymy z ich wyników łącznego rankingu.',
        'Parametry V+T obejmują tylko wizję i tekst; pozostałe enkodery oraz nasz projektor są poza sumą. EmbeddingGemma2 miała wyłączone audio.',
        'Finalista: '+parameter_cells(efficiencies['our_finalist']['parameters'])+' mln (suma V+T | wizja | tekst); projektor osobno: 0,197 mln.','']
    for dataset in registration['datasets']:
        name=dataset['name'];output[name]={}
        for mode in ('v2_no_instructions','author_recommended'):
            ours=read(BASE/registration['selected_finalist']/name/(mode+'.json'))
            entries=[];table_rows=[]
            lines += ['## '+name+' / '+mode,'',
                f"Galeria: {dataset['images']} obrazów, {dataset['texts']} tekstów. Finalista seed44: t2i R@1 **{100*ours['directions']['t2i']['metrics']['recall@1']:.2f}%**, i2t R@1 **{100*ours['directions']['i2t']['metrics']['recall@1']:.2f}%**.",'']
            for model in registration['reference_models']:
                path=BASE/'references'/model['alias']/name/(mode+'.json');reference=read(path)
                if reference['registration_sha256']!=file_sha(OUT/'reference_eval_registration_v3_20261029.json'):
                    raise ValueError('Reference result registration differs')
                comparison={d:paired_bootstrap(ours,reference,d) for d in ('t2i','i2t')}
                flags=[]
                if model['same_size_class_le_100M']:flags.append('≤100M')
                if model['polish_adapted']:flags.append('PL-adaptowany')
                if model['english_text_only_control']:flags.append('kontrola EN')
                entries.append(dict(model=model,reference_result_sha256=file_sha(path),
                    metrics={d:v['metrics'] for d,v in reference['directions'].items()},paired=comparison))
                cells=[]
                for direction in ('t2i','i2t'):
                    metric=comparison[direction]['metrics']['recall@1'];lo,hi=metric['ci95_pp']
                    cells.extend([f"{100*reference['directions'][direction]['metrics']['recall@1']:.2f}%",
                                  f"{metric['delta_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}]"])
                table_rows.append((model,'| '+model['alias']+' | '+parameter_cells(efficiencies[model['alias']]['parameters'])+' | '+(', '.join(flags) or '>100M')+' | '+' | '.join(cells)+' |'))
            for title,flag in (('Wszystkie referencje',None),('Klasa ≤100M','same_size_class_le_100M'),('Modele polsko-adaptowane','polish_adapted')):
                lines += ['### '+title,'',
                    '| Model | Enkodery V+T [M] | Wizyjny [M] | Tekstowy [M] | Klasa | t2i R@1 | Δ t2i [95% CI], p.p. | i2t R@1 | Δ i2t [95% CI], p.p. |',
                    '|---|---:|---:|---:|---|---:|---:|---:|---:|']
                lines.extend(row for model,row in table_rows if flag is None or model[flag])
                lines+=['']
            output[name][mode]=entries
            lines+=['']
            print('REFERENCE_CI_COMPLETE '+name+' '+mode,flush=True)
    lines+=['## Efektywność','',
        'Wspólne wejścia VN dev, obrazy zdekodowane przed pomiarem. CPU FP32, batch1, 2 wątki; GPU FP32 weights + BF16 autocast, batch64. 10 rozgrzewek + 100 pomiarów. Mediany ms na element.','',
        '| Model | Enkodery V+T [M] | Wizyjny [M] | Tekstowy [M] | Wymiar | CPU query | CPU obraz | GPU query | GPU obraz |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for alias,entry in efficiencies.items():
        values=[]
        for key in ('cpu_query','cpu_image','cuda_query','cuda_image'):
            value=entry['measurements'][key]
            values.append(f"{value['median_ms_per_item']:.3f}" if value['status']=='measured' else value['status'])
        lines.append('| '+alias+' | '+parameter_cells(entry['parameters'])+' | '+str(entry['dimension'])+' | '+' | '.join(values)+' |')
    freeze(OUT/'reference_new_gallery_comparisons_20261029.json',dict(status='complete',datasets=output,efficiency=efficiencies,
        registration_sha256=file_sha(OUT/'reference_eval_registration_v3_20261029.json'),XM3600_scored=False))
    with (OUT/'REFERENCE_COMPARISON_20261029.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--references',action='store_true');args=parser.parse_args()
    reference_report() if args.references else own_report()
