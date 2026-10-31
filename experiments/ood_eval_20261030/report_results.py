"""Full metrics, seed summaries and preregistered component-paired CIs on CPU."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import csv
from datetime import datetime,timezone
import io
import json
from pathlib import Path
import statistics
import tempfile

from prepare import ROOT,OUT,COORD,read,sha,load_common,committed_bytes,local_path,verify_marker
from acquire import ASSETS
from evaluate import gates,Budget


def main():
    common,registration,protocols=gates()
    from pl_mmb_benchmark.full_gallery import paired_bootstrap
    from pl_mmb_benchmark.report_parameters import encoder_counts
    budget=Budget(common,'final-report')
    own={model['id']:model for model in registration['models']}
    refs={model['alias']:model for model in registration['reference_models']}
    results={};artifacts=[];csv_rows=[]
    for model in list(own)+list(refs):
        for dataset,path in protocols.items():
            protocol=read(path)
            for mode in registration['prompt_modes']:
                target=ASSETS/'results'/model/dataset/(mode+'.json')
                result=read(target)
                if result['registration_sha256']!=sha(OUT/'execution_registration.json') or result['manifest_sha256']!=protocol['manifest_sha256']:
                    raise ValueError('Result from another frozen execution')
                if model in own and result['checkpoint_sha256']!=own[model]['sha256']:
                    raise ValueError('Checkpoint result identity differs')
                if model in refs and result['revision']!=refs[model]['revision']:
                    raise ValueError('Reference result revision differs')
                results[model,dataset,mode]=result
                artifacts.append(common.artifact('result:'+model+':'+dataset+':'+mode,target))
                parameters=encoder_counts(result['parameters'])
                for direction,values in result['directions'].items():
                    for metric,value in values['metrics'].items():
                        csv_rows.append(dict(model=model,variant=own[model]['variant'] if model in own else 'reference',
                            seed=own[model]['seed'] if model in own else '',dataset=dataset,prompt_mode=mode,
                            direction=direction,metric=metric,value=value,percentage=100*value,
                            encoder_total=parameters['total'],vision=parameters['vision'],text=parameters['text'],
                            head_separate=result['parameters'].get('head',0),shared_separate=result['parameters'].get('shared',0),
                            dimension=result['dimension'],exposure='source-OOD to projector' if model in own else 'pretraining exposure unestablished'))
    def write_text(path,text):
        data=text.encode('utf-8')
        old=path.stat().st_size if path.exists() else 0
        budget.reserve(2*len(data),'D',len(data))
        fd,temporary=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
        with os.fdopen(fd,'wb') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
        budget.added+=max(0,len(data)-old);budget.reserve()
    def write_csv(path,rows):
        stream=io.StringIO(newline='')
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
        write_text(path,stream.getvalue())
    write_csv(OUT/'metrics.csv',csv_rows)
    seed_rows=[]
    for dataset in protocols:
        for mode in registration['prompt_modes']:
            for variant in ('R0','R1','R3'):
                models=[model for model in own if own[model]['variant']==variant]
                for direction in ('t2i','i2t'):
                    for metric in ('recall@1','recall@5','recall@10','mrr','ndcg@10'):
                        values=[results[model,dataset,mode]['directions'][direction]['metrics'][metric] for model in models]
                        seed_rows.append(dict(dataset=dataset,prompt_mode=mode,variant=variant,direction=direction,
                            metric=metric,mean=statistics.mean(values),sample_SD=statistics.stdev(values),seeds=3,
                            mean_percentage=100*statistics.mean(values),SD_percentage=100*statistics.stdev(values)))
    write_csv(OUT/'seed_mean_SD.csv',seed_rows)
    representatives=registration['representative_model_ids']
    baseline=next(model for model in representatives if own[model]['variant']=='R0')
    pairs=[(model,ref) for model in representatives for ref in refs]
    pairs += [(model,baseline) for model in representatives if model!=baseline]
    conditional=[];ci_rows=[]
    for dataset in protocols:
        for mode in registration['prompt_modes']:
            for a,b in pairs:
                for direction in ('t2i','i2t'):
                    cache=ASSETS/'paired_CI'/dataset/mode/(a+'__'+b+'__'+direction+'.json')
                    if cache.exists():
                        value=read(cache)
                    else:
                        value=paired_bootstrap(results[a,dataset,mode],results[b,dataset,mode],direction,replicates=10000,seed=20261023)
                        value.update(ours=a,reference=b,dataset=dataset,prompt_mode=mode,
                            ours_result_sha256=sha(ASSETS/'results'/a/dataset/(mode+'.json')),
                            reference_result_sha256=sha(ASSETS/'results'/b/dataset/(mode+'.json')))
                        budget.json(cache,value)
                    if value['ours_result_sha256']!=sha(ASSETS/'results'/a/dataset/(mode+'.json')) or value['reference_result_sha256']!=sha(ASSETS/'results'/b/dataset/(mode+'.json')):
                        raise ValueError('Paired CI cache belongs to different results')
                    conditional.append(value)
                    for metric,ci in value['metrics'].items():
                        ci_rows.append(dict(dataset=dataset,prompt_mode=mode,ours=a,reference=b,direction=direction,
                            metric=metric,delta_pp=ci['delta_pp'],CI95_low_pp=ci['ci95_pp'][0],CI95_high_pp=ci['ci95_pp'][1],
                            components=value['components'],replicates=10000,seed=20261023,
                            conditional_on=value['conditional_on']))
            print('PAIRED_CI_COMPLETE',dataset,mode,flush=True)
    write_csv(OUT/'paired_CI.csv',ci_rows)
    ci_path=ASSETS/'paired_CI/conditional_CI.json'
    budget.json(ci_path,dict(comparisons=conditional))
    decision=read(OUT/'dataset_decision.json')
    tokenization={model:{dataset:{mode:results[model,dataset,mode]['tokenization'] for mode in registration['prompt_modes']}
                         for dataset in protocols} for model in list(own)+list(refs)}
    efficiency=[]
    prior=local_path('C:/Users/jsowa/benchmark-translations/series-2026-10-23/evaluation')
    for alias in refs:
        path=prior/'references'/alias/'efficiency.json'
        if path.exists():
            efficiency.append(common.artifact('existing_efficiency:'+alias,path))
    for path in sorted(prior.glob('*efficiency*.json')) if prior.exists() else []:
        efficiency.append(common.artifact('existing_efficiency:ours',path))
    own_efficiency=ROOT/'reports/series_2026-10-23/our_efficiency_20261029.json'
    if own_efficiency.exists():
        efficiency.append(common.artifact('existing_efficiency:ours',own_efficiency))
    summary=dict(status='complete',study_id='data-ood-20261030',generation=1,
        completed_at=datetime.now(timezone.utc).isoformat(),dataset_decision=decision,
        models=registration['models'],references=registration['reference_models'],
        registration_sha256=sha(OUT/'execution_registration.json'),B_EVAL_FROZEN_sha256=sha(COORD/'B_EVAL_FROZEN.json'),
        A_FINALISTS_FROZEN_sha256=sha(COORD/'A_FINALISTS_FROZEN.json'),A_GPU_DONE_sha256=sha(COORD/'A_GPU_DONE.json'),
        metrics=csv_rows,seed_mean_SD=seed_rows,conditional_CI=conditional,tokenization=tokenization,
        parameter_policy='V+T=V+T from pl_mmb_benchmark.report_parameters; head/shared separately',
        existing_efficiency=efficiency,no_efficiency_remeasurement=True,
        MT=read(OUT/'translation_summary.json'),Flickr_MT_QA=read(OUT/'flickr_translation_QA_summary.json'),
        model_selection='All nine A checkpoints, representatives fixed by A, three fixed references; no OOD selection.',
        Wiki_guardrail='R3 failed A VN guardrail; report only as diagnostic, no replacement winner.',
        source_OOD='New source collection relative to audited projector training; encoder pretraining exposure not established.',
        no_manual_annotations=True,XM3600_scored=False,no_external_publication=True)
    budget.json(OUT/'summary.json',summary)
    budget.json(OUT/'tokenization.json',tokenization)
    lines=['# OOD tekst ↔ obraz — sesja B', '',
        'Ukończono porównanie zamrożonych danych i modeli. Zero nowych ręcznych adnotacji. '
        'Selekcja A była zamknięta przed pierwszym wynikiem B; wyniki B nie służyły strojeniu tej serii.', '',
        '## Galerie, pochodzenie i audyt', '',
        '| Lokalna galeria | Obrazy | Unikatowe query PL | Oryginalne captiony | Komponenty |',
        '|---|---:|---:|---:|---:|']
    for name,value in decision['counts'].items():
        lines.append(f'| {name} | {value["images"]} | {value["unique_queries"]} | {value["original_captions"]} | {value["components"]} |')
    lines += ['',
        'Flickr: istniejące tłumaczenia NLLB-200-1.3B, nie native PL; 1000 źródłowych ID zweryfikowano z listą testową. '
        'To lokalna, oczyszczona galeria, bez deklaracji oficjalnego Flickr30k1K. '
        'Repack deklaruje CC BY 4.0, lecz zdjęcia zachowują prawa autorów i warunki oryginalnego dystrybutora '
        '(niekomercyjne badania/edukacja i warunki Flickr). Brak redystrybucji.', '',
        'VizWiz: oryginalne zdjęcia validation pobrano zakresami ZIP, bez całego archiwum ani miniatur. '
        'Istniejące ludzkie EN odfiltrowano przez is_rejected/is_precanned i przetłumaczono wyłącznie tekst→tekst na CPU. '
        'Oryginalne źródło deklaruje CC BY 4.0. Dokładne modele, rewizje, SHA, parametry, cache i koszty są w summary.json.', '',
        'Wykluczenia wobec całego uniwersum A i chronionych train/dev/val/eval objęły źródła/rodziców/URL, '
        'file i pixel SHA, pHash≤6 oraz identyczne znormalizowane EN/PL. Odrzucenia propagowano przez komponenty. '
        'Po MT powtórzono audyt PL. QA jest automatyczną kontrolą powierzchniową, nie ludzkim potwierdzeniem semantyki; '
        'przy niepewności pomijano cały obraz przed score, zachowując wszystkie oryginalne zaakceptowane captiony wybranych obrazów. '
        f'Liczniki QA obrazów: `{json.dumps(decision["QA_rejected_whole_images"],ensure_ascii=False)}`.', '',
        'Qrels wynikają wyłącznie z oryginalnych par. Identyczne znormalizowane EN/PL łączą znane pozytywy; '
        'pHash nie dodaje pozytywów. Galerie dobierano hashem całych komponentów, seed 20261023, bez wyników modeli. '
        'DOCCI zachowano jako rezerwę (oryginalne archiwum przekracza budżet); AIDe zablokowane przez dostęp/prawa do oryginałów bez korespondencji.', '',
        'Źródła: [Flickr repack](https://huggingface.co/datasets/aplominski/flickr30k-multilingual), '
        '[Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/), '
        '[VizWiz](https://vizwiz.org/tasks-and-datasets/image-captioning/), '
        '[DOCCI](https://google.github.io/docci/), [AIDe](https://zil.ipipan.waw.pl/Scwad/AIDe).', '',
        '## Wyniki podstawowe', '',
        'R@1 w %. Nasze warianty: mean±sample SD trzech seedów; referencje: pojedynczy zamrożony model. '
        'Pełne R@1/5/10, MRR i NDCG@10 obu kierunków oraz obu promptów: metrics.csv i summary.json. '
        'Nie łączymy rankingów różnych galerii. R3 jest diagnostyką po niezaliczeniu guardraila A.', '']
    for dataset in protocols:
        lines += ['### '+dataset,'','| Model | t2i R@1 [%] | i2t R@1 [%] |','|---|---:|---:|']
        for variant in ('R0','R1','R3'):
            cells=[]
            for direction in ('t2i','i2t'):
                value=next(row for row in seed_rows if row['dataset']==dataset and row['prompt_mode']=='v2_no_instructions' and row['variant']==variant and row['direction']==direction and row['metric']=='recall@1')
                cells.append(f'{value["mean_percentage"]:.3f} ± {value["SD_percentage"]:.3f}')
            lines.append('| '+variant+(' (diagnostyczny)' if variant=='R3' else '')+' | '+' | '.join(cells)+' |')
        for alias in refs:
            result=results[alias,dataset,'v2_no_instructions']
            lines.append(f'| {alias} | {100*result["directions"]["t2i"]["metrics"]["recall@1"]:.3f} | {100*result["directions"]["i2t"]["metrics"]["recall@1"]:.3f} |')
        lines.append('')
    lines += ['## Niepewność i ograniczenia','','Paired bootstrap komponentów: 10 000, seed 20261023, wspólne losowania, '
        'stała galeria i identyczne query/qrels dla każdej pary. Warunkowe CI różnic reprezentantów względem R0 i referencji '
        'są w paired_CI.csv; obejmują wszystkie metryki i oba tryby promptów. Są oddzielne od mean±SD seedów i '
        'nie opisują niepewności pretreningu ani nieznanych pozytywów.', '',
        'Source-OOD oznacza nową kolekcję względem treningu projektora po audycie. Nie dowodzi braku ekspozycji '
        'w pretreningu enkoderów; dla konkurentów ekspozycja pozostaje nieustalona. Flickr zmienia kolekcję, '
        'VizWiz profil zdjęć, MT styl językowy. Wiki po uczeniu na Wiki jest ID dla R3. XM3600 pozostaje FINAL_BLIND.', '',
        '## Parametry, tokenizacja i precyzja','','| Model | V+T | V | T | Head osobno | Wymiar |','|---|---:|---:|---:|---:|---:|']
    for model in [registration['representative_model_ids'][0]]+list(refs):
        result=results[model,next(iter(protocols)),'v2_no_instructions']
        counts=encoder_counts(result['parameters'])
        lines.append(f'| {"nasze R0/R1/R3" if model in own else model} | {counts["total"]} | {counts["vision"]} | {counts["text"]} | {result["parameters"].get("head",0)} | {result["dimension"]} |')
    lines += ['', 'Własne encodery: FP32 wagi + BF16 autocast, cache/wynik FP32, native cosine wspólnego scorera. '
        'Bazowe cechy obrazów i tekstów współdzielono między dziewięcioma headami; sprawdzono zgodność ze stałym adapterem. '
        'Referencje ładowano pojedynczo z historyczną precyzją, Gemma bez audio. '
        'Tokeny i truncation każdego modelu/galerii/promptu: tokenization.json. '
        'Zgodne istniejące pomiary efektywności wskazano po SHA w summary.json; nie wykonywano ich ponownie.', '',
        'Wszystkie zależności, source/revision/SHA, manifesty, qrels, komponenty, koszty MT i rejestracje: '
        'B_EVAL_FROZEN.json, B_RESULTS_READY.json i summary.json. Dane i modele nie zostały opublikowane na zewnątrz.', '']
    lines += ['### Istniejące pomiary efektywności', '',
        'Mediany ms/element. CPU batch1 FP32; GPU batch64 FP32+BF16 autocast, osobno od historycznej precyzji retrievalu.', '',
        '| Model | CPU tekst | CPU obraz | GPU tekst | GPU obraz |','|---|---:|---:|---:|---:|']
    for artifact in efficiency:
        value=read(local_path(artifact['path']))
        cells=[]
        for key in ('cpu_query','cpu_image','cuda_query','cuda_image'):
            record=value.get('measurements',{}).get(key,{})
            cells.append(f'{record["median_ms_per_item"]:.3f}' if record.get('status')=='measured' else 'n/d')
        lines.append('| '+artifact['role'].split(':',1)[1]+' | '+' | '.join(cells)+' |')
    write_text(OUT/'REPORT.md','\n'.join(lines))
    artifacts += [common.artifact('report',OUT/'REPORT.md'),common.artifact('summary',OUT/'summary.json'),
        common.artifact('metrics_csv',OUT/'metrics.csv'),common.artifact('seed_mean_SD_csv',OUT/'seed_mean_SD.csv'),
        common.artifact('paired_CI_csv',OUT/'paired_CI.csv'),common.artifact('paired_CI_json',ci_path),
        common.artifact('tokenization',OUT/'tokenization.json')]
    marker=dict(schema_version=1,study_id='data-ood-20261030',generation=1,owner='B',phase='B_RESULTS_READY',status='complete',
        contract_sha256=sha(COORD/'COORDINATION.md'),artifacts=artifacts,
        A_FINALISTS_FROZEN_sha256=sha(COORD/'A_FINALISTS_FROZEN.json'),A_GPU_DONE_sha256=sha(COORD/'A_GPU_DONE.json'),
        B_EVAL_FROZEN_sha256=sha(COORD/'B_EVAL_FROZEN.json'))
    common.atomic_json(COORD/'B_RESULTS_READY.json',marker,immutable=True)
    budget.json(COORD/'B_STATUS.json',dict(schema_version=1,study_id='data-ood-20261030',generation=1,owner='B',
        status='complete',phase='B_RESULTS_READY',models_evaluated=12,our_checkpoints=9,references=3,
        new_manual_annotations=0,XM3600_scored=False,GPU_lock_released=True,
        B_RESULTS_READY_sha256=sha(COORD/'B_RESULTS_READY.json')))
    budget.reserve()
    print('B_RESULTS_READY')


if __name__=='__main__':
    main()
