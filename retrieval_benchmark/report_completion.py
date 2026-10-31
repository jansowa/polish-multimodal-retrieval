"""Reproducible report of the preregistered parameter-class completion."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from retrieval_benchmark.core import HERE,read,sha,write
from retrieval_benchmark.completion import destination
from retrieval_benchmark.space.view import overview


def main():
    reg=read(destination()/'registration.json')
    doc=read(HERE/'space/results.json')
    names={j['alias'] for j in reg['jobs']}|{'metaclip-s16','metaclip-m16','embeddinggemma-2','R0_seed44','R1_seed44','R3_seed44'}
    own_registration=destination()/'own-legacy/registration.json'
    if own_registration.exists():
        names.update(m['alias'] for m in read(own_registration)['models'])
    selected=[(mid,m) for mid,m in doc['models'].items() if m['label'] in names]
    view=overview(doc,visible_columns=['average','task:flickr30k_pl_local_test','task:vizwiz_pl_local_validation',
                    'task:wiki_native_dev','task:VN_dev','task:xtd10_no_overlap'])
    scores={row[0]:row[1:] for row in view['rows']}
    rows=[]
    for mid,m in selected:
        count=m['parameters']['total']
        size=next(label for high,label in [(50e6,'<50M'),(100e6,'50–100M'),(200e6,'100–200M'),
            (300e6,'200–300M'),(400e6,'300–400M'),(500e6+1,'400–500M')] if count<high)
        records=[r for r in doc['results'] if r['model_id']==mid and r['profile']=='historical-native-v2']
        count_results=len(records)
        values=scores.get(m['label'],[None]*6)
        rows.append(dict(model=m['label'],size_class=size,parameters=m['parameters'],
            metrics=dict(zip(['mean','flickr','vizwiz','wiki','visualnews','xtd_clean'],values)),
            result_count=count_results,complete=count_results==12,revision=m['revision'],repository=m.get('repository')))
    rows.sort(key=lambda r:(r['parameters']['total'],-(r['metrics']['mean'] or 0)))
    text=['# Uzupełnienie benchmarku ≤500M — 31.10.2026','',
        'Dobór modeli i rewizji zamknięto przed pierwszym nowym wynikiem. To mocni kandydaci w przyjętych klasach, nie dowód znalezienia globalnie najlepszego modelu.',
        'R0/R1/R3 to wcześniej ustaleni reprezentanci seed44; R3 pozostaje diagnostyczny po niezaliczeniu guardraila VN.',
        'Średni wynik: R@1, najpierw średnia T2I/I2T, następnie równe wagi pięciu zadań. Pełne XTD10 ma osobne wyniki, ale nie dubluje wariantu oczyszczonego w średniej.',
        'Parametry V+T obejmują wyłącznie używane enkodery; V i T podano osobno. Head TinyCLIP/MobileCLIP: 196 992; head SigLIP2: 295 296 dodatkowych parametrów. Logit scale/bias i inne modalności nie są wliczane.','',
        '| Klasa | Model | V+T [M] | V [M] | T [M] | Średni R@1 [%] | Flickr | VizWiz | Wiki | VN | XTD clean | Wyniki /12 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        p=row['parameters'];v=row['metrics']
        cells=[row['size_class'],row['model'],*[f'{p[k]/1e6:.3f}' for k in ('total','vision','text')],
               *['—' if v[k] is None else f'{v[k]:.3f}' for k in ('mean','flickr','vizwiz','wiki','visualnews','xtd_clean')],str(row['result_count'])]
        text.append('| '+' | '.join(cells)+' |')
    text+=['','Wiki/VN są wcześniej używanymi dev, COCO jest źródłem ID dla naszych modeli FULL, a Flickr/VizWiz to lokalne galerie z MT. Średnia nie jest czystym wynikiem zero-shot.',
           'Każdy istniejący model przed uzupełnieniem musi dokładnie odtworzyć wszystkie per-query metryki historycznej kontroli Wiki. Nowe modele przechodzą przez ten sam scorer i natywny adapter.',
           'Nie uruchomiono XM3600 ani DOCCI. Nie oceniano nowych modeli >500M. M-CLIP 677.76M oraz SigLIP2 large/MetaCLIP huge/giant są poza limitem.',
           'Wagi referencji są ładowane pojedynczo; usuwane są wyłącznie nowo pobrane pliki wag po pełnym ukończeniu modelu, z kontrolą SHA, rozmiaru i właścicieli snapshotów. Dane, checkpointy projektu i stare wyniki pozostają.','',
           'Źródła nowych modeli (dokładne rewizje w rejestracji):']
    text += ['- ['+j['repository']+']('+j['source_url']+')' for j in reg['jobs'] if j['alias'].startswith('metaclip-')]
    target=HERE/'COMPLETION_20261031.md'
    target.write_text('\n'.join(text)+'\n',encoding='utf-8')
    write(HERE/'local/completion_summary.json',dict(registration_sha256=sha(destination()/'registration.json'),
          results_sha256=sha(HERE/'space/results.json'),selected_models=rows,
          missing_models=sorted(names-{r['model'] for r in rows}),
          all_selected_complete=len(rows)==len(names) and all(r['complete'] for r in rows)))
    print(target)


if __name__=='__main__':main()
