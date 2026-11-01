"""Finalize only after every registered comparison has completed."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

import visualnews_judge_analysis as a
import visualnews_judge_pilot as p

MULTI=['A_direct','B_taxonomy','C_claims','D_skeptical','B_taxonomy_v2','C_claims_v2','B_taxonomy_v3']

def pct(x):return f'{100*x:.1f}%' if x is not None else '—'

def main():
    a.summarize()
    summary=json.loads((p.OUT/'summary.json').read_text(encoding='utf-8'))
    results,attempts=a.read_results()
    sample=json.loads((p.OUT/'sample.json').read_text(encoding='utf-8'))
    manifest=json.loads((p.OUT/'manifest.json').read_text(encoding='utf-8'))
    ids={r['id'] for r in sample}; repeated={r['id'] for r in sample[:100]}
    controls=json.loads((p.OUT/'controls.json').read_text(encoding='utf-8'))
    cids={r['id'] for r in controls}
    checks={}
    for name in MULTI:
        rr=results[name]
        expected={(uid,0) for uid in ids|cids}|{(uid,rep) for uid in repeated for rep in [1,2]}
        if set(rr)!=expected:raise RuntimeError(f'{name}: missing or extra calls {len(set(rr)^expected)}')
        checks[name]=dict(expected_unique_requests=len(expected),observed=len(rr),http_success=sum(r.get('http_status')==200 for r in rr.values()),
            recovered_schema_valid=sum(r.get('schema_valid',False) for r in rr.values()))
    for name in ['T1','T2']:
        if set(results[name])!={(uid,0) for uid in ids}:raise RuntimeError('Text sample mismatch')
    if p.sha(p.OUT/'sample.json')!=manifest['sample_sha256']:raise RuntimeError('Sample changed')
    for record in sample:
        if p.sha(p.ROOT/record['image'])!=record['image_sha256']:raise RuntimeError('Image changed')
    for name in ['B_taxonomy_v2','C_claims_v2','B_taxonomy_v3']:
        if not all(r.get('raw_schema_valid') for r in results[name].values()):
            raise RuntimeError('Structured response failed: '+name)
    total_usage={k:sum((r.get('usage') or {}).get(k,0) for data in attempts.values() for r in data) for k in ['prompt_tokens','completion_tokens','total_tokens']}
    p.save(p.OUT/'verification.json',dict(variant_checks=checks,text_only_n=360,sample_sha256_verified=True,image_hashes_verified=360,
        original_parquet_sha256=p.sha(p.OUT/'source.parquet'),original_zip_sha256=p.sha(p.OUT/'images.zip'),
        total_http_attempts=sum(len(v) for v in attempts.values()),token_usage=total_usage,
        note='No corpus-wide judge job was launched. Original translation jobs/checkpoints untouched. Raw responses preserve formatting failures.'))
    # Rank the fully tested revised candidates using observed controls, false alarms and manual diagnostics.
    finalists=['B_taxonomy_v2','C_claims_v2','B_taxonomy_v3']
    def rank(name):
        m=summary['variants'][name]; c=summary['controls'][name]
        return (48-c['passed'],m['manual_mismatch_diagnostics']['false_alarms'],m['manual_diagnostics']['severe_distance_at_least_2'],
            m['manual_diagnostics']['mean_interval_distance'],sum(m['logical_anomalies'].values()),-m['stability']['full_label_agreement'])
    finalists.sort(key=rank); best=finalists[0]
    selection=dict(recommended_multimodal=best,second_finalist=finalists[1],recommended_text_only='T1',ranking=finalists,
        criteria='Complete controls, then fewer independently inspected natural false alarms, then severe grounding interval misses, mean interval distance, logical flags and repeat agreement. Exploratory diagnostic ranking, not statistically proven superiority.',
        schema='schema.json',request_config=best+'_config.json')
    p.save(p.OUT/'selection.json',selection)
    (p.OUT/'recommended_multimodal.txt').write_text((p.OUT/(best+'.txt')).read_text(encoding='utf-8'),encoding='utf-8')
    text_config=json.loads((p.OUT/'T1_config.json').read_text(encoding='utf-8'))
    (p.OUT/'recommended_text_only.txt').write_text(text_config['prompt'],encoding='utf-8')
    text_schema=dict(type='object',additionalProperties=False,properties={
        'potential_external_context_dependency':dict(type='integer',minimum=0,maximum=3),
        'named_entity_dependency':dict(type='string',enum=['none','incidental','essential']),
        'event_or_temporal_context':dict(type='boolean'),'precise_location_mentioned':dict(type='boolean'),
        'potentially_good_visual_query':dict(type='boolean'),'confidence':dict(type='number',minimum=0,maximum=1),
        'reasons':dict(type='array',items=dict(type='string'))})
    text_schema['required']=list(text_schema['properties']);p.save(p.OUT/'text_only_schema.json',text_schema)
    candidate_ids=summary['text_only_comparison']['T1 vs '+best]['potential_overfilter_candidates']
    text_candidates=[]
    for record in sample:
        if record['id'] not in candidate_ids:continue
        text_candidates.append(dict(record,multimodal=results[best][(record['id'],0)]['parsed'],
            text_only={name:results[name][(record['id'],0)].get('parsed') for name in ['T1','T2']},
            assessment='Potential overfilter candidate, not an independently confirmed text-only false positive.'))
    p.save(p.OUT/'text_only_false_positive_candidates.json',text_candidates)
    comparisons=[]
    for name in MULTI:
        m=summary['variants'][name]; c=summary['controls'][name]; manual=m['manual_diagnostics']; stability=m['stability']
        comparisons.append(dict(variant=name,n=m['requested_unique'],raw_json_valid_rate=m['json_valid_rate'],recovered_schema_rate=m['schema_valid_rate'],
            controls_passed=c['passed'],controls_expected=48,controls_evaluable=c['n'],natural_false_alarms=m['manual_mismatch_diagnostics']['false_alarms'],
            manual_interval_misses=manual['outside_interval'],manual_severe_misses=manual['severe_distance_at_least_2'],
            grounding_repeat_exact=stability['grounding_exact'],all_labels_repeat_exact=stability['full_label_agreement'],grounding_repeat_mae=stability['grounding_mae']))
    with (p.OUT/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    lines=[f'**Rekomendacja: {best}; text-only: T1.**',
        '',f'Przeprowadziłem {len(MULTI)} wariantów multimodalnych (4 różne strategie v1, 2 rewizje v2, 1 rewizję v3) oraz 2 text-only. Każdy dostał dokładnie te same 360 naturalnych par. Dla każdego multimodalnego: 64 podpisy kontrolne na 8 obrazach oraz 3 losowania na identycznych 100 parach (pierwsze losowanie w przebiegu głównym; dwa dodatkowe). Łącznie {sum(len(v) for v in attempts.values())} wywołań HTTP, plus 4 wcześniejsze techniczne próby w bootstrap_v0.',
        '', '**Materiały do akceptacji**', '',
        f'- [Rekomendowany prompt multimodalny](recommended_multimodal.txt), [dokładna konfiguracja wykonania]({best}_config.json); [drugi finalista]({finalists[1]}.txt).',
        '- [Rekomendowany prompt text-only](recommended_text_only.txt).',
        '- [Finalny JSON schema](schema.json), [osobny schema text-only](text_only_schema.json).',
        '- [Galeria: 50 rozbieżności / błędów i 30 losowych przykładów](review.html). Każdy przykład zawiera obraz, EN caption, JSON i confidence; starsze warianty można rozwinąć.',
        '- [Wyniki liczbowe](summary.json), [porównanie CSV](comparison.csv), [surowe odpowiedzi](raw/), [kontrole](controls.json), [weryfikacja kompletności](verification.json).',
        '', 'Pierwszy finalista ma mniej fałszywych alarmów i większą stabilność. Drugi, B v2, ma mniej wyjść poza ręczne zakresy VG (6/60 wobec 8/60), więc pozostaje wariantem porównawczym. Nie zakładam, że różnice w tym pilocie dowodzą przewagi na całym korpusie.',
        '', '**Próbka i zakres wniosków**', '',
        'Źródło to VisualNews_t2i z TIGER-Lab/MMEB-eval, revision `'+p.REV+'`. Pula zawiera '+str(manifest['source_rows'])+' rekordów; wybrano 360 różnych captionów i obrazów. 180 wybrano losowo, 180 przez rotację heurystycznych kategorii tekstowych, seed '+str(p.SEED)+'. Usunięto wyłącznie stałą instrukcję MMEB „Retrieve an image of this news caption.”; oryginalny query jest w sample.json. Obrazy powiązano z oficjalnym kandydatem pozytywnym 0, bez zgadywania ID.',
        '', 'Jest to diagnostyczny podzbiór benchmarku, a nie losowa próba 537 tys. captionów M-BEIR VisualNews. Nie estymuję na tej podstawie odsetka dobrych przykładów całego korpusu. Te obrazy należą do podzbioru ewaluacyjnego; pilot nie dodaje ich do treningu. Pierwsze 100 rekordów zamrożono jako grupę powtórzeń; pozostałe 260 brały udział w eksploracyjnej analizie błędów, więc nie stanowią ślepego holdoutu.',
        '', 'Kategorie heurystyczne zachodzą na siebie: '+', '.join(k+'='+str(v) for k,v in manifest['tag_counts'].items())+'. Etykiety są wskazówkami doboru, nie prawdą referencyjną. Przypadki czarne i uszkodzone pozostały jako trudne przykłady; samo poprawne zdekodowanie pliku nie zapewnia użytecznej treści.',
        '', '**Porównanie**', '',
        '| Wariant | Surowy JSON | Schema po usunięciu samych fence | Kontrole /48 | Fałszywe alarmy /10 | Poza ręcznym zakresem /60 | Ciężkie /60 | Zgodność VG | Wszystkie etykiety |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in comparisons:
        lines.append('| '+row['variant']+' | '+pct(row['raw_json_valid_rate'])+' | '+pct(row['recovered_schema_rate'])+' | '+str(row['controls_passed'])+' | '+str(row['natural_false_alarms'])+' | '+str(row['manual_interval_misses'])+' | '+str(row['manual_severe_misses'])+' | '+pct(row['grounding_repeat_exact'])+' | '+pct(row['all_labels_repeat_exact'])+' |')
    lines += ['', 'Surowy JSON oznacza `json.loads` na niezmienionej odpowiedzi. Odzyskiwanie usuwa tylko kompletny zewnętrzny blok Markdown i ponownie sprawdza schema; nie naprawia treści. V1 nie używał constrained decoding; v2/v3 wysyłają `response_format=json_schema`. Poprawy formatu nie przypisuję wyłącznie promptowi. B v1 ma jeden niepoprawny wynik kontrolny (pięć zależnych porównań niewykonalnych) i jeden niepoprawny wynik powtórzenia; w tabeli kontrole niewykonalne nie dostają zaliczenia.',
        '', 'Zgodność powtórzeń porównuje losowania 1 i 2 z losowaniem bazowym: 200 par porównań na 100 rekordach; B v1 ma 199 poprawnych par. „Wszystkie etykiety” obejmuje 8 pól klasyfikacyjnych, bez confidence i opisów dowodów. Samo agreement nie jest miarą trafności. Ważona kappa, MAE, dokładne mianowniki oraz podział na grupy są w summary.json.',
        '', 'Ręczny przegląd obejmuje 60 obrazów z captionami i przedziały akceptowalnego VG zapisane po diagnozie v1, przed pełnym porównaniem przypadków v2. Są to diagnostyczne oceny Codex, bez ekspertów i bez ustalonego gold standard; przypadki OCR, landmarków i źródeł obrazu bywają niejednoznaczne. „Ciężki” oznacza odległość co najmniej 2 od tego przedziału. Osobno obejrzałem wszystkie 14 naturalnych mismatchów v2: 9 oceniłem jako fałszywe alarmy, 3 jako niejednoznaczne, 2 jako nieużyteczne pliki. Te przypadki służyły projektowaniu v3, więc poprawa na nich jest wynikiem rozwojowym, nie niezależnym precision. Końcowy przegląd v3 wykrył jeszcze jeden nowy fałszywy alarm (413), dodany po zakończeniu strojenia. Stąd końcowy mianownik 10, z czego 9 stanowiło przypadki rozwojowe.',
        '', '**Iteracje i stwierdzony efekt**', '',
        'Bootstrap: cztery wywołania ujawniły echo definicji schema i bloki Markdown. Zastąpiłem definicję w prompcie szablonem instancji i usunąłem instrukcję retrieval z wejścia; techniczne wyniki zachowano osobno i wyłączono z porównania.',
        '', 'V1: A — bezpośrednie osie; B — typy informacji; C — atomowe claims; D — audyt sceptyczny. Wspólne skale i schema. Problemami były rozpoznawanie osób jako dowodu, nadmierne karanie nazw przy widocznej akcji oraz mylenie źródła obrazu z opisem treści.',
        '', 'V2: jawne kredytowanie widocznego rdzenia, reguły OCR / landmarków / source credit oraz spójność uzasadnienia ze score. B: liczba wyjść poza ręczny zakres '+str(summary['variants']['B_taxonomy']['manual_diagnostics']['outside_interval'])+' → '+str(summary['variants']['B_taxonomy_v2']['manual_diagnostics']['outside_interval'])+' /60; C: '+str(summary['variants']['C_claims']['manual_diagnostics']['outside_interval'])+' → '+str(summary['variants']['C_claims_v2']['manual_diagnostics']['outside_interval'])+' /60. Format wykonania zmieniono osobno na wymuszony JSON schema.',
        '', 'V3: doprecyzowanie rzeczywistej sprzeczności, odbić, cutoutów, szkiców, zasłonięcia, tłumaczeń tekstu i błędów wnioskowania geograficznego. Naturalne fałszywe alarmy B v2 → B v3 w końcowym zestawie 10: '+str(summary['variants']['B_taxonomy_v2']['manual_mismatch_diagnostics']['false_alarms'])+' → '+str(summary['variants']['B_taxonomy_v3']['manual_mismatch_diagnostics']['false_alarms'])+'. W pierwotnych 9 przypadkach rozwojowych spadek wyniósł 7 → 5, ale pojawił się nowy alarm 413. Jednocześnie poprawnie wykryte błędne kolory: '+str(summary['controls']['B_taxonomy_v3']['by_kind']['wrong_color']['passed'])+'/8. Poprawa jest częściowa; mismatch=true powinno kierować do ręcznego przeglądu, a nie automatycznie usuwać parę. Pozostały m.in. błędny odczyt greckiego napisu, wnioskowanie o kraju z ubioru i wyobrażenie sali sądowej. Wszystkie skutki i regresje można sprawdzić w galerii oraz raw outputs.',
        '', '**Rozkład labeli finalisty**', '', '| visual_grounding | n | % |','|---|---:|---:|']
    m=summary['variants'][best]
    for score in range(5):
        n=m['score_distribution'].get(str(score),0); lines.append(f'| {score} | {n} | {pct(n/360)} |')
    lines += ['', '| caption_type | n |','|---|---:|']
    for kind in p.TYPES:lines.append('| '+kind+' | '+str(m['caption_types'].get(kind,0))+' |')
    lines += ['', 'External context 0–3: '+json.dumps(m['external_context_distribution'])+'. Flaga mismatch: '+str(m['mismatch_count'])+'/360. Rozkład jest silnie przesunięty ku niskiej ocenie, co w news captions jest możliwe; sam rozkład nie potwierdza jakości ani nie uzasadnia jego sztucznego wyrównania.']
    for group in ['random','targeted']:
        g=m['groups'][group];lines.append(f"Grupa {group}: n={g['n']}, średni VG={g['grounding_mean']:.3f}, external={g['external_mean']:.3f}, mismatch={g['mismatch']}.")
    lines += ['', '**Kontrole**', '', '8 obrazów, każdy z czystym opisem, dopisanym nazwiskiem fotografa, usuniętym nazwiskiem, dokładną datą, nazwą wydarzenia, dokładną lokalizacją, późniejszym skutkiem i błędnym kolorem. Daje to 64 captiony i 48 porównań oczekiwanego kierunku. Dla daty, wydarzenia, miejsca i późniejszego skutku wymagano ścisłego wzrostu external context; przy usunięciu nazwiska VG nie może spaść; błędny widoczny kolor musi zmienić mismatch z false na true. Dodatkowo zapisano, czy dopisanie kontekstu nie zwiększa VG.',
        '', '| Kontrola | Zaliczone |','|---|---:|']
    for kind,counts in summary['controls'][best]['by_kind'].items():lines.append('| '+kind+' | '+str(counts['passed'])+'/'+str(counts['n'])+' |')
    lines += ['', '**Text-only**', '', '| Wariant | Poprawny schema /360 | Named entity i nadal potencjalnie dobry query | External MAE wobec finalisty | Kandydaci nadmiernego odrzucania |','|---|---:|---:|---:|---:|']
    for name in ['T1','T2']:
        t=summary['variants'][name]; cmp=summary['text_only_comparison'][name+' vs '+best]
        lines.append('| '+name+' | '+str(t['valid'])+' | '+str(cmp['good_visual_query_despite_entity'])+' | '+f"{cmp['external_mae']:.3f}"+' | '+str(len(cmp['potential_overfilter_candidates']))+' |')
    lines += ['', 'T1 wybrano za kompletny poprawny format i zachowanie query wizualnych pomimo nazw własnych. „Kandydat nadmiernego odrzucania” oznacza text-only external ≥2 przy multimodal VG ≥3; jest to rozbieżność dwóch ocen, a nie potwierdzony false-positive rate. Nie należy odrzucać przykładów wyłącznie przez nazwisko ani przez wysoki wynik text-only. External MAE porównuje potencjalną zależność tekstową z zależnością ocenianą na obrazie; nie są to identyczne zadania. [Przypadki do kontroli false positives, z captionami i wynikami obu trybów](text_only_false_positive_candidates.json).',
        '', '**Schema i wykonanie**', '', 'Zachowano wszystkie 9 wymaganych pól. `named_entity_dependency` ma enum none/incidental/essential. Dodano `evidence` i `unsupported_information`, aby audytować, co jest widoczne i co wymaga kontekstu. Confidence oznacza pewność oceny modelu, nie jakość pary i nie skalibrowane prawdopodobieństwo. Text-only ma osobny schema, bo bez obrazu nie może stwierdzać rzeczywistego grounding, OCR ani mismatch.',
        '', 'Testowana konfiguracja: model `qwen3.8-27b`, temperatura 0.2, thinking wyłączone, max_tokens=1800, obraz proporcjonalnie do maksymalnie 1280 px, JPEG quality 88. Testy nie obejmują automatycznie temperatury 0 ani innych ustawień; porównanie finalistów dotyczy zapisanej konfiguracji.',
        '', 'Łączne raportowane użycie tokenów: '+json.dumps(total_usage)+'. Nie obliczam kosztu pieniężnego bez cennika endpointu. Wszystkie raw HTTP bodies, finish_reason, usage, czasy, hashe promptów / obrazów / captionów i numery prób zapisano w raw/.',
        '', '**50 rozbieżności / błędów oraz 30 losowych przykładów**', '', '[Galeria z obrazami i pełnymi wynikami](review.html), [50 przypadków JSON](disagreements.json), [30 losowych przypadków JSON](random30.json), [60 ocen ręcznych](manual_reviews.json), [15 przeglądów mismatch](manual_mismatch_reviews.json). Pierwszeństwo na liście 50 mają ręcznie sprawdzone mismatchy, dalej największa rozbieżność ocen / typów; przypadki bez ręcznej adjudykacji są oznaczone jako kandydaci do przeglądu.',
        '', 'Pilot zakończony. Prompt i schema są przygotowane do ręcznej akceptacji; pełny VisualNews judge nie został uruchomiony.']
    (p.OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(selection=selection,verification=checks,total_usage=total_usage),indent=2),flush=True)

if __name__=='__main__':main()
