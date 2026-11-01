"""Write an honest CPU preparation report and B status; never readiness by inference."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from prepare import ROOT, OUT, COORD, read, write, load_common, committed_bytes, sha, volumes


def optional(name):
    path = OUT / name
    return read(path) if path.exists() else None


def main():
    common = load_common()
    a_status = read(COORD / 'A_STATUS.json') if (COORD / 'A_STATUS.json').exists() else {}
    markers = {name: (sha(COORD / (name+'.json')) if (COORD / (name+'.json')).exists() else None)
               for name in ('COMMON_READY','A_UNIVERSE_FROZEN','A_FINALISTS_FROZEN','A_GPU_DONE','B_EVAL_FROZEN')}
    audit = optional('candidate_overlap_audit.json')
    images = optional('vizwiz_original_inventory.json')
    translation = optional('translation_summary.json') or optional('translation_progress.json')
    manifest = optional('dataset_decision.json')
    waiting = [name for name in ('A_FINALISTS_FROZEN','A_GPU_DONE') if not markers[name]]
    summary = dict(schema_version=1, study_id='data-ood-20261030', generation=1, owner='B',
        phase='CPU_PREPARATION', status='preparing_data_waiting_for_A' if waiting else 'preparing_data',
        updated_at=datetime.now(timezone.utc).isoformat(), contract_sha256=sha(COORD/'COORDINATION.md'),
        waiting_for=waiting, marker_sha256=markers,
        marker_presence_is_not_permission=True, A_status=a_status,
        image_overlap_audit=audit, vizwiz_originals=images, translations=translation,
        dataset_decision=manifest, models_evaluated=0, CUDA_operations=0,
        manual_annotations_requested=0, disk=volumes(), committed_B_bytes=committed_bytes(),
        report_path=common.windows_path(OUT/'REPORT.md'))
    write(OUT/'summary.json', summary)
    write(COORD/'B_STATUS.json', {k:v for k,v in summary.items() if k not in ('image_overlap_audit','vizwiz_originals','translations','dataset_decision')})
    lines = ['# Sesja B: nowe testy tekst ↔ obraz', '',
        f'Stan: **{summary["status"]}**. Aktualizacja UTC: {summary["updated_at"]}.', '',
        'Nie obliczono żadnych wyników modeli ani nie użyto CUDA. Zero nowych ręcznych adnotacji. '
        'XM3600 pozostaje FINAL_BLIND; odczytywana jest wyłącznie jego chroniona blacklista.', '',
        'Sesja B czeka na zamrożonych finalistów A i zakończenie wszystkich operacji GPU A. '
        'Obecność markera nie zastępuje weryfikacji kontraktu, generacji, zależności i SHA. '
        'Wspólny helper rezerwacji i oryginalny flock pozostają jedynymi mechanizmami koordynacji.', '',
        '## Dane i licencje', '',
        '| Źródło | Stan | Pochodzenie i ograniczenia |',
        '|---|---|---|',
        '| Flickr30k-PL | Test: 1000 oryginałów w Parquet, 5000 EN/PL; train niepobrany | Gotowe tłumaczenia NLLB-200-1.3B, nie native PL. Repack CC BY 4.0 nie dowodzi praw do wszystkich zdjęć; oryginalny dystrybutor dopuszcza niekomercyjne badania/edukację z warunkami Flickr. Bez redystrybucji. |',
        '| VizWiz-Captions-PL | ' + (f'{images["images"]} oryginałów validation pobranych przez zakresy ZIP' if images else 'Pobieranie deterministycznej puli oryginałów validation') + ' | Oryginalne fotografie i istniejące ludzkie EN; źródło CC BY 4.0. Odrzucono is_rejected i is_precanned. PL: osobne MT tekst→tekst na CPU. |',
        '| DOCCI | Opisy pobrane: 5000 test; zdjęcia niepobrane | Oryginalne zdjęcia i ludzkie EN na CC BY 4.0. Archiwum 7 592 938 768 B przekracza budżet B; bez tworzenia krótkich goldów. Rezerwa, gdy VizWiz nie będzie wykonalny. |',
        '| AIDe | Blokada oryginalnych zdjęć | Strona wymaga kontaktu z autorką po obrazy. Rozbieżność: etykieta CC BY-SA 4.0, link do BY-NC-SA. Bez korespondencji i bez zakładania praw mirrorów. |', '',
        'Źródła: [Flickr repack](https://huggingface.co/datasets/aplominski/flickr30k-multilingual), '
        '[oryginalny dystrybutor Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/), '
        '[VizWiz](https://vizwiz.org/tasks-and-datasets/image-captioning/), '
        '[DOCCI](https://google.github.io/docci/), [AIDe](https://zil.ipipan.waw.pl/Scwad/AIDe). '
        'Dokładne rewizje i hashe kart/metadanych: source_metadata.json i deep_research_v2.json.', '',
        '## Audyt, qrels i tłumaczenia', '',
        'Audyt obejmuje całe zweryfikowane uniwersum A, stare trainy, chronione dev/val/eval, '
        'źródła/rodziców/URL, file i pixel SHA, pHash≤6 oraz identyczne znormalizowane EN/PL. '
        'Kolizje propagują się przez komponenty. Finalne tłumaczenia PL wymagają ponownego audytu tekstowego.', '',
        'Qrels pochodzą wyłącznie z istniejących par. Identyczne znormalizowane EN lub PL łączą znane pozytywy. '
        'Exact duplikaty zdjęć są kanonikalizowane; pHash tworzy komponenty, nigdy pozytywne etykiety. '
        'Wybór galerii: seed 20261023, hash całych komponentów, cel 1000, minimum 500; bez wyników modeli.', '']
    if audit:
        lines += [f'Audyt kandydatów: {json.dumps(audit["candidates"], ensure_ascii=False)}; '
                  f'zaakceptowane: {json.dumps(audit["accepted"], ensure_ascii=False)}. '
                  f'Wpisy ochronne: {audit["protected_index_rows"]}; unikatowe pHash: {audit["protected_unique_phashes"]}.', '']
    if translation:
        lines += ['Tłumaczenia: `' + json.dumps({k:translation[k] for k in ('status','done','total','rows','accepted','flags','QA_flags') if k in translation}, ensure_ascii=False) + '`.', '']
    lines += ['## Zarejestrowane porównanie', '',
        'Wszystkie 9 checkpointów A: po 3 seedy R0/R1/wybrany Wiki; 3 reprezentantów ustala A. '
        'Referencje przypięte przed score: polskie MetaCLIP-S16 i M16 oraz większe EmbeddingGemma2; '
        'dokładne rewizje z historycznej rejestracji v3 są w evaluation_preregistration.json. '
        'Bez wyboru najlepszego seeda na OOD ani zmiany modeli po wynikach.', '',
        'Dwa kierunki, osobne pełne galerie, multi-positive R@1/5/10, MRR, NDCG@10, dwa tryby promptów. '
        'Paired bootstrap komponentów: 10 000, seed 20261023; te same query i galerie. '
        'Mean±SD trzech seedów oddzielnie od warunkowych CI. Wspólny scorer i geometria ProjectorEncoder. '
        'Cache bazowych cech niezależny od headów; referencje po jednej. Brak wyników i CI do czasu bramek A.', '',
        'Parametry naszych enkoderów V+T = 70 470 528, V = 38 587 392, T = 31 883 136; '
        'head 196 992 osobno. Używany pl_mmb_benchmark.report_parameters. '
        'Istniejące pomiary efektywności należy odwołać po SHA, bez ponownego benchmarkowania czasu.', '',
        'Source-OOD oznacza nową kolekcję względem treningu projektora po audycie. '
        'Nie dowodzi braku ekspozycji w pretreningu; u referencji pozostaje ona nieustalona. '
        'Flickr zmienia kolekcję, VizWiz profil zdjęć, MT styl językowy, DOCCI długość opisów. '
        'Wiki po uczeniu na Wiki jest ID dla danego checkpointu.', '',
        '## Pozostałe bramki', '',
        'Dokończyć MT i QA, ponowić audyt PL, zamrozić manifesty/galerie/qrels/komponenty i B_EVAL_FROZEN. '
        'Następnie zweryfikować A_FINALISTS_FROZEN i A_GPU_DONE, dokładne 9 checkpointów oraz wspólny flock. '
        'Dopiero wtedy rejestracja wykonania i ocena. Nie opublikowano B_RESULTS_READY.', '']
    data = ('\n'.join(lines)).encode('utf-8')
    common.reserve('B','report',{'D':2*len(data)},committed_bytes=committed_bytes(),temporary_bytes=len(data))
    fd, temporary = tempfile.mkstemp(prefix='.REPORT.', dir=OUT)
    with os.fdopen(fd,'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,OUT/'REPORT.md')
    common.reserve('B','report',{'D':0},committed_bytes=committed_bytes())
    print(json.dumps(dict(status=summary['status'],waiting_for=waiting, translations=translation and translation.get('status'))))


if __name__=='__main__':
    main()
