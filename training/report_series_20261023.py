"""Finalize a truthful partial-series report; never computes model metrics."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pl_mmb_training.series_20261023 import OUT, ROOT, read, disk_inventory, freeze, file_sha


def main():
    state = read(OUT / 'preparation_state.json')
    wiki, xtd = state['wiki'], state['xtd']
    checks = read(OUT / 'validation.json')
    statuses = {
        'P0.1': ('częściowo', 'Zamrożono kandydatów provenance; brak rezerwy na oryginalne obrazy. Pełny audyt komponentów, dev i guard pozostają otwarte.'),
        'P0.2': ('częściowo', 'Źródłowe EN/PL/COCO names i licencja przypięte; policzony parent overlap. Brak zweryfikowanych pełnych obrazów; hash audit i pary z cache Qwen otwarte.'),
        'P0.3': ('częściowo', 'Gotowy lokalny formularz i 150 obrazów VN. Brak nowego Wiki dev: 150 Wiki nie wylosowano. Brak ludzkich odpowiedzi.'),
        'P0.4': ('częściowo', 'Gotowy formularz i 100 par VN: po 25 na wydawcę i split. Wiki poza dev wymaga najpierw zamknięcia dev. Brak ludzkich ocen.'),
        'P0.5.1': ('częściowo', 'Zamknięta lista 20 referencji z rewizjami i protokołem. Status model_list_frozen_execution_blocked: potrzebne prompt sources, adapter naszych modeli i cache parity.'),
        'P0.5.2': ('częściowo', 'Zdefiniowano niezmienny klucz baseline i zablokowano runner pilota Wiki. Baseline nie obliczono; historyczny pilot Wiki już istniał.'),
        'P0.5.3': ('zrobione', 'Tylko draft bramki XM3600 do decyzji użytkownika, zgodnie z zakresem dnia. Żadnej ewaluacji XM.'),
        'P1.A': ('częściowo', 'Reguła mediany VN dev i reuse seeda 42 zapisane przed nowym testem. Seedy 43/44 nie uruchomione: D poniżej 4 GiB.'),
        'P1.B0': ('nie zaczęte', 'Brak zamkniętej rejestracji wykonawczej i galerii ze zweryfikowanymi obrazami; rezerwa dyskowa niespełniona.'),
        'P1.B': ('nie zaczęte', 'Kolejność wymaga zamknięcia P1.A i P1.B0. Brak pobierania wag i cache.'),
        'P1.C': ('nie zaczęte', 'Kolejność GPU i blokada rezerwy dysku.'),
        'P1.D': ('nie zaczęte', 'Kolejność GPU i blokada rezerwy dysku.'),
        'P2.E': ('nie zaczęte', 'Warunek dopiero po P0 niespełniony. Licencja XTD10 zachowana w P0.2; nie zastępuje dossier.'),
        'P2.F': ('nie zaczęte', 'P0 niezamknięte; brak kompletnego guarda tekstów dev. Nie rozpoczęto treningu ani diagnostyki.'),
        'P2.G': ('nie zaczęte', 'P0 niezamknięte; nauczyciel P1.B nie istnieje. Nie wykonano audytu ani treningu destylacji.'),
    }
    files = {str(p.relative_to(OUT)).replace('\\','/'): file_sha(p) for p in OUT.rglob('*')
             if p.is_file() and p.name not in ('REPORT.md','summary.json') and not p.name.startswith('human_')}
    current_disk = disk_inventory()
    summary = dict(series='2026-10-23', status='partial_stopped_at_storage_prerequisite',
        tasks={k: dict(status=s, reason=r) for k, (s, r) in statuses.items()},
        wiki_native_candidates=wiki['direct_native_rows'], wiki_frozen_dev_images=0,
        xtd_parent_overlap=xtd['parent_overlap_images'], xtd_preliminary_no_parent_overlap=xtd['preliminary_clean_count'],
        reference_execution_registration_closed=False, model_list_frozen=True,
        new_retrieval_results=[], GPU_runs=[], GPU_seconds=0, human_annotations_created=0,
        protected_snapshots_unchanged=True, live_pipeline_databases_read=False, pipeline_allocation_changed=False,
        xm3600_scored=False, validation=checks, artifacts_sha256=files,
        disk_before=state['disk_before'], disk_after_preparation=state['disk_after'], disk_final=current_disk,
        new_artifacts_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),
        missing_context=['reports/consultation_2026-10-21_external.md'],
        decisions=['reference_eval_registration.json','vn_full_seeds_decision.json','xm3600_gate_registration.json'],
        requires_user_decision=['XM3600 gate draft: prohibited until a separate later approved run',
            'publication after licensing dossier and independent results',
            'new Wiki pilot after frozen dev/guard/immutable baseline',
            'distillation after teacher choice, leakage audit and explicit approval'])
    summary['code_sha256'] = {name: file_sha(ROOT/name) for name in (
        'src/pl_mmb_training/series_20261023.py', 'src/pl_mmb_training/series_holdout.py',
        'src/pl_mmb_training/wiki_pilot.py', 'src/pl_mmb_benchmark/full_gallery.py',
        'training/prepare_series_20261023.py', 'training/validate_series_20261023.py',
        'training/report_series_20261023.py', 'experiments/series_annotation.py',
        'experiments/series_annotation/index.html', 'experiments/series_annotation/app.js',
        'experiments/series_annotation/style.css', 'tests/test_series_20261023.py')}
    freeze(OUT / 'summary.json', summary)
    lines = [
        '# Seria 23.10.2026 — raport częściowy', '',
        '**Seria nie została ukończona.** Prace CPU przygotowały rejestracje i narzędzia. '
        'Nie rozpoczęto pobierania obrazów/wag, tworzenia cache ani obliczeń GPU: D: ma mniej niż wymagana rezerwa 4 GiB. '
        'Nie ma nowych wyników jakości modeli ani podstaw do wniosków o przewadze nad konkurencją.', '',
        'Nie otwierano SQLite/WAL Wiki ani VN, nie zmieniano procesów ani przydziału requestów. '
        'Weryfikacja SHA256 ośmiu chronionych plików przed/po potwierdziła niezmienność FULL_CLEAN, validation, '
        'snapshotu VN 35 918, VN train/dev i snapshotu Wiki 1322 oraz jego train/dev. '
        'Nie uruchomiono legacy test, fixed gallery ani XM3600.', '',
        '## Status każdego punktu', '', '| Punkt | Status | Powód / zakres |', '|---|---|---|',
        *[f'| {k} | {s} | {r} |' for k, (s, r) in statuses.items()], '',
        '## Wiki: kandydaci, nie gotowy dev', '',
        f"Jednorazowo odczytany eksport miał {wiki['export_rows']} wierszy, SHA256 `{wiki['source_export_sha256']}`. "
        f"Ścisła reguła provenance dała **{wiki['direct_native_rows']} kandydatów**: A={wiki['source_classes'].get('A',0)}, "
        f"B={wiki['source_classes'].get('B',0)}, C=0, D={wiki['source_classes'].get('D',0)}. "
        'Zapisano ich w `wiki_native_candidates.jsonl`. To nie manifest dev i nie wolno go użyć do ewaluacji.', '',
        f"{wiki['local_identity_records']} kandydatów ma metadane tożsamości w historycznym pilocie; "
        f"{wiki['historical_pilot_train_id_overlap']} ID było w jego train. Wczesne dokładne dopasowanie tekstu "
        f"do chronionych pul wykryło {wiki['preflight_exact_text_conflicts']} konfliktów; nie jest to pełny audyt leakage. "
        'Pełne grupowanie po canonical pixels, pHash ≤6, identyfikatorach źródła/strony i EN/PL oraz '
        'wykluczenie całych komponentów wymagają zweryfikowanych oryginalnych assetów. '
        'Nie pobierano ich poza wspólnym budżetem scrapera. Nie ustalono jeszcze finalnej liczby obrazów dev.', '',
        'Historyczny pilot Wiki 0/3/10% ma status complete. Zachowano jego pliki. '
        'Nowy dev musi jawnie rozliczyć ten wcześniejszy trening. Nowy train pilota wymaga osobnego manifestu; '
        'historyczny runner jest teraz blokowany przed ładowaniem danych/GPU. '
        'Finalnego guarda dla wszystkich przyszłych train nie ogłaszamy jako gotowego.', '',
        '## XTD10: kontrola stylu', '',
        f'Źródło: [repozytorium Adobe, rewizja {xtd["revision"]}](https://github.com/adobe-research/Cross-lingual-Test-Dataset-XTD10/tree/{xtd["revision"]}). '
        'Zachowano 1000 EN, 1000 ludzkich PL, 1000 jawnych nazw COCO oraz pełną licencję. '
        'Zweryfikowano dokładne Git blob SHA1 wszystkich trzech plików; zachowano oryginalne CRLF polskiego pliku. '
        'Parent ID wynika wyłącznie z oficjalnej nazwy obrazu, bez zgadywania mapowań.', '',
        '| Chroniona pula | Obrazy XTD10 ze wspólnym parent ID |', '|---|---:|',
        *[f'| {k} | {v} |' for k,v in xtd['parent_overlap_images'].items()], '',
        f"Wstępnie {xtd['preliminary_clean_count']} obrazów nie ma parent overlapu. Liczniki pul nie są sumą rozłączną. "
        'Zero pełnych assetów udało się powiązać z istniejącą ścieżką o dokładnej nazwie upstream. '
        'Nie uznajemy cropa o tym samym parent ID za pełny obraz. Obie galerie pozostają niegotowe: '
        'potrzebny jest audyt hashy oryginałów. Podzbioru clean nie zamrożono. '
        'Cache tłumaczeń Qwen dla dokładnie tych samych EN nie został jeszcze sprawdzony.', '',
        f'Licencja źródłowa: [Adobe Research License](https://github.com/adobe-research/Cross-lingual-Test-Dataset-XTD10/blob/{xtd["revision"]}/LICENSE.md), '
        '§1.1 ogranicza wykorzystanie do noncommercial research. Lokalny pełny tekst: `xtd10_source/LICENSE.md`. '
        'To odczyt warunków źródła, bez kwalifikacji prawnej projektu.', '',
        '## Ręczne adnotacje', '',
        'Lokalne narzędzie: `python experiments/series_annotation.py --port 8768`, adres `http://127.0.0.1:8768`. '
        'Próbki: 150 różnych obrazów VN dev do 1–2 query i 100 par VN do grounding 0–4, mismatch, komentarza '
        '(25 WaPo train, 25 WaPo dev, 25 USA Today train, 25 USA Today dev). '
        'Obrazy query i audytu są rozłączne. API query nie zwraca captionu. '
        'Dobór deterministyczny po SHA256 z canonical pixels; manifesty mają SHA256.', '',
        'Wykorzystano istniejący, zamrożony tensor RGB 224×224 przez mmap; obrazy są kodowane w RAM podczas żądania, '
        'bez nowego dyskowego cache. To ograniczenie rozdzielczości oceny. '
        'Zapisy trafiają append-only do `annotation/human_query.jsonl` i `annotation/human_audit.jsonl` '
        'z timestampem UTC, ID obrazu i SHA manifestu. Pliki powstają dopiero po ręcznym zapisie. '
        'Nie utworzono żadnych ludzkich query ani ocen. Próbki Wiki są wstrzymane.', '',
        '## Porównanie z konkurencją', '',
        'Brak wyników dla Wiki, XTD10 i nowych query. Nie ma tabel paired Δ ani efektywności. '
        'Rejestracja utrwala wszystkie 20 modeli z przypiętymi rewizjami, klasy ≤100M, alphaedge-ai i kontrole EN '
        'MobileCLIP2-S2/PE-Core. Nie dodano modeli opcjonalnych. '
        'Nie jest zamknięta do wykonania: brak audytu promptów autorów, adaptera naszych checkpointów i kontroli rang z trzech historycznych cache.', '',
        'Implementacja wspólnych metryk `pl_mmb_benchmark/full_gallery.py` obejmuje t2i/i2t R@1/5/10, MRR, '
        'multi-positive NDCG@10 i paired bootstrap po komponentach przy identycznych galeriach, qrels i query. '
        'Sprawdzono ją wyłącznie na fixture testowym. Nowe zbiory muszą najpierw dostać status frozen_image_verified; '
        'rejestracja musi mieć closed_ready_for_execution. Nie obliczono embeddingów.', '',
        'VN dev jest in-domain dla naszych modeli, z etykietami modelowymi. XTD10 to kontrola stylu i częściowo '
        'wspólny parent COCO, nie holdout. Status Wiki zależy od checkpointu: VN_FULL bez Wiki może być oceniony '
        'jako zero-shot, ale historyczne modele pilota Wiki są in-domain. '
        'Zachowujemy osobne galerie i osobne tryby v2/author sensitivity; bez łączonego rankingu.', '',
        '## Walidacja i dysk', '',
        f"Testy: {checks['tests_passed']} / {checks['tests_total']} przeszło. {checks['scope']} "
        f"Kontrola składni JS: {checks['javascript_syntax']}. {checks['http_smoke']} "
        'Nie wykonano wizualnej kontroli w przeglądarce.', '',
        '| Wolumen | Wolne przed przygotowaniem [GiB] | Wolne przy raporcie [GiB] |', '|---|---:|---:|',
        *[f"| {p} | {v['free']/2**30:.3f} | {current_disk[p]['free']/2**30:.3f} |" for p,v in state['disk_before'].items()], '',
        'Małe artefakty nowej serii: około '+f"{summary['new_artifacts_bytes']/2**20:.2f} MiB"+
        '; koszt nowych obrazów: 0 bajtów; GPU: 0 s, brak runów. Zmiana wolnego miejsca obejmuje '
        'również niezależnie działające pipeline’y. Nie wykonywano sprzątania ani relokacji ich danych.', '',
        '## SHA i decyzje', '',
        '`summary.json` zawiera SHA256 wszystkich nowych manifestów, rejestracji, plików źródłowych i walidacji. '
        '`preservation_before.json` i `preservation_after.json` są zgodne. '
        'Decyzje: `reference_eval_registration.json`, `vn_full_seeds_decision.json`, `xm3600_gate_registration.json`. '
        'Reguła P1.A: mediana VN dev t2i R@1 z trzech seedów; remis: rosnący seed. '
        'Nie dokonano wyboru ani nowego testu.', '',
        '## Wymaga decyzji użytkownika', '',
        '- Draft `xm3600_gate_registration.json`: wspólny przyszły protokół dla finalisty i referencji; '
        'proponowane +2 p.p. t2i R@1 oraz dolna granica paired 95% CI >0 wobec małego polskozdolnego konkurenta '
        'wybranego poza XM. Draft nie daje zgody na uruchomienie.',
        '- Publikacja po dossier licencyjnym i niezależnych wynikach.',
        '- Nowy pilot Wiki dopiero po dev, guardzie, niezmiennym odczycie baseline i decyzji.',
        '- Destylacja dopiero po wyborze nauczyciela, audycie leakage i akceptacji; plan P2.G jeszcze nie rozpoczęty.', '',
        'Kontekst z `project_consultation_brief.md` i przekazanego zadania. '
        '`reports/consultation_2026-10-21_external.md` nie istnieje w tym checkoutcie. '
        'Blokadą dalszego wykonania jest rezerwa D:, a następnie wymienione warunki P0/P0.5; '
        'ten raport nie deklaruje ukończenia serii.', '',
    ]
    report = OUT / 'REPORT.md'
    if report.exists():
        raise ValueError('Report already finalized; create a new dated report instead of overwriting')
    report.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Finalized {report}; {len(files)} artifact SHA256 values')


if __name__ == '__main__':
    main()
