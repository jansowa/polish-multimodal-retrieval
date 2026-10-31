"""Write a reviewable storage proposal. No deletion or relocation capabilities."""
import datetime
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/series_2026-10-23'
GIB = 1024**3
MIB = 1024**2


def main():
    inventory = json.loads((OUT / 'storage_readonly_inventory_v2.json').read_text())
    evidence = json.loads((OUT / 'storage_deletion_proposal_evidence.json').read_text())
    groups = [{key: item[key] for key in ('path', 'files', 'logical_bytes')}
              for item in inventory['groups']]
    by_path = {item['path']: item for item in groups}
    resumes = evidence['verified_resume_candidates']
    assert len(resumes) == 19 and not evidence['skipped_resume_files']
    assert all(item['links'] == 1 for item in resumes + evidence['unused_text_cache_candidates'])
    totals = {drive: sum(item['bytes'] for item in resumes if item['path'].startswith(drive))
              for drive in ('C:', 'D:')}
    totals['D:'] += evidence['unused_text_cache_bytes']
    disks = {drive: dict(free_bytes=shutil.disk_usage(drive + '/').free)
             for drive in ('C:', 'D:')}
    backups = [g for g in groups if 'before_storage_migration' in g['path']]
    # v2 file lengths and explicit hardlink checks are useful. Its aggregate
    # allocation sums are invalid due to unsupported Windows views of WSL links.
    # Exclude those fields entirely from this final report.
    result = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  no_files_deleted=True, no_files_moved=True, no_large_downloads_started=True,
                  source_inventory='storage_readonly_inventory_v2.json',
                  excluded_invalid_fields=['all v1 and v2 aggregate allocated_unique_bytes'],
                  method='logical file lengths; junctions skipped; candidate hardlink counts and best checkpoint SHAs independently checked',
                  limitations='live directories change during scanning; missing/unsupported paths are listed in source inventory; no claim of exact whole-volume physical allocation',
                  disks=disks, groups=sorted(groups, key=lambda r: -r['logical_bytes']),
                  verified_candidate_bytes_by_drive=totals,
                  historical_backups_optional=backups,
                  derived_C_caches_conditional_bytes=4803143808 + 1792035968,
                  wsl=dict(vhdx_path=r'C:\Users\jsowa\AppData\Local\wsl\{2b947200-4ccc-4b70-a439-a4689c52273b}\ext4.vhdx',
                           vhdx_logical_bytes=230738100224, hf_du_bytes=41900163072,
                           datasets_json_du_bytes=16690077696, xet_du_bytes=3197521920,
                           immediate_Windows_reclaim_not_guaranteed=True),
                  evidence='storage_deletion_proposal_evidence.json')
    with (OUT / 'storage_review.json').open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    croot = Path(r'C:\Users\jsowa\benchmark-translations')
    lines = [
        '# Przegląd miejsca — ' + datetime.datetime.now().strftime('%d.%m.%Y'), '',
        'Nic nie usunięto ani nie przeniesiono w ramach tego przeglądu. Obowiązuje Twoja decyzja: ewentualne kasowanie wykonujesz sam.', '',
        f"Wolne miejsce w chwili zapisu: **C: {disks['C:']['free_bytes']/GIB:.2f} GiB**, **D: {disks['D:']['free_bytes']/GIB:.2f} GiB**. Aktywne zbieranie danych nadal może zmieniać te wartości.", '',
        '## Co zajmuje C', '',
        'Ścieżki poniżej są rzeczywistymi lokalizacjami. Dowiązania z D nie są dodatkowymi kopiami.', '',
        '| Katalog | Rozmiar plików | Ocena |', '|---|---:|---|',
    ]
    for relative, note in [
        ('series-2026-10-23/relocated', '11,44 GiB cache’y i zamrożonych danych przeniesionych wcześniej z D; szczegóły niżej'),
        ('series-2026-10-23/wiki-native-dev', 'Nowe oryginalne obrazy do audytu i zamrożonego dev — zachować'),
        ('visualnews-image-cache', 'Źródłowe obrazy/parquet oraz indeks — zachować na potrzeby danych i pipeline’u'),
        ('pipeline-storage', 'Aktywne Wiki/VN, w tym SQLite/WAL — zachować'),
    ]:
        path = croot / relative
        lines.append(f"| `{path}` | {by_path[str(path)]['logical_bytes']/GIB:.2f} GiB | {note} |")
    lines += ['', '## Pliki, które proponuję do ręcznego usunięcia', '',
              f"Łącznie: **{totals['C:']/MIB:.2f} MiB na C** i **{totals['D:']/MIB:.2f} MiB na D**. Są to rozmiary plików; odzyskane miejsce może nieznacznie różnić się przez przydział NTFS.", '',
              '### 1. Nieużywany cache po odrzuconym uruchomieniu — D', '',
              f"**{evidence['unused_text_cache_bytes']/MIB:.2f} MiB**, folder bazowy:", '',
              f"`{ROOT / 'training/runs/tinyclip-visualnews-v1/text_cache/embeddings'}`", '',
              'Wyłącznie trzy pozycje o wspólnym identyfikatorze:', '',
              '- `d88614a4d82ad83396067a00e02bc9af2a0cdfd66e93120886da159363975f71.pt`',
              '- `d88614a4d82ad83396067a00e02bc9af2a0cdfd66e93120886da159363975f71.json`',
              '- `d88614a4d82ad83396067a00e02bc9af2a0cdfd66e93120886da159363975f71.chunks` — cały ten katalog.', '',
              'Powstały przy zmianie wersji bibliotek, a uruchomienie odrzucono przed treningiem. Udane seedy korzystały z historycznego cache’a `dbd7f7...`, którego SHA256 ponownie sprawdzono. Nie usuwać całego `embeddings` ani pozostałych identyfikatorów. Utrata: możliwość ponownego wykorzystania tego odrzuconego przeliczenia.', '',
              '### 2. Stany wznowienia 19 zakończonych treningów', '',
              'Poniżej wyłącznie `last.pt`. Dla każdego sprawdzono SHA256 zachowanego `best.pt` względem `result.json`. Usunięcie nie zmienia wybranych modeli ani zapisanych wyników; usuwa stan optymalizatora/RNG potrzebny do wznowienia danego treningu.', '',
              'Na C:', '']
    for item in resumes:
        if item['path'].startswith('C:'):
            lines.append(f"- `{item['path']}` — {item['bytes']/MIB:.2f} MiB")
    lines += ['', f'Na D ścieżki względem `{ROOT}`:', '']
    for item in resumes:
        if item['path'].startswith('D:'):
            lines.append(f"- `{Path(item['path']).relative_to(ROOT)}` — {item['bytes']/MIB:.2f} MiB")
    lines += ['', '## Większe możliwości, wymagające świadomego kompromisu', '',
              'Na C dwa odtwarzalne cache’e zajmują łącznie **6,14 GiB**:', '',
              f"- `{croot / 'series-2026-10-23/relocated/tinyclip-adaptation-cache/prefix.npy'}` — **4,47 GiB**.",
              f"- `{croot / 'series-2026-10-23/relocated/tinyclip-generalization-pixels/native_rgb.npy'}` — **1,67 GiB**.", '',
              'To nie są zbędne duplikaty: starsze moduły `tinyclip_visualnews.Engine`, generalization i częściowego fine-tuningu jawnie je wczytują lub weryfikują ich SHA. Ich usunięcie spowoduje błąd tych uruchomień do czasu odbudowania cache’y w zgodnym środowisku. Na razie proponuję je zachować. Nie usuwać sąsiednich `features.pt`, metadanych ani zamrożonych danych VN.', '',
              'Na D istnieją dwie historyczne kopie sprzed migracji — łącznie około **0,89 GiB**:', '']
    for backup in backups:
        lines.append(f"- `{backup['path']}` — {backup['logical_bytes']/MIB:.2f} MiB")
    lines += ['', 'Aktywne `v1` wskazują na C. Usunięcie kopii `before_storage_migration` oznacza utratę możliwości powrotu do tego historycznego stanu. Nie sprawdzałem pełnej równoważności historycznych baz z obecnymi, dlatego nie klasyfikuję tych kopii jako bezwarunkowo zbędnych. Ich usunięcie nie zwolni miejsca na C.', '',
              '## WSL na C', '',
              f"`{result['wsl']['vhdx_path']}` ma długość pliku około **214,89 GiB**. To cały dysk Ubuntu, obejmujący również inne projekty — nie usuwać tego pliku.", '',
              '- `/home/jsowa/.cache/huggingface` — około **39,02 GiB** wg `du`.',
              '- W tym `datasets/json` — około **15,54 GiB**; największe pochodzą z `multilang-kd`, a część wskazanych źródeł już nie istnieje. Nie proponuję kasowania całości.',
              '- `.../huggingface/xet` — około **2,98 GiB** transferowego cache’a, głównie `chunk-cache`. Kandydat do osobnego ręcznego porządku po upewnieniu się, że nie trwa pobieranie w innych projektach; utrata oznacza możliwe ponowne pobieranie fragmentów.', '',
              'Zwolnienie miejsca wewnątrz WSL nie gwarantuje natychmiastowego zmniejszenia pliku VHDX ani wzrostu wolnego C. Ewentualne odzyskiwanie miejsca z VHDX wymaga osobnej procedury; nie zatrzymywałem WSL ani aktywnych pipeline’ów.', '',
              '## Sprawdzone pułapki i ograniczenia', '',
              '- Dwa pliki `evaluation_examples.json` po około 793 MiB w `p0-e2` i `vision-four/reference/artifacts` są już hardlinkami do tych samych bajtów. Usunięcie jednej ścieżki nie odzyska tych 793 MiB.',
              '- Nie usuwać aktywnych plików `.tmp`, SQLite/WAL, źródłowych obrazów, zamrożonych manifestów, wybranych checkpointów ani całych folderów `.cache`.',
              '- Rozmiary folderów są sumami długości plików, nie dokładnym przydziałem całego woluminu. Przegląd pomija dowiązania katalogowe; aktywne katalogi zmieniają się w czasie odczytu.',
              '- Pierwotne techniczne inwentarze v1/v2 mają niepoprawne agregaty przydziału NTFS; nie użyto ich w tym raporcie. Poprawne długości plików i oddzielne sprawdzenie hardlinków są zachowane w `storage_review.json` i dowodach.', '',
              'Dowody: `storage_deletion_proposal_evidence.json` (dokładne ścieżki i SHA256), `storage_wsl_cache_provenance.json` (źródła cache’y), `storage_review.json` (bieżący przegląd).', '',
              'Do dalszej pracy: bez automatycznego usuwania plików; bez nowych dużych pobrań w ramach tego przeglądu; przed kolejnymi dużymi zapisami ponowny pomiar fizycznego C i D.', '']
    with (OUT / 'STORAGE_REVIEW.md').open('x', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    print(json.dumps(dict(report=str(OUT / 'STORAGE_REVIEW.md'), verified_candidates_bytes=totals,
                          disks=disks, no_deletions=True), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
