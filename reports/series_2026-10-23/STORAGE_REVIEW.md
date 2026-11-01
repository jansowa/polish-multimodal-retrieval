# Przegląd miejsca — 24.10.2026

Nic nie usunięto ani nie przeniesiono w ramach tego przeglądu. Obowiązuje Twoja decyzja: ewentualne kasowanie wykonujesz sam.

Wolne miejsce w chwili zapisu: **C: 8.91 GiB**, **D: 4.20 GiB**. Aktywne zbieranie danych nadal może zmieniać te wartości.

## Co zajmuje C

Ścieżki poniżej są rzeczywistymi lokalizacjami. Dowiązania z D nie są dodatkowymi kopiami.

| Katalog | Rozmiar plików | Ocena |
|---|---:|---|
| `C:\Users\jsowa\benchmark-translations\series-2026-10-23\relocated` | 11.44 GiB | 11,44 GiB cache’y i zamrożonych danych przeniesionych wcześniej z D; szczegóły niżej |
| `C:\Users\jsowa\benchmark-translations\series-2026-10-23\wiki-native-dev` | 2.57 GiB | Nowe oryginalne obrazy do audytu i zamrożonego dev — zachować |
| `C:\Users\jsowa\benchmark-translations\visualnews-image-cache` | 10.49 GiB | Źródłowe obrazy/parquet oraz indeks — zachować na potrzeby danych i pipeline’u |
| `C:\Users\jsowa\benchmark-translations\pipeline-storage` | 1.43 GiB | Aktywne Wiki/VN, w tym SQLite/WAL — zachować |

## Pliki, które proponuję do ręcznego usunięcia

Łącznie: **12.45 MiB na C** i **203.71 MiB na D**. Są to rozmiary plików; odzyskane miejsce może nieznacznie różnić się przez przydział NTFS.

### 1. Nieużywany cache po odrzuconym uruchomieniu — D

**112.64 MiB**, folder bazowy:

`D:\work\translate-vision-embedder-benchmark\pl-multimodal-benchmark-translator\pl-multimodal-benchmark-translator\training\runs\tinyclip-visualnews-v1\text_cache\embeddings`

Wyłącznie trzy pozycje o wspólnym identyfikatorze:

- `d88614a4d82ad83396067a00e02bc9af2a0cdfd66e93120886da159363975f71.pt`
- `d88614a4d82ad83396067a00e02bc9af2a0cdfd66e93120886da159363975f71.json`
- `d88614a4d82ad83396067a00e02bc9af2a0cdfd66e93120886da159363975f71.chunks` — cały ten katalog.

Powstały przy zmianie wersji bibliotek, a uruchomienie odrzucono przed treningiem. Udane seedy korzystały z historycznego cache’a `dbd7f7...`, którego SHA256 ponownie sprawdzono. Nie usuwać całego `embeddings` ani pozostałych identyfikatorów. Utrata: możliwość ponownego wykorzystania tego odrzuconego przeliczenia.

### 2. Stany wznowienia 19 zakończonych treningów

Poniżej wyłącznie `last.pt`. Dla każdego sprawdzono SHA256 zachowanego `best.pt` względem `result.json`. Usunięcie nie zmienia wybranych modeli ani zapisanych wyników; usuwa stan optymalizatora/RNG potrzebny do wznowienia danego treningu.

Na C:

- `C:\Users\jsowa\benchmark-translations\series-2026-10-23\runs\vn_full_seeds\seed-44\VN_FULL\last.pt` — 6.23 MiB
- `C:\Users\jsowa\benchmark-translations\series-2026-10-23\runs\vn_full_seeds\seed-43\VN_FULL\last.pt` — 6.23 MiB

Na D ścieżki względem `D:\work\translate-vision-embedder-benchmark\pl-multimodal-benchmark-translator\pl-multimodal-benchmark-translator`:

- `training\runs\tinyclip-wiki-pilot-v1\WIKI_10pct\last.pt` — 5.03 MiB
- `training\runs\tinyclip-wiki-pilot-v1\WIKI_03pct\last.pt` — 5.03 MiB
- `training\runs\tinyclip-wiki-pilot-v1\WIKI_00pct\last.pt` — 4.95 MiB
- `training\runs\tinyclip-source-ablation-v1\VN_S\last.pt` — 5.08 MiB
- `training\runs\tinyclip-source-ablation-v1\VN_R\last.pt` — 5.43 MiB
- `training\runs\tinyclip-source-ablation-v1\VN_only\last.pt` — 4.93 MiB
- `training\runs\tinyclip-source-ablation-v1\VN_nonCOCO_R\last.pt` — 5.19 MiB
- `training\runs\tinyclip-source-ablation-v1\VN_nonCOCO_AR\last.pt` — 5.34 MiB
- `training\runs\tinyclip-source-ablation-v1\VN_nonCOCO_A\last.pt` — 5.19 MiB
- `training\runs\tinyclip-source-ablation-v1\VN_FULL_drop_S\last.pt` — 6.14 MiB
- `training\runs\tinyclip-source-ablation-v1\VN_FULL_drop_R\last.pt` — 6.10 MiB
- `training\runs\tinyclip-source-ablation-v1\VN_FULL_drop_C\last.pt` — 5.98 MiB
- `training\runs\tinyclip-source-ablation-v1\VN_FULL_drop_A\last.pt` — 6.15 MiB
- `training\runs\tinyclip-source-ablation-v1\VN_FULL\last.pt` — 6.23 MiB
- `training\runs\tinyclip-source-ablation-v1\VN_C\last.pt` — 5.33 MiB
- `training\runs\tinyclip-source-ablation-v1\VN_A\last.pt` — 5.41 MiB
- `training\runs\tinyclip-source-ablation-v1\FULL_only\last.pt` — 3.58 MiB

## Większe możliwości, wymagające świadomego kompromisu

Na C dwa odtwarzalne cache’e zajmują łącznie **6,14 GiB**:

- `C:\Users\jsowa\benchmark-translations\series-2026-10-23\relocated\tinyclip-adaptation-cache\prefix.npy` — **4,47 GiB**.
- `C:\Users\jsowa\benchmark-translations\series-2026-10-23\relocated\tinyclip-generalization-pixels\native_rgb.npy` — **1,67 GiB**.

To nie są zbędne duplikaty: starsze moduły `tinyclip_visualnews.Engine`, generalization i częściowego fine-tuningu jawnie je wczytują lub weryfikują ich SHA. Ich usunięcie spowoduje błąd tych uruchomień do czasu odbudowania cache’y w zgodnym środowisku. Na razie proponuję je zachować. Nie usuwać sąsiednich `features.pt`, metadanych ani zamrożonych danych VN.

Na D istnieją dwie historyczne kopie sprzed migracji — łącznie około **0,89 GiB**:

- `D:\work\translate-vision-embedder-benchmark\pl-multimodal-benchmark-translator\pl-multimodal-benchmark-translator\reports\visualnews_pipeline\v1.before_storage_migration_20261021_194724` — 882.11 MiB
- `D:\work\translate-vision-embedder-benchmark\pl-multimodal-benchmark-translator\pl-multimodal-benchmark-translator\reports\wikimedia_pipeline\v1.before_storage_migration_20261021_194724` — 26.86 MiB

Aktywne `v1` wskazują na C. Usunięcie kopii `before_storage_migration` oznacza utratę możliwości powrotu do tego historycznego stanu. Nie sprawdzałem pełnej równoważności historycznych baz z obecnymi, dlatego nie klasyfikuję tych kopii jako bezwarunkowo zbędnych. Ich usunięcie nie zwolni miejsca na C.

## WSL na C

`C:\Users\jsowa\AppData\Local\wsl\{2b947200-4ccc-4b70-a439-a4689c52273b}\ext4.vhdx` ma długość pliku około **214,89 GiB**. To cały dysk Ubuntu, obejmujący również inne projekty — nie usuwać tego pliku.

- `/home/jsowa/.cache/huggingface` — około **39,02 GiB** wg `du`.
- W tym `datasets/json` — około **15,54 GiB**; największe pochodzą z `multilang-kd`, a część wskazanych źródeł już nie istnieje. Nie proponuję kasowania całości.
- `.../huggingface/xet` — około **2,98 GiB** transferowego cache’a, głównie `chunk-cache`. Kandydat do osobnego ręcznego porządku po upewnieniu się, że nie trwa pobieranie w innych projektach; utrata oznacza możliwe ponowne pobieranie fragmentów.

Zwolnienie miejsca wewnątrz WSL nie gwarantuje natychmiastowego zmniejszenia pliku VHDX ani wzrostu wolnego C. Ewentualne odzyskiwanie miejsca z VHDX wymaga osobnej procedury; nie zatrzymywałem WSL ani aktywnych pipeline’ów.

## Sprawdzone pułapki i ograniczenia

- Dwa pliki `evaluation_examples.json` po około 793 MiB w `p0-e2` i `vision-four/reference/artifacts` są już hardlinkami do tych samych bajtów. Usunięcie jednej ścieżki nie odzyska tych 793 MiB.
- Nie usuwać aktywnych plików `.tmp`, SQLite/WAL, źródłowych obrazów, zamrożonych manifestów, wybranych checkpointów ani całych folderów `.cache`.
- Rozmiary folderów są sumami długości plików, nie dokładnym przydziałem całego woluminu. Przegląd pomija dowiązania katalogowe; aktywne katalogi zmieniają się w czasie odczytu.
- Pierwotne techniczne inwentarze v1/v2 mają niepoprawne agregaty przydziału NTFS; nie użyto ich w tym raporcie. Poprawne długości plików i oddzielne sprawdzenie hardlinków są zachowane w `storage_review.json` i dowodach.

Dowody: `storage_deletion_proposal_evidence.json` (dokładne ścieżki i SHA256), `storage_wsl_cache_provenance.json` (źródła cache’y), `storage_review.json` (bieżący przegląd).

Do dalszej pracy: bez automatycznego usuwania plików; bez nowych dużych pobrań w ramach tego przeglądu; przed kolejnymi dużymi zapisami ponowny pomiar fizycznego C i D.
