# Cache Hugging Face — identyfikacja z 24.10.2026

Przegląd odczytowy. Nie usunięto ani nie przeniesiono żadnych plików. Ewentualne usuwanie użytkownik wykonuje sam.

## Główna konfiguracja

- Vision: `wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M`, rewizja `07a4b0bc751cb64fecd2b661c048c1dd98d69444`.
- Text: `OPI-PIB/PolDense-32M`, rewizja `ca70494a25dad90d980db7c735658dc7bf88ed25`.
- Z TinyCLIP korzystamy z vision tower i oryginalnej projekcji dającej 512 cech. Własny projektor odwzorowuje je do 384 wymiarów PolDense. W głównej obecnej recepcie oba encodery są zamrożone; trenujemy projektor.
- `Text-19M` w nazwie repo opisuje dołączoną oryginalną wieżę tekstową TinyCLIP. Nie zastępuje używanego przez nas PolDense.

Zachować w `/home/jsowa/.cache/huggingface/hub/`:

- `models--wkcn--TinyCLIP-ViT-39M-16-Text-19M-YFCC15M` — główny model.
- `models--OPI-PIB--PolDense-32M` — główny model tekstowy.
- `models--google--siglip2-base-patch16-256` — wcześniejszy encoder/nauczyciel i zaplanowane porównanie; około 1,43 GiB.
- `models--apple--MobileCLIP2-S2` — zaplanowana diagnostyka/porównanie.
- Pozostałe modele z zarejestrowanej listy 20 referencji: w większości katalogi zawierają obecnie głównie konfiguracje/tokenizery. Nie usuwać ich z samego powodu dużej liczby nazw katalogów.

## Kandydaci wśród modeli po zakończonych porównaniach

Poniższe modele nie należą do aktualnie zaplanowanej listy 20 referencji ani do listy backbone'ów P1.B. Usunięcie ich lokalnego cache’a nie usuwa zapisanych wyników, wybranych checkpointów projektorów ani manifestów. Powrót do dawnych eksperymentów będzie wymagał ponownego pobrania przypiętych wag. Nie analizowano zależności wszystkich innych projektów użytkownika.

Wszystkie katalogi są pod `/home/jsowa/.cache/huggingface/hub/`:

| Katalog | Przybliżony rozmiar wg `du` |
|---|---:|
| `models--wkcn--TinyCLIP-ViT-61M-32-Text-29M-LAION400M` | 440 MiB |
| `models--wkcn--TinyCLIP-ViT-40M-32-Text-19M-LAION400M` | 321 MiB |
| `models--timm--resnet50_clip.cc12m` | 389 MiB |
| `models--timm--resnet50_clip.yfcc15m` | 389 MiB |
| `models--OFA-Sys--chinese-clip-rn50` | 294 MiB |
| `models--facebook--dinov2-small` | 84 MiB |

Łącznie około **1,87 GiB**. Nie dotyczy to głównego TinyCLIP **39M**, który ma pozostać.

## `/home/jsowa/.cache/huggingface/datasets/json`

Rozmiar wg `du`: **15,54 GiB**. Zidentyfikowano 52 konfiguracje; wszystkie wskazują na źródła w `/home/jsowa/work/multilang-kd/`.

To cache’e tabel Arrow do MetricX i consistency filtering: m.in. MS MARCO, FEVER, FiQA-PL, HotpotQA-PL, SQuADv2. Nie są danymi tego projektu obraz–tekst. W 46 wpisach wskazana lokalizacja źródłowa obecnie nie istnieje. Cache może zawierać również pochodne tabele map/filter.

**Dla obecnego projektu ten katalog jest zbędny.** Usunięcie całego katalogu jest sensowne, jeżeli nie potrzebujesz tych pośrednich danych i odtwarzania prac `multilang-kd`. Nie ma podstaw, aby obiecać ich odbudowę bez ponownego pozyskania brakujących źródeł lub wykonania dawnych obliczeń.

## `/home/jsowa/.cache/huggingface/datasets/parquet`

Rozmiar wg `du`: **2,92 GiB**. To przetworzone cache’e Arrow utworzone przez loader Parquet, a nie jeden nazwany zbiór.

Katalog bazowy dla poniższych pozycji: `/home/jsowa/.cache/huggingface/datasets/parquet/`.

| Podkatalog | Zawartość | Rozmiar |
|---|---|---:|
| `default-150f3fa686223f16` | Nasz FashionIQ, 1000 rekordów | 67,9 MiB |
| `default-29208fb3fa54cf44` | Nasz RefCOCO, 1000 rekordów | 92,0 MiB |
| `default-60bf5d7cc28c2fb2` | Nasz CIRR, 1000 rekordów | 66,9 MiB |
| `default-7b107485bf79574b` | Nasz MSCOCO t2i, 1000 rekordów | 83,1 MiB |
| `default-396a72d4c16801e1` | Fragment ARO Visual Relation, 6840 rekordów | 915,8 MiB |
| `default-271f336f85783519` | `mine-hard-negatives`: korpus 1 555 806 rekordów | 584,6 MiB |
| `default-c03e25db7f881a0a` | `mine-hard-negatives`: korpus 1 555 806 rekordów | 584,6 MiB |
| `default-f1492b0e5f8849cb` | `mine-hard-negatives`: korpus 1 555 806 rekordów | 584,6 MiB |
| `default-c03de3610ef0daf9` | `mine-hard-negatives`: NFCorpus PL | 6,1 MiB |
| `default-43defb71f43dd255`, `default-de51a2415d794cee` | Małe testowe korpusy `mine-hard-negatives` | po około 28 KiB |
| `default-8121358a0931d855` | Puste/uszkodzone metadane, bez rozpoznanej zawartości | 4 KiB |

Nasza część: **1,20 GiB**. Inne projekty: około **1,72 GiB**.

Nasza część jest pochodnym cache’em, który można odbudować. Cztery zestawy MMEB rozpoznano po liczbie rekordów oraz trzech początkowych tekstach/ścieżkach, z zachowanych oryginałów w `hub/datasets--TIGER-Lab--MMEB-eval/snapshots/eb3a8af230a66be5c9308cd2be5779b9d6de4cb3/`. To identyfikacja źródła, nie porównanie hashy całych tabel.

Dla ARO zachowane jest wszystkich 7 źródłowych shardów w `hub/datasets--mteb--ARO-Visual-Relation/snapshots/53c1115ab5871ab33b09d5bb8856236d46bdaf2d/`; cache 6840 rekordów rozpoznano jako fragment tego zestawu po schemacie i początkowych captionach. Nie dopasowano pełnej zawartości cache’a do konkretnego podzbioru shardów.

Usunięcie naszych pochodnych cache’y nie usuwa oryginałów ani zamrożonych danych treningowych, ale kolejne użycie loadera może odtworzyć te cache’e i znów zająć miejsce. **Zachować źródłowe katalogi `hub/datasets--...`, oryginalne obrazy i manifesty.** Źródła zapisane w metadanych części `mine-hard-negatives` już nie istnieją; ocena usunięcia tej części zależy od tego, czy potrzebujesz danych tamtego projektu.

## Miejsce na C a WSL

Usunięcie plików wewnątrz WSL zwolni miejsce w systemie plików Ubuntu. Nie gwarantuje natychmiastowego zmniejszenia `ext4.vhdx` i wzrostu wolnego miejsca na C. Nie wykonywano kompaktowania ani zatrzymywania WSL/pipeline’ów.

Dowody techniczne: `hf_storage_inspection_2026-10-24.json` oraz `parquet_cache_source_matches_2026-10-24.json`. Nie zmieniano cache’y HF.
