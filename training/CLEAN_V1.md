# CLEAN-v1: audyt guarda i zależność od COCO

Aktualny sprint korzysta z istniejącego leakage guarda. Nie zastępuje jego
blacklisty, source/parent IDs ani podziału train/validation nowym frameworkiem.

## Wyniki

[Pełny raport i metryki](../reports/clean_v1.md),
[odrzucenia per źródło i benchmark](../reports/clean_v1_leakage.csv),
[kontrola z wyrównanym budżetem](../reports/clean_v1_matched_budget.md).

Po istniejącej filtracji nie znaleziono znanego source/parent/hash overlapu.
Historyczne wyniki zachowują etykietę **LEGACY / pre-audit**; nie ma podstaw do
oznaczenia ich jako contaminated na podstawie tego audytu.

| Train | Rekordy | Unique image-text pairs | Unique images | MSCOCO R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|---:|
| FULL_CLEAN | 70289 | 67445 | 10645 | 45,2% | 79,1% | 88,3% |
| NO_MBEIR_MSCOCO | 52420 | 49601 | 7102 | 35,2% | 67,4% | 80,7% |
| NON_COCO_CLEANROOM | 24473 | 23601 | 3037 | 22,3% | 49,1% | 66,0% |

Jeden seed: 42. Zamrożony TinyCLIP-39M i PolDense, dotychczasowy linear,
multi-positive sigmoid, natywne preprocessing/cache, batch 256, AdamW, 20 epok.
Validation i galerie benchmarków są takie same dla trzech wariantów.
Usuwanie COCO lineage propaguje się przez połączone parent IDs i aliasy pikseli.
Brak znanego mapowania VG→COCO nie dowodzi braku każdej możliwej relacji źródłowej.

Nowe checkpointy wybiera val t2i R@1; średnia t2i/i2t R@1 rozstrzyga remisy.
Historyczne checkpointy używały minimum val loss. FULL_CLEAN odtworzył dokładnie
całą historię starego seeda 42 oraz wagi w jego starej wybranej epoce 6;
nowa reguła wybrała epokę 9. Potwierdzenie:
[reproduction.json](../reports/clean_v1_legacy_reproduction.json).

Kontrola przy identycznym limicie **5500 kroków i 1405780 prezentacji** dała
45,2% / 32,4% / 22,0%. Mniejsze zbiory były wielokrotnie tasowane;
liczba różnych obrazów pozostaje różna. Limity są równe, ale wybrane na validation
checkpointy mogą pochodzić z różnych wcześniejszych kroków. Kontrola nie dowodzi,
że całą lukę powoduje sama domena, niezależnie od liczby różnych zdjęć.

## Odrzucenia

Liczby dotyczą pełnego przygotowania train+validation; powody nie są rozłączne.
M-BEIR liczy pary rozwinięte przez pozytywne qrels, więc liczba par może być
większa niż liczba unikalnych rekordów tłumaczeń.
Dla MSCOCO_t2i zapisano parent overlap:

- ARO Relation: 119 rekordów przez mapowanie źródłowe;
- ARO Attribution: 139;
- SugarCrepe: 762;
- M-BEIR MSCOCO: 5010; łącznie wykluczono 5027 rekordów tego źródła,
  licząc także przypadki odrzucone wyłącznie przez tekst.

Pozostałe wykluczenia ARO przez parent overlap dotyczą RefCOCO: 592 i 560.
Zero odrzuceń z powodu hashy nie oznacza braku zabezpieczenia hashami:
powód parent może wystarczyć do odrzucenia przed dekodowaniem obrazu.
Raport opisuje rzeczywiście zapisane powody, bez dopisywania hipotetycznych.

## Near duplicates

Porównano **202886780** par: 10685 lokalnych ścieżek train (10645 unique pixels)
z 18988 ścieżkami wszystkich legacy query/candidate images. Użyto 64-bitowego
DCT pHash, po normalizacji EXIF. Nie znaleziono par o odległości <=6;
minimum wyniosło 8. Dla 200 najbliższych kandydatów dopisano cosine z istniejących
cech TinyCLIP. Nie wykonano pełnego przeszukiwania embeddingów.

Obejrzano 48 najbliższych par; nie potwierdzono nowych overlapów:
[zapis przeglądu](../reports/clean_v1_visual_review.json).
To przegląd obrazów przez Codex, a nie niezależna adjudykacja człowieka.
Nie filtrowano automatycznie danych na podstawie pHash. Nieznane cropy
i silniejsze transformacje nadal mogą pozostać niewykryte.

Duże artefakty w `training/runs/clean-v1/`:

- `audit.json`, `rejection_evidence.jsonl`, `provenance.jsonl`;
- `FULL_CLEAN.jsonl`, `NO_MBEIR_MSCOCO.jsonl`, `NON_COCO_CLEANROOM.jsonl`;
- wspólny `validation.jsonl`;
- `near_duplicate_candidates.csv/json`, `perceptual_hashes.json`, contact sheets;
- wyniki, checkpointy i logi każdego wariantu oraz `matched-budget/`.

## XM3600-PL

Pobrano [oficjalne dane](https://google.github.io/crossmodal-3600/), podpisy PL
autorów i atrybucje zdjęć. Przygotowano **FINAL BLIND clean subset**:
3597 obrazów i 7133 podpisów po deduplikacji identycznych podpisów w obrębie obrazu.
Trzy obrazy wyłączono przez znane relacje Flickr/COCO lub overlap z istniejącymi
pulami. Są wyszczególnione w [raporcie wykluczeń](../reports/xm3600_pl_exclusions.json).
Nie jest to niezmodyfikowana oficjalna galeria 3600 obrazów.

[Manifest przygotowania](../reports/xm3600_pl_preparation.json) zawiera SHA256
oficjalnych pobrań, licencję adnotacji i informacje o pochodzeniu. Licencje zdjęć
są zachowane w atrybucjach. Nie policzono żadnego wyniku modelu na tym zbiorze.
W przyszłym ingestion należy dołączyć `training/runs/xm3600-pl/blacklist.json`
do istniejącej blacklisty, aby utrzymać rozdzielenie danych.

Evaluator używa pełnej galerii tego subsetu, multi-positive qrels z oryginalnych
polskich captionów i mierzy oba kierunki. Nie należy bezpośrednio porównywać
jego liczb z legacy MSCOCO o lokalnych pulach kandydatów.

## Uruchomienie

Ze środowiska WSL, z katalogu projektu:

```bash
PY=~/.venvs/pl-mmb-benchmark/bin/python
$PY training/cleanroom_study.py audit
$PY training/cleanroom_study.py near
$PY training/cleanroom_study.py train
$PY training/cleanroom_study.py report
$PY training/matched_cleanroom.py
$PY training/matched_cleanroom.py report
$PY training/xm3600.py prepare
PYTHONPATH=src $PY -m unittest discover -s tests -p 'test_cleanroom*.py' -v
```

Run identities zawierają hashe kodu i danych. Zmiana kodu nie pozwala po cichu
ponownie wykorzystać ukończonego treningu. Polecenie `report` tylko eksportuje
zapisane wyniki i nie uruchamia ponownego treningu ani benchmarku.

Wyłącznie dla wcześniej wybranego finalisty:

```bash
$PY training/xm3600.py evaluate --checkpoint PATH_TO_PROJECTOR --release-final-blind
```

Nie uruchomiono tego polecenia na prawdziwym zbiorze. Regresyjne testy treningu
(29) i testy CLEAN-v1/budżetu/blind (6) przeszły.

## Decyzja

Zachować MSCOCO_t2i jako **IN-DOMAIN / LEGACY**, nie jako jedyną główną miarę
ogólnej jakości. Wynik jest wyraźnie zależny od miksu danych COCO; to inna kwestia
niż instance leakage. Dodać niezależne dane image-text i niezależny benchmark,
zanim rozszerzymy poszukiwanie architektur. XM3600-PL zachować dla finalistów.
CIRR/FashionIQ pozostają diagnostyką composition, RefCOCO — grounding.
