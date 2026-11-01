# Trening polskiego multimodalnego embeddera

**Przed nowymi treningami:** [wspólna polityka splitów i dekontaminacji](DATA_SPLITS.md).
Aktualna kolejka Wiki ma bramkę wspólnego rejestru holdoutów, obejmującą także Flickr/VizWiz.

Aktualny audyt guarda i ablacja COCO exposure: [CLEAN-v1](CLEAN_V1.md).
Wyniki historyczne pozostają LEGACY / pre-audit; [raport](../reports/clean_v1.md)
nie wykazał znanego instance leakage. XM3600-PL użytkownik włączył 31.10.2026 do zwykłego leaderboardu.

Trzecia, niezależna część projektu: `train.py` i `src/pl_mmb_training/`.
Trening nie dodaje zależności GPU do środowiska tłumaczeń i nie dopisuje wyników
eksperymentalnych do istniejącego leaderboardu. Korzysta z jego zamrożonego
protokołu v2 oraz dokładnie tych samych czterech zestawów i kandydatów.

Zaimplementowane: E0/E1 oraz runner P0 + E2a/E2b.
[Aktualny plan P0–P11](PLAN.md); wcześniejsza kolejność w `EXPERIMENTS.md` jest historyczna.

Wykonano także [pierwsze cztery frozen vision backbone'y](VISION_STUDY.md)
z sigmoid loss i liniowym projectorem. [Jakość vs liczba parametrów](../reports/training_vision_four.md):
DINOv2-Small, MobileCLIP2-S2, ConvNeXt-Tiny, oficjalny iBOT ViT-S/16 oraz
referencyjny SigLIP2. Po tej czwórce eksperymenty zatrzymano.

Kolejna zlecona seria: [pięć modeli vision z deklaracją MIT/Apache](VISION_COMMERCIAL_STUDY.md).
TinyCLIP 39M/16 YFCC, TinyCLIP 40M/32 LAION, Chinese-CLIP RN50,
RN50 CC12M i RN50 YFCC — identyczne dane i recepta sigmoid.
[Raport jakości, parametrów i licencji](../reports/training_vision_commercial_five.md).
Po tych pięciu stop; RN101 i TinyCLIP 61M pozostają planem.

P0/E2 bez pobierania danych i wag, na dokładnie tym samym splicie i cache E1:

```bash
PY=~/.venvs/pl-mmb-benchmark/bin/python
$PY training/freeze_e1.py
$PY -u training/loss_ablation.py all
$PY training/export_ablation.py
```

`freeze_e1.py` pierwszy raz uruchomić przed zmianą kodu, który wygenerował E1.
Przy wznowieniu weryfikuje istniejący lock. Nie wymaga ponownego treningu E1.
Runner dodaje dwukierunkowy retrieval, scouting obu lossów seedem 42, replikację
po ≥1 p.p. poprawy na walidacji i test tylko lossu wybranego na walidacji.
Szczegóły protokołów, temperatury, multi-positive i wyboru checkpointów: `PLAN.md`.

P0 oraz E2a/E2b wykonano; oba lossy otrzymały po trzy seedy na podstawie poprawy
walidacji. Wybrany na val **E2b (sigmoid)** osiągnął lokalne MSCOCO t2i R@1
**49,70% ± 0,70 p.p.**, +8,00 p.p. wobec pełnego E1. E2a ma wyniki val;
jego test nie został uruchomiony. [Raport P0/E2](../reports/training_p0_e2.md)
zawiera R@1/5/10 w obu kierunkach, pełne logowanie i historię loss.

Poprzednie 5435 train / 565 validation było pilotem 6k. Oryginalny config
pozostaje w `configs/e1-pilot6k.json`, a historyczny raport w
`../reports/training_e0_e1.md`. Domyślny config `configs/e1.json` nie ogranicza
liczby par (`max_pairs=null`).

Pełny snapshot E0 zawiera 95 088 przetłumaczonych rekordów. Po powiązaniu z
obrazami i filtracji: **70 289 train / 7 695 validation**, łącznie 77 984 par
i 11 861 różnych obrazów według hasha pikseli. VisualNews ma w tym snapshocie
10 083 tłumaczenia, ale nie wnosi gotowych par z dostępnym obrazem.
`data_audit.json` zawiera dokładne powody wykluczeń. Jeden rekord może mieć
kilka powodów, więc sumy liczników powodów nie oznaczają liczby wykluczonych par.

Pełny E1 wykonano dla seedów 42/43/44. MSCOCO_t2i R@1: **41,70% ± 0,20 p.p.**
(średnia ± odchylenie między seedami), wobec 13,07% dla pilota.
[Raport pełnego E0/E1](../reports/training_e0_e1_full.md) zawiera cztery benchmarki,
różnice i loss od epoki 0. Wynik MSCOCO dotyczy lokalnych pul MMEB.

Priorytet użytkowy: **query = polski tekst, wynik = obraz**. Główny bieżący
wynik to `MSCOCO_t2i`; CIRR/FashionIQ/RefCOCO i AVG są pomocniczą diagnostyką
zadań obraz+tekst. Ich wynik może korzystać z samego podobieństwa wizualnego
i nie powinien samodzielnie decydować o wyborze modelu T2I.

## Uruchomienie w WSL

Użyj istniejącego środowiska `~/.venvs/pl-mmb-benchmark` przygotowanego zgodnie
z `benchmark/README.md` (PyTorch, Transformers, sentence-transformers >=5.4,
datasets, Pillow, huggingface-hub, pyarrow). Wagi i feature cache pozostają
w natywnym cache Linux, domyślnie `~/.cache/huggingface/pl_mmb_training/`.

Z katalogu głównego projektu:

```bash
PY=~/.venvs/pl-mmb-benchmark/bin/python
$PY train.py fetch-data
$PY train.py e0
$PY train.py fetch
$PY train.py cache
$PY train.py e1
$PY training/export_report.py
```

`fetch-data` pobiera przypięte źródła danych, metadane parent IDs i 5000 zdjęć
COCO wskazanych w oryginalnych qrels M-BEIR. Nie pobiera całego archiwum M-BEIR.
`fetch` dodatkowo pobiera wagi; zapisuje pełne SHA rewizji w `model_lock.json`.
Pozostałe polecenia wymuszają tryb offline. `all` wykonuje powyższy proces
w jednym wywołaniu, z pobieraniem na początku.

```bash
$PY train.py all --config training/configs/e1.json --output training/runs/e1-full
```

Każdy output jest przypisany do jednego configu. Zmiana configu wymaga nowego
`--output`. Ukończone runy seedów i chunki feature cache są wznawialne;
nieukończony trening danego seeda zaczyna się od jego deterministycznej inicjalizacji.
Nie uruchamiaj dwóch procesów z tym samym outputem równocześnie.

## Dane i brak leakage

Domyślny config uwzględnia pełne ARO Visual Relation, ARO Visual Attribution,
SugarCrepe oraz ukończone tłumaczenia M-BEIR MSCOCO i VisualNews. Nie stosuje
limitu liczby par. ARO używa `true_caption`, SugarCrepe — `caption`; negatywne
podpisy nigdy nie stają się pozytywnymi parami treningowymi.

Tłumaczenia z SQLite są czytane w jednej transakcji i zamrażane w snapshotach.
Trwająca kontynuacja VisualNews nie dopisuje rekordów do już rozpoczętego runu.
Powiązania podpisów M-BEIR z obrazami pochodzą z oryginalnych positive qrels;
nie są zgadywane na podstawie kolejności czy sąsiadujących ID.

Liczba przetłumaczonych rekordów nie jest automatycznie liczbą gotowych par.
`data_audit.json` raportuje osobno każde źródło i powody wykluczenia: leakage,
nieznana linia pochodzenia, brak zweryfikowanego pozytywnego obrazu lub brak
samego pliku zdjęcia. Dla VisualNews config jawnie ustawia `missing_images=report`:
brakujące zdjęcia nie są cicho traktowane jako dostępne dane. Pozostałe źródła
kończą przygotowanie błędem, jeśli wymaganego obrazu nie ma. Jeden podpis może
mieć wiele pozytywnych obrazów, więc liczba rozpatrywanych par może przewyższać
liczbę przetłumaczonych rekordów.

E0 wymaga pełnych, przypiętych tłumaczeń czterech zadań MMEB i lokalnych cache
embeddingów dwóch istniejących baseline'ów. Odtwarza wszystkie cztery zadania
dla każdego modelu, sprawdza każdą rangę oraz wszystkie sześć metryk.
Brak cache lub rozbieżność blokuje E0. Nie wystarcza przepisanie starych wyników.

Blacklista zawiera wszystkie ID zapytań, wszystkie obrazy zapytań i kandydatów,
hashe plików i zdekodowanych pikseli RGB, teksty EN/PL oraz parent IDs.
Visual Genome jest mapowane na COCO przy użyciu autorskich metadanych
`image_data.json.zip`, weryfikowanych SHA256. Wykluczenie rodzica usuwa wszystkie
jego cropy z ARO. Rekord bez ustalonego rodzica jest odrzucany.
Nazwy `RefCOCO/image_<ann_id>_crop.jpg` są rozwiązywane przez oficjalne
`instances_train2014.json`: ID adnotacji → ID obrazu COCO. Bez tego mapowania
E0 kończy się błędem; numery cropów nie są traktowane jako ID zdjęć.
Raport pokazuje też nakładanie się samych zbiorów eval między sobą.

Train/validation są dzielone deterministycznie po połączonych parent IDs
i identycznych pikselach, zatem cropy i aliasy tego samego zdjęcia nie przekraczają
granicy splitu. Validation służy wyłącznie do wyboru epoki po stracie regresji.
Leaderboard eval nie wybiera epoki ani hiperparametrów.

Do treningu nie można wybrać datasetu, który ma pełny, poprawny wynik v2
w obecnym leaderboardzie. Zmiana listy datasetów leaderboardu unieważnia E0.
ARO użyte w treningu nie może następnie być niezależną ewaluacją tych modeli.
Dokładne hashe i parent IDs nie certyfikują dowolnych przekształconych kopii
zdjęć bez metadanych pochodzenia; nowe adaptery muszą wnosić zweryfikowaną linię
pochodzenia, zanim zostaną dopuszczone do treningu.

## Konfiguracja runu

`training/configs/e1.json` jest pełnym, wersjonowalnym configiem:

- `data.sources`: lista źródeł, ścieżek tłumaczeń i jawnych shardów. Obecne adaptery:
  `aro_visual_relation`, `aro_visual_attribution`, `sugarcrepe`, `mbeir_mscoco`
  i `mbeir_visualnews`. `checkpoint` wskazuje SQLite zamiast eksportu JSONL.
  Shardy HF muszą być ciągłym
  prefiksem od sharda 0; join nie zgaduje globalnych offsetów wierszy.
- `data.include_ids`: lista dokładnych ID albo `null` dla wszystkich dostępnych;
  `data.exclude_ids`: jawne wykluczenia. Nieistniejące żądane ID są błędem.
- `data.max_pairs`, `selection_seed`, `validation_fraction`, `split_seed`:
  rozmiar i odtwarzalny wybór danych. Filtr leakage zawsze ma pierwszeństwo.
- `text`: domyślnie `OPI-PIB/PolDense-32M`, frozen, prefiks `[query]: `,
  limit 128 tokenów. Natywny pooling Sentence Transformers.
- `vision`: frozen `google/siglip2-base-patch16-256`, natywne 256 px,
  surowy `SiglipVisionModel.pooler_output` (768 wymiarów).
- `optimization`: 20 epok, batch 256, AdamW, lr 0.001, weight decay 0.01.
- `seeds`: domyślnie 42/43/44; trzy runy od początku, z tym samym podziałem danych.
- `previous_run`: ścieżka do `result.json` albo `summary.json` poprzednika.
  Pełny E1 porównuje się ze średnią pilota 6k. Bez poprzednika referencją jest
  nietrenowany projektor o identycznym seedzie. Baseline'y są osobnymi referencjami.

PolDense zaleca `[query]: ` dla retrieval:
[karta autora](https://huggingface.co/OPI-PIB/PolDense-32M).
Mapowanie identyfikatorów VG/COCO opisuje
[dokumentacja Visual Genome](https://visualgenome.org/api/v0/api_readme).

## Artefakty

`training/runs/e1-full/` (ignorowane przez Git):

- `config.json`, `model_lock.json`, `status.json`;
- `e0.json`: odtworzone baseline'y, protokół i hashe artefaktów;
- `blacklist.json`: pełny snapshot eval i raport cross-dataset;
- `data.jsonl`: wybrane rekordy z równoległym tekstem EN/PL, parent IDs i splitem;
- `data_audit.json`: pochodzenie shardów, odrzucenia i ich powody;
- `translation_snapshots/`: spójne, zamrożone eksporty ukończonych tłumaczeń;
- `cache.json`: mapowanie kluczy do macierzy offline, rewizje modeli i hashe;
- `E1-seed-*/config.json`, `projector.pt`, `result.json`, cztery pliki predictions;
- `E1-seed-*/epochs.jsonl`: bieżący train/validation loss po każdej epoce;
- `comparison.md`, `comparison.csv`, `summary.json`: cztery benchmarki, AVG,
  różnice względem poprzednika i średnia/odchylenie między seedami.

`training/export_report.py` zapisuje mały raport i JSON w `reports/training_e0_e1_full.*`,
do przeglądu i wersjonowania bez dużych cache i checkpointów.

E1 optymalizuje tylko liniowy projektor obrazu, na CPU z cache FP32.
Strata: `mean(1 - cosine(projector(image), text_pl))`. Powtarzające się targety
nie są negatywami. Dla E2 jest testowany helper multi-positive InfoNCE,
ale CLI nie pozwala przypadkowo uruchomić E2 pod etykietą E1.

Główna metryka: Recall@1; dodatkowo R@5/R@10, MRR, P@1, NDCG@10.
AVG istnieje tylko dla kompletu czterech zadań. CIRR/FashionIQ/RefCOCO w E1
stosują dotychczasową surową sumę obrazu i tekstu jako kontrolę diagnostyczną;
composed-query encoder jest osobną, późną fazą P9 aktualnego planu.

Testy bez pobierania wag/datasetów:

```bash
PYTHONPATH=src $PY -m unittest discover -s tests -p 'test_training*.py' -v
PYTHONPATH=src $PY -m unittest discover -s tests -p 'test_benchmark*.py' -v
```

Proces treningowy wypisuje `seed`, `epoch`, `train` i `val` po każdej epoce
z natychmiastowym flush, także pomiar w epoce 0 przed pierwszą aktualizacją wag.
Pełna historia pozostaje w `result.json` dla każdego
seeda. Log na żywo można też zachować przez:

```bash
$PY -u train.py e1 2>&1 | tee training/runs/e1-full/train.log
```

Diagnostyka uwagi PolDense (bez pobierania, na CPU):

```bash
$PY training/check_attention.py
```
