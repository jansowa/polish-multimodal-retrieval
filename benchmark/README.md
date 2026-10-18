# Benchmarkowanie na GPU w WSL

Kod znajduje się w `src/pl_mmb_benchmark/`, niezależnie od `src/pl_mmb_translate/`.
Tłumaczenie oraz przeglądarka mogą pracować jednocześnie z benchmarkiem.

Z katalogu projektu w WSL:

```bash
bash benchmark/setup_wsl.sh
~/.venvs/pl-mmb-benchmark/bin/python benchmark.py --list
~/.venvs/pl-mmb-benchmark/bin/python benchmark.py --datasets mscoco_t2i --models metaclip-s16 metaclip-m16 mobileclip2-s2 --batch-size 64
~/.venvs/pl-mmb-benchmark/bin/python benchmark.py --datasets mscoco_t2i cirr fashioniq refcoco --batch-size 64
~/.venvs/pl-mmb-benchmark/bin/python benchmark.py --datasets all --batch-size 64
```

`--model` jest aliasem `--models`. Można podać alias lub repozytorium HF kompatybilne
z interfejsem CLIP w Transformers. Domyślnie uruchamianych jest wszystkich sześć modeli.
`all` oznacza wszystkie **zaimplementowane, ukończone** zbiory: cztery warianty MMEB
po 1000 rekordów oraz dwa ARO. Kolejność rośnie według liczby przykładów; wszystkie
modele kończą mały dataset przed przejściem do kolejnego. Nie uruchamiamy automatycznie
dużych ARO w pierwszej serii. SugarCrepe nadal się tłumaczy. Winoground, ImageCoDe
i M-BEIR wymagają osobnych adapterów ewaluacyjnych; nie wchodzą obecnie do `all`.

Środowisko i modele znajdują się na natywnym systemie plików Linux:
`~/.venvs/pl-mmb-benchmark` oraz domyślnie `~/.cache/huggingface`.
Benchmark nie czyta `.env` tłumaczenia. Odrzuca zmienne cache wskazujące na `/mnt/`
(również przez symlinki). Istniejące obrazy i źródła MMEB są czytane z cache projektu,
bez ponownego kopiowania. Nowe modele zawsze pobierają się do natywnego cache HF.

RTX 5070 Ti: początkowy batch 64; MetaCLIP używa CUDA z autocast FP16, a MobileCLIP2
FP32, obrazy 0–1 i reparametryzację FastViT. Przy CUDA OOM batch jest
zmniejszany dwukrotnie i ponawiany. Rzeczywisty końcowy batch, GPU, szczyt VRAM,
czas i rewizja modelu trafiają do wyniku. Brak cichego przejścia na CPU.

## Protokół i interpretacja

- Polskie tłumaczenia są łączone ze źródłem przez `metadata.row_index`, nigdy przez
  kolejność JSONL. Wymagane jest 100% rekordów bez duplikatów i identyczny tekst źródła.
- MMEB: lokalna pula kandydatów z każdego wiersza, poprawna odpowiedź na pozycji 0.
  Metryka bazowa: Precision@1 (równe Recall@1 i Hit@1 przy jednej oznaczonej poprawnej
  odpowiedzi). Dodatkowo: NDCG@10, Recall@5/10 i MRR. To nie globalne wyszukiwanie w pełnym COCO/M-BEIR.
- Wariant tekst → obraz używa całego przetłumaczonego zapytania, z zachowaniem instrukcji.
  Usuwane są wyłącznie znaczniki obrazów. Limit tekstu zależy od architektury:
  MetaCLIP/MobileCLIP2 — 77, SigLIP2 — 64, multilingual CLIP — 128 tokenów.
- CIRR/FashionIQ/RefCOCO: zapytanie jest znormalizowaną sumą jednostkowego embeddingu
  tekstu i obrazu. To jawny baseline CLIP, nie oficjalny wynik modelu multimodalnego MMEB.
  Instrukcje kandydatów obrazowych nie są kodowane.
- ARO: podpis prawdziwy i fałszywy porównywane z obrazem; pair accuracy.
- Remisy liczone są jako błędy. Średnia w tabeli jest nieważoną średnią wyników
  datasetów dla wybranej metryki; pojawia się dopiero po ukończeniu wszystkich
  datasetów w tabeli. Nie jest to oficjalny zbiorczy wynik MMEB ani MIEB.
- `--limit N` służy wyłącznie pilotom. Wynik ma `full_dataset=false` i osobne ID.
  Nie należy porównywać pilota z pełnym przebiegiem.

Repozytoria modeli i ich instrukcje:
[MetaCLIP S16](https://huggingface.co/alphaedge-ai/metaclip-2-worldwide-s16-384-pol-16384),
[MetaCLIP M16](https://huggingface.co/alphaedge-ai/metaclip-2-worldwide-m16-384-pol-16384),
[MobileCLIP2-S2](https://huggingface.co/apple/MobileCLIP2-S2).
MetaCLIP używa Transformers; MobileCLIP2 używa architektury `MobileCLIP2-S2` w OpenCLIP
i oficjalnego checkpointu Apple. Zgodnie z [instrukcją Apple](https://github.com/apple/ml-mobileclip#openclip-support),
S2 wymaga `image_mean=(0,0,0)`, `image_std=(1,1,1)`; interpolacja bilinear
odpowiada konfiguracji `dfndr2b` w OpenCLIP. Reparametryzację realizuje timm.
Kontrola ładowania na 10 angielskich parach (oddzielna od benchmarku):
`~/.venvs/pl-mmb-benchmark/bin/python benchmark/check_mobileclip.py`.
Dane ARO zawierają pojedynczy ręczny fallback
tłumaczenia opisany w `reports/aro_manual_fallback.json`.

## Wyniki i leaderboard

`benchmark/results/<run_id>/result.json` zawiera metryki i pochodzenie danych,
`predictions.jsonl` — ID, pozycję odpowiedzi, liczbę kandydatów i wyniki podobieństwa.
ID obejmuje protokół, dataset, SHA256 tłumaczeń, model, rewizję i zakres. Ponowne
uruchomienie pomija ukończone identyczne wyniki. Niedokończony dataset jest liczony od nowa.
Poprawiony MobileCLIP2 ma dodatkowe `numerics=mobile-fp32-unit-range-reparam-v3` w identyfikatorze;
wcześniejsze pliki pozostają nienaruszone. `validation.json` wskazuje przebiegi
wyłączone z leaderboardu z powodu NaN lub błędnego preprocessingu MobileCLIP2. Nowy evaluator odrzuca nieprawidłowe
embeddingi i podobieństwa, a zapis JSON nie dopuszcza NaN ani nieskończoności.

`benchmark/results/index.html` zawiera jedną tabelę: pełny identyfikator modelu HF,
liczba parametrów [M], licencja, średnia, a następnie kolumny datasetów. Każdy model
ma jeden wiersz. Parametry obejmują model obrazu i tekstu, przed reparametryzacją;
dla pierwszych trzech modeli zostały policzone z wag na CPU. Licencje mają linki
do dokumentów wydawców. Wszystkie kolumny można sortować; domyślnie średnia malejąco.
Tabela nie wyświetla czasu obliczeń ani batch size.

Przełącznik metryki aktualizuje komórki, średnią i sortowanie. NDCG@10 dla historycznych
przebiegów jest liczone z zapisanych rang na CPU i zapisywane w `derived_metrics.json`;
oryginalne `result.json` i `predictions.jsonl` nie są modyfikowane.
`leaderboard.csv` odzwierciedla tabelę dla metryki bazowej, a `metrics.csv` zawiera
wszystkie metryki najnowszych pełnych wyników. Oba eksporty pomijają czas i batch size.
Uzasadnienie metryk i źródła: [metric_review.md](metric_review.md).
Odbudowa i serwer (port odrębny od reviewera):

```bash
~/.venvs/pl-mmb-benchmark/bin/python benchmark.py --leaderboard
python3 -m http.server 8766 --bind 0.0.0.0 --directory benchmark/results
```

Otwórz `http://localhost:8766`. HTML odświeża się co minutę; pliki powstają po każdym
ukończonym modelu/datasetcie. Wyniki są lokalne, ignorowane przez Git.

Odłączona kolejka (przeżywa zamknięcie terminala/czatu):

```bash
python3 benchmark/manage.py start --datasets mscoco_t2i cirr fashioniq refcoco --batch-size 64
python3 benchmark/manage.py status
tail -f benchmark/results/worker.log
```

`start --wait` ustawia kolejne zadanie w kolejce za już działającym workerem,
bez przerywania jego obliczeń. Piloty nie wchodzą do tabeli ani średniej.

Blokada zapobiega uruchomieniu dwóch kolejek przez `manage.py`. Procesy uruchomione
bezpośrednio przez `benchmark.py` nie korzystają z tej blokady. Po restarcie WSL
uruchom tę samą komendę kolejki. Ukończone wyniki zostaną pominięte.
Dokładne wersje z przygotowanego środowiska zapisano w `environment-wsl.lock.txt`.

Sprawdzenie regresji:
```bash
PYTHONPATH=src ~/.venvs/pl-mmb-benchmark/bin/python -m unittest discover -s tests -p test_benchmark.py -v
```

## Dodatkowe modele CLIP i SigLIP2

```bash
python3 benchmark/manage.py start --datasets mscoco_t2i cirr fashioniq refcoco \
  --models clip-b32-pl siglip2-pl siglip2-base --batch-size 32
```

Aliasy odpowiadają pełnym identyfikatorom:

| Alias | Model | Parametry [M] | Licencja |
|---|---|---:|---|
| `clip-b32-pl` | `alphaedge-ai/clip-ViT-B-32-multilingual-v1-pl-16384` | 143,75 | MIT |
| `siglip2-pl` | `alphaedge-ai/siglip2-base-patch16-512-pol-16384` | 191,80 | Apache-2.0 |
| `siglip2-base` | `google/siglip2-base-patch16-512` | 375,82 | Apache-2.0 |

Multilingual CLIP zawiera osobny enkoder tekstowy DistilBERT z mean pooling
i projekcją Dense w Sentence Transformers. Obrazy koduje przypięty
`openai/clip-vit-base-patch32`, zgodnie z zalecanym połączeniem w karcie modelu.
Liczba parametrów obejmuje oba używane enkodery (143747328), bez nieużywanego
angielskiego enkodera tekstowego OpenAI. Rewizja enkodera obrazu jest częścią ID wyniku.

Oba SigLIP2 używają oryginalnego preprocessingu 512×512, szybkiego tokenizera
z przypiętego repozytorium, paddingu do 64 tokenów i CUDA BF16 autocast.
Batch 32 daje zapas pamięci dla większej rozdzielczości; przy OOM działa redukcja.
Geometria porównania pozostaje taka sama: znormalizowane embeddingi i suma
embeddingów obrazu oraz tekstu w zapytaniach mieszanych. Nie zmieniamy starszych wyników.

Kontrola 10 par EN/PL, oddzielna od pełnych benchmarków:

```bash
~/.venvs/pl-mmb-benchmark/bin/python benchmark/check_models.py --models clip-b32-pl siglip2-pl siglip2-base
```

Kontrole są zapisywane pod `benchmark/results/checks/<revision>/paired_control.json`.
Wszystkie pobrania modeli, w tym dodatkowy enkoder obrazu, trafiają do natywnego cache HF.
