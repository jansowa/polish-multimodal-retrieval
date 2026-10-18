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
z interfejsem CLIP w Transformers. Domyślnie uruchamiane są wszystkie trzy modele.
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

RTX 5070 Ti: początkowy batch 64, CUDA z autocast FP16. Przy CUDA OOM batch jest
zmniejszany dwukrotnie i ponawiany. Rzeczywisty końcowy batch, GPU, szczyt VRAM,
czas i rewizja modelu trafiają do wyniku. Brak cichego przejścia na CPU.

## Protokół i interpretacja

- Polskie tłumaczenia są łączone ze źródłem przez `metadata.row_index`, nigdy przez
  kolejność JSONL. Wymagane jest 100% rekordów bez duplikatów i identyczny tekst źródła.
- MMEB: lokalna pula kandydatów z każdego wiersza, poprawna odpowiedź na pozycji 0.
  Metryki: Recall@1/5/10 i MRR. To nie globalne wyszukiwanie w pełnym COCO/M-BEIR.
- Wariant tekst → obraz używa całego przetłumaczonego zapytania, z zachowaniem instrukcji.
  Usuwane są wyłącznie znaczniki obrazów. Tekst jest obcinany do kontekstu 77 tokenów.
- CIRR/FashionIQ/RefCOCO: zapytanie jest znormalizowaną sumą jednostkowego embeddingu
  tekstu i obrazu. To jawny baseline CLIP, nie oficjalny wynik modelu multimodalnego MMEB.
  Instrukcje kandydatów obrazowych nie są kodowane.
- ARO: podpis prawdziwy i fałszywy porównywane z obrazem; pair accuracy.
- Remisy liczone są jako błędy. Nie ma średniej między nieporównywalnymi zadaniami.
- `--limit N` służy wyłącznie pilotom. Wynik ma `full_dataset=false` i osobne ID.
  Nie należy porównywać pilota z pełnym przebiegiem.

Repozytoria modeli i ich instrukcje:
[MetaCLIP S16](https://huggingface.co/alphaedge-ai/metaclip-2-worldwide-s16-384-pol-16384),
[MetaCLIP M16](https://huggingface.co/alphaedge-ai/metaclip-2-worldwide-m16-384-pol-16384),
[MobileCLIP2-S2](https://huggingface.co/apple/MobileCLIP2-S2).
MetaCLIP używa Transformers; MobileCLIP2 używa architektury `MobileCLIP2-S2` w OpenCLIP
i oficjalnego checkpointu Apple. Dane ARO zawierają pojedynczy ręczny fallback
tłumaczenia opisany w `reports/aro_manual_fallback.json`.

## Wyniki i leaderboard

`benchmark/results/<run_id>/result.json` zawiera metryki i pochodzenie danych,
`predictions.jsonl` — ID, pozycję odpowiedzi, liczbę kandydatów i wyniki podobieństwa.
ID obejmuje protokół, dataset, SHA256 tłumaczeń, model, rewizję i zakres. Ponowne
uruchomienie pomija ukończone identyczne wyniki. Niedokończony dataset jest liczony od nowa.

`benchmark/results/index.html` to samodzielny graficzny leaderboard z filtrem;
`leaderboard.csv` zawiera wszystkie metryki. Odbudowa i serwer (port odrębny od reviewera):

```bash
~/.venvs/pl-mmb-benchmark/bin/python benchmark.py --leaderboard
python3 -m http.server 8766 --bind 0.0.0.0 --directory benchmark/results
```

Otwórz `http://localhost:8766`. HTML odświeża się co minutę; pliki powstają po każdym
ukończonym modelu/datasetcie. Wyniki są lokalne, ignorowane przez Git.

Sprawdzenie regresji:
```bash
PYTHONPATH=src ~/.venvs/pl-mmb-benchmark/bin/python -m unittest discover -s tests -p test_benchmark.py -v
```
