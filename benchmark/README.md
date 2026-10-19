# Benchmarkowanie na GPU w WSL

**Bieżący protokół: [v2](protocol_v2.md)** — bez instrukcji, suma surowych
embeddingów CLIP i cosine po połączeniu. Poprzednie wyniki pozostają w archiwum.
EmbeddingGemma-2 używa natywnego wspólnego kodowania tekstu i obrazu, bez audio.

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
z interfejsem CLIP w Transformers. Dostępnych jest dwadzieścia modeli. EmbeddingGemma-2
jest automatycznie kierowane do oddzielnego środowiska `pl-mmb-modern`.
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
- Wariant tekst → obraz używa treści przetłumaczonego zapytania po usunięciu instrukcji
  zadania i znaczników obrazów. Limit tekstu zależy od architektury:
  MetaCLIP/MobileCLIP2 — 77, SigLIP2 — 64, multilingual CLIP — 128 tokenów.
- CIRR/FashionIQ/RefCOCO: dla CLIP zapytanie jest sumą surowych embeddingów tekstu
  i obrazu; normalizacja zachodzi dopiero podczas cosine similarity. Referencja:
  `get_pred(normalization=True)` w kodzie MMEB v1. To nie deklaracja reprodukcji
  wszystkich ustawień modeli z angielskiej tabeli. EmbeddingGemma-2 koduje zapytanie wspólnie.
  Instrukcje kandydatów obrazowych nie są kodowane.
- ARO: podpis prawdziwy i fałszywy porównywane z obrazem; pair accuracy.
- Remisy rozstrzyga pierwsza pozycja, zgodnie z oryginalnym `np.argmax`. Średnia w tabeli jest nieważoną średnią wyników
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
W protokole v2 sumujemy surowe embeddingi obrazu i tekstu, a cosine liczymy
dopiero po połączeniu. Starsze wyniki pozostają w archiwum.

Kontrola 10 par EN/PL, oddzielna od pełnych benchmarków:

```bash
~/.venvs/pl-mmb-benchmark/bin/python benchmark/check_models.py --models clip-b32-pl siglip2-pl siglip2-base
```

Kontrole są zapisywane pod `benchmark/results/checks/<revision>/paired_control.json`.
Wszystkie pobrania modeli, w tym dodatkowy enkoder obrazu, trafiają do natywnego cache HF.

## Cztery kolejne modele i przeliczenie v2

Najpierw zainstaluj środowiska CUDA 13: `bash benchmark/setup_modern_wsl.sh`.

```bash
python3 benchmark/manage.py start --wait \
  --datasets mscoco_t2i cirr fashioniq refcoco \
  --models metaclip-s16 metaclip-m16 mobileclip2-s2 clip-b32-pl siglip2-pl siglip2-base \
    siglip2-pl-32768 embeddinggemma-2 siglip2-naflex clip-b32-pl-32768 \
  --batch-size 8
```

CLI automatycznie dobiera środowisko HF dla modelu. Gemma ma ograniczenie batcha
do 16, pozostałe modele zmniejszają batch po OOM. Ukończone przebiegi tego samego
protokołu, rewizji i tłumaczenia są pomijane przed inicjalizacją GPU.
Parametry i licencje wszystkich piętnastu modeli znajdują się w leaderboardzie.
Szczegóły selektywnego ładowania Gemma (438,76M, bez audio), NaFlex i protokołu
znajdują się w [protocol_v2.md](protocol_v2.md).

## M-CLIP oraz dodatkowe warianty SigLIP

```bash
python3 benchmark/manage.py start --wait \
  --datasets mscoco_t2i cirr fashioniq refcoco \
  --models siglip2-base-384 siglip2-base-256 siglip-i18n-256 siglip2-b32-256 mclip-b16plus \
  --batch-size 8
```

| Alias | Model |
|---|---|
| `mclip-b16plus` | `M-CLIP/XLM-Roberta-Large-Vit-B-16Plus` + `timm/vit_base_patch16_plus_clip_240.laion400m_e31` |
| `siglip2-base-384` | `google/siglip2-base-patch16-384` |
| `siglip2-base-256` | `google/siglip2-base-patch16-256` |
| `siglip-i18n-256` | `timm/ViT-B-16-SigLIP-i18n-256` |
| `siglip2-b32-256` | `timm/ViT-B-32-SigLIP2-256` |

M-CLIP używa maskowanego mean poolingu XLM-R i projekcji liniowej do 640 wymiarów,
zgodnie z [kodem autora](https://github.com/FreddeFrallan/Multilingual-CLIP/blob/main/multilingual_clip/pt_multilingual_clip.py).
Ładowany jest dokładnie wskazany wariant obrazu `e31` (karta tekstowego M-CLIP
pokazuje przykład z `e32`). Liczba parametrów obejmuje wyłącznie używany model
tekstowy i wizualny, bez angielskiej wieży tekstowej OpenCLIP.
Wagi tekstowe M-CLIP nie mają deklaracji licencji w karcie HF; tabela pokazuje
brak deklaracji dla tekstu oraz MIT dla wskazanego enkodera wizji.

Wagi M-CLIP i nowych modeli OpenCLIP są przechowywane na GPU w BF16, a Google
SigLIP używa BF16 autocast. Wszystkie warianty stosują protokół v2: brak
instrukcji MMEB, surowe embeddingi, suma tekst+obraz i końcowy cosine.
OpenCLIP zachowuje oficjalne `canonicalize`, padding i kontekst 64 tokenów oraz
preprocessing checkpointu. Modele 256 i 384 korzystają ze swojej natywnej rozdzielczości.

Checkpointy i tokenizery są pobierane z przypiętych rewizji. OpenCLIP ładuje je
przez `local-dir:` z natywnego cache HF, bez pobierania innej wersji `main`.

Dla M-CLIP CLI automatycznie wybiera środowisko `pl-mmb-benchmark` z PyTorch
2.11.0+cu128. Pozostałe nowe modele korzystają z `pl-mmb-legacy-cu130`.
`PL_MMB_MCLIP_PYTHON` pozwala zmienić interpreter M-CLIP. Jego wagi tekstowe
trafiają do bloków GPU transferami do 8 MB, bez zmiany wartości wag.

## TIPSv2, BGE-VL, Perception Encoder i FashionSigLIP

```bash
python3 benchmark/manage.py start --wait \
  --datasets mscoco_t2i cirr fashioniq refcoco \
  --models tipsv2-b14 bge-vl-base tipsv2-l14 pe-core-b16 fashion-siglip \
  --batch-size 8
```

Nowe aliasy odpowiadają `google/tipsv2-b14`, `BAAI/BGE-VL-base`,
`google/tipsv2-l14`, `facebook/PE-Core-B16-224`, `Marqo/marqo-fashionSigLIP`.
CLI dobiera dla nich `pl-mmb-benchmark` (PyTorch 2.11.0+cu128); interpreter można
zmienić przez `PL_MMB_CU128_PYTHON`. Dodatkowe niewielkie zależności:
`sentencepiece==0.2.1` i `einops==0.8.1`.

- TIPS: przypięty kod z repozytorium HF, rozdzielczość 448, bilinear resize,
  piksele w [0,1], pierwszy CLS jako globalny embedding, natywny SentencePiece,
  lowercase i 64 tokeny. FP32 i batch najwyżej 2; Torch attention bez xFormers.
- BGE-VL: przypięty oficjalny kod HF, CLIPProcessor 224, maksymalnie 77 tokenów.
  Używane są surowe cechy. Oficjalne `encode_multimodal` również sumuje surowe
  cechy i normalizuje dopiero wynik, co odpowiada obecnemu protokołowi.
- PE: dokładnie checkpoint Facebooka i oficjalny kod
  `facebookresearch/perception_models` przypięty do
  `3e352cca660658d4b5c90f42a7808b11469e4c66`. Oba enkodery są ładowane ze ścisłą
  kontrolą kluczy; natywny preprocessing 224 z bilinear squash, zakres [-1,1],
  kontekst 32. Nie jest używany zamiennik wag z repozytorium timm.
- Marqo: OpenCLIP z lokalnego snapshotu przypiętej rewizji HF, 224 piksele,
  canonicalization i 64 tokeny, surowe cechy z `normalize=False`.

Wszystkie modele zachowują protokół v2 (bez instrukcji MMEB, suma surowych cech,
końcowy cosine). Wagi są pobierane wyłącznie do natywnego cache HF w WSL.
Po zakończeniu i walidacji tej serii wagi zostają usunięte dla oszczędności miejsca;
wyniki, konfiguracje, tokenizery i embeddingi pozostają dostępne.

Transfery parametrów używają wyrównanych bloków GPU do 256 MiB (większy pojedynczy parametr zachowuje wymagane ciągłe miejsce) i fragmentów host→GPU do 8 MiB. Po błędzie alokacji CUDA worker jest ponawiany najwyżej trzy razy w osobnym procesie; inne błędy kończą przebieg od razu. Gotowe wyniki są zachowywane.
