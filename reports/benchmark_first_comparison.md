# Pierwsze porównanie modeli — 2026-10-18

Zakończono 12 pełnych przebiegów na RTX 5070 Ti w WSL. Każdy dataset ma 1000
przykładów. Leaderboard: http://localhost:8766 (serwer działa niezależnie od czatu).

Recall@1, procenty (większa wartość jest lepsza):

| Dataset | MetaCLIP S16 PL | MetaCLIP M16 PL | MobileCLIP2-S2 |
|---|---:|---:|---:|
| MSCOCO_t2i | 18,2 | 28,4 | 8,7 |
| CIRR | 9,3 | 9,2 | 9,6 |
| FashionIQ | 8,4 | 10,4 | 6,6 |
| RefCOCO | 54,4 | 57,5 | 57,4 |

To lokalne pule kandydatów MMEB. Dla CIRR, FashionIQ i RefCOCO zapytanie jest
sumą jednostkowych embeddingów tekstu oraz obrazu, po czym jest normalizowane.
Jest to jawny baseline CLIP, nie oficjalny protokół modeli obsługujących wspólne
zapytania multimodalne. Wyniki różnych datasetów nie są uśredniane.
MSCOCO wykorzystuje tłumaczenia v2, pozostałe zbiory v3b.

## Pliki i wznowienie

- `benchmark/results/<run_id>/result.json`: metryki, rewizja modelu, hash tłumaczeń,
  batch, precyzja, czas i szczytowe wykorzystanie VRAM.
- `predictions.jsonl` w tym samym katalogu: wynik każdego przykładu.
- `benchmark/results/index.html`, `leaderboard.csv`: graficzne porównanie i eksport.
- `benchmark/results/validation.json`: przebiegi wyłączone z leaderboardu.
- `benchmark/results/preservation_manifest.json`: SHA256 zachowanych plików wynikowych.
- `benchmark/results/workers/515651.json`: zakończona poprawnie kolejka MobileCLIP2.

Modele są w natywnym `/home/jsowa/.cache/huggingface/hub`, środowisko w
`/home/jsowa/.venvs/pl-mmb-benchmark`. Początkowy batch 64; CIRR/M16 w pierwszym
przebiegu automatycznie zmniejszył go do 32 po CUDA OOM. Pozostałe wyniki mają batch 64.
MetaCLIP używa FP16 autocast, MobileCLIP2 FP32.

Wcześniejsze wyniki MobileCLIP2 zachowano. Pierwsza wersja używała nieprawidłowego
preprocessingu OpenCLIP; FP16 prowadziło też do NaN i błędnej oceny rankingu.
FP32 samo nie wystarczyło. Poprawna konfiguracja używa skali obrazu 0–1,
interpolacji bilinear i reparametryzacji FastViT, zgodnie z instrukcją Apple.
Tylko przebiegi `numerics=mobile-fp32-unit-range-reparam-v3` trafiają do porównania.
Nowy kod odrzuca NaN i nieskończoności. Oddzielny test kontrolny modelu dał
10/10 trafień na 10 angielskich parach przy 10 kandydatach; nie jest to wynik
pełnego benchmarku. Oryginalnych plików błędnych przebiegów nie nadpisano.

Tłumaczenie SugarCrepe, dalsze kolejki tłumaczeń i przeglądarka na porcie 8765
pozostały uruchomione. Większych datasetów ARO jeszcze nie liczono.
Opcja `--datasets all` obejmuje obecnie cztery powyższe zbiory i dwa ARO;
inne rodziny wymagają kolejnych adapterów ewaluacyjnych.

Weryfikacja: 48 testów regresyjnych przeszło, kontrola GPU i serwer leaderboardu
działają. Instrukcja CLI i protokół: [../benchmark/README.md](../benchmark/README.md).
