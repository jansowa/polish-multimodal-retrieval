# Pięć frozen vision backbones — sigmoid / PolDense-32M

Zakończono dokładnie pierwszą piątkę. RN101 i TinyCLIP 61M nie uruchomiono. Wszystkie nowe modele: seed 42; dodatkowo 43/44 zgodnie z zapisaną decyzją po pięciu scoutach.

Stałe: 70 289 train / 7695 validation, 67 445 unikalnych par train, frozen PolDense i vision, liniowy projector, multi-positive sigmoid z uczoną skalą/biasem, 20 epok, batch 256, AdamW lr=.001, wd=.01 (parametry lossu wd=0), bez schedulera. Checkpoint: minimum validation loss. Ten sam manifest, blacklisty i cache tekstów EN/PL. Natywny preprocessing, standardowy image embedding z oryginalną projekcją, offline cache.

Metryki R@k w %. ± to odchylenie standardowe między seedami. Jest to dotychczasowy lokalny benchmark MSCOCO MMEB, nie oficjalny COCO 5k.

| Model | Vision M | Projector M | Cały embedder M | Seedy | t2i R@1 | Δ vs SigLIP2 p.p. | Licencja wag |
|---|---:|---:|---:|---|---:|---:|---|
| SigLIP2 Base/16 256 | 92.930 | 0.295 | 125.109 | [42, 43, 44] | 49.70 ± 0.70 | +0.00 | [Apache-2.0 (reference)](https://huggingface.co/google/siglip2-base-patch16-256) |
| MobileCLIP2-S2 | 35.815 | 0.197 | 67.895 | [42] | 49.10 | -0.60 | [Apple custom (reference)](https://huggingface.co/apple/MobileCLIP2-S2) |
| TinyCLIP 39M/16 YFCC | 38.587 | 0.197 | 70.668 | [42, 43, 44] | 44.67 ± 0.42 | -5.03 | [mit](https://huggingface.co/wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M/blob/07a4b0bc751cb64fecd2b661c048c1dd98d69444/README.md) |
| TinyCLIP 40M/32 LAION | 39.692 | 0.197 | 71.772 | [42] | 36.70 | -13.00 | [mit](https://huggingface.co/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M/blob/95ec8197b3f2fe7f747865c61ca556cf0768b2f7/README.md) |
| Chinese-CLIP RN50 | 38.317 | 0.394 | 70.594 | [42] | 39.10 | -10.60 | [apache-2.0](https://huggingface.co/OFA-Sys/chinese-clip-rn50/blob/717ba215769231e53b9b7c6b9d329b9cc5944418/README.md) |
| RN50 CC12M | 38.317 | 0.394 | 70.594 | [42] | 27.70 | -22.00 | [mit](https://huggingface.co/timm/resnet50_clip.cc12m/blob/d72a2fdb0278498f216ca970010d730b95249beb/README.md) |
| RN50 YFCC | 38.317 | 0.394 | 70.594 | [42] | 31.90 | -17.80 | [mit](https://huggingface.co/timm/resnet50_clip.yfcc15m/blob/c94a9e3bde942b58c1cd13bb0a03d3c1ff4cb056/README.md) |

Najlepszy z nowych kandydatów według średniej testowej: **TinyCLIP 39M/16 YFCC**, **44.67 ± 0.42% R@1** przy **38.587M vision parameters**. Różnica do MobileCLIP2-S2: -4.43 p.p.; do SigLIP2: -5.03 p.p.

- Cel ≥45% przy ≤40M: brak w pierwszej piątce.
- Cel ≥47% przy ≤50M: brak w pierwszej piątce.
- Cel ≥49% przy ≤60M: brak w pierwszej piątce.

Cały embedder = używany image encoder + PolDense (31 883 136 parametrów) + nowy projector. Oryginalne text towers i dwa skalary lossu pominięte w inference. Deklaracje MIT/Apache-2.0 w kartach modeli umożliwiają rozpatrywanie użycia komercyjnego z zachowaniem warunków licencji; nie są audytem praw do danych pretrainingowych ani całego produktu. Karty modeli i kod mają zapisane rewizje i SHA. Chinese-CLIP: karta wag Apache-2.0, repozytorium kodu MIT. [MobileCLIP2](https://huggingface.co/apple/MobileCLIP2-S2) ma apple-amlr; [licencja modeli Apple](https://github.com/apple-aiml-research/ml-mobileclip/blob/main/LICENSE_MODELS) ogranicza użycie do badań niekomercyjnych i wyłącza rozwój produktów. Nie traktujemy go jako kandydata komercyjnego na tej licencji. [PolDense-32M](https://huggingface.co/OPI-PIB/PolDense-32M) deklaruje Gemma, więc cały embedder nie jest MIT/Apache-only; obejmują go również [warunki Gemma](https://ai.google.dev/gemma/terms), w tym obowiązki dystrybucji i ograniczenia użycia.

## Wyniki scoutów i replikacja

| Model | Scout seed 42 R@1 | Klasa vision ≤M | Seedy | Powody |
|---|---:|---:|---|---|
| TinyCLIP 39M/16 YFCC | 44.20 | 40 | [42, 43, 44] | best_declared_permissive_candidate_in_size_class |
| TinyCLIP 40M/32 LAION | 36.70 | 40 | [42] | brak spełnionego progu |
| Chinese-CLIP RN50 | 39.10 | 40 | [42] | brak spełnionego progu |
| RN50 CC12M | 27.70 | 40 | [42] | brak spełnionego progu |
| RN50 YFCC | 31.90 | 40 | [42] | brak spełnionego progu |

Klasy przed testem: ≤40M, (40M,50M], (50M,60M]. „Istotnie mniej” = ≥20% mniej niż MobileCLIP2-S2. Kryterium najlepszego modelu w klasie obejmuje nowe modele z deklaracją MIT/Apache-2.0 i wcześniejsze DINOv2/ConvNeXt/iBOT; nie obejmuje Apple custom. To jawna selekcja dalszych seedów z testu na zlecenie użytkownika. Nie zmieniano LR/lossu/danych ani nie wybierano epok według testu.

## MSCOCO lokalne pule

| Model | t2i R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| SigLIP2 Base/16 256 | 49.70 ± 0.70 | 81.17 ± 0.57 | 89.53 ± 1.10 |
| MobileCLIP2-S2 | 49.10 | 78.20 | 88.80 |
| TinyCLIP 39M/16 YFCC | 44.67 ± 0.42 | 78.00 ± 0.85 | 88.00 ± 0.36 |
| TinyCLIP 40M/32 LAION | 36.70 | 67.80 | 81.20 |
| Chinese-CLIP RN50 | 39.10 | 70.30 | 82.40 |
| RN50 CC12M | 27.70 | 59.80 | 74.80 |
| RN50 YFCC | 31.90 | 63.80 | 75.70 |

## Osobna stała galeria multi-positive

1002 obrazy / 1000 captionów, 1000 zapytań t2i, 201 zapytań i2t.

| Model | t2i R@1 | R@5 | R@10 | i2t R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|---:|
| SigLIP2 Base/16 256 | 49.63 ± 0.74 | 81.13 ± 0.51 | 89.50 ± 1.08 | 81.92 ± 1.44 | 98.18 ± 0.76 | 99.00 ± 0.00 |
| MobileCLIP2-S2 | 49.10 | 78.20 | 88.80 | 80.60 | 97.51 | 99.00 |
| TinyCLIP 39M/16 YFCC | 44.60 ± 0.46 | 77.93 ± 0.87 | 88.00 ± 0.36 | 76.78 ± 1.25 | 97.18 ± 0.76 | 99.17 ± 0.29 |
| TinyCLIP 40M/32 LAION | 36.70 | 67.80 | 81.10 | 71.64 | 95.02 | 99.00 |
| Chinese-CLIP RN50 | 39.10 | 70.30 | 82.40 | 69.65 | 93.03 | 95.52 |
| RN50 CC12M | 27.60 | 59.80 | 74.80 | 64.68 | 91.54 | 94.53 |
| RN50 YFCC | 31.90 | 63.70 | 75.70 | 65.17 | 90.55 | 94.03 |

## Cztery benchmarki

CIRR/FashionIQ/RefCOCO/AVG diagnostycznie. Δ do SigLIP2 E2b i do poprzedniej pozycji w zleconej serii (średnie).

| Model | CIRR | FashionIQ | MSCOCO_t2i | RefCOCO | AVG |
|---|---:|---:|---:|---:|---:|
| SigLIP2 Base/16 256 | 9.40 ± 0.52 | 6.20 ± 0.69 | 49.70 ± 0.70 | 61.47 ± 1.04 | 31.69 ± 0.24 |
| MobileCLIP2-S2 | 21.30 | 7.90 | 49.10 | 36.90 | 28.80 |
| TinyCLIP 39M/16 YFCC | 10.07 ± 0.55 | 3.93 ± 0.32 | 44.67 ± 0.42 | 47.70 ± 0.95 | 26.59 ± 0.09 |
| Δ vs SigLIP2 p.p. | +0.67 | -2.27 | -5.03 | -13.77 | -5.10 |
| Δ vs poprzedni test p.p. | +0.67 | -2.27 | -5.03 | -13.77 | -5.10 |
| TinyCLIP 40M/32 LAION | 10.90 | 5.00 | 36.70 | 42.70 | 23.82 |
| Δ vs SigLIP2 p.p. | +1.50 | -1.20 | -13.00 | -18.77 | -7.87 |
| Δ vs poprzedni test p.p. | +0.83 | +1.07 | -7.97 | -5.00 | -2.77 |
| Chinese-CLIP RN50 | 13.30 | 5.80 | 39.10 | 39.40 | 24.40 |
| Δ vs SigLIP2 p.p. | +3.90 | -0.40 | -10.60 | -22.07 | -7.29 |
| Δ vs poprzedni test p.p. | +2.40 | +0.80 | +2.40 | -3.30 | +0.58 |
| RN50 CC12M | 9.80 | 2.70 | 27.70 | 29.00 | 17.30 |
| Δ vs SigLIP2 p.p. | +0.40 | -3.50 | -22.00 | -32.47 | -14.39 |
| Δ vs poprzedni test p.p. | -3.50 | -3.10 | -11.40 | -10.40 | -7.10 |
| RN50 YFCC | 8.00 | 2.60 | 31.90 | 31.80 | 18.57 |
| Δ vs SigLIP2 p.p. | -1.40 | -3.60 | -17.80 | -29.67 | -13.12 |
| Δ vs poprzedni test p.p. | -1.80 | -0.10 | +4.20 | +2.80 | +1.27 |

## Historia treningu

| Model | Seed | Val loss początek | Val loss minimum | Epoka | Train loss koniec |
|---|---:|---:|---:|---:|---:|
| TinyCLIP 39M/16 YFCC | 42 | 14.00461 | 4.60649 | 6 | 1.90830 |
| TinyCLIP 39M/16 YFCC | 43 | 14.05711 | 4.63118 | 4 | 1.89490 |
| TinyCLIP 39M/16 YFCC | 44 | 13.91049 | 4.63765 | 6 | 1.91796 |
| TinyCLIP 40M/32 LAION | 42 | 15.29390 | 4.87945 | 6 | 1.99096 |
| Chinese-CLIP RN50 | 42 | 13.65389 | 4.73349 | 15 | 2.38660 |
| RN50 CC12M | 42 | 15.84351 | 5.08629 | 7 | 2.56496 |
| RN50 YFCC | 42 | 13.61815 | 5.00610 | 5 | 2.43169 |

TinyCLIP 39M vs 40M zmienia zarówno patch, jak i dane pretrainingowe. Para RN50 ma identyczną architekturę i preprocessing; to porównanie gotowych checkpointów, bez gwarancji identycznych pozostałych ustawień ich pretrainingu. MobileCLIP2 ma jeden seed referencyjny; małe różnice nie dowodzą przewagi.
