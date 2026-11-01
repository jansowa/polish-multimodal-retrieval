# Aktualnie dostępne dane

Odczyt tłumaczeń: 2026-10-20T10:11:38.351642+00:00. Wyłącznie aktualne wersje produkcyjne. Liczby to rekordy, nie liczba wszystkich zdań ani unikalnych obrazów. VisualNews jest w trakcie tłumaczenia.

| Zbiór | Źródło EN: rekordy | Gotowe PL: rekordy | Pary w obecnym train+val |
|---|---:|---:|---:|
| ARO Visual Relation | 23,937 | 23,937 | 23226 |
| ARO Visual Attribution | 28,748 | 28,748 | 28049 |
| SugarCrepe | 7,511 | 7,511 | 6747 |
| M-BEIR MSCOCO | 24,809 | 24,809 | 19962 |
| M-BEIR VisualNews | 537,568 | 25,641 | 0 |
| MMEB CIRR | 1,000 | 1,000 | eval — wykluczone z train |
| MMEB FashionIQ | 1,000 | 1,000 | eval — wykluczone z train |
| MMEB MSCOCO_t2i | 1,000 | 1,000 | eval — wykluczone z train |
| MMEB RefCOCO | 1,000 | 1,000 | eval — wykluczone z train |

Razem 114,646 przetłumaczonych rekordów, w tym 4000 rekordów ewaluacyjnych. ARO i SugarCrepe mają dodatnie oraz kontrastowe ujemne podpisy; nie liczymy ich jako oddzielnych rekordów. VisualNews: opisy EN/PL i metadane, brak lokalnych obrazów potrzebnych do gotowej puli treningowej; nie policzono na nowo audytu rozszerzonego zbioru.

Obecny train: 70 289 rekordów, 67 445 unikalnych par, 10 645 różnych target images. Validation: 7695 rekordów, 1216 różnych target images. Razem 77 984 pary i 11 861 obrazów według pikseli; audyt nie wykazuje overlapu z chronionym eval.

## Dodatkowe lokalne dane tekstowe, nieużywane w treningach multimodalnych

mFAQ PL: 63 157 train + 2025 validation = 65 182 pary pytanie–odpowiedź, bez obrazów.

MAUPQA: 3 660 140 rekordów pytanie–passage z etykietą trafności, bez obrazów. 454 234 pozytywne i 3 205 906 negatywnych wierszy; nie są to liczby unikalnych pytań.

| MAUPQA subset | Rekordy |
|---|---:|
| 1z10 | 166,845 |
| czy-wiesz-v2 | 172,384 |
| gpt3-cc | 99,380 |
| gpt3.5-cc | 281,692 |
| gpt3.5-wiki | 145,319 |
| mkqa | 23,782 |
| mqa | 1,428,596 |
| multilingual-NLI | 812,097 |
| poleval2021-pairs | 19,723 |
| poquad | 346,052 |
| templates | 61,221 |
| wiki-def | 103,049 |

## Obrazy i cache

Źródłowe obrazy ARO i SugarCrepe są lokalnie w Parquet; gotowe obrazy treningowe są wyodrębnione. Cache plików MMEB zawiera CIRR 4530, FashionIQ 17 148, MSCOCO_t2i 2004, RefCOCO 16 832 plików. To liczby plików na dysku, nie unikalnych semantycznie obrazów ani rozmiary train.

Wspólny manifest encoderów obejmuje 30 893 ścieżki obrazów i 137 904 teksty EN/PL, z train, val i eval. Nie są to dodatkowe niezależne przykłady treningowe. Offline image embeddings są policzone dla dziesięciu przetestowanych backbone’ów; tekstowy cache wykorzystuje frozen PolDense-32M.

Mamy też oryginalne qrels, zapytania, caption pools i odwołania do obrazów dla użytych wariantów M-BEIR MSCOCO/VisualNews, oraz blacklisty eval, identyfikatory źródłowe, hashe i split train/val. Winoground ma w cache tylko README; nie zaliczamy go do gotowych danych. ImageCoDe i inne opcje rejestru nie są gotowymi przetłumaczonymi zbiorami.

Lokalizacje aktualnych danych: `outputs/` w projekcie; `C:/Users/jsowa/benchmark-translations/` dla SugarCrepe, M-BEIR MSCOCO i VisualNews; `.cache/absence-v3/` dla źródeł EN; `training/runs/e1-full/` dla aktualnego splitu i audytu; cache Hugging Face dla źródłowych Parquet/CSV i embeddingów.
