# Nasze modele na nowych galeriach — 29.10.2026

Parametry podano w milionach: **Enkodery V+T = wizyjny + tekstowy**. Pozostałe enkodery oraz osobny projektor nie wchodzą do tej sumy. Nasz model: **70,47 mln = 38,59 mln wizja + 31,88 mln tekst**, projektor osobno **0,197 mln**. EmbeddingGemma2: **438,76 mln = 167,76 mln wizja + 271,00 mln tekst**; enkoder audio był wyłączony.


Checkpoint finalisty: seed 44, wskazany wcześniej przez medianę VN dev. Nie zmieniano go po nowych wynikach.
Wiki jest zero-shot dla tych checkpointów; VN dev jest in-domain z etykietami modelowymi. XTD10 jest kontrolą, nie niezależnym holdoutem.
Każda tabela ma własną galerię. Nie porównywać wyników między tabelami jako rankingu.

## wiki_native_dev

Obrazy: 397; unikalne query tekstowe: 331.

| Model | Enkodery V+T [M] | Wizyjny [M] | Tekstowy [M] | t2i R@1 | i2t R@1 | t2i MRR | i2t MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VN_FULL_seed42 | 70.47 | 38.59 | 31.88 | 16.62% | 13.10% | 0.2566 | 0.2261 |
| VN_FULL_seed43 | 70.47 | 38.59 | 31.88 | 15.11% | 14.86% | 0.2505 | 0.2378 |
| VN_FULL_seed44 | 70.47 | 38.59 | 31.88 | 15.41% | 14.86% | 0.2556 | 0.2366 |
| VN_only | 70.47 | 38.59 | 31.88 | 16.01% | 16.62% | 0.2665 | 0.2652 |
| historical_A1 | 70.47 | 38.59 | 31.88 | 9.37% | 7.30% | 0.1625 | 0.1427 |

VN_FULL, 3 seedy, t2i R@1: 15.71% ± 0.80 p.p. SD.

| Seed 44 minus kontrola | Enkodery V+T [M] | Wizyjny [M] | Tekstowy [M] | t2i Δ R@1 [95% CI], p.p. | i2t Δ R@1 [95% CI], p.p. |
| --- | ---: | ---: | ---: | ---: | ---: |
| VN_only | 70.47 | 38.59 | 31.88 | -0.60 [-3.76, +2.82] | -1.76 [-5.06, +1.27] |
| historical_A1 | 70.47 | 38.59 | 31.88 | +6.04 [+2.98, +9.20] | +7.56 [+4.13, +11.29] |

## xtd10_full_1k

Obrazy: 1000; unikalne query tekstowe: 1000.

| Model | Enkodery V+T [M] | Wizyjny [M] | Tekstowy [M] | t2i R@1 | i2t R@1 | t2i MRR | i2t MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VN_FULL_seed42 | 70.47 | 38.59 | 31.88 | 49.70% | 45.30% | 0.6282 | 0.5975 |
| VN_FULL_seed43 | 70.47 | 38.59 | 31.88 | 49.90% | 45.10% | 0.6355 | 0.5939 |
| VN_FULL_seed44 | 70.47 | 38.59 | 31.88 | 49.30% | 43.60% | 0.6283 | 0.5809 |
| VN_only | 70.47 | 38.59 | 31.88 | 45.00% | 38.90% | 0.5818 | 0.5310 |
| historical_A1 | 70.47 | 38.59 | 31.88 | 42.80% | 41.10% | 0.5748 | 0.5641 |

VN_FULL, 3 seedy, t2i R@1: 49.63% ± 0.31 p.p. SD.

| Seed 44 minus kontrola | Enkodery V+T [M] | Wizyjny [M] | Tekstowy [M] | t2i Δ R@1 [95% CI], p.p. | i2t Δ R@1 [95% CI], p.p. |
| --- | ---: | ---: | ---: | ---: | ---: |
| VN_only | 70.47 | 38.59 | 31.88 | +4.30 [+1.60, +7.00] | +4.70 [+1.90, +7.50] |
| historical_A1 | 70.47 | 38.59 | 31.88 | +6.50 [+3.60, +9.30] | +2.50 [-0.10, +5.10] |

## xtd10_no_overlap

Obrazy: 897; unikalne query tekstowe: 897.

| Model | Enkodery V+T [M] | Wizyjny [M] | Tekstowy [M] | t2i R@1 | i2t R@1 | t2i MRR | i2t MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VN_FULL_seed42 | 70.47 | 38.59 | 31.88 | 51.39% | 47.27% | 0.6434 | 0.6154 |
| VN_FULL_seed43 | 70.47 | 38.59 | 31.88 | 51.39% | 46.27% | 0.6497 | 0.6066 |
| VN_FULL_seed44 | 70.47 | 38.59 | 31.88 | 51.39% | 45.37% | 0.6447 | 0.5978 |
| VN_only | 70.47 | 38.59 | 31.88 | 46.60% | 40.69% | 0.5996 | 0.5490 |
| historical_A1 | 70.47 | 38.59 | 31.88 | 44.15% | 42.59% | 0.5862 | 0.5772 |

VN_FULL, 3 seedy, t2i R@1: 51.39% ± 0.00 p.p. SD.

| Seed 44 minus kontrola | Enkodery V+T [M] | Wizyjny [M] | Tekstowy [M] | t2i Δ R@1 [95% CI], p.p. | i2t Δ R@1 [95% CI], p.p. |
| --- | ---: | ---: | ---: | ---: | ---: |
| VN_only | 70.47 | 38.59 | 31.88 | +4.79 [+1.78, +7.80] | +4.68 [+1.78, +7.58] |
| historical_A1 | 70.47 | 38.59 | 31.88 | +7.25 [+4.12, +10.26] | +2.79 [+0.11, +5.57] |

## VN_dev

Obrazy: 2000; unikalne query tekstowe: 2003.

| Model | Enkodery V+T [M] | Wizyjny [M] | Tekstowy [M] | t2i R@1 | i2t R@1 | t2i MRR | i2t MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VN_FULL_seed42 | 70.47 | 38.59 | 31.88 | 74.09% | 70.40% | 0.8128 | 0.7886 |
| VN_FULL_seed43 | 70.47 | 38.59 | 31.88 | 73.74% | 70.35% | 0.8132 | 0.7885 |
| VN_FULL_seed44 | 70.47 | 38.59 | 31.88 | 73.94% | 69.15% | 0.8150 | 0.7789 |
| VN_only | 70.47 | 38.59 | 31.88 | 78.08% | 74.45% | 0.8450 | 0.8202 |
| historical_A1 | 70.47 | 38.59 | 31.88 | 32.60% | 33.70% | 0.4305 | 0.4386 |

VN_FULL, 3 seedy, t2i R@1: 73.92% ± 0.18 p.p. SD.

| Seed 44 minus kontrola | Enkodery V+T [M] | Wizyjny [M] | Tekstowy [M] | t2i Δ R@1 [95% CI], p.p. | i2t Δ R@1 [95% CI], p.p. |
| --- | ---: | ---: | ---: | ---: | ---: |
| VN_only | 70.47 | 38.59 | 31.88 | -4.14 [-5.64, -2.61] | -5.30 [-6.76, -3.85] |
| historical_A1 | 70.47 | 38.59 | 31.88 | +41.34 [+39.04, +43.66] | +35.45 [+33.18, +37.75] |

Wszystkie R@5/10, NDCG@10 oraz przedziały pozostałych metryk są w JSON. Bootstrap: 10 000 replik, seed 20261023, jednostka: komponent.
Tryb autorów jest identyczny dla naszych modeli: `[query]: ` stanowi część wytrenowanej przestrzeni. Nie powtarzano encoderów dla tego trybu.
Kontrola stylu XTD10 human-vs-Qwen niedostępna: nie znaleziono jednoznacznych istniejących tłumaczeń dokładnie tych samych EN. Nie generowano zastępczych danych.
