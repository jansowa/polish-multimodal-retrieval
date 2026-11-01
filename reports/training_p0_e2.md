# P0 + E2a/E2b

Pełny E1 pozostaje zamrożonym baseline: lokalne MSCOCO MMEB t2i R@1 **41,70% ± 0,20 p.p.** (3 seedy). PolDense i vision są frozen. Zmiana E2 dotyczy wyłącznie lossu i jego temperatury/biasu.

Train: **70289 rekordów**, 67445 unikalnych par PL–obraz, 10645 obrazów. Validation: 7695 rekordów; retrieval wyboru lossu: 2090 unikalnych captionów / 419 obrazów z 2093 rekordów M-BEIR MSCOCO w istniejącym validation splicie.

Budżet każdego treningu: 20 epok, 5500 kroków, 1405780 ekspozycji rekordów, batch=256, AdamW lr=0.001, wd=0.01 dla projectora, scheduler=none. Te same inicjalizacje i permutacje batchy dla odpowiadających sobie seedów.

Checkpoint każdego E2 wybiera minimum jego validation loss, tak jak w E1. Losy i dodatkowe seedy wybiera wyłącznie validation retrieval. Wartości różnych funkcji loss nie są bezpośrednio porównywalne.

Wybrany nowy loss: **E2b**. Replikacje po progu ≥1 p.p. na validation t2i R@1: E2a, E2b. Decyzje zapisano przed ewaluacją testu w `replication_decision.json` i `selection.json`.

## Walidacja retrieval — wszystkie konfiguracje

| Model | Seedy | t2i R@1 | t2i R@5 | t2i R@10 | i2t R@1 | i2t R@5 | i2t R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 frozen | 3 | 55.49 ± 0.58 | 85.77 ± 0.34 | 93.80 ± 0.17 | 45.03 ± 0.96 | 77.33 ± 1.09 | 88.78 ± 0.41 |
| E2a | 3 | 59.89 ± 0.60 | 89.43 ± 0.29 | 95.50 ± 0.25 | 74.78 ± 0.60 | 95.86 ± 0.96 | 99.20 ± 0.14 |
| E2b | 3 | 61.56 ± 1.05 | 90.30 ± 0.37 | 95.65 ± 0.25 | 75.82 ± 1.08 | 96.26 ± 0.73 | 99.12 ± 0.14 |

## Benchmark lokalny MMEB — tylko konfiguracja wybrana na val

| Model | Seedy testu | CIRR | FashionIQ | MSCOCO_t2i | RefCOCO | AVG |
|---|---:|---:|---:|---:|---:|---:|
| E1 frozen | 3 | 8.37 ± 0.35 | 4.53 ± 0.15 | 41.70 ± 0.20 | 35.70 ± 0.46 | 22.57 ± 0.18 |
| E2a | 0 | — | — | — | — | — |
| E2b | 3 | 9.40 ± 0.52 | 6.20 ± 0.69 | 49.70 ± 0.70 | 61.47 ± 1.04 | 31.69 ± 0.24 |
| Δ vs pełny E1 (p.p.) | — | +1.03 | +1.67 | +8.00 | +25.77 | +9.12 |

Nie wybrany loss ma wyniki validation i status `not_selected`; jego test nie został uruchomiony.

| MSCOCO lokalny | t2i R@1 | t2i R@5 | t2i R@10 |
|---|---:|---:|---:|
| E1 frozen | 41.70 ± 0.20 | 70.63 ± 0.64 | 81.63 ± 0.49 |
| E2a | — | — | — |
| E2b | 49.70 ± 0.70 | 81.17 ± 0.57 | 89.53 ± 1.10 |

## Rozszerzony benchmark dwukierunkowy — osobny protokół

`fixed-gallery-multi-positive-v1`: galeria 1002 obrazów i 1000 unikalnych captionów z przykładów MSCOCO MMEB. Liczba zapytań: 1000 tekstowych oraz **201 obrazowych**. i2t pyta wyłącznie o obrazy ze znanym pozytywnym captionem; obrazy bez qrels pozostają negatywnymi kandydatami t2i. To osobny protokół, **nie oficjalne COCO 5k** i nie odwrócenie lokalnych pul. Multi-positive i tie policy są zapisane w grafie.

| Model | t2i R@1 | t2i R@5 | t2i R@10 | i2t R@1 | i2t R@5 | i2t R@10 |
|---|---:|---:|---:|---:|---:|---:|
| E1 frozen | 41.70 ± 0.20 | 70.57 ± 0.59 | 81.53 ± 0.40 | 68.16 ± 1.49 | 88.89 ± 0.76 | 95.19 ± 0.57 |
| E2a | — | — | — | — | — | — |
| E2b | 49.63 ± 0.74 | 81.13 ± 0.51 | 89.50 ± 1.08 | 81.92 ± 1.44 | 98.18 ± 0.76 | 99.00 ± 0.00 |

## Loss i temperatura

| Model | Seed | Val loss przed treningiem | Najlepsza val loss | Epoka | Train loss final | Wybrana temperatura |
|---|---:|---:|---:|---:|---:|---:|
| E2a | 42 | 5.39183 | 1.79538 | 5 | 0.45104 | 0.02258 |
| E2a | 43 | 5.40393 | 1.76046 | 5 | 0.44542 | 0.02258 |
| E2a | 44 | 5.36969 | 1.77897 | 5 | 0.44896 | 0.02258 |
| E2b | 42 | 13.88645 | 4.30983 | 8 | 1.59035 | 0.02039 |
| E2b | 43 | 14.61649 | 4.34324 | 4 | 1.58081 | 0.03195 |
| E2b | 44 | 13.78053 | 4.32475 | 9 | 1.63425 | 0.01864 |

Pełne run_id, rewizje modeli, trainable params, best/selected validation scores, temperatura i bias oraz wszystkie R@1/5/10 są w JSON. Oryginalny E1 nie mierzył retrieval co epokę; P0 mierzy zachowane checkpointy i nie dopisuje fikcyjnej historii.

Dalsze etapy pozostają planem: [P0–P11](../training/PLAN.md). Oryginalny wynik E1: [raport](training_e0_e1_full.md). Reprodukcja: `training/freeze_e1.py`, `training/loss_ablation.py all`, `training/export_ablation.py`.
