# Nasz finalista i modele <100 mln — 30.10.2026

Model własny to **VN_FULL seed44**, finalista wybrany wcześniej medianą VN dev; nie dokonywano nowej selekcji po wynikach. Parametry obejmują tylko enkodery wizji i tekstu; osobny projektor ma 0,197 mln. MobileCLIP2-S2 jest kontrolą z tekstem EN. W komórkach z parą wyników kolejność to **tekst→obraz / obraz→tekst**. „—” oznacza brak odczytu dla tego checkpointu/protokołu. Nie liczymy średniej między różnymi galeriami.

| Benchmark / parametr | VN_FULL seed44 | MetaCLIP-S16 PL | MetaCLIP-M16 PL | MobileCLIP2-S2 (EN) |
|---|---:|---:|---:|---:|
| Parametry enkoderów V+T [mln] | 70.47 | 49.72 | 85.30 | 99.13 |
| Enkoder wizyjny [mln] | 38.59 | 21.96 | 38.78 | 35.70 |
| Enkoder tekstowy [mln] | 31.88 | 27.76 | 46.52 | 63.43 |
| Wiki dev R@1 t→i / i→t [%] | 15.41 / 14.86 | 29.31 / 30.98 | 33.84 / 32.75 | 18.73 / 24.69 |
| VN dev R@1 t→i / i→t [%] | 73.94 / 69.15 | 56.07 / 60.00 | 63.65 / 67.75 | 27.31 / 27.20 |
| XTD10 pełne 1000 R@1 t→i / i→t [%] | 49.30 / 43.60 | 38.00 / 39.50 | 42.90 / 43.50 | 28.90 / 30.80 |
| XTD10 bez znanego overlapu 897 R@1 t→i / i→t [%] | 51.39 / 45.37 | 39.69 / 41.47 | 44.15 / 44.70 | 30.66 / 31.77 |
| Lokalny test MSCOCO P@1 [%] | 50.70 | 30.00 | 34.60 | 27.20 |
| COCO val R@1 t→i / i→t [%] | 63.30 / 71.36 | — | — | — |
| COCO test — fixed gallery R@1 t→i / i→t [%] | 50.70 / 79.60 | — | — | — |
| Wymiar embeddingu | 384 | 384 | 512 | 512 |
| CPU query [ms] | 14.498 | 18.898 | 29.589 | 33.960 |
| CPU obraz [ms] | 77.105 | 137.686 | 212.994 | 87.014 |
| GPU query [ms/element] | 0.277 | 0.211 | 0.243 | 0.233 |
| GPU obraz [ms/element] | 2.103 | 3.632 | 3.800 | 1.529 |

Zakres: wyłącznie retrieval tekst↔obraz. Testy z zapytaniem obraz+tekst i grounding są poza tym porównaniem. COCO val i test fixed gallery nie mają jeszcze odpowiedników konkurencji w tych samych galeriach.

Wiki: 397 obrazów / 331 tekstów; VN: 2000 / 2003; XTD10: 1000 / 1000 i 897 / 897. VN jest in-domain z podpisami modelowymi; Wiki to odczyt przed treningiem Wiki. XTD10 pozostaje kontrolą z ujawnionym overlapem.
Lokalny MSCOCO ma 1000 zapytań i pule po 1000 kandydatów; nie jest oficjalnym COCO5k. COCO val: 419 obrazów / 2090 tekstów. COCO test fixed gallery: 1002 obrazy / 1000 tekstów, i2t obejmuje 201 uprawnionych zapytań obrazowych. Są to osobne protokoły.
Czasy: mediana; CPU FP32, batch 1, 2 wątki; GPU BF16 autocast, batch 64, RTX 5070 Ti. 10 rozgrzewek i 100 pomiarów; GPU podano na element batcha.

## Wszystkie metryki

R@1/5/10 i NDCG@10 poniżej w procentach; MRR w skali 0–1. W lokalnym MMEB P@1=R@1 (jedna pozytywna odpowiedź). CI to zapisane wcześniej sparowane 95% CI różnicy R@1: finalista minus referencja; nie liczono nowych przedziałów.

### Wiki dev / t2i

Protokół: v2 full gallery.

| Model | V+T [M] | Wizyjny [M] | Tekstowy [M] | R@1 | R@5 | R@10 | MRR | NDCG@10 | Δ R@1 [95% CI], p.p. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VN_FULL seed44 | 70.47 | 38.59 | 31.88 | 15.41 | 34.44 | 43.81 | 0.2556 | 27.94 | — |
| MetaCLIP-S16 PL | 49.72 | 21.96 | 27.76 | 29.31 | 50.15 | 60.12 | 0.3934 | 41.97 | -13.90 [-19.87, -8.58] |
| MetaCLIP-M16 PL | 85.30 | 38.78 | 46.52 | 33.84 | 56.50 | 66.77 | 0.4440 | 47.85 | -18.43 [-24.33, -12.99] |
| MobileCLIP2-S2 (EN) | 99.13 | 35.70 | 63.43 | 18.73 | 31.72 | 41.39 | 0.2566 | 27.61 | -3.32 [-8.16, +1.38] |

### Wiki dev / i2t

Protokół: v2 full gallery.

| Model | V+T [M] | Wizyjny [M] | Tekstowy [M] | R@1 | R@5 | R@10 | MRR | NDCG@10 | Δ R@1 [95% CI], p.p. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VN_FULL seed44 | 70.47 | 38.59 | 31.88 | 14.86 | 30.48 | 42.82 | 0.2366 | 26.80 | — |
| MetaCLIP-S16 PL | 49.72 | 21.96 | 27.76 | 30.98 | 52.90 | 64.48 | 0.4168 | 45.96 | -16.12 [-21.04, -11.38] |
| MetaCLIP-M16 PL | 85.30 | 38.78 | 46.52 | 32.75 | 59.70 | 71.03 | 0.4513 | 50.49 | -17.88 [-23.85, -11.66] |
| MobileCLIP2-S2 (EN) | 99.13 | 35.70 | 63.43 | 24.69 | 41.56 | 53.90 | 0.3385 | 37.45 | -9.82 [-15.38, -4.67] |

### VN dev / t2i

Protokół: v2 full gallery.

| Model | V+T [M] | Wizyjny [M] | Tekstowy [M] | R@1 | R@5 | R@10 | MRR | NDCG@10 | Δ R@1 [95% CI], p.p. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VN_FULL seed44 | 70.47 | 38.59 | 31.88 | 73.94 | 90.91 | 94.06 | 0.8150 | 84.41 | — |
| MetaCLIP-S16 PL | 49.72 | 21.96 | 27.76 | 56.07 | 77.93 | 83.72 | 0.6591 | 69.78 | +17.87 [+15.38, +20.33] |
| MetaCLIP-M16 PL | 85.30 | 38.78 | 46.52 | 63.65 | 82.08 | 88.07 | 0.7210 | 75.65 | +10.28 [+7.90, +12.73] |
| MobileCLIP2-S2 (EN) | 99.13 | 35.70 | 63.43 | 27.31 | 44.78 | 52.02 | 0.3597 | 38.94 | +46.63 [+44.00, +49.30] |

### VN dev / i2t

Protokół: v2 full gallery.

| Model | V+T [M] | Wizyjny [M] | Tekstowy [M] | R@1 | R@5 | R@10 | MRR | NDCG@10 | Δ R@1 [95% CI], p.p. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VN_FULL seed44 | 70.47 | 38.59 | 31.88 | 69.15 | 89.05 | 92.80 | 0.7789 | 81.33 | — |
| MetaCLIP-S16 PL | 49.72 | 21.96 | 27.76 | 60.00 | 80.65 | 86.50 | 0.6944 | 73.17 | +9.15 [+6.65, +11.66] |
| MetaCLIP-M16 PL | 85.30 | 38.78 | 46.52 | 67.75 | 85.40 | 91.00 | 0.7589 | 79.28 | +1.40 [-1.05, +3.90] |
| MobileCLIP2-S2 (EN) | 99.13 | 35.70 | 63.43 | 27.20 | 44.45 | 52.75 | 0.3578 | 38.98 | +41.95 [+39.11, +44.70] |

### XTD10 pełne 1000 / t2i

Protokół: v2 full gallery.

| Model | V+T [M] | Wizyjny [M] | Tekstowy [M] | R@1 | R@5 | R@10 | MRR | NDCG@10 | Δ R@1 [95% CI], p.p. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VN_FULL seed44 | 70.47 | 38.59 | 31.88 | 49.30 | 80.30 | 89.30 | 0.6283 | 68.75 | — |
| MetaCLIP-S16 PL | 49.72 | 21.96 | 27.76 | 38.00 | 64.90 | 75.60 | 0.5011 | 55.44 | +11.30 [+7.80, +14.90] |
| MetaCLIP-M16 PL | 85.30 | 38.78 | 46.52 | 42.90 | 69.80 | 79.40 | 0.5513 | 60.26 | +6.40 [+2.90, +9.90] |
| MobileCLIP2-S2 (EN) | 99.13 | 35.70 | 63.43 | 28.90 | 51.50 | 61.40 | 0.3951 | 43.72 | +20.40 [+16.80, +23.90] |

### XTD10 pełne 1000 / i2t

Protokół: v2 full gallery.

| Model | V+T [M] | Wizyjny [M] | Tekstowy [M] | R@1 | R@5 | R@10 | MRR | NDCG@10 | Δ R@1 [95% CI], p.p. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VN_FULL seed44 | 70.47 | 38.59 | 31.88 | 43.60 | 75.90 | 88.40 | 0.5809 | 64.85 | — |
| MetaCLIP-S16 PL | 49.72 | 21.96 | 27.76 | 39.50 | 68.30 | 80.10 | 0.5292 | 58.66 | +4.10 [+0.50, +7.70] |
| MetaCLIP-M16 PL | 85.30 | 38.78 | 46.52 | 43.50 | 71.10 | 82.50 | 0.5626 | 61.88 | +0.10 [-3.40, +3.60] |
| MobileCLIP2-S2 (EN) | 99.13 | 35.70 | 63.43 | 30.80 | 55.30 | 67.40 | 0.4288 | 47.78 | +12.80 [+9.20, +16.40] |

### XTD10 bez znanego overlapu 897 / t2i

Protokół: v2 full gallery.

| Model | V+T [M] | Wizyjny [M] | Tekstowy [M] | R@1 | R@5 | R@10 | MRR | NDCG@10 | Δ R@1 [95% CI], p.p. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VN_FULL seed44 | 70.47 | 38.59 | 31.88 | 51.39 | 82.16 | 90.52 | 0.6447 | 70.34 | — |
| MetaCLIP-S16 PL | 49.72 | 21.96 | 27.76 | 39.69 | 67.22 | 77.03 | 0.5177 | 57.09 | +11.71 [+7.92, +15.50] |
| MetaCLIP-M16 PL | 85.30 | 38.78 | 46.52 | 44.15 | 71.57 | 80.38 | 0.5630 | 61.40 | +7.25 [+3.34, +11.04] |
| MobileCLIP2-S2 (EN) | 99.13 | 35.70 | 63.43 | 30.66 | 52.40 | 64.21 | 0.4129 | 45.80 | +20.74 [+16.83, +24.53] |

### XTD10 bez znanego overlapu 897 / i2t

Protokół: v2 full gallery.

| Model | V+T [M] | Wizyjny [M] | Tekstowy [M] | R@1 | R@5 | R@10 | MRR | NDCG@10 | Δ R@1 [95% CI], p.p. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VN_FULL seed44 | 70.47 | 38.59 | 31.88 | 45.37 | 77.48 | 89.86 | 0.5978 | 66.55 | — |
| MetaCLIP-S16 PL | 49.72 | 21.96 | 27.76 | 41.47 | 69.68 | 81.72 | 0.5443 | 60.22 | +3.90 [+0.00, +7.69] |
| MetaCLIP-M16 PL | 85.30 | 38.78 | 46.52 | 44.70 | 72.69 | 83.28 | 0.5755 | 63.06 | +0.67 [-3.01, +4.35] |
| MobileCLIP2-S2 (EN) | 99.13 | 35.70 | 63.43 | 31.77 | 57.19 | 69.79 | 0.4417 | 49.39 | +13.60 [+9.92, +17.39] |

### Lokalny test MSCOCO / task-specific

Protokół: local MMEB candidate protocol v2.

| Model | V+T [M] | Wizyjny [M] | Tekstowy [M] | R@1 | R@5 | R@10 | MRR | NDCG@10 | Δ R@1 [95% CI], p.p. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VN_FULL seed44 | 70.47 | 38.59 | 31.88 | 50.70 | 79.00 | 88.30 | 0.6350 | 68.99 | — |
| MetaCLIP-S16 PL | 49.72 | 21.96 | 27.76 | 30.00 | 56.80 | 66.70 | 0.4251 | 47.46 | — |
| MetaCLIP-M16 PL | 85.30 | 38.78 | 46.52 | 34.60 | 60.10 | 72.00 | 0.4685 | 52.07 | — |
| MobileCLIP2-S2 (EN) | 99.13 | 35.70 | 63.43 | 27.20 | 51.70 | 62.10 | 0.3876 | 43.28 | — |

### COCO val / t2i

Protokół: fixed-gallery-multi-positive-v1.

| Model | V+T [M] | Wizyjny [M] | Tekstowy [M] | R@1 | R@5 | R@10 | MRR | NDCG@10 | Δ R@1 [95% CI], p.p. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VN_FULL seed44 | 70.47 | 38.59 | 31.88 | 63.30 | 90.72 | 96.08 | — | — | — |
| MetaCLIP-S16 PL | 49.72 | 21.96 | 27.76 | — | — | — | — | — | — |
| MetaCLIP-M16 PL | 85.30 | 38.78 | 46.52 | — | — | — | — | — | — |
| MobileCLIP2-S2 (EN) | 99.13 | 35.70 | 63.43 | — | — | — | — | — | — |

### COCO val / i2t

Protokół: fixed-gallery-multi-positive-v1.

| Model | V+T [M] | Wizyjny [M] | Tekstowy [M] | R@1 | R@5 | R@10 | MRR | NDCG@10 | Δ R@1 [95% CI], p.p. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VN_FULL seed44 | 70.47 | 38.59 | 31.88 | 71.36 | 94.99 | 97.85 | — | — | — |
| MetaCLIP-S16 PL | 49.72 | 21.96 | 27.76 | — | — | — | — | — | — |
| MetaCLIP-M16 PL | 85.30 | 38.78 | 46.52 | — | — | — | — | — | — |
| MobileCLIP2-S2 (EN) | 99.13 | 35.70 | 63.43 | — | — | — | — | — | — |

### COCO test — fixed gallery / t2i

Protokół: fixed-gallery-multi-positive-v1.

| Model | V+T [M] | Wizyjny [M] | Tekstowy [M] | R@1 | R@5 | R@10 | MRR | NDCG@10 | Δ R@1 [95% CI], p.p. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VN_FULL seed44 | 70.47 | 38.59 | 31.88 | 50.70 | 79.00 | 88.30 | — | — | — |
| MetaCLIP-S16 PL | 49.72 | 21.96 | 27.76 | — | — | — | — | — | — |
| MetaCLIP-M16 PL | 85.30 | 38.78 | 46.52 | — | — | — | — | — | — |
| MobileCLIP2-S2 (EN) | 99.13 | 35.70 | 63.43 | — | — | — | — | — | — |

### COCO test — fixed gallery / i2t

Protokół: fixed-gallery-multi-positive-v1.

| Model | V+T [M] | Wizyjny [M] | Tekstowy [M] | R@1 | R@5 | R@10 | MRR | NDCG@10 | Δ R@1 [95% CI], p.p. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VN_FULL seed44 | 70.47 | 38.59 | 31.88 | 79.60 | 97.51 | 99.50 | — | — | — |
| MetaCLIP-S16 PL | 49.72 | 21.96 | 27.76 | — | — | — | — | — | — |
| MetaCLIP-M16 PL | 85.30 | 38.78 | 46.52 | — | — | — | — | — | — |
| MobileCLIP2-S2 (EN) | 99.13 | 35.70 | 63.43 | — | — | — | — | — | — |

## Tożsamości modeli

- Nasz: `wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M` + `OPI-PIB/PolDense-32M` + Linear 512→384; seed44, SHA `4bf977049098ca6d8bcc7abc177183aace8c03c5f08e0ecaf011d82aad5b6844`.
- `alphaedge-ai/metaclip-2-worldwide-s16-384-pol-16384`, revision `006223472923ba925226b59b6acd1e3e266c2382`.
- `alphaedge-ai/metaclip-2-worldwide-m16-384-pol-16384`, revision `bc15d9f5ddabe0497898578493a6e81288057c90`.
- `apple/MobileCLIP2-S2`, revision `72424e7025436db18f15c3eff6ee8c7c15ad4481`.
