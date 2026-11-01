# Stan serii — 30.10.2026

Parametry podano w milionach: **Enkodery V+T = wizyjny + tekstowy**. Pozostałe enkodery oraz osobny projektor nie wchodzą do tej sumy. Nasz model: **70,47 mln = 38,59 mln wizja + 31,88 mln tekst**, projektor osobno **0,197 mln**. EmbeddingGemma2: **438,76 mln = 167,76 mln wizja + 271,00 mln tekst**; enkoder audio był wyłączony.


**P1.B0 ukończone: 20 zarejestrowanych modeli porównawczych, 4 galerie i 2 tryby promptów.** Finalista pozostaje VN_FULL seed44, wybrany wcześniej medianą VN dev.

Nasza konfiguracja: vision `wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M`, text `OPI-PIB/PolDense-32M`, Linear 512→384; oba encodery frozen. Około 70,67 mln parametrów inference.

## Porównanie z konkurencją

[Pełne tabele paired Δ i 95% CI, osobno ≤100M, PL-adaptowane i tryby promptów, oraz efektywność](REFERENCE_COMPARISON_20261029.md).
[Wszystkie nasze checkpointy, średnia/SD trzech seedów i porównania kontroli](OUR_NEW_GALLERIES_20261029.md).

Poniżej wyróżnione odczyty trybu v2. Δ = nasz seed44 minus referencja. Jednostka CI: komponent; 10 000 replik, seed 20261023. Przedziały są warunkowe; nie obejmują niepewności treningu/pretreningu ani korekty wielokrotnych porównań.

### wiki_native_dev

Galeria: 397 obrazów / 331 tekstów. Nasz seed44: t2i R@1 **15.41%**, i2t **14.86%**.

| Referencja | Enkodery V+T [M] | Wizyjny [M] | Tekstowy [M] | t2i R@1 | Seed44 minus referencja [95% CI], p.p. |
| --- | ---: | ---: | ---: | ---: | ---: |
| metaclip-s16 | 49.72 | 21.96 | 27.76 | 29.31% | -13.90 [-19.87, -8.58] |
| metaclip-m16 | 85.30 | 38.78 | 46.52 | 33.84% | -18.43 [-24.33, -12.99] |
| embeddinggemma-2 | 438.76 | 167.76 | 271.00 | 54.98% | -39.58 [-46.33, -32.71] |

### xtd10_full_1k

Galeria: 1000 obrazów / 1000 tekstów. Nasz seed44: t2i R@1 **49.30%**, i2t **43.60%**.

| Referencja | Enkodery V+T [M] | Wizyjny [M] | Tekstowy [M] | t2i R@1 | Seed44 minus referencja [95% CI], p.p. |
| --- | ---: | ---: | ---: | ---: | ---: |
| metaclip-s16 | 49.72 | 21.96 | 27.76 | 38.00% | +11.30 [+7.80, +14.90] |
| metaclip-m16 | 85.30 | 38.78 | 46.52 | 42.90% | +6.40 [+2.90, +9.90] |
| embeddinggemma-2 | 438.76 | 167.76 | 271.00 | 71.10% | -21.80 [-25.00, -18.60] |

### xtd10_no_overlap

Galeria: 897 obrazów / 897 tekstów. Nasz seed44: t2i R@1 **51.39%**, i2t **45.37%**.

| Referencja | Enkodery V+T [M] | Wizyjny [M] | Tekstowy [M] | t2i R@1 | Seed44 minus referencja [95% CI], p.p. |
| --- | ---: | ---: | ---: | ---: | ---: |
| metaclip-s16 | 49.72 | 21.96 | 27.76 | 39.69% | +11.71 [+7.92, +15.50] |
| metaclip-m16 | 85.30 | 38.78 | 46.52 | 44.15% | +7.25 [+3.34, +11.04] |
| embeddinggemma-2 | 438.76 | 167.76 | 271.00 | 72.58% | -21.18 [-24.75, -17.73] |

### VN_dev

Galeria: 2000 obrazów / 2003 tekstów. Nasz seed44: t2i R@1 **73.94%**, i2t **69.15%**.

| Referencja | Enkodery V+T [M] | Wizyjny [M] | Tekstowy [M] | t2i R@1 | Seed44 minus referencja [95% CI], p.p. |
| --- | ---: | ---: | ---: | ---: | ---: |
| metaclip-s16 | 49.72 | 21.96 | 27.76 | 56.07% | +17.87 [+15.38, +20.33] |
| metaclip-m16 | 85.30 | 38.78 | 46.52 | 63.65% | +10.28 [+7.90, +12.73] |
| embeddinggemma-2 | 438.76 | 167.76 | 271.00 | 93.76% | -19.82 [-21.73, -17.91] |

Wiki: odczyt przed nowym treningiem Wiki. VN: in-domain dla nas, podpisy modelowe. XTD10: kontrola stylu, nie niezależny holdout. Nie ustalono rozłączności z pretreningiem konkurencji.

## P1.A — trzy seedy i zamknięty test

Seedy 42/43/44: VN dev t2i R@1 **73.92% ± 0.18 p.p. SD**; COCO val post hoc **62.31% ± 0.98 p.p. SD**.
Seed44 wybrano wcześniej regułą mediany VN dev. Lokalny legacy MSCOCO: t2i R@1 50,7%, R@5 79,0%, R@10 88,3%. To lokalny protokół kandydatów, nie oficjalny COCO5k. Fixed-gallery zapisano osobno w `vn_full_seeds_posthoc.json`; testów nie powtarzano.

## Stan zadań

| Punkt | Status | Uwagi |
|---|---|---|
| P0.1 | zrobione | Wiki397 / 331 tekstów, guard i immutable baseline seed44; brak nowego treningu Wiki. |
| P0.2 | częściowo | XTD10 1000 i clean897 gotowe i ocenione; brak istniejących dokładnych par EN do Qwen-vs-human style. |
| P0.3 | częściowo | 300 zadań i narzędzie gotowe; rzeczywiste ludzkie query pozostają do zebrania. |
| P0.4 | częściowo | 200 zadań audytu i narzędzie gotowe; rzeczywiste ludzkie oceny pozostają do zebrania. |
| P0.5 | częściowo | Rejestracja, prompt sources, adapter parity, cache ranks, full-gallery i efficiency gotowe; lokalne CIRR/FashionIQ/RefCOCO finalisty niezamknięte. |
| P0.5.2 | zrobione | wiki_dev_zero_shot_pre_wiki_training.json zachowany bez nadpisania. |
| P0.5.3 | zrobione | Draft XM istnieje; execution_allowed=false; zero wyników XM3600. |
| P1.A | zrobione | Seedy42 reuse,43/44 fresh; mediana44; legacy i fixed-gallery test już wykonane. |
| P1.B0 | zrobione | 20 modeli, 4 galerie, 2 tryby, 2 kierunki; paired CI i efficiency; usunięto ukończone wagi. |
| P1.B | nie zaczęte | Po zamknięciu B0: ekstrakcja i 1-seed scout według pierwotnego planu. |
| P1.C | nie zaczęte | Po B: sampler i ARO0/10/20/50, 3 seedy. |
| P1.D | nie zaczęte | Po C: Linear / LNLinear / residual MLP, 3 seedy. |
| P2.E | częściowo | Dossier z linkami i cytatami; USA Today niedostępne, kwalifikacje prawne nierozstrzygnięte. |
| P2.F | nie zaczęte | Text-only modality gap, licencjonowany OPUS i alignment pilot pozostają. |
| P2.G | nie zaczęte | Plan i image audit po B; brak treningu destylacyjnego bez akceptacji. |
| expanded_corpus | częściowo | 157000 zamrożonych rekordów, metadata audit; training_ready_rows=0, image/rights audit pending. |

## Większy korpus

Zamrożono 132 875 rekordów VN i 24 125 Wiki, razem 157 000. Wstępnie nowe, niepowiązane z chronionymi komponentami po metadanych: 96 930 VN i 22 580 Wiki. To nie są gotowe niezależne pary treningowe: brak pełnego audytu obrazów, pixel SHA, pHash≤6 i praw. `training_ready_rows=0`.
Oryginalne FULL_CLEAN, validation, snapshot VN 35 918 i stary Wiki 1322 zachowały SHA. Nie podmieniono ich nowymi eksportami. Nie otwierano aktywnych SQLite/WAL ani nie zmieniano działających pipeline’ów.

## Decyzje użytkownika

- XM3600 pozostaje FINAL_BLIND. [Draft bramki v2](xm3600_gate_registration_v2_20261030.json) proponuje +2 p.p. i dolną granicę paired 95% CI > 0 wobec najlepszego małego polskojęzycznego modelu na Wiki dev; wymaga domknięcia protokołu i akceptacji przed wspólnym uruchomieniem. Zero wyników XM w tej serii.
- Publikacja: [dossier](LICENSING_DOSSIER_20261030.md), w tym Gemma/PolDense, źródła newsowe i luka USA Today. Brak zatwierdzenia publikacji.
- Nowy trening Wiki: baseline zapisany, ale wymagany rozłączny train loader, pełny audyt i decyzja. Historyczny pilot 975/201 nie jest nowym Wiki 397.
- Destylacja: najpierw plan i audyt; brak treningu bez akceptacji. FT/LoRA/KD-λ na starej mieszance nie uruchamiano.

## Dysk, czas i dowody

Wagi referencji ładowano pojedynczo i usuwano po wszystkich galeriach oraz timingach. Zachowano konfiguracje, embeddingi, wyniki i modele bazowe TinyCLIP/PolDense. Wyjątek formatu wspólnych blobów HF obsłużono z kontrolą SHA i brakiem odwołań z innych repozytoriów. Nie wykonywano ogólnego czyszczenia cache.
Natywny cache Xet miał podczas przeglądu około 3 GiB; nie usuwano jego chunków. Aktywne pipeline’y również zmieniają zajętość dysku, więc różnica wolnego miejsca nie jest wyłącznie kosztem benchmarku.

| Worker referencji | Enkodery V+T [M] | Wizyjny [M] | Tekstowy [M] | Czas ścienny, s | Usunięte pliki wag, GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| metaclip-s16 | 49.72 | 21.96 | 27.76 | 199.3 | 0.185 |
| metaclip-m16 | 85.30 | 38.78 | 46.52 | 211.1 | 0.318 |
| mobileclip2-s2 | 99.13 | 35.70 | 63.43 | 142.6 | 0.371 |
| clip-b32-pl | 143.75 | 87.85 | 55.90 | 179.8 | 0.772 |
| siglip2-pl | 191.80 | 93.52 | 98.28 | 343.6 | 0.715 |
| siglip2-base | 375.82 | 93.52 | 282.30 | 239.4 | 1.400 |
| siglip2-pl-32768 | 204.38 | 93.52 | 110.86 | 244.3 | 0.761 |
| embeddinggemma-2 | 438.76 | 167.76 | 271.00 | 1239.5 | 1.387 |
| siglip2-naflex | 375.23 | 92.93 | 282.30 | 179.3 | 1.398 |
| clip-b32-pl-32768 | 156.33 | 87.85 | 68.48 | 88.2 | 0.819 |
| mclip-b16plus | 677.76 | 117.21 | 560.55 | 239.1 | 2.865 |
| siglip2-base-384 | 375.48 | 93.18 | 282.30 | 251.5 | 1.399 |
| siglip2-base-256 | 375.23 | 92.93 | 282.30 | 196.1 | 1.398 |
| siglip-i18n-256 | 370.63 | 92.93 | 277.70 | 185.9 | 1.381 |
| siglip2-b32-256 | 376.86 | 94.55 | 282.30 | 163.8 | 1.404 |
| tipsv2-b14 | 195.95 | 86.32 | 109.63 | 400.7 | 0.730 |
| bge-vl-base | 149.62 | 86.19 | 63.43 | 185.6 | 0.279 |
| tipsv2-l14 | 487.94 | 304.02 | 183.92 | 1687.6 | 1.818 |
| pe-core-b16 | 447.66 | 93.67 | 353.99 | 176.2 | 1.668 |
| fashion-siglip | 203.16 | 92.88 | 110.27 | 163.0 | 0.757 |

Czas workera obejmuje CPU/GPU, dekodowanie, ładowanie i pomiar CPU. Nie jest pomiarem aktywnego czasu kerneli GPU; nie rekonstruujemy go z median timingów.

Raport treningu P1.A podaje `GPU_seconds_new_runs`: seed 43: 18.34 s, seed 44: 17.94 s. Seed42: reuse.
- Wolne po wykonaniu: `C:\` **24.70GiB**.
- Wolne po wykonaniu: `D:\` **4.07GiB**.

SHA nowych manifestów i artefaktów: [artifact_index_20261030.json](artifact_index_20261030.json). Lista rejestracji/decision JSON, czasy i dowody: [summary.json](summary.json). Poprzedni raport i summary zachowane w kopiach `*_before_20261030`.
Weryfikacja: parity 3000/3000 historycznych rang wspólnym scorerem; adapter dokładnie odtworzył 16 historycznych próbek; 15 testów serii oraz 4 testy ochrony współdzielonych wag przeszły. Przedziały sprawdzają tożsamości galerii, qrels, query i komponentów.

Najbliższy etap GPU: P1.B — frozen features i scout backbone’ów; potem P1.C/P1.D. Kończymy obecny podpunkt w całości, zgodnie z zasadą przerwania po pełnym etapie.
