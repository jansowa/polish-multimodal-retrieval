# 05 — Eksperymenty FE0–FE6 (Faza 1) i ME1–ME2 (Faza 2)

## 0. Zasady wspólne

- **Jedna zmiana naraz**; wszystko inne stałe i zapisane w configu przed startem.
- **Screening (seed 42) tylko odrzuca ramiona; decyzje wymagają potwierdzenia na 42/43/44.**
- **Budżet w prezentacjach**, jednakowy dla porównywanych ramion. Budżet bazowy serii F = budżet serii R: 5 500 aktualizacji, batch 256 (co 275 kroków batch 145) = 1 405 780 prezentacji; AdamW, LR 1e‑3 dla głowic, WD 0,01 (0 dla skalarów lossu); sigmoid wielopozytywny z `src/pl_mmb_training/losses.py`; ocena co 275 kroków i w kroku 0.
- **Checkpoint:** wyłącznie najwyższy SEL ([02 §3](02_PROTOKOL_EWALUACJI.md)); remis → `SEL_i2t`, prostszy, wcześniejszy.
- **GUARD** czyta się tylko dla potwierdzonych kandydatów, w limitach.
- **Identyczność PolDense** sprawdzana w każdym runie Fazy 1.
- **Inicjalizacja kandydatów** wyłącznie z runów na `f-data-v*`; checkpointy `RESEARCH_ONLY` są wykluczone (dotyczy poziomu STRICT; na poziomie WEIGHTS R1 jest `COMMERCIAL_BASELINE`).
- **Dlaczego te eksperymenty, a nie strojenie LR:** LR/schedule/LP→FT dały +0,14 p.p. (CI [−0,43; +0,72]) przy luce kilkunastu punktów. Każdy eksperyment FE zamyka gałąź drzewa decyzji: wymiar i sufit przestrzeni tekstowej (FE0), wieża wizji i bariera (FE1), źródło celów (FE2), wartość odmrożenia i destylacji (FE3), głowica tekstowa (FE4), dane (FE5), alokacja (FE6).

---

## FE0 — Sufity przestrzeni tekstowej i geometria (bez treningu)

**Pytanie:** ile wyszukiwania obraz↔tekst mieści się w zamrożonej przestrzeni PolDense‑17M/32M/68M i czy i2t jest ograniczone geometrią galerii.

**Pomiary (tylko zbiory SEL; żadnego modelu obrazu nie trenujemy):**
1. **Sufit centroidów t2i:** na zbiorach z ≥ 2 podpisami na obraz (COCO val 419/2 090; VizWiz‑A) dla każdego podpisu c zapytanie = PolDense(c), „obraz” = centroid osadzeń pozostałych podpisów tego obrazu (leave‑one‑out), pozostałe obrazy = pełne centroidy; R@1. Znana wartość: COCO val 70,53% dla 32M (`training/consultation_export_intro.md`).
2. **Sufit centroidów i2t:** obraz = centroid k−1 podpisów; galeria = wszystkie podpisy poza użytymi w centroidzie; trafienie, gdy odłożony podpis jest pierwszy.
3. **Hubność i anizotropia:** średni cos par losowych tekstów, rozkład N10, odsetek hubów; na Wiki native dev, VN dev, COCO val, VizWiz‑A.
4. **Prefiks:** `[query]: ` vs `[sts]: ` vs brak prefiksu dla podpisów galerii — te same pomiary 1–3.
5. **Referencja:** te same miary dla wieży tekstowej MetaCLIP‑2 M16 PL (REF, tylko tekst — po H1 pkt 3) i dla tekstu EG2.

**Modele tekstowe:** PolDense‑17M (256‑D), 32M (384‑D), 68M (512‑D), wszystkie z przypiętymi rewizjami i zapisanymi SHA osadzeń.

**Wejście do DF1:** sufit 68M vs 32M (próg +5 p.p. na SEL lub wyraźnie lepszy i2t); czy sufit i2t jest niski (→ głowica h obowiązkowa); czy `[sts]: ` poprawia sufity (→ do użycia dla tekstów galerii w FE4, bez zmiany wyjścia tekst↔tekst).

---

## FE1 — Drabina komercyjnych wież wizji do zamrożonego PolDense‑32M

**Hipotezy:** (a) mocniejsza komercyjna wieża przy tym samym moście liniowym daje ≥ +3 p.p. SEL względem TinyCLIP‑39M; (b) zamrożona P‑space mieści większość wyrównania nauczyciela (R ≥ 0,90).

**Ramiona (zmienia się wyłącznie ekstraktor cech obrazu; wszystkie z cache z F04):**

| Ramię | Ekstraktor (rewizja z `results.json`) | Wizja | Rola |
|---|---|---:|---|
| A0 | TinyCLIP‑ViT‑39M/16 YFCC @224 (`07a4b0bc…`) | 38,59M | kontrola |
| A1 | TinyCLIP‑ViT‑40M/32 LAION (`95ec8197…`) | 39,69M | kandydat |
| A2 | TinyCLIP‑ViT‑61M/32 LAION (`94cbbea5…`) | 61,41M | kandydat (mieści się tylko z PolDense‑32M) |
| A3 | PE‑Core‑T16‑384 (`8b472274…`) | 6,14M | kandydat |
| A4 | PE‑Core‑S16‑384 (`aabf3b99…`) | 23,78M | kandydat główny |
| A5 | ViTamin‑S (`a3316130…`) | 22,03M | kandydat |
| A6 | OpenVision‑ViT‑S/16‑384 (`74f6c001…`) | 21,96M | kandydat (po sanity adaptera) |
| A7 | uform3‑english‑small, wieża obrazu | 21,77M | opcjonalny |
| A8 | EG2, embedding obrazu 768‑D | > 100M | sufit mostu `W_pairs`; nie jest kandydatem |

Na każdym: świeży `Linear(d, 384)` z biasem, sigmoid wielopozytywny (skala 1/0,07, bias −10, ograniczenia jak w serii R), dane `f-data-v1`, sampler z F06, budżet bazowy. Bez augmentacji (cache).

**Seedy:** screening 42 dla A0–A8; potwierdzenie 43/44 dla A0, dwóch najlepszych z A1–A7 i A8.

**Metryki:** SEL, SEL‑ID, SEL‑OOD, `SEL_i2t`, hubność N10, parametry (wizja + PolDense‑32M + head), latencja ekstraktora.

**Decyzje (DF1):**
- Inicjalizacja = najwyższy SEL spośród A1–A7 w budżecie parametrów i latencji; różnica < 1 p.p. → szybszy. Jeśli żaden nie bije A0 o ≥ 1 p.p. → wizja nie jest pierwszą dźwignią; FE3 startuje z A0.
- Wskaźnik bariery `R = SEL‑OOD(A8) / SEL‑OOD(EG2 zero‑shot)`: `R ≥ 0,90` → h opcjonalna; `R < 0,80` → h obowiązkowa; pomiędzy → oba warianty w FE3.
- Jeśli FE0 wskazało 68M: powtórz A0 i zwycięzcę z PolDense‑68M (`Linear(d, 512)`) jako ramiona A0‑68/Ax‑68 — wejście do FE6.

**Przerwanie ramienia:** brak zgodności preprocessingu, brak rekordu licencji, negatywny sanity angielski.

---

## FE2 — Cele z nauczyciela: który most daje najlepsze cele dla obrazów bez podpisów

**Pytanie:** jak przenieść osadzenie obrazu EG2 do P‑space (lub H‑space) tak, aby służyło jako cel S1 dla obrazów bez polskiego podpisu.

**Ramiona (wszystkie liniowe 768→384 lub 768→512, dopasowane bez GPU lub w minutach):**

| Most | Dopasowanie | Dane | Uwagi |
|---|---|---|---|
| `W_pairs` | sigmoid/MSE: `W·EG2_img(x) ≈ PolDense(t)` na parach | pary `f-data-v1` (dziesiątki tysięcy obrazów) | = głowica ramienia A8 z FE1 |
| `g_text` | najmniejsze kwadraty / kontrastywnie: `g·EG2_txt(t) ≈ PolDense(t)` | tylko tekst: podpisy treningowe PL, teksty Wiki, FineWeb‑2 PL, korpusy po H1 (miliony zdań) | nie wymaga obrazów; narażony na lukę modalności EG2 |
| `W_hybrid` | start od `g_text`, dostrojenie na parach z małym LR | jak wyżej | łączy skalę tekstu z korektą luki |

**Pomiary:**
1. **Jakość celu na odłożonych parach** (10% obrazów `f-data-v1` wyłączonych z dopasowania): R@1 celu `most(EG2_img(x))` wobec PolDense(t) w galerii tekstów odłożonych par i odwrotnie; porównanie z sufitem centroidów FE0.
2. **Luka modalności:** średni cos między `EG2_img(x)` a `EG2_txt(t)` dla par oraz odległość centroidów modalności; wpływ na `g_text` vs `W_pairs`.
3. **Zgodność z centroidami:** cos między `most(EG2_img(x))` a centroidem podpisów obrazu na COCO val.

**Decyzja (DF2, część mostu):** generator celów S1 = most o najwyższym R@1 na odłożonych parach; różnica < 0,5 p.p. → `W_hybrid` (większe pokrycie tekstu). Zapisać `W` z SHA; most nie wchodzi do eksportu.

---

## FE3 — Destylacja S1/S2 z odmrożoną wizją

**Hipotezy:** (a) odmrożenie wizji z S1 + S2 daje ≥ +3 p.p. SEL względem najlepszej zamrożonej wieży; (b) sama destylacja wnosi ≥ 1 p.p. ponad odmrożenie; (c) głowica h wnosi ≥ 1 p.p., jeśli FE1 wskazało barierę.

**Punkt startowy:** zwycięzca FE1 (wizja + head) z `f-data-v1`; most z FE2 (zamrożony); h z FE4 (jeśli wybrana).

**Ramiona**

| Ramię | Wizja | Tekst | Etapy | Rola |
|---|---|---|---|---|
| C | zwycięzca FE1, zamrożona | PolDense zamrożony | — | kontrola (istniejące checkpointy 42/43/44) |
| D0 | odmrożona | PolDense zamrożony | tylko S2 bez KD | izoluje efekt odmrożenia |
| D1 | odmrożona | PolDense zamrożony | S1 (cos do celu z mostu) + S2 z KD afiniczności | wariant A |
| D2 | odmrożona | PolDense zamrożony | S1 relacyjny (sigmoid: pozytyw = własny cel z mostu, negatywy = cele innych obrazów w batchu) + S2 z KD | wariant A′ |
| D3 | odmrożona | PolDense zamrożony + h (FE4) | jak lepszy z D1/D2, cele i pary w H‑space | wariant B |

**S1:** obrazy puli U (cel ≥ 50 000 obrazów; mniej → zapisać liczbę) + obrazy par, widoki z cache F04; 1 000 000 prezentacji obrazów. **S2:** pary `f-data-v1`, sigmoid wielopozytywny + `λ_aff·KL(softmax(S_T/τ) ‖ softmax(S_S/τ))` w obu kierunkach (S_T = macierz podobieństw EG2 w batchu, τ = 0,07, λ_aff = 1); 700 000 prezentacji par. Optymalizacja: wizja AdamW LR 1e‑5, WD 0,05, warmup 5%, cosine; głowice LR 1e‑3; BF16 autocast, gradient checkpointing; batch największy mieszczący się w 16 GB dla największego ramienia, jednakowy dla wszystkich. Ramię na seedzie s startuje z checkpointu C dla tego samego seeda.

**Checkpoint:** punkty 25/50/75/100% S2 × WiSE‑FT α ∈ {0,5; 0,75; 1,0} (interpolacja wizji i head między startem a stanem po treningu; h skalowana przez α); wybór najwyższy SEL; remis → `SEL_i2t`, mniejsze α, wcześniejszy punkt.

**Guardraile w SEL:** SEL‑OOD ≥ C − 0,5 p.p.; `SEL_i2t` ≥ C − 1,0 p.p.; identyczność PolDense pozytywna.

**Sukces:** screening — najlepsze ramię ≥ C + 3,0 p.p. SEL; potwierdzenie (42/43/44) — średnio ≥ +3,0, każdy seed ≥ +1,5, CI > 0. **Stop:** D0, D1, D2, D3 wszystkie < C + 1,0 → stop skalowania Fazy 1, eskalacja z diagnozą (jakość celów, bariera tekstu, dane) i rekomendacja otwarcia Fazy 2.

**Decyzje (DF3):** `D1/D2 − D0 ≥ 1,0` → KD zostaje (lepszy z D1/D2); `D3 − max(D1, D2) ≥ 1,0` → h zostaje; w pozostałych przypadkach prostszy wariant. Po potwierdzeniu odczyt GUARD #1 z F‑base.

---

## FE4 — Głowica po stronie tekstu h na zamrożonym PolDense

**Pytanie:** czy reprojekcja galerii tekstowej przywraca i2t ≥ t2i i podnosi SEL bez dotykania wag PolDense.

**Warianty (wszystkie zostawiają surowe wyjście P‑space dla tekst↔tekst; identyczność PolDense sprawdzana):**

| h | Opis | Parametry | Uczone? |
|---|---|---:|---|
| H0 | brak (P‑space) | 0 | — |
| H1 | whitening w formie zamkniętej (PCA/ZCA ze statystyk tekstów treningowych), potem L2 | ~0,15M (macierz) | nie — liczone ze statystyk |
| H2 | uczona liniowa 384→384 (lub →512) | ~0,15M | tak, w S2 |
| H3 | rezydualny MLP 384→768→384, ostatnia warstwa od zer | ~0,6M | tak, w S2 |
| H4 | attention‑pooling (1 zapytanie, 4 głowy) na zamrożonych stanach tokenów PolDense + liniowa | ~0,6M | tak, w S2; wymaga cache stanów tokenów |

**Etap 1 (tanio, na cache):** każdy wariant h w recepcie zamrożonej (jak FE1, wizja = zwycięzca FE1): obraz → `Linear(d, dim_h)`, tekst → h(P‑space); budżet bazowy; seed 42 → potwierdzenie 43/44 dla najlepszego. **Etap 2:** zwycięzca h wchodzi jako D3 do FE3.

**Metryki:** SEL, `SEL_i2t` (główna dla tej hipotezy), hubność N10, panel kompozycyjny, identyczność PolDense.

**Decyzja (DF2, część głowicy):** h zostaje, jeśli `SEL ≥ H0 + 1,0` lub `SEL_i2t ≥ H0 + 2,0` przy SEL ≥ H0 − 0,3; inaczej H0. Koszt produktowy (osobny indeks tekstów) zapisać w rekordzie decyzji.

---

## FE5 — Dane: ilość, jakość, obrazy bez podpisów

**FE5a — skala puli U w S1** (recepta zwycięzcy FE3): U0 tylko obrazy par; U1 50 tys.; U2 200 tys.; U3 maksimum po bramce. Ta sama liczba prezentacji S1 (1 000 000) i identyczne S2. Screening 42; potwierdzenie dla U0 i najlepszego. **Decyzja:** `U3 ≥ U2 + 1,0 p.p. SEL‑OOD` → wniosek H5 o pobranie kolejnych obrazów Wikimedia; inaczej nie rozszerzać.

**FE5b — pary: ilość czy jakość** (ten sam budżet): D0 `f-data-v1` (N obrazów); D1 losowe N obrazów z `f-data-v2` (po dopuszczeniu pełnej puli Wiki/VN ready); D2 N obrazów z `f-data-v2` po filtrze skalibrowanym w QA (próg ustalony przed treningiem, nie na SEL); D3 cały `f-data-v2`. Interpretacja: D1 − D0 = różnorodność źródeł; D2 − D1 = filtr; D3 − max(D1, D2) = ilość. **Decyzja:** `D3 ≥ max(D1, D2) + 1,5` → priorytet dla zatwierdzania kolejnych par; `D2 ≤ D1` → poza guardem i twardymi błędami nie filtrować.

**FE5c — sampler (jednorazowo, w recepcie zamrożonej):** sampler serii R (wagi wierszy) vs `UniformImages` z limitem ARO ≤ 10% — potwierdza zmianę bazową z F06; nie jest początkiem strojenia.

---

## FE6 — Alokacja parametrów i rozdzielczość

**Ramiona (tylko kombinacje < 100M w budżecie latencji, recepta finalna z FE3):** zwycięzca wizji + PolDense‑32M; zwycięzca wizji + PolDense‑68M (jeśli FE0/FE1 to uzasadniły; `Linear(d, 512)`, osobna referencja identyczności); TinyCLIP‑61M/32 + PolDense‑32M; dla PE‑Core‑S16: 224 px vs 384 px (pozycyjne interpolowane, zapisane). Screening 42; potwierdzenie dla dwóch najlepszych.

**Decyzja (DF5):** konfiguracja na froncie SEL × CPU ms/obraz × ms/tekst × parametry; latencja tekstu dla 68M ≈ 2× 32M — właściciel potwierdza budżet.

---

## ME1 — LoRA w dwóch trybach z replayem (Faza 2)

**Warunki wejścia:** DF3 zamknięte; M01 gotowe (protokół v3‑M, referencja PolDense na panelu retencji, dane replay zweryfikowane w rejestrze).

**Ramiona (wizja i head = zwycięzca Fazy 1, zamrożone na start):**

| Ramię | Tekst | Replay | Kotwica | Rola |
|---|---|---|---|---|
| P1 | zwycięzca Fazy 1 (PolDense nienaruszony) | — | — | kontrola |
| L0 | LoRA r=16 na ciele PolDense, bez prefiksu, bez replayu | nie | nie | ile daje sama swoboda (diagnostyka ryzyka) |
| L1 | LoRA r=16 + prefiks `[image]: ` dla tekstów zadań z obrazem | tak (1:1) | tak (λ_d = 1) | wariant główny |
| L2 | jak L1 bez kotwicy | tak | nie | izoluje kotwicę |

**Strata:** `L = L_it (sigmoid, pary) + λ_t·L_tt (InfoNCE z 10 trudnymi negatywami + negatywy z batcha; zapytania `[query]: `, pasaże bez prefiksu) + λ_d·L_anchor (cos do osadzeń zamrożonego oryginalnego PolDense na tekstach replayu)`; batch tekstowy i obrazowy naprzemiennie; LR ciała 1e‑5, LoRA 1e‑4, głowice 1e‑3; budżet S2 jak w FE3 plus równa liczba prezentacji tekstowych.

**Panel retencji co 500 kroków:** podzbiór PIRB + odłożone części replayu; progi: średnio ≤ −0,5 p.p. nDCG@10, żadne zadanie ≤ −1,0 p.p.; przekroczenie → rollback do ostatniego checkpointu w progu, bez dalszego strojenia λ. Tryb tekstowy (adapter off) = bitowo oryginalny PolDense — test identyczności jak w Fazie 1.

**Sukces:** L1 ≥ P1 + 3,0 p.p. SEL przy panelu w progu i `SEL_i2t` ≥ P1. **Decyzja DM1:** sukces → wdrożenie dwutrybowe; L1 lepsze, ale panel poza progiem → ME2; L1 < P1 + 1,0 → pozostać przy Fazie 1.

## ME2 — Siostrzana wieża tekstowa (Faza 2)

PolDense‑32M zamrożony do tekst↔tekst; do obrazów osobna kopia PolDense‑17M (256‑D) lub 32M trenowana swobodnie w układzie LiT (wizja zamrożona lub WiSE‑FT, tekst uczony) z KD od EG2; budżet jak ME1; bez replayu (tekst↔tekst obsługuje nietknięty oryginał). Limit parametrów: 32 + 17 + wizja (≤ ~45M) lub 32 + 32 + wizja (≤ ~30M). **Decyzja DM2:** ME2 vs ME1 vs Faza 1 na froncie SEL × parametry × liczba przebiegów tekstu; scalenie do jednych wag wyłącznie po pełnym PIRB i jednej zaplanowanej ewaluacji TEST.
