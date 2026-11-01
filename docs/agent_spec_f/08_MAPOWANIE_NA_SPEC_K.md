# 08 — Mapowanie na specyfikację K (konfrontacja dwóch przeglądów)

Specyfikacja K (`docs/agent_spec_k/`, 31.10.2026) i niniejsza F wychodzą z tej samej diagnozy: liniowy most między dwiema zamrożonymi wieżami jest wyczerpany, a jedyną komercyjną drogą jest własny dual‑encoder z nauczycielem EmbeddingGemma‑2. Różnice dotyczą domyślnego poziomu licencyjnego, kilku pomiarów wstępnych, szerszej drabiny wież, sposobu produkcji celów z nauczyciela oraz jawnego wydzielenia Fazy 2.

## 1. Tabela mapowania

| Element K | Odpowiednik F | Status | Powód |
|---|---|---|---|
| Cel: komercyjny dual‑encoder < 100M; MetaCLIP‑2/MobileCLIP2 wyłączone z treningu | 01 §1, §5 | **przyjęte** | zgodne z wymogiem właściciela |
| Definicja „komercyjnie użyteczny” = wagi + generator + dane (K 01 §4) | 01 §4: dwa poziomy WEIGHTS (domyślny) / STRICT | **zmienione** | właściciel traktuje R1 jako komercyjnie użyteczny; STRICT zostaje jako opcja H1 z pełnym zapisem pochodzenia, bez blokowania Fazy 1 |
| R0/R1/R3 = `RESEARCH_ONLY`, nowy baseline R1‑K na `commercial-v0` | F‑base = istniejący R1 przeliczony na protokół v3‑F; R1‑K tylko przy STRICT | **zmienione** | oszczędza GPU; na poziomie WEIGHTS R1 jest `COMMERCIAL_BASELINE` |
| Generator `qwen3.8-27b` jako bloker P0 wszystkich tekstów syntetycznych | F02: dokumentacja P0; bloker tylko dla STRICT i dla **nowego** generowania | **zmienione** | na poziomie WEIGHTS istniejące teksty pozostają; nowe generowanie wymaga H1 |
| Zakaz nowych ocen modeli `BLOCKED` (także ewaluacyjnych) | 01 §5: modele NC mają zakres `REF` po H1 pkt 3 | **zmienione** | właściciel chce „ścigać” modele NC — potrzebne są ich aktualne wyniki jako poprzeczki; zakaz użycia w treningu/destylacji/filtrach pozostaje |
| Bramki H1–H6 | 01 §8 | **przyjęte** | ta sama semantyka, rozszerzony zakres H1 (poziom, REF, korpusy) |
| Protokół v3: SEL‑ID/SEL‑OOD/GUARD/TEST/REF, wzór SEL, limity odczytów, kryterium +3 p.p., guardraile, budżet latencji | 02 | **przyjęte z rozszerzeniem** | dodano: część dev holdoutu do SEL‑OOD (zapytania i zdania osobno), GUARD #0 dla XM3600, poprzeczki zewnętrzne A/B (S16/M16 NC), raport truncation, test identyczności PolDense jako warunek ważności wyniku |
| K‑A (PolDense zamrożony), K‑B1 (adapter wyjściowy), K‑B2 (LoRA) | Faza 1 = K‑A/K‑B1 (FE4 rozszerza warianty h o whitening i attention‑pooling); K‑B2 → Faza 2 (ME1) z dwoma trybami i replayem | **przyjęte / przeniesione** | priorytet właściciela: najpierw wersja bez modyfikacji wag |
| KE1 drabina wież (A0–A5: TinyCLIP 39/61/63, PE‑Core T16/S16, EG2) | FE1 (A0–A8: + TinyCLIP 40M/32, ViTamin‑S, OpenVision‑S16, uform3; bez TinyCLIP‑63M auto bez karty HF) | **przyjęte z rozszerzeniem** | nowe wieże są już zarejestrowane w `results.json` z licencjami i rewizjami |
| Wskaźnik bariery R z ramienia A5 (EG2 + linear) | FE1 ramię A8 + FE0 sufity centroidów | **przyjęte + dodane FE0** | sufit tekstowy liczy się bez treningu i rozstrzyga pytanie o PolDense‑68M |
| Mapa `W` EG2→PolDense z ramienia A5 | FE2: `W_pairs` / `g_text` / `W_hybrid` | **rozszerzone** | `g_text` uczy się na milionach zdań bez obrazów; porównanie na odłożonych parach wybiera generator celów |
| S1 (cos do `W·EG2(obraz)`) + S2 (sigmoid + KL afiniczności) + WiSE‑FT α | FE3 (dodane ramię D2 z relacyjnym S1) | **przyjęte z rozszerzeniem** | strata relacyjna testuje, czy regresja do pojedynczego celu nie zubaża reprezentacji |
| „Wyczerpane: loss, LR i sampler — nie stroić” | 03 §6 pkt 1; F06 zmienia sampler **raz** (`UniformImages`, limit ARO) i FE5c to potwierdza | **częściowo zmienione** | ~1/3 budżetu w ARO to wada recepty, nie kwestia strojenia; K również przyjmuje `UniformImages` w R1‑K |
| Zakaz treningu na trudnych negatywach ARO | 03 §6 pkt 4 (Faza 1); do rozważenia w Fazie 2 | **przyjęte dla Fazy 1** | zamrożony tekst ich nie rozróżnia |
| KE3 skala nieopisanych obrazów, KE4 ilość vs jakość | FE5a, FE5b | **przyjęte** | numeracja i progi bez zmian merytorycznych |
| KE5 alokacja parametrów i rozdzielczość | FE6 (+ warunek wejścia 68M z FE0/FE1) | **przyjęte z warunkiem** | 68M wchodzi tylko po dowodzie wyższego sufitu |
| Filtr EG2 + mSigLIP z sygnałami (a)–(e) | 06 §5 | **przyjęte, zdegradowane** | flagi tylko do QA; automatyczne wykluczenie wyłącznie dla flag o zmierzonej precyzji ≥ 70% |
| Ludzki QA ~600 par, narzędzie `reviewer/` | 06 §6 (+ próbka MT z Flickr/VizWiz) | **przyjęte z rozszerzeniem** | ocena jakości galerii testowych MT |
| Nowy holdout: ~600 zdjęć własnych + ~400 Commons, 2 annotatorów, zapytanie + zdanie, 40/60, rejestr v6, pieczęć | 06 §7 | **przyjęte** | praca równoległa; nie blokuje FE0–FE3 |
| Wiki‑K dev ~1 000 obrazów, rejestr v5 | Wiki‑F dev, F05 | **przyjęte** | |
| KE‑ALT wieża tekstu bez Gemma (gdy H1 odrzuci PolDense) | brak odpowiednika | **nieprzyjęte** | właściciel wybrał PolDense; w razie odrzucenia Gemma ToU — nowa wersja specyfikacji |
| Operacje: cykl, katalogi, config, `result.json`, raport, log, rekord decyzji, eskalacja, `STATUS.md` | 07 | **przyjęte z rozszerzeniem** | dodane pola `tier`, `poldense_identity`, `unlabeled_pool`, `bridge`, `sampler`, `text_retention` |

## 2. Co F dodaje ponad K

1. **FE0 — sufity przestrzeni tekstowej bez treningu** (centroidy t2i/i2t, hubność, prefiks `[sts]: `) dla PolDense‑17/32/68M; wejście do decyzji o większym PolDense i o głowicy h.
2. **Most `g_text`** uczony na samym tekście — odsprzęga produkcję celów dla obrazów od par; porównany z `W_pairs` na odłożonych parach (FE2).
3. **Szersza drabina wież** (ViTamin‑S, OpenVision‑S16, uform3, TinyCLIP‑40M) z sanity angielskim adaptera.
4. **Głowica tekstowa w czterech wariantach** (whitening, liniowa, rezydualny MLP, attention‑pooling na tokenach) z twardym testem identyczności PolDense.
5. **Faza 2 jawnie wydzielona** (ME1 LoRA w dwóch trybach + prefiks + replay + kotwica; ME2 siostrzana wieża), z harnessem retencji i warunkami wejścia (dane od właściciela, decyzja o wspólnej przestrzeni).
6. **Poprzeczki zewnętrzne A/B** względem MetaCLIP‑2 S16/M16 PL 32768 w kryterium końcowym — zgodnie z celem „ścigania” modeli NC.
7. **Status wyjściowy** w 00 — co w repo i w `results.json` już istnieje, żeby agent nie powtarzał wykonanej pracy.

## 3. Co K ma, a F świadomie pomija lub odkłada

- Ścieżka KE‑ALT (tekst bez Gemma) — odłożona do ewentualnej decyzji H1.
- Rygor STRICT jako domyślny — odłożony do decyzji H1; F zapisuje wszystko, co STRICT wymaga, ale nie egzekwuje.
- Retrening baseline’u R1‑K — tylko przy STRICT.

## 4. Jak używać obu specyfikacji razem

- Jeśli Codex dostaje obie: obowiązuje F; K służy jako źródło sformułowań tam, gdzie F odsyła („jak w K §…”).
- Identyfikatory nie kolidują: K01–K11/KE1–KE5 vs F01–F12/FE0–FE6/M01–M03/ME1–ME2; rekordy decyzji D0–D7 (K) vs DF0–DF6, DM1–DM2 (F).
- Wspólne są bramki H1–H6, role zbiorów, format rejestru licencyjnego, układ katalogów (z zamianą `k_series` → `f_series`).
