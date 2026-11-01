# 04 — Pakiety pracy

## 1. Ścieżka krytyczna

| ID | Pakiet | Prio | Wykonawca | Zależy od | Bramki | Koszt (szac.) |
|---|---|---|---|---|---|---|
| F01 | Protokół v3‑F i infrastruktura pomiaru | P0 | agent | — | — | CPU, 1 dzień |
| F02 | Rejestr licencyjny v1, poziomy WEIGHTS/STRICT, generator, przegląd rynku | P0 | agent | — | H1 | CPU, 1 dzień |
| F03 | FE0: sufity przestrzeni tekstowej i geometria | P0 | agent | F01 | — | CPU/GPU < 2 h |
| F05 | Wiki‑F dev + kandydaci holdoutu z Commons → rejestr v5 | P0 | agent | F02 | — | CPU + pobieranie |
| F04 | Cache cech wież wizji + cache nauczyciela EG2 | P0 | agent | F02, F05 | H5 jeśli > limit | GPU 0,5–1 dnia |
| F06 | Manifest `f-data-v1` (bramka v5, pochodzenie, sampler) | P0 | agent | F02, F05 | H2 | CPU, audyt |
| F07 | FE1: drabina wież wizji | P0 | agent | F01, F04, F06 | — | GPU 0,5–1 dnia |
| F08 | FE2 mosty + FE4 głowica tekstowa | P0 | agent | F03, F04, F07 | — | GPU < 1 dnia |
| F09 | FE3: destylacja S1/S2 z odmrożoną wizją | P1 | agent | F07, F08 | — | GPU 2–3 dni |
| F10 | Nowy holdout: zdjęcia własne + Commons, annotacja, QA, rejestr v6, pieczęć TEST | P1 | agent + ludzie | F05 | H3 | ludzie ~50–60 h |
| F11 | FE5 (dane, skala U) + FE6 (alokacja, rozdzielczość) | P1 | agent | F09, F06 | H2, H5 | GPU 2–4 dni |
| F12 | Trening finalny, eksport, GUARD #3, TEST, pakiet zgodności | P2 | agent + ludzie | F09–F11 | H4, H5, H6 | GPU 1 dzień |
| M01 | Harness retencji tekstu i dane replay | P2 (Faza 2) | agent + właściciel | DF3 zamknięte | H1 (dane) | CPU/GPU < 1 dnia |
| M02 | ME1: LoRA w dwóch trybach + prefiks + replay + kotwica | P2 | agent | M01 | H5 | GPU 2–3 dni |
| M03 | ME2: siostrzana wieża; decyzja dwa tryby vs wspólne wagi | P2 | agent | M02 | — | GPU 1–2 dni |

**Równoległość:** F02, F03, F05, F10 i audyt F06 są pracą CPU/ludzką i biegną równolegle z zadaniami GPU. Zadania GPU wykonuje się sekwencyjnie pod wspólną blokadą.

## 2. Pakiety Fazy 1

### F01 — Protokół v3‑F i infrastruktura pomiaru (P0)

**Cel:** zamrozić pomiar, zanim pojawi się pierwszy wynik serii F.

**Kroki**
1. Zapisz `protocols/protocol-v3-f.md` i `.json` (role, SEL, kryterium, limity z [02](02_PROTOKOL_EWALUACJI.md)); commit, SHA.
2. Podziel VizWiz‑PL na połowy A/B po komponentach (seed 20261101, ~500 obrazów każda) w **nowym** katalogu `retrieval_benchmark/catalog-f-v3.json`; nie modyfikuj `catalog-current.json` ani `catalog.json`; uruchom kontrolę spójności rejestru.
3. Zaimplementuj log odczytów GUARD/TEST i wrapper odmawiający odczytu bez wpisu/zgody ([07 §6](07_OPERACJE_ARTEFAKTY.md)); wpisz GUARD #0 (XM3600, 31.10.2026) jako historyczny.
4. Zaimplementuj licznik parametrów eksportu (test: R1 = 70 667 520) z rozbiciem `vision / text / text_head / image_head / other / total`.
5. Zaimplementuj kalkulator SEL i `SEL_i2t`; policz dla R1 seed 42/43/44 (= F‑base) i dla EG2 (sufit). Dla VizWiz‑A potrzebne są rankingi per zapytanie z istniejących wyników serii/OOD.
6. Zaimplementuj test identyczności PolDense ([02 §5](02_PROTOKOL_EWALUACJI.md)): sonda 1 000 tekstów, referencja FP32, SHA.
7. Zaimplementuj panel kompozycyjny (5 602 pary walidacyjne z `training/runs/e1-full/translation_snapshots/`, połączenie po `translation_id`, tylko ID walidacji, remisy = 0,5); policz dla R1 i dla EG2.
8. Zmierz latencję w harnessie dla wież kandydujących jako ekstraktorów (PE‑Core S16/T16, ViTamin‑S, OpenVision‑S16, TinyCLIP 40/61) — same wieże obrazu, bez tekstu.

**Gotowe, gdy:** licznik odtwarza 70 667 520; wrapper blokuje nieautoryzowany odczyt (test); protokół zacommitowany przed jakimkolwiek treningiem serii F; tabela SEL dla F‑base (mean ± SD) i EG2; panel kompozycyjny policzony; test identyczności przechodzi dla R1.

### F02 — Rejestr licencyjny v1, poziomy, generator, przegląd rynku (P0, H1)

**Kroki**
1. Utwórz rejestr i snapshoty dowodów dla wszystkich pozycji z [01 §5](01_CEL_ZASADY_LICENCJE.md).
2. Udokumentuj generator `qwen3.8-27b`: konfiguracja serwera, provenance w SQLite pipeline’u tłumaczeń, dokładny model/rewizja, licencja, warunki użycia wyjść do trenowania. Zapisz wynik niezależnie od tego, czy jest rozstrzygający.
3. Policz pokrycie poziomów: ile rekordów każdej puli (FULL, VN użyte, VN ready, Wiki ready) spełnia WEIGHTS i ile STRICT, osobno dla `native-original / native-recycled / translated / generated`.
4. Przegląd rynku: modele tekst↔obraz < 100M z polskim i licencją komercyjną, ze źródeł pierwotnych; każdy `CLEARED` dołącz jako `REF` do leaderboardu. Stan na 01.11.2026: brak takiego modelu w benchmarku.
5. Wniosek H1: poziom WEIGHTS/STRICT; Gemma ToU dla PolDense; polityka EG2; zgoda na oceny REF NC; Wikimedia BY‑SA; VN; FULL; korpusy mostu/replay; zbiory `EVAL-INTERNAL`.

**Gotowe, gdy:** rejestr, dowody, raport pokrycia; decyzje H1 wpisane albo jawnie oczekujące z listą blokerów.

### F03 — FE0: sufity przestrzeni tekstowej i geometria (P0)

Specyfikacja w [05 FE0](05_EKSPERYMENTY.md). **Gotowe, gdy:** tabela sufitów centroidów (t2i, i2t) dla PolDense‑17M/32M/68M na COCO val i VizWiz‑A, miary hubności/anizotropii na wszystkich zbiorach SEL, wynik porównania prefiksów `[query]: ` vs `[sts]: `, rekord wejściowy do DF1.

### F05 — Wiki‑F dev i kandydaci holdoutu z Commons → rejestr v5 (P0)

**Kroki**
1. Wiki‑F dev: ~1 000 obrazów z puli Wikimedia w całych komponentach, teksty `native-original` A/B, warstwowo po kategoriach; wyklucz komponenty użyte w treningu Wiki G2 i 1 159 obrazów zarezerwowanych komponentów.
2. Kandydaci holdoutu z Commons: ~400 plików wgranych po 2026‑10‑06 (premiera EG2; inna data tylko po H1), licencje z allowlisty, spoza uniwersum kolektora.
3. Oryginały + fingerprinty (SHA bajtów, SHA pikseli po EXIF, pHash DCT 32×32/8×8/mediana, Hamming ≤ 6); komponenty przechodnie.
4. `retrieval_benchmark/holdouts/registry-v5.json` jako nowy plik; aktywacja jawna tylko dla nowych runnerów; zamrożone generacje zachowują swoje rejestry.

**Gotowe, gdy:** v5 aktywny dla serii F; Wiki‑F dev ma galerię, qrels i SHA; kandydaci chronieni.

### F04 — Cache cech wież wizji i cache nauczyciela (P0)

**Kroki**
1. Wieże wizji (przypięte rewizje z `results.json`): TinyCLIP‑39M/16 (kontrola), TinyCLIP‑40M/32, TinyCLIP‑61M/32, PE‑Core‑T16‑384, PE‑Core‑S16‑384, ViTamin‑S, OpenVision‑S16‑384 (opcjonalnie uform3‑small). Dla każdej: test zgodności preprocessingu z implementacją autorów na 100 obrazach (cos ≥ 0,999) i sanity angielski zero‑shot na 50 obrazach (wyklucza błąd adaptera — OpenVision ma w benchmarku ~0% po polsku, co wymaga sprawdzenia, że obrazy są kodowane poprawnie). Cache końcowego embeddingu z natywną projekcją (512‑D dla TinyCLIP/PE/OpenVision; ViTamin wg konfiguracji) dla obrazów manifestu i galerii SEL.
2. Nauczyciel EG2 (rev `914f7f89…`): smoke 1 000 obrazów (przepustowość, BF16 vs FP32, brak NaN). Zakoduj: obrazy manifestu w 3 widokach (środkowy + 2 RandomResizedCrop z zapisanymi parametrami), obrazy puli U w widoku środkowym, teksty PL (i EN, jeśli są) z prefiksami wg karty (zapytania: `task: search result | query: `; dokumenty: `title: none | text: `; porównać także bez prefiksu — zapis obu). Szacunek: 41,7 ms/obraz CUDA → 100k obrazów ≈ 1,2 h.
3. Galerie SEL/GUARD/TEST koduj osobno w `eg2_eval/`; trenery mają allowlistę ścieżek i NIE MOGĄ czytać tego katalogu.
4. Policz sygnały audytu z [06 §5](06_DANE_HOLDOUT_QA.md) (bliskość holdoutu, zgodność rangowa, false negatives).

**Gotowe, gdy:** manifesty cache z SHA, raport pokrycia, lista sąsiadów holdoutów do przeglądu, EG2 zero‑shot na SEL (referencja wskaźnika R).

### F06 — Manifest `f-data-v1` (P0, H2)

**Kroki**
1. Zbuduj `f-data-v1` z: FULL_CLEAN (70 289 wierszy), VN użyte w R1 (53 794 pary), Wiki ready po bramce v5 (do 46 968 par; tylko rekordy z lokalnym oryginałem i fingerprintem), z polami pochodzenia i `tier` per rekord. Przy poziomie STRICT z H1 — manifest alternatywny `f-data-v1-strict` z filtrem.
2. Guard v5 na manifeście; zapisz rekordy, unikalne pary, unikalne obrazy (SHA pikseli), komponenty, powody wykluczeń.
3. Sampler `UniformImages`: jednostajnie po obrazach w obrębie źródła, jeden podpis na wylosowany obraz, udziały źródeł ustalone z góry (FULL 35% / VN 45% / Wiki 20% prezentacji — do zapisania w protokole przed FE1), limit ARO ≤ 10% wszystkich prezentacji, limit ≤ 10 obrazów z jednego komponentu Wiki na epokę budżetową.
4. Zamroź manifest (SHA) i złóż wniosek H2.

**Gotowe, gdy:** manifest z SHA i raportem; H2 zatwierdzone lub jawnie oczekuje. **Uwaga:** F‑base = R1 nie jest trenowany ponownie na `f-data-v1`; porównania FE1 zawierają własną kontrolę A0 na tym manifeście.

### F07 — FE1: drabina wież wizji (P0)

Specyfikacja w [05 FE1](05_EKSPERYMENTY.md). **Gotowe, gdy:** rekord DF1 (inicjalizacja, wskaźnik R, decyzja o 68M, wariant h do FE3).

### F08 — FE2 mosty + FE4 głowica tekstowa (P0)

Specyfikacja w [05](05_EKSPERYMENTY.md). **Gotowe, gdy:** rekord DF2 (generator celów S1, wariant h, dowód identyczności PolDense dla każdego h).

### F09 — FE3: destylacja S1/S2 z odmrożoną wizją (P1)

Specyfikacja w [05 FE3](05_EKSPERYMENTY.md). **Gotowe, gdy:** rekord DF3; odczyt GUARD #1 razem z F‑base; jeśli DF3 = stop, eskalacja z diagnozą i propozycja otwarcia Fazy 2.

### F10 — Nowy holdout i QA (P1, H3)

Kroki w [06 §6–7](06_DANE_HOLDOUT_QA.md). **Gotowe, gdy:** rejestr v6 aktywny dla nowych runnerów; manifesty serii F ponownie zaudytowane względem v6; część testowa zapieczętowana (SHA qrels w logu, katalog z ograniczonym dostępem); część dev dodana do SEL‑OOD w rekordzie decyzji; raport QA z błędami per warstwa.

### F11 — FE5 i FE6 (P1, H2, H5)

Specyfikacja w [05](05_EKSPERYMENTY.md). Pobieranie dodatkowych obrazów puli Wikimedia wymaga H5. **Gotowe, gdy:** rekordy DF4 i DF5.

### F12 — Trening finalny, eksport, pakiet zgodności (P2, H4–H6)

**Kroki**
1. Wniosek H5 (koszt, oczekiwany zysk).
2. Trening finalny wg DF1–DF5 na `f-data-v1` (lub `v2`) i puli U, seedy 42/43/44; α i checkpoint na SEL.
3. Eksport (dwa wyjścia tekstowe, jeśli h ≠ identyczność), licznik parametrów, latencja CPU/GPU, test identyczności PolDense na eksporcie.
4. Odczyt GUARD #3 z F‑base; wniosek H4; odczyt TEST (finał) z F‑base i, po H1, z REF NC.
5. Karta modelu PL/EN (w tym pochodzenie danych treningowych i poziom WEIGHTS/STRICT), pakiet licencyjny (MIT / Apache‑2.0 / Gemma ToU, NOTICE, atrybucje Wikimedia); wniosek H6.

**Gotowe, gdy:** raport końcowy spełnia [02 §6](02_PROTOKOL_EWALUACJI.md) lub jawnie dokumentuje niespełnione warunki; H6 zatwierdzone albo oczekuje.

## 3. Pakiety Fazy 2 (seria M; osobna wersja modelu)

### M01 — Harness retencji tekstu i dane replay (P2, H1)

**Kroki:** wybrać i zamrozić podzbiór PIRB (6–8 zadań różnych typów, lista w protokole v3‑M przed treningiem) i odłożone części danych replay dostarczonych przez właściciela (polskie MS MARCO, FiQA, SQuAD v2, TriviaQA, FEVER; 10 trudnych negatywów na pozytyw); zweryfikować licencje źródeł w rejestrze (zakres TRAIN‑DATA Fazy 2); zmierzyć oryginalny PolDense‑32M jako referencję (nDCG@10 per zadanie, SHA osadzeń); zdefiniować wskaźnik kompatybilności. **Gotowe, gdy:** protokół v3‑M zamrożony, referencja policzona.

### M02 — ME1: LoRA w dwóch trybach (P2, H5)

Specyfikacja w [05 ME1](05_EKSPERYMENTY.md). **Gotowe, gdy:** rekord DM1 z wynikiem SEL, panelem retencji co N kroków, decyzją o trybach.

### M03 — ME2: siostrzana wieża i decyzja o wspólnych wagach (P2)

Specyfikacja w [05 ME2](05_EKSPERYMENTY.md). **Gotowe, gdy:** rekord DM2: który wariant Fazy 2 (jeśli którykolwiek) trafia do eksportu „wersja M”, z pełnym panelem retencji i GUARD.
