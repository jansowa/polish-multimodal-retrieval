# 02 — Protokół ewaluacji v3‑F

## 1. Zasady ogólne

- Protokół MUSI być zamrożony przed pierwszym wynikiem serii F (pakiet F01): commit + SHA w `protocols/protocol-v3-f.md` i `protocols/protocol-v3-f.json`.
- Zmiana protokołu = nowa wersja z datą i uzasadnieniem, wprowadzana tylko między seriami. Zmiana progów po zamrożeniu unieważnia status TEST.
- Liczby z różnych protokołów (stare lokalne COCO/MMEB, pełne galerie v2, połówki VizWiz) raportuje się w osobnych tabelach.
- Scorer: `src/pl_mmb_benchmark/full_gallery.py` — pełna galeria, qrels wielopozytywne, FP32, oba kierunki, R@1/5/10, MRR, nDCG@10. Tryb promptów `v2_no_instructions` jest trybem głównym (identyczny z `author_recommended` dla naszych modeli).

## 2. Role zbiorów

| Rola | Zbiory (obrazy / teksty) | Do czego służy |
|---|---|---|
| **SEL‑ID** | Wiki native dev v2 (397 / 331); po F05 dodatkowo Wiki‑F dev (~1 000 obrazów, teksty `native-original` A/B) | wybór checkpointów, wariantów, α, alokacji |
| **SEL‑OOD** | VizWiz‑PL‑A (połowa komponentów, seed podziału 20261101), COCO val (419 / 2 090), VN dev (2 000 / 2 003); po F10 dodatkowo część dev nowego holdoutu (krótkie zapytania i zdania — osobne kolumny) | jak wyżej |
| **GUARD** | VizWiz‑PL‑B, Flickr30k‑PL (952 / 4 746), XM3600 PL (3 597 / 7 111), XTD897 (897 / 897), panel kompozycyjny (5 602 pary walidacyjne ARO/SugarCrepe), w Fazie 2 panel retencji tekstu | wyłącznie nieinferiority potwierdzonych kandydatów; limit odczytów |
| **TEST** | część testowa nowego holdoutu (F10) | ocena końcowa; najwyżej 2 odczyty |
| **REF** | wyniki historyczne bez nowych treningów: MetaCLIP‑2 S16/M16/B16/L14 PL 32768 (NC), EG2, mSigLIP, SigLIP2, modele natywne EN, R0/R1/R3; nowe oceny modeli NC tylko po H1 pkt 3 | kontekst i poprzeczki; nie są celem optymalizacji ani selekcji |

- COCO val i VN dev są w serii F zbiorami, których **domena** była w treningu R1 (FULL zawiera M‑BEIR COCO; VN train i VN dev pochodzą z jednego pipeline’u). Pozostają SEL‑OOD względem nowych źródeł, ale ich wagę w SEL ogranicza wzór z §3. W wariancie STRICT (bez FULL/VN w manifeście) są prawdziwym OOD.
- XM3600 był raz oceniony (31.10.2026) dla 7 modeli i modeli natywnych; ten odczyt liczy się jako GUARD #0. Pozostaje chroniony przed treningiem.
- Panel kompozycyjny: odsetek par walidacyjnych z `score(obraz, pozytyw) > score(obraz, negatyw)`, osobno dla relacji, atrybutów i SugarCrepe, remisy liczone jako 0,5; ślepy baseline = 50%. Zbiór zawiera część treningową ARO/SugarCrepe, więc panel jest kontrolą, nie testem niezależnym.
- Jeśli H1 wykluczy zbiór z użytku wewnętrznego, wypada on z protokołu w następnej wersji, nie w trakcie serii.

## 3. Wynik selekcyjny SEL

```text
SEL = ½ · średnia(t2i R@1 na zbiorach SEL‑ID) + ½ · średnia(t2i R@1 na zbiorach SEL‑OOD)
```

- Do SEL wchodzi wyłącznie **t2i**. Wartość i2t liczy się równolegle jako `SEL_i2t` i służy do rozstrzygania remisów oraz jako guardrail (§6).
- Remis (|Δ SEL| < 0,3 p.p.) rozstrzyga kolejno: `SEL_i2t`, prostszy wariant, wcześniejszy checkpoint.
- Dopóki nie istnieje Wiki‑F dev, różnice |Δ SEL‑ID| < 2 p.p. traktuje się jako remis (331 tekstów to za mało). Po dodaniu części dev holdoutu jej dwie kolumny (zapytania, zdania) wchodzą do średniej SEL‑OOD jako dwa osobne zbiory.
- Wszystkie runy serii F MUSZĄ używać tej samej definicji SEL. Zmiana składu SEL (dodanie Wiki‑F dev lub holdout‑dev) następuje tylko między eksperymentami i jest zapisana w rekordzie decyzji; porównania między eksperymentami wykonuje się po przeliczeniu kontroli na nowy skład.

## 4. Metryki i statystyka

- **Bootstrap:** 10 000 losowań całych komponentów, seed 20261023 (`paired_bootstrap` w `full_gallery.py`); CI różnicy jest warunkowe (nie obejmuje wyboru recepty).
- **Seedy:** screening 42; potwierdzenie 42/43/44. Raport: mean ± SD seedów i bootstrap per seed. NIE WOLNO wybierać seeda.
- **Hubność i2t:** skośność rozkładu N10 (ile razy tekst pojawia się w top‑10 sąsiadów obrazów) i odsetek tekstów‑hubów; liczone na zbiorach SEL.
- **Latencja:** `src/pl_mmb_benchmark/efficiency.py`, ta sama maszyna: CPU FP32 batch 1, 2 wątki; GPU BF16 batch 64; 10 rozgrzewek, 100 pomiarów, mediany. Odniesienie: R0/R1 CPU obraz 77,1 ms, tekst 14,5 ms; CUDA 2,10 / 0,28 ms.
- **Parametry:** z eksportu ([01 §3](01_CEL_ZASADY_LICENCJE.md)).
- **Truncation:** odsetek tekstów przyciętych do 128 tokenów PolDense per zbiór, raportowany zawsze.

## 5. Dowód nienaruszenia PolDense (Faza 1)

- Sonda: 1 000 tekstów `protocols/poldense_identity_probe_v1.jsonl` (różne długości, prefiks `[query]: `, także teksty bez prefiksu), zamrożona w F01 z SHA.
- Referencja: osadzenia sondy oryginalnym `OPI-PIB/PolDense-32M` (rev `ca70494a…`) w FP32, batch 1 i batch 32, zapisane jako `protocols/poldense_identity_reference_v1.pt` + SHA256 tensora.
- Test w każdym runie Fazy 1 i w eksporcie: wyjście trybu tekst↔tekst modelu MUSI mieć **identyczne tensory** (SHA256 równe; dodatkowo maks. |Δ| = 0 w FP32 przy tym samym batchu; przy innej precyzji/batchu dopuszczalne maks. |Δ| ≤ 1e‑5 z adnotacją).
- Wynik w `result.json` → `poldense_identity: {probe_sha256, reference_sha256, max_abs_diff, passed}`. `passed=false` unieważnia run.
- Dla większego wariantu PolDense (68M) tworzy się osobną referencję z własną rewizją.

## 6. Kryterium „najmocniejszy komercyjny < 100M” (ustalone z góry)

**Warunki konieczne**
1. < 100 000 000 parametrów eksportu.
2. Składniki i manifest spełniają poziom licencyjny wybrany w H1 (WEIGHTS lub STRICT); test identyczności PolDense pozytywny (Faza 1).

**Kryterium główne (TEST = nowy holdout; punkt odniesienia F‑base = R1, 3 seedy, ten sam protokół)**
3. t2i R@1 ≥ F‑base + 3,0 p.p. z dolną granicą 95% CI (bootstrap po komponentach) powyżej 0; liczone osobno dla krótkich zapytań i dla zdań: obie różnice ≥ 0 i co najmniej jedna ≥ +3,0.
4. i2t R@1 ≥ F‑base − 1,0 p.p.

**Poprzeczki zewnętrzne (REF, modele NC; raportowane na GUARD XM3600 i na TEST, jeśli REF zostaną na TEST ocenione w tym samym odczycie)**
- **Poziom A:** kandydat ≥ MetaCLIP‑2 S16 PL 32768 (56,0M) na XM3600 t2i (46,0) i na co najmniej 3 z 5 zadań Flickr/VizWiz/Wiki/VN/XTD897 (t2i).
- **Poziom B:** kandydat ≥ MetaCLIP‑2 M16 PL 32768 (93,7M) na XM3600 t2i (50,3).
- Poziomy A/B nie są warunkami zaliczenia; są publicznie deklarowanym celem i wchodzą do raportu końcowego z CI.

**Guardraile (GUARD, nieinferiority względem F‑base)**
5. XM3600 t2i i i2t: najwyżej −1,0 p.p.; Flickr30k‑PL i VizWiz‑PL‑B: najwyżej −1,5 p.p.; XTD897 i panel kompozycyjny: najwyżej −2,0 p.p.
6. Retencja tekstu: Faza 1 — wyjście tekst↔tekst identyczne (§5). Faza 2 — panel retencji (podzbiór PIRB + odłożone części danych replay): średnie nDCG@10 najwyżej −0,5 p.p., żadne zadanie poniżej −1,0 p.p.

**Koszt inferencji**
7. CPU ms/obraz i ms/tekst ≤ budżet ustalony przez właściciela przed FE3; do tego czasu domyślnie ≤ 2× R1 (obraz ≤ 154 ms, tekst ≤ 29 ms). Uwaga: wieże 384 px (PE‑Core‑S16) mogą przekroczyć budżet obrazu — wtedy decyzja właściciela albo wariant 224 px (FE6).

**Raportowane informacyjnie:** dystans do EG2 (73,8% średniej 5 zadań, XM 68,2/76,6), do mSigLIP (58,8%), do modeli `REF` komercyjnych > 100M (siglip2‑pl‑32768 45,2%).

## 7. Limity odczytów

- **GUARD:** najwyżej 3 odczyty na serię F (plus GUARD #0 z 31.10). Odczyt obejmuje tylko kandydatów potwierdzonych na 3 seedach, zawsze razem z F‑base; każdy odczyt w logu ([07 §6](07_OPERACJE_ARTEFAKTY.md)).
- **TEST:** najwyżej 2 odczyty łącznie (kamień milowy po FE3 lub FE6 i finał); każdy wymaga H4; lista modeli deklarowana przed odczytem: kandydat, F‑base, opcjonalnie EG2 i REF NC (po H1).
- **Skażenie:** jeśli wynik TEST doprowadzi do zmiany recepty, TEST staje się dev, a nowy test wymaga nowej annotacji.
- **Wybory tylko na SEL:** checkpoint, α, wariant głowicy, most, filtr, próg, alokacja parametrów.
- Miary czysto tekstowe (FE0: sufity, hubność, prefiksy) wolno liczyć na zbiorach SEL bez limitu; na GUARD/TEST — nie.

## 8. Raportowanie

- t2i i i2t zawsze osobno, per zbiór; „średnia wszystkiego” nie jest wynikiem głównym (średnia 5 zadań z leaderboardu może być podana jako kontekst z tą etykietą).
- Każda tabela: parametry (V / T / głowice), latencja, poziom licencyjny, SHA manifestu, flaga `COMMERCIAL_CANDIDATE` / `COMMERCIAL_BASELINE` / `RESEARCH_ONLY` / `REF`, wynik testu identyczności PolDense.
- Fakty, hipotezy i interpretacje wyraźnie oznaczone.

## 9. Wymagana wielkość TEST

Błąd standardowy różnicy dwóch modeli ≈ √(d/n), gdzie d = odsetek zapytań z różnym wynikiem (zwykle ~0,2), n = liczba zapytań. Dla n = 1 000: ~1,4 p.p.; dla n = 2 400: ~0,9 p.p. Aby wykazać +3 p.p., część TEST POWINNA mieć ≥ 1 500 zapytań t2i; zalecane ~2 400 (ok. 600 obrazów testowych × 2 annotatorów × 2 typy tekstu — [06 §7](06_DANE_HOLDOUT_QA.md)).
