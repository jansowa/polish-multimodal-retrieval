# Specyfikacja agentowa F — komercyjny polski enkoder tekst↔obraz < 100M przy nienaruszonych wagach PolDense

**Wersja:** F-1.2 · **Stan wiedzy:** 01.11.2026 (załącznik KONTEKST z 31.10.2026, `results.json` z 01.11.2026, snapshot repo `e1b9676`) · **Repozytorium:** `jansowa/polish-multimodal-retrieval`
**Proponowane miejsce w repo:** `docs/agent_spec_f/` (wszystkie pliki w jednym katalogu; linki względne). Praca agenta przebiega w małych sesjach wg [09](09_PROTOKOL_SESJI_CODEX.md): jedna sesja = jeden pakiet lub krok, stan w `reports/f_series/STATUS.md` i handoffach.
**Relacja do specyfikacji K** (`docs/agent_spec_k/`, inny przegląd z 31.10.2026): ta specyfikacja przejmuje z niej bramki ludzkie H1–H6, role zbiorów i format operacyjny, a różni się w kilku decyzjach opisanych w [08](08_MAPOWANIE_NA_SPEC_K.md). Przy sprzecznych zapisach obowiązuje **F**, a tam, gdzie F milczy — K.

## Cel

Zbudować **najmocniejszy komercyjnie użyteczny** polski enkoder tekst↔obraz o **mniej niż 100 000 000 parametrów inferencji**, w dwóch fazach:

- **Faza 1 (seria F, priorytet):** wagi `OPI-PIB/PolDense-32M` (lub większego wariantu PolDense) pozostają **bitowo nienaruszone**. Uczymy się wyłącznie wieży obrazu, głowic/adapterów poza PolDense oraz mostów do nauczyciela. Oryginalne wyjście PolDense do wyszukiwania tekst↔tekst musi być w eksporcie dostępne i identyczne z oryginałem.
- **Faza 2 (seria M, później, osobna wersja modelu):** PolDense modyfikowany pod kontrolą retencji (LoRA w dwóch trybach, replay na polskich zbiorach retrievalowych, kotwica destylacyjna). Faza 2 startuje dopiero po decyzji DF3 z Fazy 1 i po dostarczeniu przez właściciela danych tekstowych.

Poprzeczką są modele niekomercyjne, których nie wolno użyć w treningu: MetaCLIP‑2 S16 PL 32768 (56,0M) i M16 PL 32768 (93,7M). Wśród modeli komercyjnie użytecznych poniżej 100M najlepszy jest dziś własny R1 (41,4% średniej 5 zadań).

## Kolejność czytania (obowiązkowa)

| Zanim… | Przeczytaj |
|---|---|
| wykonasz jakąkolwiek akcję | 01, 02 |
| zaplanujesz lub uruchomisz eksperyment | 03, 05, 07 |
| dotkniesz danych, manifestów lub rejestru holdoutów | 06 oraz `training/DATA_SPLITS.md`, `AGENTS.md` |
| napiszesz raport, rekord decyzji lub prośbę o decyzję | 07 |
| porównasz tę specyfikację z `docs/agent_spec_k/` | 08 |
| zaczynasz lub kończysz sesję agenta | 09 (protokół sesji, handoff, szablony promptów) |

## Spis treści

1. [01 — Cel, twarde zasady, limit parametrów, licencje](01_CEL_ZASADY_LICENCJE.md)
   - §1 Cel i definicja „PolDense nienaruszony” · §2 Słowa kluczowe · §3 Limit parametrów i licznik · §4 Dwa poziomy „komercyjnie użyteczny” (WEIGHTS / STRICT) · §5 Rejestr licencyjny i wstępna klasyfikacja · §6 Zasoby · §7 Twarde zasady · §8 Bramki ludzkie H1–H6 · §9 Kiedy przerwać
2. [02 — Protokół ewaluacji v3‑F](02_PROTOKOL_EWALUACJI.md)
   - §1 Zasady · §2 Role zbiorów · §3 Wynik selekcyjny SEL · §4 Metryki i statystyka · §5 Dowód nienaruszenia PolDense · §6 Kryterium „najmocniejszy komercyjny < 100M” i poprzeczki NC · §7 Limity odczytów · §8 Raportowanie · §9 Wielkość TEST
3. [03 — Diagnoza i strategia](03_DIAGNOZA_I_STRATEGIA.md)
   - §1 Stan wyjściowy w liczbach · §2 Diagnoza · §3 Strategia Fazy 1 (trzy dźwignie) · §4 Drzewo decyzji DF0–DF6 · §5 Zarys Fazy 2 · §6 Czego nie robić
4. [04 — Pakiety pracy](04_PAKIETY_PRACY.md)
   - §1 Ścieżka krytyczna · §2 Pakiety F01–F12 · §3 Pakiety M01–M03 (Faza 2)
5. [05 — Eksperymenty](05_EKSPERYMENTY.md)
   - §0 Zasady wspólne · FE0 sufity przestrzeni tekstowej · FE1 drabina wież wizji · FE2 cele z nauczyciela (mosty) · FE3 destylacja S1/S2 z odmrożoną wizją · FE4 głowica po stronie tekstu · FE5 dane: ilość, jakość, obrazy bez podpisów · FE6 alokacja parametrów i rozdzielczość · ME1/ME2 (Faza 2)
6. [06 — Dane, holdout, QA](06_DANE_HOLDOUT_QA.md)
   - §1 Warstwy danych i poziomy licencyjne · §2 Ścieżka dopuszczenia · §3 Wikimedia · §4 Pula obrazów bez podpisów · §5 Sygnały nauczyciela · §6 Ludzki QA · §7 Nowy holdout · §8 Wiki‑F dev · §9 Korpusy tekstowe do mostu · §10 Co zbierać dalej
7. [07 — Operacje, artefakty, raporty](07_OPERACJE_ARTEFAKTY.md)
   - §1 Cykl pracy · §2 Katalogi · §3 Config · §4 Wyniki · §5 Raport · §6 Log odczytów · §7 Rekord decyzji · §8 Eskalacja · §9 `STATUS.md` · §10 Higiena · §11 Definicja ukończenia
8. [08 — Mapowanie na specyfikację K](08_MAPOWANIE_NA_SPEC_K.md)
9. [09 — Protokół sesji Codex i prompt stały](09_PROTOKOL_SESJI_CODEX.md)
   - §1 Zasady sesji (bez gałęzi i PR) · §2 Identyfikatory i pliki stanu (`NEXT.md`) · §3 Szablony handoffu i `NEXT.md` · §4 Plan sesji · §5 Prompt stały z jednym polem · §6 Sygnały, że sesja jest za duża

## Status wyjściowy (co już istnieje — agent weryfikuje w repo przed startem)

Statusy pochodzą z załącznika KONTEKST (31.10.2026), `results.json` (01.11.2026) i snapshotu repo. „Gotowe” nie oznacza „zatwierdzone do treningu”.

| Obszar | Stan | Dowód / gdzie |
|---|---|---|
| Benchmark retrieval PL v2 (7 galerii gotowych + DOCCI planowane), wspólny scorer, bootstrap, leaderboard | **gotowe** | `retrieval_benchmark/`, `src/pl_mmb_benchmark/full_gallery.py`, `results.json`: 50 modeli, 584 wyniki |
| XM3600 PL oceniony (R0/R1/R3 seed44, MetaCLIP S16/M16/B16‑32768, EG2, modele natywne EN) | **gotowe**; liczy się jako odczyt GUARD #0 serii F | `results.json`, `reports/benchmark_decontamination_20261031/xm3600_completion.json` |
| Rejestr holdoutów v4 (`current_holdouts`), generacja Wiki2 przypięta do v2 | **gotowe** | `src/pl_mmb_training/current_holdouts.py`, `training/DATA_SPLITS.md` |
| Rejestr holdoutów v5 (Wiki‑F dev, kandydaci holdoutu) i v6 (nowy holdout) | do zrobienia | F05, F10 |
| Modele frozen‑linear R0/R1/R3 G1 (3 seedy), VN_only, A1 — w leaderboardzie; R3 G2: w leaderboardzie tylko `R3_wiki2_seed43`, 3 seedy wyłącznie w raporcie G2 | **gotowe**, wyniki w leaderboardzie | `results.json`; `reports/data_study_20261030/wiki_recovery_20261031/REPORT.md` |
| Pomiar latencji w harnessie: R0_seed44 (CPU obraz 77,1 ms, tekst 14,5 ms), większość referencji (w tym PE‑Core‑B16) | **częściowo** (brak dla PE‑Core S16/T16, TinyCLIP natywnych, ViTamin, OpenVision, uform3) | `results.json` → `efficiency`, `src/pl_mmb_benchmark/efficiency.py` |
| Kandydaci wież wizji zarejestrowani z przypiętą rewizją i licencją: PE‑Core S16/T16 (Apache), TinyCLIP 8/39/40/61M (MIT), ViTamin‑S (MIT), OpenVision S16‑384/S8‑224/T8‑384 (Apache), uform3 small (Apache) | **gotowe jako referencje zero‑shot** (angielskie wieże tekstu → 0–11% po polsku, zgodnie z oczekiwaniem); **brak** cache cech i sondy z PolDense | `results.json`; historyczne studia wizji (`training/VISION_STUDY.md`, `VISION_COMMERCIAL_STUDY.md`) dotyczyły tylko FULL i COCO val |
| Nauczyciel `google/embeddinggemma-2` zarejestrowany, oceniony, latencja zmierzona (CUDA 41,7 ms/obraz) | **gotowe**; cache nauczyciela dla danych treningowych: do zrobienia | `results.json`; F04 |
| Pule danych: VN 165 316 par „ready” / 283 012 w puli; Wiki 46 968 par „ready” / 417 353 obrazów; status `stopped_by_request` | **gotowe w pipeline’ie, niedopuszczone** | `reports/visualnews_pipeline_state.json`, `reports/wikimedia_pipeline_state.json` |
| Dane użyte dotąd: FULL 70 289 wierszy / 10 645 obrazów; VN 33 906 + 19 888 = 53 794 par (liczby z załącznika, do potwierdzenia w manifestach serii R); Wiki G2 3 539 obrazów | **gotowe (zamrożone)** | KONTEKST §3–4; `training/runs/tinyclip-visualnews-v1/data/`, manifesty `data_study_20261030` |
| Audyt licencyjny 30.10 (dossier) | **częściowo**; rejestr z zakresami i poziomami: do zrobienia | `reports/series_2026-10-23/LICENSING_DOSSIER_20261030.md`; F02 |
| Tożsamość i warunki generatora `qwen3.8-27b` | nieustalone | F02 |
| Sufity przestrzeni tekstowej (oracle), hubness, głowica po stronie tekstu | do zrobienia | F03, F08 |
| Panel ARO/SugarCrepe (5 602 pary walidacyjne) | dane gotowe, panel niepoliczony | F01 |
| Nowy holdout (zdjęcia własne + Commons, ludzkie zapytania), ludzki QA | do zrobienia | F10 |
| Harness retencji tekstu (PIRB) i dane replay (MS MARCO PL, FiQA, SQuAD v2, TriviaQA, FEVER z 10 negatywami) | do zrobienia; dane dostarcza właściciel | M01 |

## Ścieżka krytyczna

| Etap | Pakiety | Bramki | GPU | Uwagi |
|---|---|---|---|---|
| 1 | F01, F02, F03 | H1 (licencje, poziom WEIGHTS/STRICT) | ~0 | protokół, rejestr, sufity tekstowe — wszystko bez treningu |
| 2 | F05 → F04, F06 | H2 (manifest `f-data-v1`) | ekstrakcja cech i cache nauczyciela | F05 musi poprzedzać każdy trening na puli Wiki |
| 3 | F07 (FE1) → F08 (FE2 + FE4) | — | ~1–2 dni | decyzje DF1, DF2 |
| 4 | F09 (FE3) | — | ~2–3 dni | decyzja DF3; odczyt GUARD #1 |
| 5 | F10 (równolegle od etapu 1) | H3 | 0 | ludzie: zdjęcia, annotacja, QA; pieczęć TEST |
| 6 | F11 (FE5, FE6) | H2, H5 | ~2–4 dni | decyzje DF4, DF5 |
| 7 | F12 | H4, H5, H6 | ~1 dzień | trening finalny 3 seedy, GUARD #3, TEST, pakiet zgodności |
| 8 | M01 → M02 → M03 (Faza 2) | H1 (dane), H5 | ~2–4 dni | tylko po DF3 i po dostarczeniu danych tekstowych |

## Pierwsze kroki agenta

1. **F01:** zamroź protokół v3‑F, zbuduj licznik parametrów eksportu (test: R1 = 70 667 520), kalkulator SEL, log odczytów GUARD/TEST z wrapperem, test bitowej identyczności wyjścia PolDense, panel ARO/SugarCrepe. Policz SEL dla istniejących R1 seed 42/43/44 i dla EG2.
2. **F02:** utwórz rejestr licencyjny v1 z zakresami i dwoma poziomami (WEIGHTS/STRICT); udokumentuj generator `qwen3.8-27b`; przygotuj wniosek H1.
3. **F03 (FE0):** sufity centroidów i miary geometrii dla PolDense‑17M/32M/68M na zbiorach SEL; sprawdź prefiks `[sts]: `.
4. **F05, potem F04:** rejestr v5 (Wiki‑F dev, kandydaci holdoutu) → cache cech wież wizji i cache nauczyciela.
5. **F06:** manifest `f-data-v1` przez bramkę v5, wniosek H2.
6. **F07 (FE1):** drabina wież wizji → rekord decyzji DF1.

Każdy krok kończy się wpisem w `reports/f_series/STATUS.md`.

## Decyzje, które czekają na właściciela

1. **Poziom komercyjny (H1):** WEIGHTS (licencje wag inferencji; dotychczasowe dane i R1 pozostają dopuszczalne) czy STRICT (także pochodzenie danych, generator, tłumacz, nauczyciel). Domyślnie w tej specyfikacji: **WEIGHTS**, z pełnym zapisem pochodzenia, tak aby wariant STRICT dało się później wyprodukować bez ponownego audytu.
2. Akceptacja Gemma Terms of Use dla PolDense i polityki użycia EmbeddingGemma‑2 jako nauczyciela (H1).
3. Zgoda na ocenianie modeli NC (MetaCLIP‑2) jako referencji REF — bez żadnego użycia w treningu, destylacji i filtrach (H1).
4. Budżet latencji (przed FE3): czy 384 px po stronie obrazu jest dopuszczalne (PE‑Core‑S16), czy obowiązuje 224 px.
5. Dostępność ludzi: zdjęcia własne, annotacja, QA (H3). Budżet GPU/API ponad limity (H5).
6. Faza 2: czy jedna wspólna przestrzeń tekst+obraz jest wymogiem produktu, czy wystarczą dwa tryby/dwa indeksy; dostarczenie danych tekstowych replay.

## Słownik

| Termin | Znaczenie |
|---|---|
| SEL / SEL‑ID / SEL‑OOD | zbiory i wynik selekcyjny (02 §2–3) |
| GUARD / TEST / REF | kontrola nieinferiority z limitem odczytów / zapieczętowany test / wyniki referencyjne bez nowych treningów |
| WEIGHTS / STRICT | poziomy „komercyjnie użyteczny” (01 §4) |
| `f-data-v*` | zatwierdzony w H2 manifest danych serii F z polami pochodzenia i poziomu |
| F‑base | istniejący R1 (seed 42/43/44) przeliczony na protokół v3‑F; punkt odniesienia Fazy 1 |
| P‑space / H‑space | przestrzeń wyjściowa PolDense / przestrzeń po głowicy tekstowej h (FE4) |
| most (bridge) | mapa z przestrzeni nauczyciela do P‑space lub H‑space: `W_pairs`, `g_text`, `W_hybrid` (FE2) |
| S1 / S2 | etap destylacji wizji na obrazach bez podpisów / etap treningu na parach z KD afiniczności |
| U | pula obrazów bez podpisów po bramce holdoutów |
| WiSE‑FT α | interpolacja wag między punktem startowym a modelem po treningu |
| identyczność PolDense | test SHA256 osadzeń 1 000 tekstów sondy przed i po (02 §5) |
| DF0–DF6 / DM1–DM2 | rekordy decyzji Fazy 1 / Fazy 2 |
| H1–H6 | bramki ludzkie, znaczenie jak w specyfikacji K |
