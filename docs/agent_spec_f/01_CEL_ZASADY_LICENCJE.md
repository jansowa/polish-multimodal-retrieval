# 01 — Cel, twarde zasady, limit parametrów, licencje

## 1. Cel i definicja „PolDense nienaruszony”

- **Produkt:** enkoder dwuwieżowy kodujący polski tekst i obraz do wspólnej przestrzeni wyszukiwania. W inferencji nie działa żaden VLM ani LLM.
- **Kierunek główny:** polskie zapytanie → obraz (t2i). **Drugi:** obraz → tekst (i2t). Oba raportowane osobno.
- **Faza 1 (seria F) — wymóg nadrzędny:** wagi wieży tekstowej PolDense są **bitowo identyczne** z przypiętą rewizją (`OPI-PIB/PolDense-32M`, `ca70494a25dad90d980db7c735658dc7bf88ed25`, albo większy wariant z własną rewizją). Dotyczy to także tokenizera, prefiksu `[query]: ` i poolingu dla wyjścia tekst↔tekst.
  - **Wolno:** dodać wieżę obrazu i jej głowicę; dodać głowicę/adapter **po** wyjściu PolDense albo na jego zamrożonych stanach tokenów (głowica tekstowa h, FE4), pod warunkiem że surowe wyjście PolDense pozostaje w eksporcie i jest zwracane dla trybu tekst↔tekst; używać nauczycieli i mostów w treningu.
  - **Nie wolno:** zmieniać żadnego tensora PolDense, dodawać adapterów wewnątrz jego bloków (także LoRA o zerowym wpływie), zmieniać prefiksów trybu tekst↔tekst, zmieniać długości kontekstu ani precyzji w sposób zmieniający wyjście tekst↔tekst.
  - **Dowód:** test identyczności z [02 §5](02_PROTOKOL_EWALUACJI.md) w każdym `result.json` i w eksporcie.
- **Faza 2 (seria M):** osobna wersja modelu, w której PolDense może być modyfikowany; obowiązuje bramka retencji ([02 §6](02_PROTOKOL_EWALUACJI.md)) i gwarancja trybu tekstowego (adapter wyłączony = oryginał). Faza 2 nie zastępuje artefaktów Fazy 1.
- Który model jest „najmocniejszy”, rozstrzyga wyłącznie kryterium z [02 §6](02_PROTOKOL_EWALUACJI.md), zapisane przed pierwszym wynikiem serii F.

## 2. Słowa kluczowe

| Słowo | Znaczenie |
|---|---|
| **MUSI** | obowiązek bez wyjątków |
| **NIE WOLNO** | zakaz bez wyjątków |
| **POWINIEN** | zachowanie domyślne; odstępstwo wymaga uzasadnienia w raporcie |
| **MOŻE** | dopuszczalne, jeśli nie łamie innych reguł |

Przy konflikcie z `AGENTS.md` lub `training/DATA_SPLITS.md` wygrywa reguła bardziej restrykcyjna.

## 3. Limit parametrów i licznik

- **Próg twardy:** mniej niż **100 000 000** parametrów w eksportowanym modelu inferencyjnym.
- **Wliczamy:** wieżę obrazu z natywną projekcją, wieżę tekstu z tablicą embeddingów, głowice, adaptery, normalizacje — wszystko, co jest w eksporcie. Dwa wyjścia tekstowe (surowe PolDense i głowica h) liczą się raz dla ciała i raz dla głowicy.
- **Nie wliczamy:** skali i biasu funkcji celu, nauczyciela, mostów `W`/`g` używanych wyłącznie do produkcji celów treningowych — pod warunkiem, że fizycznie nie ma ich w eksporcie.
- **Liczenie:** automatycznie ze `state_dict` eksportu (unikalne tensory; wagi współdzielone raz). Wynik w `result.json` jako `inference_parameters` z rozbiciem `vision / text / text_head / image_head / other / total`.
- **Test licznika:** R1 = 70 667 520 = PolDense‑32M 31 883 136 + TinyCLIP‑39M wizja 38 587 392 + `Linear(512, 384)` 196 992.
- **Kombinacje w budżecie (liczby z `results.json`, do potwierdzenia licznikiem):** PE‑Core‑S16 wizja 23,78M + PolDense‑32M ≈ 56M; PE‑Core‑S16 + PolDense‑68M (≈68M, wyjście 512‑D) ≈ 92M; ViTamin‑S (22,03M) lub OpenVision‑S16‑384 (21,96M) + PolDense‑68M ≈ 90M; TinyCLIP‑61M/32 (61,41M) + PolDense‑32M ≈ 93,5M. **Poza budżetem:** TinyCLIP‑39M + PolDense‑68M ≈ 107M.

## 4. Dwa poziomy „komercyjnie użyteczny”

Właściciel uznaje (31.10/01.11.2026), że R1 jest dziś najlepszym komercyjnie użytecznym modelem < 100M. Wynika z tego **domyślny** poziom WEIGHTS. Poziom STRICT jest opcją do włączenia w H1.

| Poziom | Warunki | Skutek dla planu |
|---|---|---|
| **WEIGHTS (domyślny)** | każdy składnik **wag inferencji** ma status `CLEARED` lub `CONDITIONAL` z zaakceptowanymi obowiązkami; każdy rekord treningowy ma zapisane pochodzenie (licencja obrazu/tekstu, URL, autor, `native-original / native-recycled / translated / generated`, model generujący) | dotychczasowe dane (FULL, VN, Wiki) i R1 pozostają dopuszczalne; pochodzenie jest zapisywane, nie egzekwowane |
| **STRICT** | dodatkowo: każdy model użyty do wytworzenia danych lub sygnału (generator, tłumacz, sędzia, filtr, nauczyciel) i każde źródło rekordów ma status dopuszczalny; manifest zawiera tylko rekordy spełniające warunek | dane z generatora o nieznanym statusie, VN i FULL wypadają z manifestu; R0/R1/R3 stają się `RESEARCH_ONLY`; główną pulą są Wikimedia (teksty `native-original`) i obrazy bez podpisów do destylacji |

Agent **nie rozstrzyga** kwestii prawnych. Zapisuje pochodzenie każdego rekordu i każdego sygnału tak, aby manifest STRICT dało się wyprowadzić z manifestu WEIGHTS filtrem, bez nowego audytu. Model wydany na poziomie WEIGHTS ma w karcie jawny opis pochodzenia danych treningowych.

## 5. Rejestr licencyjny i wstępna klasyfikacja

**Plik:** `licensing/commercial_register-v1.json`; dowody w `licensing/evidence/<id>/<revision>/` (karta przy przypiętej rewizji, LICENSE, data, SHA256). Format rekordu jak w specyfikacji K §2 (pola `id`, `kind`, `revision`, `declared_license`, `evidence[]`, `scopes{}`, `obligations[]`, `decision_id`, `decided_status`). Zakresy: `INFERENCE`, `INIT`, `TEACHER`, `BRIDGE` (dopasowanie mostów), `FILTER`, `DATA-GEN`, `TRAIN-DATA`, `EVAL-INTERNAL`, `REF` (wyłącznie porównanie w leaderboardzie).

Wstępna klasyfikacja (propozycja do H1; dowody = pola `license` w `results.json` i karty HF z 31.10.2026):

| Składnik | Dowód | Proponowany status | Uwagi |
|---|---|---|---|
| `OPI-PIB/PolDense-17M/-32M/-68M` | karta HF: `license: gemma`; 32M rev `ca70494a…` | `CONDITIONAL` (INFERENCE, INIT) | Gemma Terms of Use: przeniesienie ograniczeń, kopia warunków, NOTICE, Prohibited Use Policy. 68M = ModernBERT 19 warstw, wymiar 512 |
| `wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M` (rev `07a4b0bc…`), `-40M-32-…-LAION400M`, `-61M-32-…-LAION400M`, `-8M-16-…` | karty HF: MIT | `CLEARED` (INFERENCE, INIT) | prawa do danych pretreningu nieaudytowane (jak w `VISION_COMMERCIAL_STUDY.md`) |
| `facebook/PE-Core-S16-384` (rev `aabf3b99…`), `PE-Core-T16-384` (rev `8b472274…`) | karty HF: apache-2.0; README perception_models: „all checkpoints Apache 2.0” | `CLEARED` (INFERENCE, INIT) | wizja 23,78M / 6,14M; 384 px |
| `jienengchen/ViTamin-S` (rev `a3316130…`) | karta HF: MIT | `CLEARED` (INFERENCE, INIT) | wizja 22,03M |
| `UCSC-VLAA/openvision-vit-small-patch16-384` (rev `74f6c001…`), `-small-patch8-224`, `-tiny-patch8-384` | karty HF: Apache-2.0 | `CLEARED` (INFERENCE, INIT) | zero‑shot PL ≈ 0 — przed użyciem wizji sprawdzić poprawność adaptera na zadaniu angielskim |
| `unum-cloud/uform3-image-text-english-small` | karta HF: Apache-2.0 | `CLEARED` (INIT), opcjonalny | wizja 21,77M |
| `google/embeddinggemma-2` (rev `914f7f89…`) | karta HF: apache-2.0; obowiązuje Gemma Prohibited Use Policy | `CLEARED` (TEACHER, BRIDGE, FILTER) | nie wchodzi do produktu; CUDA 41,7 ms/obraz |
| `timm/ViT-B-16-SigLIP-i18n-256` (rev `ec35f945…`) | Apache-2.0 w `models.py` | `CLEARED` po snapshotcie karty (FILTER, REF) | drugi głos filtra; 370,6M |
| `google/siglip2-*`, `alphaedge-ai/siglip2-*-pol-*` | Apache-2.0 | `REF`; ewentualnie `TEACHER` | > 100M |
| `alphaedge-ai/metaclip-2-worldwide-*-pol-*`, `facebook/metaclip-2-*` | karty: cc-by-nc-4.0 | **`REF` wyłącznie** (ocena w leaderboardzie jako poprzeczka) | NIE WOLNO: INIT, TEACHER, BRIDGE, FILTER, DATA-GEN. Zgoda na nowe oceny REF — decyzja H1 pkt 3 |
| `apple/MobileCLIP2-*` | Apple AMLR (research only) | `BLOCKED` poza `REF` | |
| `M-CLIP/XLM-Roberta-Large-Vit-B-16Plus` | brak deklaracji tekstu | `UNKNOWN` | tylko wynik historyczny |
| Endpoint `qwen3.8-27b` (tłumacz, judge, recycle, recheck) | alias nie identyfikuje wag | `UNKNOWN` dla DATA‑GEN; **dokumentacja P0** | na poziomie WEIGHTS nie blokuje użycia istniejących tekstów; na STRICT blokuje; nowe generowanie wymaga H1 |
| NLLB-200-1.3B (MT Flickr30k PL), `opus-mt-en-zlw` (MT VizWiz PL) | do zweryfikowania | `EVAL-INTERNAL` | tylko ewaluacja |
| Wikimedia Commons / Wikidata (allowlista CC0, CC BY, CC BY‑SA per rekord) | licencja, autor, URL per rekord | `CONDITIONAL` (TRAIN-DATA) | atrybucja; skutki BY‑SA dla zbioru i modelu — H1 |
| VisualNews (zdjęcia prasowe) | lokalny audyt: prawa do oceny | WEIGHTS: `CONDITIONAL` (TRAIN-DATA, bez redystrybucji); STRICT: `UNKNOWN` | pochodzenie zapisane per rekord |
| FULL: ARO (Visual Genome), SugarCrepe, M‑BEIR COCO | obrazy Flickr o mieszanych licencjach; splity `test` źródeł | WEIGHTS: `CONDITIONAL` (TRAIN-DATA); STRICT: `UNKNOWN` | |
| Korpusy tekstowe do mostu `g_text` i replay (MS MARCO PL, FiQA PL, SQuAD v2 PL, TriviaQA PL, FEVER PL, FineWeb‑2 PL) | licencje źródeł do zweryfikowania (MS MARCO: badawcza; SQuAD CC BY‑SA 4.0; TriviaQA Apache‑2.0; FEVER CC BY‑SA 3.0; FineWeb‑2 ODC‑By) | `BRIDGE` / `TRAIN-DATA` do decyzji H1 | most nie wchodzi do eksportu; replay dotyczy tylko Fazy 2 |
| Zbiory ewaluacyjne: XM3600, Flickr30k PL, VizWiz PL, XTD10, VN dev, Wiki native dev, COCO val, panel ARO/SugarCrepe | — | `EVAL-INTERNAL` | bez redystrybucji; potwierdza H1 |

Procedura dodania składnika i szkic obowiązków do pakietu zgodności: jak w specyfikacji K §5–6 (MIT: tekst licencji; Apache: LICENSE/NOTICE, oznaczenie zmian; PolDense: Gemma ToU; dane Wikimedia: polityka atrybucji — H6).

## 6. Zasoby i środowisko

- WSL, istniejące środowiska Pythona, **jedna RTX 5070 Ti 16 GB** współdzielona z innymi sesjami; blokada GPU `gpu_lock()` z `experiments/data_ood_common_20261030/common.py` (POSIX flock na `training/runs/tinyclip-visualnews-v1/worker.lock`).
- Presja RAM hosta przerywa CUDA mimo wolnej VRAM (`training/VISION_COMMERCIAL_STUDY.md`): runy wznawialne, ekstrakcja cech w osobnych workerach (`training/cache_commercial_backbone.py`, `training/observe_commercial_study.py`).
- Dysk współdzielony: duże zapisy przez `reserve(...)` z tego samego API; nigdy ręczna edycja rezerwacji.
- Środowisk NIE WOLNO aktualizować w trakcie runów; nowe zależności tylko w nowym, wersjonowanym środowisku po zgodzie człowieka. Implementacja PE‑Core wymaga `facebookresearch/perception_models` (rewizja `3e352cca…` z `results.json`) — używać już zarejestrowanego adaptera z benchmarku.
- API tłumaczeń/sędziego: osobny zasób o nieznanym budżecie; użycie do **nowych** danych wymaga H1 i H5.
- Limity: run > 24 GPU‑h lub seria > 72 GPU‑h wymaga H5; czas szacuje się po smoke teście.

## 7. Twarde zasady

### 7.1. Dane, pliki, procesy (z `AGENTS.md`)
1. NIE WOLNO usuwać danych ani plików; „wykluczenie” = brak rekordu w nowym manifeście z zapisanym powodem.
2. NIE WOLNO zatrzymywać cudzych procesów ani zmieniać zamrożonych generacji.
3. NIE WOLNO edytować `src/pl_mmb_training/holdouts.py` ani starego `retrieval_benchmark/catalog.json`.
4. Przed dopuszczeniem danych: `PYTHONPATH=src python -m pl_mmb_training.current_holdouts --manifest <manifest.jsonl>`; kod 2 = blokada. Nowe rejestry (v5, v6) to nowe pliki, aktywowane jawnie w `current_holdouts.DEFAULT_REGISTRY` tylko dla nowych runnerów.
5. Chronione przed treningiem: wszystkie obrazy galerii (także bez zapytań), podpisy EN/PL, negatywne podpisy walidacyjne, 1 159 obrazów zarezerwowanych komponentów Wiki, XM3600, Wiki‑F dev, kandydaci i część testowa nowego holdoutu.
6. NIE WOLNO naprawiać wycieku przez usuwanie przykładów z benchmarku; bramka nie „oczyszcza” historycznych modeli.
7. NIE WOLNO publikować niczego (HF Space, wag, zbiorów) bez H6.

### 7.2. Licencje
8. NIE WOLNO używać składnika `BLOCKED`/`UNKNOWN` w zakresie INIT, TEACHER, BRIDGE, FILTER, DATA‑GEN. Modele NC (`REF`) wolno jedynie oceniać w leaderboardzie, po zgodzie H1 pkt 3.
9. Kandydat komercyjny MUSI być trenowany wyłącznie na manifeście `f-data-v*` zatwierdzonym w H2, z polem `tier` (`WEIGHTS`/`STRICT`) per rekord.
10. Każdy nowy tekst generowany modelem (podpisy w stylu zapytań, regeneracja opisów) wymaga statusu generatora `CLEARED` w zakresie DATA‑GEN albo decyzji H1 dopuszczającej konkretny endpoint.

### 7.3. Ewaluacja i selekcja
11. Role zbiorów i limity odczytów z [02](02_PROTOKOL_EWALUACJI.md) są wiążące; NIE WOLNO niczego wybierać na GUARD ani TEST.
12. Każdy odczyt GUARD/TEST trafia do logu ([07 §6](07_OPERACJE_ARTEFAKTY.md)).
13. NIE WOLNO: raportować jako wyniku głównego średnich mieszających t2i z i2t albo dev z testem; mieszać protokołów (stare lokalne COCO, MMEB, pełne galerie); mieszać generacji (R3 G1 ≠ R3 G2); wybierać najlepszego seeda.
14. W Fazie 1 każdy `result.json` MUSI zawierać pozytywny test identyczności PolDense ([02 §5](02_PROTOKOL_EWALUACJI.md)); brak = wynik nieważny.

### 7.4. Eksperymenty
15. Każdy run ma config ([07 §3](07_OPERACJE_ARTEFAKTY.md)) z SHA manifestu, rejestrów, protokołu, kodu i wag startowych, seedem, budżetem prezentacji i regułą wyboru checkpointu zapisaną przed startem.
16. Jedna zmiana naraz względem kontroli; wyjątki tylko zadeklarowane w [05](05_EKSPERYMENTY.md).
17. Hiperparametry z góry w configu; bez gridów LR. Zmiana = nowy eksperyment z nowym ID.
18. Nowe artefakty wyłącznie w nowych, wersjonowanych katalogach; NIE WOLNO nadpisywać.

## 8. Bramki ludzkie (znaczenie jak w specyfikacji K)

| ID | Rozstrzyga | Kto | Agent przygotowuje |
|---|---|---|---|
| H1 | poziom WEIGHTS/STRICT; statusy składników, generatora, źródeł; zgoda na oceny REF modeli NC; zgoda na korpusy mostu/replay | właściciel / prawnik | dowody, propozycja statusów, obowiązki |
| H2 | dopuszczenie manifestu `f-data-v*` | właściciel | raport audytu, wynik guarda, liczby, QA |
| H3 | annotacja holdoutu i QA | annotatorzy / właściciel | wytyczne, narzędzie `reviewer/`, próbki |
| H4 | zgoda na odczyt TEST | właściciel | lista modeli, uzasadnienie, licznik |
| H5 | większy trening, GPU ponad limit, API, pobieranie puli | właściciel | koszt, oczekiwany zysk, ryzyka |
| H6 | pakiet zgodności i publikacja | właściciel / prawnik | karta modelu, NOTICE, atrybucje |

Agent nie obchodzi bramek. Zadania niezależne od zablokowanej bramki MOŻE kontynuować.

## 9. Kiedy przerwać i eskalować

- składnik lub źródło bez statusu w rejestrze w wymaganym zakresie;
- `current_holdouts` zwraca kod 2 albo brakuje oryginałów/fingerprintów;
- test identyczności PolDense negatywny w jakimkolwiek runie Fazy 1 — stop natychmiast, raport przyczyny;
- konieczność edycji zamrożonego pliku lub zmiany rejestru poza zaplanowanym pakietem;
- wynik GUARD/TEST nieoczekiwanie wysoki (ponad +10 p.p. bez znanego mechanizmu) — przed raportem kontrola wycieku;
- przerwania CUDA/OOM po dwóch wznowieniach;
- dalsza praca wymagałaby złamania reguły z §7.

Format eskalacji: [07 §8](07_OPERACJE_ARTEFAKTY.md).
