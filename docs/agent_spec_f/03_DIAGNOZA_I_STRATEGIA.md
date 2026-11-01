# 03 — Diagnoza i strategia

## 1. Stan wyjściowy w liczbach (`results.json`, tryb `v2_no_instructions`, R@1 w %, t2i / i2t)

| Model | Licencja | Parametry V+T | Flickr | VizWiz | Wiki | VN | XTD897 | XM3600 | Średnia 5 zadań* |
|---|---|---:|---|---|---|---|---|---|---:|
| **R1 (3 seedy, mean)** | MIT + Gemma | 70,47M (+0,20M head) | 48,9 / 52,2 | 21,7 / 21,6 | 15,1 / 13,7 | 74,5 / 69,8 | 50,9 / 47,1 | 38,6 / 37,0 (seed44) | 41,4 (seed44) |
| R0 (3 seedy) | MIT + Gemma | 70,47M | 47,6 / 51,4 | 21,4 / 20,4 | 15,7 / 14,3 | 73,9 / 70,0 | 51,4 / 46,3 | 38,5 / 35,8 (seed44) | 41,0 |
| R3 G1 (3 seedy) | MIT + Gemma | 70,47M | 47,9 / 51,1 | 20,6 / 20,6 | 17,2 / 15,8 | 72,4 / 68,3 | 51,1 / 46,6 | 37,4 / 36,7 (seed44) | 40,8 |
| MetaCLIP‑2 S16 PL 32768 | **CC BY‑NC** | 56,01M | 45,6 / 65,3 | 31,7 / 36,2 | 36,6 / 40,1 | 68,6 / 68,7 | 47,5 / 50,9 | 46,0 / 58,2 | 49,1 |
| MetaCLIP‑2 M16 PL 32768 | **CC BY‑NC** | 93,69M | 51,8 / 71,3 | 35,1 / 43,4 | 42,6 / 44,8 | 74,7 / 75,0 | 53,8 / 54,5 | 50,3 / 62,7 | 54,7 |
| siglip2‑pl‑32768 | Apache | 204,38M | 52,8 / 74,3 | 35,5 / 49,9 | 14,5 / 16,1 | 38,7 / 41,5 | 63,4 / 65,4 | — | 45,2 |
| mSigLIP i18n‑256 | Apache | 370,63M | 59,7 / 79,7 | 45,5 / 54,1 | 35,0 / 39,3 | 73,5 / 77,2 | 60,1 / 63,8 | — | 58,8 |
| EmbeddingGemma‑2 (nauczyciel) | Apache | 438,76M | 76,0 / 89,5 | 58,5 / 73,1 | 55,0 / 59,2 | 93,8 / 90,8 | 72,6 / 69,7 | 68,2 / 76,6 | 73,8 |
| PE‑Core‑S16‑384 natywny (EN) | Apache | 87,19M (wizja 23,78M) | 3,5 / 10,0 | 10,8 / 19,3 | 13,0 / 17,1 | 8,2 / 10,5 | 6,6 / 9,6 | 4,7 / 9,3 | 10,9 |
| TinyCLIP‑39M natywny (EN) | MIT | 83,10M (wizja 38,59M) | 2,9 / 7,7 | 4,2 / 9,8 | 7,9 / 12,8 | 6,9 / 10,1 | 4,2 / 5,9 | 2,7 / 6,1 | 7,2 |

\*średnia leaderboardu: najpierw średnia kierunków, potem równe wagi Flickr/VizWiz/Wiki/VN/XTD897 — kontekst, nie wynik główny.

Fakty wynikające z tabeli:
- Wśród modeli **komercyjnie użytecznych < 100M** R1 jest najlepszy z dużym zapasem; następny (PE‑Core‑S16 z angielską wieżą tekstu) ma 10,9. Modele natywne EN nie rozumieją polskiego — ich wartość to wyłącznie wieże wizji.
- Poprzeczki NC: S16 (56,0M) wygrywa z R1 na 4 z 5 zadań i o 7,4 p.p. w XM t2i; M16 (93,7M) o 11,7 p.p. w XM t2i i 25,7 p.p. w XM i2t.
- R1 przegrywa z M16 nawet w domenach, na których był trenowany: XTD897 (COCO) 50,9 vs 53,8 t2i; VN dev (zbiór selekcyjny) 74,5 vs 74,7 t2i i 69,8 vs 75,0 i2t.
- R1 ma i2t < t2i (XM: 37,0 < 38,6; VN: 69,8 < 74,5; XTD: 47,1 < 50,9), podczas gdy każda referencja ma i2t wyraźnie wyżej.
- Przyrosty danych w obecnej recepcie są małe: R0→R1 (+19 876 obrazów VN) = +1,25 p.p. Flickr, +0,33 VizWiz, +0,55 VN dev, −0,48 XTD (t2i, 3 seedy); Wiki G2 (+3 539 obrazów) = +1,4 Wiki dev przy −1,5 VN dev.

## 2. Diagnoza (hipotezy robocze z poziomem pewności; weryfikują je FE0–FE3)

1. **Klasa funkcji, nie strojenie** (pewność wysoka). 196 992 uczonych parametrów między dwiema zamrożonymi wieżami nie wchłania więcej danych ani sygnału nauczyciela. LR/schedule/LoRA/MLP/KD na tej recepcie dały ≤ +0,7 p.p. z CI zawierającym zero. Dlatego seria F zmienia klasę funkcji (odmrożona wizja, mosty, głowica tekstowa), a nie hiperparametry.
2. **Geometria zamrożonej przestrzeni tekstowej ogranicza i2t i kompozycyjność** (pewność średnio‑wysoka). Galerią w i2t są osadzenia PolDense; żadna głowica po stronie obrazu ich nie rozsunie. Mediany cos(prawdziwy, fałszywy podpis ARO) w PolDense: 0,995 / 0,987. Jedyne dźwignie bez ruszania wag: reprojekcja/whitening po stronie tekstu (FE4) i wyższy wymiar (PolDense‑68M, 512‑D). Sufit mierzy FE0.
3. **Wieża wizji i pokrycie domen** (pewność wysoka dla Wiki/VizWiz). TinyCLIP‑39M (YFCC‑15M, IN‑1k 63,5%) vs wieże z wielokrotnie większego pretreningu (PE‑Core‑S16: IN‑1k 72,7% przy 23,78M). Luka Wiki −27,5 p.p. t2i do M16 nie zniknie od 3,5k obrazów Wiki.
4. **Dane: wiersze ≠ obrazy; podpisy to w 88% tekst modelowy; ~1/3 budżetu idzie w ARO** (pewność wysoka dla księgowości, średnia dla wpływu). Sampler jednostajny po obrazach z limitem ARO wchodzi do bazowej recepty serii F (jednorazowa zmiana, bez dalszego strojenia).
5. **Selekcja po VN dev nie przewidywała generalizacji** (proces wykazany). Obowiązuje protokół v3‑F.

## 3. Strategia Fazy 1 — „zamknięty tekst, uczona wizja, mosty do nauczyciela”

Trzy dźwignie, w kolejności kosztu informacji:

**Dźwignia 1 — głowica po stronie tekstu h (FE4), zero ryzyka dla tekstu.** Z tego samego zamrożonego przebiegu PolDense obliczana druga projekcja (whitening w formie zamkniętej, uczona liniowa, rezydualny MLP od zer, attention‑pooling na zamrożonych tokenach). Obrazy są dopasowywane do H‑space = h(P‑space). Surowe wyjście P‑space pozostaje dla tekst↔tekst. Koszt produktowy: osobny indeks tekstów dla wyszukiwania obrazów.

**Dźwignia 2 — wieża wizji i jej odmrożenie (FE1, FE3).** Drabina komercyjnych wież (PE‑Core S16/T16, ViTamin‑S, OpenVision‑S16, TinyCLIP 40/61) z zamrożoną sondą liniową wybiera inicjalizację; potem odmrożenie z WiSE‑FT. Odmrożenie bez nowych danych nie pomagało (10,6k obrazów); z celami z nauczyciela i obrazami bez podpisów ma czym się uczyć.

**Dźwignia 3 — mosty do nauczyciela i obrazy bez podpisów (FE2, FE3, FE5).** EmbeddingGemma‑2 ma wspólną przestrzeń tekst‑obraz, więc dla dowolnego obrazu można wyprodukować cel w P‑space/H‑space przez most: `W_pairs` (uczony na parach, jak ramię A5 w K), `g_text` (uczony na samym tekście — miliony polskich zdań — bez obrazów) albo `W_hybrid` (g_text jako start, dostrojony na parach). To odsprzęga trening wizji od wąskiego gardła zweryfikowanych polskich podpisów: pula Wikimedia 417k obrazów i VN 283k obrazów mogą uczyć wizji bez tekstu (po bramce holdoutów i po H1 dla praw do obrazów).

Trening dwuetapowy: **S1** — wizja (odmrożona) + głowica uczą się celów z mostu na puli U i obrazach par (strata cosinusowa lub relacyjna — FE3 porównuje); **S2** — pary z wielopozytywnym sigmoidem i destylacją afiniczności od EG2; na końcu WiSE‑FT α wybrane na SEL.

Elementy stałe serii F: loss sigmoid (normalizacja przez B, uczona skala i bias), sampler jednostajny po obrazach z limitem ARO ≤ 10% prezentacji, budżety w prezentacjach, cache cech dla zamrożonych części, nauczyciel nigdy na GPU razem z uczniem.

Dlaczego nie ścieżka „adaptacja MetaCLIP‑2”: licencja CC BY‑NC wyklucza INIT/TEACHER/BRIDGE/FILTER. MetaCLIP służy wyłącznie jako poprzeczka REF.

## 4. Drzewo decyzji

**DF0 — po F02 i H1:** poziom WEIGHTS → manifest `f-data-v1` = FULL + VN (użyte) + Wiki ready po bramce; poziom STRICT → manifest tylko z rekordów spełniających STRICT (głównie Wiki `native-original`), F‑base przeliczany na nowy manifest jako `COMMERCIAL_BASELINE`, a R1 oznaczony `RESEARCH_ONLY`. Zgoda na REF NC → nowe oceny MetaCLIP dozwolone; brak zgody → tylko wyniki historyczne.

**DF1 — po FE0 i FE1 (wizja, wymiar, bariera):**
- Inicjalizacja wizji = najwyższy SEL spośród komercyjnych ramion w budżecie parametrów i latencji; różnica < 1 p.p. → szybszy model.
- Wskaźnik bariery `R = SEL‑OOD(ramię EG2‑obraz + linear do P‑space) / SEL‑OOD(EG2 zero‑shot)`: `R ≥ 0,90` → P‑space wystarcza, głowica h opcjonalna; `R < 0,80` → h obowiązkowa w FE3; pomiędzy → oba warianty.
- Sufity FE0 dla 68M wyższe o ≥ 5 p.p. od 32M na SEL (lub wyraźnie w i2t) → wariant 68M wchodzi do FE6 jako równorzędny kandydat; inaczej 68M tylko jako ramię kontrolne.

**DF2 — po FE2 i FE4 (most i głowica):** wybór generatora celów S1 (`W_pairs` / `g_text` / `W_hybrid`) po jakości celów na odłożonych parach; wybór h po SEL i hubności przy pozytywnym teście identyczności.

**DF3 — po FE3 (czy odmrożenie + destylacja działa):** najlepsze ramię ≥ kontrola + 3,0 p.p. SEL przy guardrailach → FE5/FE6 i wniosek H5; `D1 − D0 < 1,0` → KD nic nie wnosi, zostaje samo odmrożenie; wszystkie ramiona < kontrola + 1,0 → stop skalowania, eskalacja z diagnozą i **otwarcie Fazy 2 jako jedynej drogi do celu**. Po DF3 odczyt GUARD #1.

**DF4 — po FE5 (dane):** czy inwestować w obrazy bez podpisów, w pary, czy w filtr (reguły w [05](05_EKSPERYMENTY.md)).

**DF5 — po FE6:** konfiguracja finalna na froncie SEL × CPU ms × parametry (32M vs 68M, 224 vs 384 px).

**DF6 — H4/H5/H6:** trening finalny, odczyt TEST, pakiet zgodności.

## 5. Zarys Fazy 2 (seria M; osobna wersja modelu)

Warunek wejścia: DF3 zamknięte (sukces lub stop) **i** właściciel dostarczył dane tekstowe (polskie MS MARCO, FiQA, SQuAD v2, TriviaQA, FEVER z 10 trudnymi negatywami na pozytyw) **i** decyzja, czy produkt wymaga jednej wspólnej przestrzeni.

- **M01 — harness retencji:** stały podzbiór PIRB (6–8 zadań różnych typów) + odłożone części danych replay; nDCG@10; progi: średnio ≤ −0,5 p.p., pojedyncze zadanie ≤ −1,0 p.p.; wskaźnik kompatybilności (cos nowych i starych osadzeń, R@1 stary‑indeks/nowe‑zapytanie).
- **ME1 — LoRA w dwóch trybach:** adapter (rząd 8–16) na ciele PolDense + nowy prefiks `[image]: ` dla tekstów w zadaniach z obrazem; strata = sigmoid obraz‑tekst + λ_t·strata retrievalowa na replayu (10 trudnych negatywów + negatywy z batcha, prefiksy jak w PolDense: `[query]: ` dla zapytań, pasaże bez prefiksu) + λ_d·kotwica do zamrożonego oryginału na tekstach replayu. Adapter wyłączony = bitowo oryginalny PolDense (tryb tekstowy). Wdrożenie dwutrybowe jest gwarancją; scalenie do jednego trybu tylko po pozytywnym panelu retencji na pełnym PIRB.
- **ME2 — siostrzana wieża:** jeśli LoRA nie daje ≥ +3 p.p. SEL nad zwycięzcą Fazy 1 lub nie mieści się w progu retencji: osobna kopia PolDense‑17M/32M trenowana swobodnie do obrazów przy zamrożonym PolDense‑32M do tekstu (32 + 17 + 22 ≈ 71M).
- Decyzje DM1 (LoRA vs siostrzana vs pozostanie przy Fazie 1) i DM2 (dwa tryby vs wspólne wagi).

## 6. Czego nie robić

1. Strojenia LR, MLP ani LoRA na zamrożonym moście TinyCLIP→PolDense. Sampler zmienia się raz (jednostajnie po obrazach, limit ARO) i nie jest dalej strojony.
2. Używania MetaCLIP‑2, MobileCLIP2 ani innych składników `BLOCKED`/`UNKNOWN` w jakimkolwiek kroku treningowym, destylacji czy filtrze.
3. Selekcji na GUARD/TEST; optymalizacji średniej leaderboardu.
4. Trenowania na trudnych negatywach ARO w Fazie 1 (zamrożony tekst ich nie rozróżnia; w ablacji ARO pogarszał VN). Powrót do tematu w Fazie 2.
5. Generowania nowych podpisów (także „w stylu zapytań”) endpointem bez statusu DATA‑GEN albo decyzji H1.
6. Promowania checkpointów `RESEARCH_ONLY` do kandydata; promowania artefaktów Fazy 2 jako „Fazy 1”.
7. Pobierania całej puli 417k obrazów Wikimedia przed wynikiem FE5 (skala U) i H5.
