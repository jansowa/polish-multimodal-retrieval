# Polski embedder multimodalny — dokument do niezależnej konsultacji

Stan na **20 października 2026**, po eksperymentach A–F i serii LR/warmup/LP→FT. Dokument opracowano z lokalnego kodu, manifestów i raportów. Liczniki procesów danych są datowanymi odczytami lokalnych plików statusu, nie potwierdzeniem bieżącego stanu odrębnego serwera. Dodatkowy odczyt manifestów: **21:50 czasu Europe/Warsaw**.

Dokument jest samodzielnym kontekstem dla innego modelu: zawiera wyniki i ograniczenia bez konieczności dostępu do repozytorium. Ścieżki na końcu służą audytowi, jeżeli repozytorium będzie dostępne. Wyniki procentowe oznaczają procent poprawnych zapytań; różnice podajemy w punktach procentowych. `±` w historycznych tabelach oznacza odchylenie między seedami, nie przedział ufności.

**Najważniejsze fakty w jednym miejscu:**

- Obecny trening ma 70 289 wierszy, ale tylko **10 645 niezależnych target images według pikseli**; VisualNews nie uczestniczy jeszcze w treningu.
- Mapping frozen SigLIP2→PolDense z sigmoid osiągnął historycznie **49,70 ± 0,70% test MSCOCO**, a mały TinyCLIP→PolDense około **45%**; protokół jest lokalny i in-domain.
- Ostatnie TinyCLIP KD wygrało val (**58,76%**), lecz dostało **44,50% test**. Najlepsze późniejsze partial FT ma **58,71% val**, tylko **+0,14 p.p.** nad odpowiadającą kontrolą frozen; nie ma jego nowego testu.
- Nie znaleziono znanego overlapu train/eval images/parents, ale audyt ma granice, source domains współdzielą COCO, część tekstów train/val jest wspólna, a test historycznie wielokrotnie oglądano.
- **XM3600 pozostaje bez wyników** i nie służy do wyboru ustawień. Największe otwarte kwestie to niezależna walidacja, wartość nowych danych oraz dobór lossu/samplingu do małej liczby różnych obrazów.

## 1. Zadanie dla modelu konsultującego

Prosimy o niezależną ocenę celu, danych, protokołu i eksperymentów. Należy rozdzielić obserwacje od hipotez oraz wskazać potencjalne błędy implementacji lub metodologii. Nie zakładać, że dotychczasowe interpretacje autora projektu albo agenta wykonującego eksperymenty są poprawne.

Najważniejsze pytanie: **jak najlepiej rozszerzyć mały, dobry polski encoder tekstowy o obrazy, zachowując jego oryginalną przestrzeń tekstową, przy niewielkiej liczbie niezależnych obrazów treningowych?** Czy obecny bottleneck to dane, loss, optymalizacja, architektura mapowania, czy ewaluacja?

Oczekiwany rezultat konsultacji: diagnoza oparta na poniższych dowodach oraz krótki, uporządkowany plan kolejnych eksperymentów. Na końcu dokumentu znajdują się konkretne pytania i aktualne ograniczenia.

## 2. Cel i obowiązujące decyzje

- Główny produkt: **polskie zapytanie tekstowe → wyszukanie obrazu**. Image→text jest dodatkowym kryterium i diagnostyką.
- Zachować oryginalny `OPI-PIB/PolDense-32M`, jego preprocessing i embedding tekstowy. Dotychczas wszystkie eksperymenty mają **frozen text tower**; nie ma trenowanego mapowania po stronie tekstu.
- Dążyć do małego modelu i sensownej jakości, a nie wyłącznie maksymalnego wyniku dużego encodera. Ważne są również warunki wykorzystania checkpointów; same deklaracje MIT/Apache dla vision nie przesądzają warunków całego produktu.
- Bieżąca mała architektura referencyjna: **TinyCLIP-39M vision + PolDense-32M + linear 512→384**.
- **Nie zmieniać aktualnego train/validation snapshotu.** Ostatnie eksperymenty używały wyłącznie istniejącego `FULL_CLEAN`.
- Aktualnie wybierać checkpointy i konfiguracje po **validation retrieval**, nie po testowym MSCOCO.
- **XM3600-PL pozostaje FINAL BLIND.** Nie wolno używać go do strojenia ani uruchamiać bez osobnej decyzji użytkownika.
- VisualNews jest przygotowywany/oznaczany poza bieżącymi treningami; nie czekaliśmy na niego z eksperymentami modelowymi. Żadne nowe pary VisualNews nie weszły do opisanych treningów.
- CIRR/FashionIQ i grounding to osobne zdolności; nie należy zastępować dobrego bazowego t2i wysokim wynikiem ich średniej.

## 3. Architektura i loss

### 3.1. Text tower i model obrazowy

**PolDense-32M:** 31 883 136 parametrów, embedding **384D**, natywny pooling Sentence Transformers, prefiks `[query]: `, maksymalnie 128 tokenów w obecnej recepcie. Rewizja: `ca70494a25dad90d980db7c735658dc7bf88ed25`.

**TinyCLIP-39M:** `wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M`, rewizja `07a4b0bc751cb64fecd2b661c048c1dd98d69444`. Używamy wyłącznie vision tower i jej oryginalnej CLIP projection; oryginalna TinyCLIP text tower nie jest używana. Vision: **38 587 392 parametrów**, obraz 224×224, patch 16, 12 transformer blocks, hidden 512, MLP intermediate 2048, 8 attention heads, dropout i attention dropout 0. Końcowe cechy mają 512D.

Nowy liniowy projector z biasem: `Linear(512,384)`, **196 992 parametrów**. Cały embedder w inference ma **70 667 520 parametrów**. Uczone skala/bias lossu dodają dwa parametry treningowe, niewymagane w inference.

Schemat:

```text
PL caption → [query]: caption → frozen PolDense → t ∈ R^384
image → native TinyCLIP vision + original projection → v ∈ R^512
v → learned Linear(512,384) → z ∈ R^384
retrieval score = cosine(t,z)
```

Przy partial finetuningu odmrażano **cały ostatni transformer block albo ostatnie dwa bloki**, nie tylko ostatnią macierz MLP w bloku. Natywna końcowa LayerNorm poza blokami i oryginalna CLIP projection pozostawały frozen.

### 3.2. Multi-positive sigmoid

Macierz logitów dla batcha B:

```text
S_ij = exp(a) · cosine(t_i, z_j) + b
y_ij = +1 dla każdej znanej pozytywnej relacji caption_i ↔ image_j
y_ij = -1 dla pozostałych par w batchu
L_gold = (1/B) · sum_ij softplus(-y_ij · S_ij)
```

To normalizacja przez **B, nie B²**. `a` i `b` są uczone. Początek: temperatura 0,07, czyli `exp(a)=1/0,07`, bias −10. Po update: `a` ograniczane do `[0,ln(100)]`, bias do `[−30,30]`. Weight decay loss scalars = 0.

Pozytywne relacje są budowane po całym danym splicie, nie tylko z przekątnej batcha. To samo zdjęcie pod aliasami jest jednym targetem według hasha pikseli. Ten sam polski caption może mieć kilka znanych obrazów pozytywnych; wszystkie są dodatnie, jeśli wystąpią w batchu.

**Granica tej definicji:** nieznana relacja zostaje negatywem. Nie wiemy, czy każdy taki negatyw jest semantycznie poprawny. Multi-positive nie eliminuje wszystkich false negatives.

ARO `false_caption` i SugarCrepe `negative_caption` są zachowane w tłumaczeniach, ale **nie były dotychczas jawnie używane w lossie**. Do treningu dodatnich par wchodzą odpowiednio `true_caption` i `caption`.

### 3.3. Wspólna recepta bazowa

Seed 42, chyba że podano 42/43/44. AdamW, batch 256, maksymalnie 20 epok, projector LR 1e-3, weight decay 0,01, brak schedulera w starszej recepcie. FULL_CLEAN daje 275 batchy na epokę, czyli **5500 kroków i 1 405 780 ekspozycji rekordów** przy 20 epokach.

Frozen cechy są cache'owane; vision extraction używa FP32 weights + BF16 autocast. Projector/loss działają w FP32. W partial FT cache obejmuje tylko niezmienną część vision. Nie stosowano dodatkowego gridu augmentacji, dropout, clippingu ani regularizacji dryfu reprezentacji. Przy tej implementacji preprocessing obrazu jest stały.

Zachowanie text↔text oznacza niezmienność samej funkcji PolDense i jej wejść. Nie należy tego mylić z wykonaniem nowego, szerokiego audytu polskich benchmarków tekstowych po każdej próbie — takich nowych wyników tu nie raportujemy.

## 4. Dane używane i dostępne

### 4.1. Faktyczny, zamrożony snapshot treningowy

| Źródło | Źródłowe rekordy EN/PL | Po filtracji train+val | Train | Validation |
|---|---:|---:|---:|---:|
| ARO Visual Relation | 23 937 | 23 226 | 20 883 | 2343 |
| ARO Visual Attribution | 28 748 | 28 049 | 25 434 | 2615 |
| SugarCrepe | 7511 | 6747 | 6103 | 644 |
| M-BEIR MSCOCO | 24 809 | 19 962 | 17 869 | 2093 |
| VisualNews w starym snapshocie | 10 083 ukończone tłumaczenia | **0 gotowych par** | 0 | 0 |
| **Razem gotowe pary** | — | **77 984** | **70 289** | **7695** |

Train: **67 445 unikalnych par PL–obraz, 59 955 unikalnych surowych captionów PL, 10 645 unikalnych obrazów według pikseli**. Validation: 7415 unikalnych par, 7145 unikalnych surowych tekstów PL, **1216 obrazów**. Łącznie 11 861 obrazów. To znacznie mniej niezależnej informacji wizualnej niż sugeruje liczba 70 tys. rekordów.

Duplicate rows/pairs nie zostały wszystkie usunięte przed treningiem: rozróżniamy 70 289 wierszy od 67 445 różnych par. Targety identycznych pikseli są jednak wspólnie rozpoznawane jako pozytywne.

Nazwy snapshotów: `training/runs/e1-full/data.jsonl`, `training/runs/clean-v1/FULL_CLEAN.jsonl`, `training/runs/clean-v1/validation.jsonl`. FULL_CLEAN zachowuje dane wcześniejszego splitu; „CLEAN” nie oznacza, że później arbitralnie usunięto wszystkie podobne do COCO zdjęcia z wariantu FULL.

### 4.2. Źródła i wersje

| Źródło | Repozytorium / wariant | Przypięta rewizja |
|---|---|---|
| ARO Relation | `mteb/ARO-Visual-Relation`, source split `test` | `53c1115ab5871ab33b09d5bb8856236d46bdaf2d` |
| ARO Attribution | `mteb/ARO-Visual-Attribution`, source split `test` | `25fe9b460426ff67cdb0608a737c94f8f49a901e` |
| SugarCrepe | `mteb/SUGARCREPE_fmt`, source split `train` | `e9e0e3ce40c94023c819d3413edb5d090ec9f0d8` |
| Lokalny eval MMEB | `TIGER-Lab/MMEB-eval`, cztery configi poniżej | `eb3a8af230a66be5c9308cd2be5779b9d6de4cb3` |
| M-BEIR | oryginalne query/candidate pools i positive qrels dla MSCOCO/VisualNews | zapisane źródła/SHA w planach i manifestach |

**Istotne ujawnienie:** używana pula M-BEIR MSCOCO pochodzi z `query/test` i task3 test candidate pool oryginalnego loadera; ARO też ma źródłowy split nazwany `test`. Projekt wydziela własny train/validation i filtruje overlap z lokalnym eval. To **nie uprawnia do nazywania oryginalnych oficjalnych testów tych korpusów niezależnymi od treningu**.

Nie trenujemy na całym COCO ani całym Visual Genome. Nie pobrano ani nie wykorzystano automatycznie wszystkich zbiorów dostępnych w rejestrze adapterów.

### 4.3. Inne lokalne dane, nieużyte w tych treningach

- **mFAQ PL:** 63 157 train + 2025 validation = 65 182 pary pytanie–odpowiedź; bez obrazów.
- **MAUPQA:** 3 660 140 wierszy pytanie–passage, 454 234 pozytywne i 3 205 906 negatywnych; to wiersze, nie liczba różnych pytań. Subsety m.in. 1z10, czy-wiesz-v2, gpt3-cc, gpt3.5-cc, gpt3.5-wiki, mkqa, mqa, multilingual-NLI, poleval2021-pairs, poquad, templates, wiki-def.
- **Winoground:** w sprawdzonym cache tylko README; nie raportować jako gotowego przetłumaczonego benchmarku. **ImageCoDe** ma adapter, ale nie gotowy korpus użyty w opisanych eksperymentach.
- Więcej configów MMEB jest zarejestrowanych, ale rejestr ≠ gotowe dane ≠ wykonana ewaluacja.
- Wspólny stary manifest cech zawiera 30 893 ścieżki obrazów i 137 904 teksty EN/PL, **łącznie z train, val i eval**. To cache obliczeniowy, nie dodatkowa pula przykładów treningowych. Indeksy train/val/test są osobne.

## 5. Filtracja i kontrola leakage

### 5.1. Powiązanie źródeł i tekstów

- Tłumaczenia są łączone z przypiętymi source rows po stabilnych identyfikatorach/row index i zgodności oryginalnych pól. Nie zakłada się zgodności kolejności eksportu.
- Eksport SQLite czyta spójną transakcję. Snapshot istniejącego runu nie rośnie wraz z tłumaczeniami.
- M-BEIR caption→image pochodzi z **positive qrels**, a nie z sąsiedztwa ID czy wierszy. Jeden rekord może rozszerzyć się na kilka pozytywnych obrazów.
- Brak wiarygodnego mapowania, brak wymaganego obrazu i brak zweryfikowanej linii pochodzenia nie stają się pozytywnymi parami.
- Do gotowych par nie są dopisywane ARO/SugarCrepe negative captions jako pozytywy.

### 5.2. Co obejmuje blacklist

Chronione są cztery lokalne benchmarki MMEB: **CIRR, FashionIQ, MSCOCO_t2i, RefCOCO**. Blacklista obejmuje:

1. ID zapytań i rekordów eval.
2. **Wszystkie obrazy zapytań i wszystkich kandydatów**, nie tylko pozycję pozytywną.
3. SHA256 bajtów pliku oraz hash zdekodowanych pikseli RGB po EXIF transpose, z uwzględnieniem rozmiaru obrazu.
4. Zweryfikowane parent IDs: COCO, Visual Genome, Flickr oraz identyfikatory odpowiednich źródeł.
5. Teksty EN i PL po `casefold` oraz scaleniu białych znaków. To exact text matching po normalizacji, nie semantyczny dedup.

ARO/VG jest mapowane do COCO przez oficjalne `image_data.json`. RefCOCO `image_<ann_id>_crop.jpg` jest mapowane z **ID adnotacji do COCO image ID** przez oficjalne `instances_train2014.json`. Nie wolno traktować numeru cropa jako ID zdjęcia. Wykrycie parent overlap usuwa całą powiązaną rodzinę cropów. Brak niezbędnego mapowania zatrzymuje przygotowanie albo powoduje wykluczenie; system nie zgaduje.

### 5.3. Rzeczywiste wykluczenia gotowych źródeł

| Źródło | Wykluczone | Powód parent | Powód text | Zachowane train+val |
|---|---:|---:|---:|---:|
| ARO Relation | 711 | 711 | 0 | 23 226 |
| ARO Attribution | 699 | 699 | 0 | 28 049 |
| SugarCrepe | 764 | 762 | 143 | 6747 |
| M-BEIR MSCOCO | 5027 | 5010 | 1024 | 19 962 |

Powody nie są rozłączne. Zero dodatkowych odrzuceń po hashach w tej tabeli nie znaczy, że hashowanie było wyłączone: rekord mógł już zostać wykluczony po parent IDs. Stary pełny audyt raportował również brakujące obrazy i brak qrels dla VisualNews; jego sumy nie są po prostu sumą gotowych źródeł powyżej.

### 5.4. Train/validation

Split jest grupowy i deterministyczny: komponenty połączonych parent IDs i hashy pikseli, seed **1701**, fraction 0,1. Powiązane cropy i aliasy pikselowe nie przekraczają granicy splitu. Odczyt 20.10 potwierdził **0 wspólnych parent IDs i 0 wspólnych canonical pixels** train/validation.

**Dodatkowy odczyt na potrzeby tej konsultacji:** split nie grupuje automatycznie całych komponentów identycznego tekstu między różnymi obrazami. Po normalizacji istnieją przecięcia captionów:

| Zakres validation | Wspólne różne EN z train | Wiersze val z takim EN | Wspólne różne PL z train | Wiersze val z takim PL |
|---|---:|---:|---:|---:|
| Wszystkie 7695 wierszy | 1033 | 1337 | 1012 | 1301 |
| Właściwe 2093 wiersze retrieval MSCOCO | 20 | 20 | 23 | 25 |

To **nie jest wspólny obraz ani automatyczny dowód instance leakage**. Oznacza natomiast, że walidacja nie mierzy wyłącznie generalizacji do nieznanych ciągów tekstowych. Wcześniejsze „brak overlapu” odnosiło się do chronionych obrazów/parent IDs i guarda eval, a nie do zera identycznych captionów między train i validation. Nie zmieniono splitu ani wyników po tym odczycie.

### 5.5. Near-duplicates

Porównano 10 685 ścieżek obrazów train (10 645 różnych pikseli) z 18 988 ścieżkami eval: **202 886 780 par**. DCT pHash 64-bit po EXIF normalizacji: brak par o odległości ≤6, minimalna odległość 8. Dla 200 najbliższych kandydatów dodano cosine z TinyCLIP; obejrzano 48 najbliższych par i nie potwierdzono nowych overlapów.

To przegląd diagnostyczny Codex, nie niezależna ludzka adjudykacja. **Nie zastosowano automatycznego odrzucania po pHash. Nie wykonano pełnego embedding-nearest-neighbor audytu.** pHash i exact hashes mogą nie wykryć cropów, mocnych transformacji i nieznanych relacji źródłowych. Audyt nie obejmuje pełnej historii danych pretrainingowych encoderów.

## 6. Ewaluacja: trzy odrębne rzeczy

### 6.1. Validation loss i validation retrieval

Validation loss jest liczone na istniejących **7695 wierszach** ze wszystkich czterech gotowych źródeł. Natomiast retrieval do wyboru konfiguracji wykorzystuje tylko **2093 rekordy M-BEIR MSCOCO**: **2090 unikalnych captionów PL i 419 obrazów**. Galeria i znane multi-positive edges są stałe.

Raportujemy t2i R@1/5/10 oraz i2t R@1/5/10. Jeden wynik t2i R@1 zmienia się o ok. **0,04785 p.p. na jedno zapytanie**. Captiony nie są niezależnymi obserwacjami: wiele odnosi się do tych samych obrazów.

Aktualny wybór checkpointu: maksymalne **val t2i R@1**, remis rozstrzyga średnia t2i/i2t R@1, następnie wcześniejsza epoka. Historyczny E1/E2 i porównania backbone’ów wybierały **minimum validation loss**. Nie wolno mieszać tych reguł w jednej deklaracji „identyczny baseline”.

### 6.2. Lokalny legacy test MMEB

Protokół: `mmeb-v1-no-instructions-raw-sum-cosine-v2`. Cztery configi, każdy **1000 zapytań**, każde z własną przypiętą pulą **1000 kandydatów**:

| Config | Zapytanie | Liczba zapytań | Unia ścieżek obrazów kandydatów |
|---|---|---:|---:|
| CIRR | obraz + tekst modyfikacji | 1000 | 2197 |
| FashionIQ | obraz + tekst modyfikacji | 1000 | 5992 |
| MSCOCO_t2i | tekst | 1000 | 1002 |
| RefCOCO | obraz + tekst w wariancie MMEB | 1000 | 8416 |

Instrukcje MMEB są usuwane po pełnym znanym prefiksie przed tokenizacją. Dla encoderów CLIP/SigLIP oraz naszego prostego fusion: zapytanie mieszane to suma **surowych** embeddingów tekstu i obrazu, z końcowym cosine. Nie normalizuje się osobno obu części przed sumowaniem. Native joint multimodal encoders, np. EmbeddingGemma, mają własną wspólną ścieżkę.

Pozytywny kandydat lokalnego zadania ma index 0; remisy rozstrzyga porządek kandydatów. Główne P@1/R@1 w tych zadaniach są równe przy pojedynczym oznaczonym pozytywie. Dodatkowo R@5/10, MRR i NDCG@10. NaN/Inf/zero vectors są odrzucane.

**To nie jest oficjalne COCO 5k ani globalny retrieval po pełnym COCO.** Lokalny RefCOCO nie jest pomiarem przewidywania bounding boxes. Raw image+text sum może korzystać z podobieństwa wizualnego bez dobrego rozumienia modyfikacji; dlatego te wyniki i AVG są diagnostyczne.

Oryginalny MMEB evaluator oferuje zarówno cosine, jak i raw dot product, a odtworzenie nieudokumentowanych flag tabeli angielskiej nie jest deklarowane. Wyniki mają własny jawny identyfikator protokołu v2.

### 6.3. Dodatkowy fixed-gallery test

`fixed-gallery-multi-positive-v1`: z przykładów lokalnego MSCOCO powstała galeria **1002 obrazów / 1000 unikalnych captionów**, 1000 zapytań tekstowych i tylko **201 zapytań obrazowych** ze znanym pozytywnym captionem. Pozostałe obrazy mogą być negatywnymi kandydatami t2i. Qrels są multi-positive. Tie policy: malejący cosine, następnie kanoniczny index galerii.

Ten wynik jest osobnym protokołem, nie prostym odwróceniem lokalnych pul i nie oficjalnym COCO 5k. Drobne różnice R@1 wobec lokalnego testu są możliwe.

### 6.4. Jak traktować historyczne testy

E2 loss selection/replikacje odbyły się po val. Jednak starszy vision scouting otwierał test każdego backbone’u, a decyzje o dalszych seedach DINOv2 i wyborze/repikacji TinyCLIP częściowo opierały się jawnie na testowym MSCOCO. Nie wybierano na nim epok ani LR, ale **nie można już uważać tego testu za całkowicie ślepy wobec rozwoju całego projektu**.

Późniejsze A–F wybierały po val; nowy test dostał tylko wybrany finalista KD. Ostatnia seria LR/warmup w ogóle nie otwierała testu. Instance leakage, domain exposure i wielokrotne oglądanie testu to różne problemy. MSCOCO traktujemy obecnie jako **in-domain / legacy**.

## 7. Początkowe treningi: wielkość danych i loss

### 7.1. Pilot i pełny E1

| Eksperyment | Vision | Train / val | Loss | Seedy | Lokalny MSCOCO R@1 |
|---|---|---|---|---:|---:|
| E1 pilot ARO 6k | frozen SigLIP2 Base/16 256 | 5435 / 565 | positive-pair cosine regression | 3 | 13,07 ± 0,71 |
| Pełny E1 | frozen SigLIP2 Base/16 256 | 70 289 / 7695 | ten sam | 3 | 41,70 ± 0,20 |

Pełny E1 seed 42/43/44: MSCOCO **41,50 / 41,90 / 41,70**; wybrane epoki **17 / 19 / 19** po minimum val loss. Średnie CIRR/FashionIQ/RefCOCO: 8,37 / 4,53 / 35,70. Pilot i FULL różnią się liczbą oraz składem danych, więc nie jest to czysta learning curve z zagnieżdżonych podzbiorów.

### 7.2. E2: zmiana samego lossu

Vision SigLIP2 Base/16 256 + frozen PolDense + linear. Te same dane, seedy, permutacje, inicjalizacje i budżet 20 epok co E1. E2a: symmetric multi-positive InfoNCE; E2b: sigmoid. Checkpoint każdego wybierał minimum własnej val loss; replikację uruchamiał ≥1 p.p. gain val t2i względem E1. Oba warianty dostały trzy seedy. Wybór lossu: średnia val t2i/i2t z wybranych checkpointów, przed testem.

| Wariant | Val t2i R@1 | Val i2t R@1 | Test MSCOCO R@1 | Seedy |
|---|---:|---:|---:|---:|
| E1 cosine regression | 55,49 ± 0,58 | 45,03 ± 0,96 | 41,70 ± 0,20 | 3 |
| E2a InfoNCE | 59,89 ± 0,60 | 74,78 ± 0,60 | **nie uruchomiono** | 3 |
| E2b sigmoid | **61,56 ± 1,05** | **75,82 ± 1,08** | **49,70 ± 0,70** | 3 |

E2b test R@5/10: **81,17 ± 0,57 / 89,53 ± 1,10**. Jego średnie CIRR/FashionIQ/RefCOCO: **9,40 / 6,20 / 61,47**. E2b epoki seed 42/43/44: **8 / 4 / 9**, temperatury ok. **0,02039 / 0,03195 / 0,01864**.

Wynik SigLIP2 49,70% dotyczy **naszego projectora do PolDense**, a nie natywnej pełnej pary SigLIP2 vision/text. W osobnym benchmarku gotowego pełnego modelu SigLIP2 Base/16 256 uzyskano 43,70%.

## 8. Frozen vision backbones — wyniki historyczne

Stałe: PolDense-32M, sigmoid, linear, te same train/val, 20 epok, LR 1e-3, batch 256, bez schedulera; checkpoint po minimum val loss. Każdy model ma swój natywny preprocessing, rozdzielczość i wymiar cech. `Vision M` liczy tylko używaną część obrazową, wraz z oryginalną projekcją, jeżeli występuje.

| Vision | Vision M | Dim | Input | Seedy | Test MSCOCO R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SigLIP2 Base/16 256 | 92,930 | 768 | 256 | 3 | 49,70 ± 0,70 | 81,17 | 89,53 |
| DINOv2 Small | 22,057 | 384 | 224 | 1 | 25,80 | 62,00 | 77,30 |
| MobileCLIP2-S2 | 35,815 | 512 | 256 | 1 | 49,10 | 78,20 | 88,80 |
| ConvNeXt Tiny | 27,820 | 768 | 224 | 1 | 20,20 | 48,40 | 63,00 |
| iBOT ViT-S/16 | 21,666 | 384 | 224 | 1 | 23,30 | 54,30 | 70,10 |
| TinyCLIP 39M/16 YFCC | 38,587 | 512 | 224 | 3 | 44,67 ± 0,42 | 78,00 | 88,00 |
| TinyCLIP 40M/32 LAION | 39,692 | 512 | 224 | 1 | 36,70 | 67,80 | 81,20 |
| Chinese-CLIP RN50 | 38,317 | 1024 | 224 | 1 | 39,10 | 70,30 | 82,40 |
| RN50 CC12M | 38,317 | 1024 | 224 | 1 | 27,70 | 59,80 | 74,80 |
| RN50 YFCC | 38,317 | 1024 | 224 | 1 | 31,90 | 63,80 | 75,70 |

Dokładne checkpointy nowych pięciu: `wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M`, `wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M`, `OFA-Sys/chinese-clip-rn50`, `timm/resnet50_clip.cc12m`, `timm/resnet50_clip.yfcc15m`.

Reprezentacje: SigLIP2 native vision pooler; DINOv2/iBOT końcowy normalized CLS; ConvNeXt GAP + final LayerNorm przed classifierem; CLIP-family standardowy projected image embedding. MobileCLIP2 zachowuje oryginalną image projection.

TinyCLIP seed 42 miał minimum val loss w epoce 6 i test **44,20%**. Później ten sam FULL_CLEAN, wybrany po val retrieval w epoce 9, dał **45,20%**. Zmiana ta nie jest dowodem zysku z dodatkowego czyszczenia danych — zmieniła się reguła wyboru checkpointu.

Ograniczenia: backbone’y różnią się również pretrainingiem, patch size i preprocessingiem; to porównanie gotowych reprezentacji, nie izolowany test metody pretrainingu. Historyczny wybór TinyCLIP jako małego kandydata uwzględniał też testową jakość i screening licencji. Raporty odnotowują MIT/Apache w kartach części wag, odrębne warunki Apple dla MobileCLIP2 oraz Gemma dla PolDense; nie jest to aktualna opinia prawna ani audyt całego łańcucha danych/licencji.

## 9. CLEAN-v1 — ablacje zależności od danych COCO

Audyt zachowanych danych nie znalazł znanego source/parent/hash overlapu z chronionymi czterema benchmarkami. Historyczne wyniki pozostają **LEGACY / pre-audit**, ale nie zostały automatycznie uznane za contaminated. Regułę checkpoint selection zmieniono na aktualne val retrieval.

Wszystkie poniższe: TinyCLIP-39M frozen + PolDense frozen + linear + sigmoid, seed 42; validation bez zmian.

| Wariant train | Rekordy | Unikalne pary | Obrazy | Val t2i R@1 | Val i2t R@1 | Test MSCOCO R@1 |
|---|---:|---:|---:|---:|---:|---:|
| FULL_CLEAN | 70 289 | 67 445 | 10 645 | 58,23 | 69,69 | 45,20 |
| NO_MBEIR_MSCOCO | 52 420 | 49 601 | 7102 | 49,09 | 56,09 | 35,20 |
| NON_COCO_CLEANROOM | 24 473 | 23 601 | 3037 | 35,79 | 44,63 | 22,30 |

NO_MBEIR usuwa całe źródło M-BEIR MSCOCO, ale pozostawia inne COCO-related dane. NON_COCO wyklucza również SugarCrepe i wszystkie znane COCO lineage/aliasy pikselowe, propagując usunięcie przez komponenty parent IDs. Brak znanego VG→COCO mappingu nie dowodzi braku każdej możliwej relacji do COCO.

Kontrola wyrównująca dostępny budżet: **5500 kroków i 1 405 780 prezentacji** dla wszystkich wariantów. Mniejsze datasety są wielokrotnie tasowane. FULL_CLEAN jest ponownie użytym runem:

| Wariant | Val t2i R@1 | Val i2t R@1 | Test MSCOCO R@1 |
|---|---:|---:|---:|
| FULL_CLEAN | 58,23 | 69,69 | 45,20 |
| NO_MBEIR_MSCOCO, matched budget | 47,51 | 55,37 | 32,40 |
| NON_COCO_CLEANROOM, matched budget | 34,11 | 44,87 | 22,00 |

Budżety to ukończone limity runów; checkpointy wybierane na val mogą pochodzić z różnych wcześniejszych kroków. Nadal różnią się liczba różnych obrazów i skład źródeł. Wyniki pokazują silną zależność od źródła/domeny/liczby różnych obrazów, **nie dowodzą instance leakage ani czystego efektu samej domeny**.

## 10. TinyCLIP A–F — nowsza seria na niezmienionym FULL_CLEAN

Wszystkie próby mają seed **42**, ten sam split i obecny wybór po val retrieval. B1 i C0 to dokładne ponowne użycie A1, nie kolejne niezależne treningi. `—` w teście oznacza brak nowego testu tego wariantu w tej serii.

| Run | Feature source | Projector | Trainable vision | Trainable params z loss | Val t2i R@1 | Val i2t R@1 | Epoka | Nowy test MSCOCO |
|---|---|---|---|---:|---:|---:|---:|---:|
| A1 | po native CLIP projection | linear | frozen | 196 994 | 58,23 | 69,69 | 9 | — |
| A2 | pooled CLS przed projection | linear | frozen | 196 994 | 57,85 | 63,01 | 2 | — |
| B1 = A1 | projected | linear | frozen | 196 994 | 58,23 | 69,69 | 9 | — |
| B2 | projected | MLP 512→768→GELU→384 | frozen | 689 282 | 54,31 | 63,72 | 3 | — |
| C0 = A1 | projected | linear | frozen | 196 994 | 58,23 | 69,69 | 9 | — |
| C1 | projected | linear | ostatni blok | 3 349 378 | 57,56 | 68,74 | 4 | — |
| C2 | projected | linear | ostatnie 2 bloki | 6 501 762 | 56,46 | 65,63 | 2 | — |
| D LoRA | projected | linear | q/v LoRA w ostatnich 2 blokach | 229 762 | 56,41 | 68,50 | 9 | — |
| E KD λ=0,1 | projected | linear | frozen | 196 994 | 58,37 | 69,93 | 9 | — |
| E KD λ=0,3 | projected | linear | frozen | 196 994 | **58,76** | **70,41** | 9 | **44,50** |
| F TinyCLIP-61M | projected | linear | frozen | 196 994 | 51,34 | 66,35 | 10 | — |

Szczegóły niezbędne do interpretacji:

- **A2** to natywny pooled CLS po końcowej native LayerNorm, ale przed oryginalną CLIP visual projection. Nie jest to średnia wszystkich patch tokens. Obydwa wejścia projectora mają tu 512D.
- **B2** ma dokładnie dwie macierze linear z GELU, bez dodatkowego LayerNorm/residual. Sprawdzono hidden 768, nie 1024. MLP osiągało niski train loss, a pogarszającą się walidację; nie uruchomiono trzech seedów.
- **C1/C2:** projector LR 1e-3, vision LR **1e-5**, od losowego projectora, stały LR, bez warmupu. AdamW, bez gradient clippingu. Early stopping patience 5, maksymalnie 20 epok. C1 zatrzymano po 9, C2 po 7. W C1 train loss spadł z 4,687 w epoce 1 do 1,622 w epoce 9, podczas gdy val loss po minimum rosło; val t2i po maksimum 57,56 spadło do 52,06. W C2 final val t2i 52,87. To negatywny wynik **tych recept**, nie dowód nieskuteczności dowolnego FT.
- **D:** uruchomione po overfittingu C. Rank **8**, alpha **16**, tylko `q_proj` i `v_proj`, ostatnie **2** bloki, LR LoRA **1e-4**, projector 1e-3; pozostałe vision weights frozen. Bez dużego gridu. Final val t2i 52,68 mimo spadku train loss; najlepsza epoka 9. Nie sprawdzono LoRA na 4 blokach ani wariantów LoRA z nową receptą aligned-head.
- **E:** teacher `google/siglip2-base-patch16-256`, student TinyCLIP-39M + PolDense. Teacher dostaje oryginalne **EN**, student **PL**. Teacher frozen, native image/text features cache. Student w tych dwóch runach miał **frozen vision**, trenował projector i loss scalars. Nie łączono KD z partial FT.
- **F:** `wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M`, jeden frozen run na najlepszej frozen recepcie A/B. Bez specjalnego strojenia. To zmiana również patch size **16→32** i pretrainingu **YFCC→LAION**, nie czysta izolacja liczby parametrów.

### 10.1. Dokładna implementacja KD

```text
U_T = cosine_matrix(teacher_EN_texts, teacher_images) / 0.07
U_S = cosine_matrix(student_PL_texts, student_projected_images) / 0.07
L_KD = 0.5 · [ KL(softmax_rows(U_T) || softmax_rows(U_S))
             + KL(softmax_rows(U_T^T) || softmax_rows(U_S^T)) ]
L = L_gold_sigmoid + λ · L_KD, λ ∈ {0.1, 0.3}
```

KL jest sumowane po kandydatach i uśredniane po query (`batchmean`). Nie ma regresji raw embeddings, kopiowania teacher vectorów, native teacher logit_scale ani dodatkowego mnożnika temperatury². Temperatura KD jest stała 0,07, niezależna od uczonej skali gold lossu. Są to oba kierunki rozkładu teacher→student, a nie symetryzacja KL przez zamianę teacher/student.

Dla λ=0,3 train gold loss wynosił ok. 4,739 w epoce 1 i 1,898 na końcu; niepomnożony KD ok. 0,356 i 0,118. Ważony dodatek KD był więc ok. 0,107 i 0,035. Te wartości opisują skalę faktycznej implementacji, nie gwarantują optymalnej kalibracji λ.

KD λ=0,3 wygrało tę serię **na val** o 0,53 p.p. wobec A1. Dopiero po zapisaniu decyzji dostało test: MSCOCO R@1/5/10 **44,50 / 79,50 / 88,60**. Historyczny FULL_CLEAN ma **45,20 / 79,10 / 88,30**. Mały zysk val nie przełożył się na testowy R@1. Nie wybierano potem innej konfiguracji na podstawie tego testu i nie otwierano XM3600.

### 10.2. Kontrole implementacji/cache

Cache A–C obejmuje 11 905 ścieżek obrazów wyłącznie z istniejących train/val (więcej ścieżek niż unikalnych pikseli). A1 zweryfikowano względem starego TinyCLIP cache: minimalny cosine **0,9999997**.

Partial FT korzysta z FP32 cache tokenów na wejściu do ostatnich dwóch bloków. Zamrożony prefix nie zmienia się; trainable suffix jest liczony z autogradem, a cechy validation są aktualizowane co epokę. Testy na małym CLIP potwierdziły zgodność native forward/suffix oraz gradienty wyłącznie w wymaganych blokach. Testy LoRA i KD potwierdziły zerowy początkowy residual LoRA, brak gradientu do teacher i możliwość porównywania similarity przy różnych wymiarach embeddings.

Przykładowy peak **allocated** CUDA VRAM: frozen A1 ok. 0,16 GiB, C1 1,57 GiB, C2 2,57 GiB, D 1,94 GiB. Nie są to całkowite zajętości urządzenia ani pełny profiling deploymentu.

## 11. Dodatkowe 10 prób LR/warmup i LP→FT

Motywacja: początkowe C miało stały vision LR i losowy projector. Sprawdziliśmy, czy łagodniejsza optymalizacja i wstępne dopasowanie projectora zmienią wniosek. Wszystko nadal na **FULL_CLEAN, seed 42**, bez KD i bez otwierania testów.

S = od losowego projectora. T = **ten sam projector A1 z epoki 9 i te same learned loss scalars**, natywne vision na starcie, świeży optimizer. T0 jest kontrolą dalszego treningu samej głowy; bez niej łatwo błędnie przypisać zysk odmrażaniu.

| Run | Init | Vision bloki | Projector LR peak | Vision LR peak | Warmup epoki | Cosine | Val t2i R@1 | Val i2t R@1 | Best epoch |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| S0 frozen schedule | random | 0 | 1e-3 | 0 | 2 | tak | 58,13 | 69,69 | 9 |
| S1 last1 low constant | random | 1 | 1e-3 | 3e-6 | 0 | nie | 57,75 | 67,30 | 4 |
| S2 last1 schedule | random | 1 | 1e-3 | 1e-5 | 2 | tak | 57,75 | 67,54 | 4 |
| S3 last1 low schedule | random | 1 | 1e-3 | 3e-6 | 2 | tak | 57,70 | 68,97 | 7 |
| S4 last2 low schedule | random | 2 | 1e-3 | 3e-6 | 2 | tak | 57,56 | 66,59 | 4 |
| T0 frozen aligned | A1 epoch 9 | 0 | 1e-4 | 0 | 1 | tak | 58,56 | 69,69 | 1 |
| T1 last1 aligned | A1 epoch 9 | 1 | 1e-4 | 1e-6 | 1 | tak | 58,61 | 70,41 | 1 |
| T2 last2 aligned | A1 epoch 9 | 2 | 1e-4 | 1e-6 | 1 | tak | 58,61 | 70,64 | 4 |
| T3 last1 aligned midpoint | A1 epoch 9 | 1 | 1e-4 | 3e-6 | 1 | tak | 58,66 | 70,41 | 1 |
| T4 last2 aligned midpoint | A1 epoch 9 | 2 | 1e-4 | 3e-6 | 1 | tak | **58,71** | **69,93** | 1 |

Trainable params: 0 bloków **196 994**, 1 blok **3 349 378**, 2 bloki **6 501 762**, razem z loss scalars. Feature source i projector są wszędzie identyczne: native CLIP-projected 512D → linear 384D.

Harmonogram jest nakładany **na wszystkie grupy**: projector, loss scalars i vision. Warmup liniowy per optimizer update, potem cosine do 10% peak LR w ostatnim planowanym kroku. Rzeczywisty pierwszy LR vision w S2 to **1,82e-8**, nie od razu 1e-5. Unit tests sprawdziły pierwszy update, granice i proporcje LR między grupami.

S: ceiling 20 epok; T: ceiling 12 **dodatkowych** epok po 9 epokach A1. Early stopping patience 6, minimum 10 epok S / 8 T. Epoch 0 jest eligible, żeby nie utracić dobrego starting checkpointu. **Wskutek early stopping część runów skończyła przed końcem cosine** — nie jest to sprawdzenie każdego pełnego tailu schedulera. Warmup i cosine testowano jako pakiet, bez osobnej ablacji każdego z nich.

T3/T4 są **warunkowym followupem**, nie od początku niezależnym holdoutem: uruchomiono je, bo 1e-6 dawało <0,5 p.p. ponad T0. Oryginalną decyzję po ośmiu runach zachowano, następnie zaktualizowano wybór wyłącznie po tej samej val. Wszystkie runy mają epoch-boundary resumable optimizer/RNG checkpointy i weryfikację hashy.

Logowano per-group LR first/last/min/max, gradient norms mean/max, próbkowaną rzeczywistą zmianę AdamW względem normy wag co 25 kroków, drift wag i cosine drift validation features. Brak clippingu i nowych regularizerów. Przykład T4: train loss **2,108→1,712**, val loss **4,768→4,993**, val t2i **58,71→56,70** od pierwszej do ósmej dodatkowej epoki.

### 11.1. Co rzeczywiście wynika z porównania

- Najlepsze S partial FT: **57,75%** wobec matching frozen S0 **58,13%**: **−0,38 p.p.**
- Najlepsze T partial FT: **58,71%** wobec matching frozen T0 **58,56%**: **+0,14 p.p.**
- Wobec pierwotnego A1 jest **+0,48 p.p.**, ale większości tego porównania nie wolno automatycznie przypisać vision: samo dalsze uczenie projectora daje 58,56%.
- T4 vs T0: **19 poprawionych i 16 pogorszonych query**, netto 3 / 2090. Sparowany bootstrap 10 tys. losowań po **416 komponentach wspólnych pozytywnych obrazów**, przy stałej galerii: warunkowy przedział 95% **[−0,43; +0,72] p.p.**
- T4 vs A1: 82 poprawione / 72 pogorszone, netto 10; analogiczny przedział **[−0,92; +1,92] p.p.**
- Bootstrap **nie uwzględnia selekcji po wielu konfiguracjach, niepewności między seedami ani zmiany galerii**. Nie jest dowodem wyższej jakości populacyjnej.

Wniosek roboczy: łagodniejsze ustawienia i aligned head ograniczają regresję, lecz **nie wykazano przekonującej przewagi trenowania vision nad porównywalnym frozen**. Nie jest to twierdzenie, że każde FT musi zawieść. Wybrany checkpoint T4 odtworzono i potwierdzono dokładnie jego zapisane metryki walidacyjne. W tej serii nie mierzono nowego testowego MSCOCO ani XM3600.

## 12. VisualNews — osobny strumień przygotowania danych

### 12.1. Zakres i problem

Pula używanego wariantu M-BEIR VisualNews ma **537 568 captionów**. W obecnych assets **283 012** ma zweryfikowane powiązanie z dostępnym obrazem; **254 556** nie ma takiego mapowania i nie jest wysyłane do multimodalnego judge’a bez zdjęcia.

Caption prasowy często opisuje osobę, datę, miejsce, przyczynę albo wydarzenie, którego nie da się ustalić z samych pikseli. Chcemy zachować dobry sygnał wizualny, zamiast zakładać, że każde poprawne językowo tłumaczenie jest dobrym pozytywem image↔text. Nie odrzucamy automatycznie przykładów tylko za nazwisko czy obecność kontekstu.

### 12.2. Judge — pilot i ograniczenia

Model endpointu nazywa się `qwen3.8-27b`. Multimodalny pilot używał 360 różnych captionów/obrazów z **MMEB VisualNews_t2i**: 180 random, 180 targeted heurystykami, seed 20261020. To próbka diagnostyczna z puli 1000 benchmarkowych przykładów, **nie losowa reprezentacja całych 537 tys. captionów**. Użyte do rozwoju promptów przypadki nie są ślepym zbiorem potwierdzającym.

Porównano A_direct, B_taxonomy, C_claims, D_skeptical oraz rewizje v2/v3. Wybrano `B_taxonomy_v3`. Oceny: visual_grounding **0–4**, external context **0–3**, zależność od named entities, event/temporal/location/OCR, rzeczywisty image-caption mismatch, typ captionu, confidence oraz evidence/unsupported information. **Confidence nie jest skalibrowanym prawdopodobieństwem jakości.** Brak wsparcia w obrazie nie jest automatycznie sprzecznością.

Finalista B v3: schema-valid 100%; kontrole kierunkowe 48/48, w tym wrong color 8/8. W diagnostycznym przeglądzie 60 przypadków 8 ocen poza dopuszczalnym ręcznym zakresem, 0 ciężkich odchyleń o ≥2. Na 200 porównaniach powtórzeń VG agreement 82%, wszystkie etykiety agreement 60%. Ręczny przegląd wykonał Codex, **nie panel ekspertów**; nie ma uzgodnionego human gold ani pełnego pomiaru trafności.

Rozkład VG w tym pilocie: **0:56, 1:149, 2:134, 3:13, 4:8**. Mismatch 8/360. Sam taki rozkład nie jest dowodem poprawnej kalibracji. W zbiorze 10 diagnostycznych naturalnych false-alarm przypadków v3 wciąż alarmował w 6; większość tej grupy służyła już rozwojowi promptu. Nie wolno czytać tego jako niezależnego precision/FPR populacji.

Ustawienia pilota multimodal: temperatura 0,2, thinking off, max output 1800, obraz proporcjonalnie do max 1280 px, JPEG quality 88, schema-constrained output. Text-only T1/T2 miały służyć screeningowi; **nie zastępują** sprawdzania rzeczywistego grounding i nie są podstawą automatycznego odrzucania par.

### 12.3. Obecny pipeline grounding → recycling → PL

Etapy mają oddzielne wywołania i checkpointy:

1. **Judge** oryginalnego EN + obrazu, frozen B_taxonomy_v3; kompatybilne wcześniejsze oceny są importowane.
2. **VG 3–4 bez mismatch:** zachowanie oryginalnego captionu i przejście do tłumaczenia/reuse.
3. **VG 1–2 bez mismatch:** osobny **recycle** prompt tworzy nowy EN caption wsparty obrazem, zapisując zachowane i usunięte informacje oraz decyzję `usable`.
4. **Recheck** nowego EN + tego samego obrazu niezależnym ponownym wywołaniem judge’a. Dopiero VG 3–4 bez mismatch przechodzi do tłumaczenia. Brak nieograniczonej pętli rewrite.
5. **Translate** zaakceptowanego final EN na PL przez istniejący image-aware translator: thinking on, temperatura 0, output max 8192. Rozumowanie nie jest zachowywane w tekstach odpowiedzi; liczniki tokenów są przechowywane.

VG 0 → `rejected_grounding`; rzeczywisty mismatch → `review_mismatch`; nieskuteczny recycle → `review_recycling_failed`. Odrzucenie oznacza routing w nowym zbiorze, **nie skasowanie danych źródłowych**. Błędy etapów nie są ready training pairs.

Istniejący cache tłumaczeń obejmuje **44 955 unikalnych tekstów EN z 44 998 ukończonych historycznych wierszy**. Reuse wymaga exact final English string po usunięciu stałej instrukcji i białych znaków na brzegach. Nowy recycled EN nie dostaje automatycznie starego PL. Cache i per-text locks zapobiegają równoległemu tłumaczeniu tego samego tekstu.

Jest punktowy terminology guard dla zaobserwowanego `riot gear` → `antyradziecki`: jedna próba korekty i zachowanie obu odpowiedzi. To nie kompletny semantyczny QA.

Benchmark wydajności na tych samych 64 source pairs, kolejność concurrency **16,32,32,16**, fresh initial judge, stały translation cache: ok. **3163 vs 3486 ready par/h**. Wybrano **32 requesty łącznie dla wszystkich etapów**, nie 32 na etap. Mała próbka i zmienne decyzje judge’a ograniczają interpretację przepustowości. To nie pomiar rozkładu jakości całego źródła.

### 12.4. Datowany lokalny status — nie wynik treningu

Odczyt pliku pipeline status z **20.10.2026, 21:50:01 Europe/Warsaw**:

| Wielkość | Liczba |
|---|---:|
| Eligible image-caption pairs | 283 012 |
| Zakończone terminalnie | 4973 |
| Ready pary EN/PL | 3912 |
| Ready recycled | 3455 |
| Ready reused original | 457 |
| Rejected grounding | 980 |
| Rejected no visual core | 1 |
| Review mismatch | 73 |
| Review recycling failed | 7 |

To snapshot lokalnego statusu procesu opisanego jako running; nie wykonano w tej konsultacji kontroli stanu odległego serwera. Starszy samodzielny judge worker ma osobny status stopped i 3094 ukończone oceny — nie wolno sumować ich z nowym pipeline’em bez uwzględnienia importów.

**Ready nie znaczy po leakage guardzie gotowe do aktualnego train.** Zbiór wymaga zamrożenia, kontroli lineage/overlap, przypisania splitu i nowej decyzji przed integracją. Nie policzono jeszcze docelowej liczby unikalnych obrazów nowego, dopuszczonego treningowego VisualNews. Oryginał/rewrite/PL, odpowiedzi etapów, hashe, provenance reuse i błędy są zachowane w SQLite i eksportach.

Ryzyka do konsultacji: model judge i generator/rechecker należą do tego samego ekosystemu; powtórna ocena nie jest niezależną adjudykacją człowieka. Przepisywanie może wprowadzać hallucination lub styl modelu, a filtr VG może usuwać trudne, lecz użyteczne query i zmienić zadanie z news retrieval w zwykłe visual caption retrieval. Konieczny jest pomiar jakości, różnorodności i korzyści po treningu, nie tylko schema validity i ready count.

## 13. Tłumaczenia i jakość językowa

Projekt obejmuje osobny resumable translation pipeline, benchmark harness i moduł treningowy. Tłumaczenia używają endpointu `qwen3.8-27b`, zapisują EN i PL, metadata, model/prompt/config hashes, próby, czasy i tokeny w SQLite/JSONL. Przebiegi pilot/production i rewizje promptów są rozdzielone.

ARO i SugarCrepe tłumaczono **text-only**, zachowując kontrasty pozytyw/negatyw. Lokalny czterozbiorowy eval MMEB przygotowano z obrazem; jego źródła i gotowe tłumaczenia są zamrożone. M-BEIR MSCOCO w istniejącym train pochodzi z tłumaczeń text-only. Nie należy utożsamiać wszystkich generacji przekładu z jednym identycznym trybem.

Starsza kontrola jakości porównała 302 ukończone rekordy MSCOCO v1/v2: brak błędów struktury, 25 flag do przeglądu, 173 zmienione captiony. Występowały zarówno poprawy (`glasses`, `winery`, `briefcase`), jak i regresje znaczenia/relacji oraz błędy źródła. **Brak flagi i `reviewed=true` nie gwarantują poprawności.** Ocena była diagnostyką Codex, bez pełnego human gold.

Osobny historyczny draft-vs-review export miał tylko **22/100** przykładów; nie traktujemy go jako zakończonego dowodu korzyści self-review. Późniejsze standardowe tłumaczenia nie zakładały obowiązkowego drugiego review. Jeden trudny rekord ARO Attribution po powtarzających się pustych odpowiedziach uzupełnił jawnie oznaczony `codex-manual-fallback`; nie przypisano go Qwen.

Stan inventory **12:11 Europe/Warsaw 20.10** raportował 23 937 ARO Relation, 28 748 Attribution, 7511 SugarCrepe, 24 809 M-BEIR MSCOCO, 25 641 VisualNews i 4000 przykładów czterech eval: 114 646 przetłumaczonych rekordów. Ten historyczny licznik nie jest liczbą gotowych train pairs i nie jest bieżącym licznikiem nowego pipeline’u VisualNews. Aktualnego FULL_CLEAN nie powiększał.

## 14. XM3600-PL — przygotowane, bez wyników

Protokół przygotowania: `xm3600-pl-clean-subset-full-gallery-v1`, rola **FINAL_BLIND**, `scores_computed=false`.

- Oryginalnie 3600 obrazów; po udokumentowanych wykluczeniach **3597 obrazów**, **7133 PL captions**, **7111 unikalnych tekstów**.
- Trzy całe obrazy wykluczono po known exact hash/source matches do frozen train/val/legacy eval albo znanym Flickr→COCO mappingu.
- Źródło Open Images / znane Flickr IDs, bez celowo wybranego źródła COCO; brak znanego overlapu po tej filtracji nie daje gwarancji całkowitego braku współdzielonego pretrainingu.
- Galerie, captions, obrazy i attributions mają SHA/provenance; znane wykluczenia zachowano. To **udokumentowany subset**, nie niezmieniony oficjalny benchmark 3600 obrazów.
- Nie wykonano tutaj retrieval scores. Nie użyto XM do żadnego wyboru modelu, LR, λ, epoki ani seeda. Przyszłe dane wymagają także respektowania jego blacklisty przed ingestion, nie dopiero po obejrzeniu wyniku.

## 15. Co wykonano, a co pozostaje planem

### Wykonane

1. Translation infrastructure, cztery pełne lokalne eval tłumaczenia i benchmark wielu gotowych modeli.
2. Pilot E1, pełny E1, loss ablation E2a/E2b z trzema seedami.
3. Cztery frozen backbone’y, potem pięć kolejnych, z replikacją TinyCLIP-39M.
4. CLEAN-v1 audit, near-duplicate diagnostic, natural-size i matched-budget data ablations.
5. A1/A2 feature source, linear/MLP, C1/C2 partial FT, jeden LoRA, dwa KD λ, jeden frozen TinyCLIP-61M.
6. Osiem planowych prób LR/schedule/aligned head oraz dwa warunkowe midpoint runs.
7. Przygotowanie blind XM3600-PL bez scoringu.
8. VisualNews judge pilot, techniczny benchmark pipeline’u i częściowe przygotowanie nowych par; **bez treningu na nowym VisualNews**.

### Niewykonane / niepotwierdzone

- Żaden z nowych A–F ani LR runów nie ma trzech seedów. Nie wykonano replikacji małych gainów KD/FT.
- Brak wyuczonego composition head dla CIRR/FashionIQ i odrębnego grounding head.
- Brak jawnych train-only hard-negative mining i wykorzystania wszystkich kontrastowych captionów ARO/SugarCrepe w lossie.
- Brak kontrolowanej skali PolDense 17M/32M/68M.
- Brak poprawnej zagnieżdżonej learning curve po **unikalnych obrazach/parach** na stałym val.
- Brak KD + trainable vision, większego przeglądu temperatur/loss calibration czy innego KD; sprawdzono tylko opisany wariant λ=0,1/0,3.
- Brak pełnego vision FT, last-4 block FT, last-4 LoRA, nowych regularizerów zachowania oryginalnej geometrii i kontrolowanego gridu augmentacji.
- Brak finetuningu text tower. Brak nowego niezależnego human-annotated PL visual validation set.
- Brak final cross-dataset scoringu XM3600, final deployment/ONNX/quantization/latency study.
- RN101 z wcześniejszej listy nie był uruchamiany. TinyCLIP-61M, który w starych planach widnieje jako „future”, **już wykonano w F**.

### Kierunki planowane wcześniej, obecnie wymagające priorytetyzacji

1. **Dane VisualNews:** dokończyć ocenę/recycle/translation, audytować próbki niezależnie, zamrozić nową wersję danych i wykonać kontrolowany data ablation. Nie dopisywać bieżących rekordów do FULL_CLEAN. Przed ingestion ponownie zweryfikować lineage, wszystkie chronione eval i blind XM.
2. **Train-only hard negatives:** rozważyć top-K=16/32 z teachera/studenta z filtrowaniem znanych positive IDs, alternatywnych captionów, duplikatów i bliskich aliasów. Nie używać testu do mining.
3. **Data scaling/source mixing:** stałe validation, nested subsets, osobne osie unique images, unique pairs i exposures; porównać oryginalne/mixed/recycled dane, nie tylko „więcej rekordów”.
4. **Replikacja:** przy istotnym praktycznie gainie powtórzyć konfigurację i matching control na seedach 42/43/44; uwzględnić wariancję oraz wiele prób na jednym val. To propozycja do oceny, nie wykonana replikacja.
5. **Text encoder scaling** dopiero po ustaleniu sensownej recepty core. Text FT późno i warunkowo, z replay oraz distillation do oryginalnego PolDense i osobnym text↔text benchmarkiem.
6. **Composed retrieval:** osobny head przy frozen core, np. `normalize(r+αt)`, `Linear([r,t])` lub residual MLP z `[r,t,r*t,r-t]`; wyłącznie po zapewnieniu core t2i.
7. **Grounding** jako osobna zdolność i odrębny pomiar; nie traktować lokalnego RefCOCO raw-sum jako kompletnego testu rozumienia relacji.
8. **XM3600 final:** dopiero po zamrożeniu małej listy finalistów i osobnej decyzji użytkownika. Wynik nie powinien uruchamiać kolejnego gridu pod ten sam blind test.

Powyższe są kierunkami do krytycznej konsultacji. Nie ma obowiązku wykonania ich wszystkich ani aktualnego uzasadnienia kolejnego dużego gridu LR/backbone’ów bez diagnozy.

## 16. Najważniejsze ograniczenia do uwzględnienia

1. **Mało niezależnych obrazów:** 10,6 tys. w train mimo 70 tys. wierszy. Wielokrotne podpisy, cropy i source overlap zmieniają efektywną liczebność.
2. **Walidacja wyboru jest COCO-related**, ma 419 obrazów i wiele captionów na obraz. Znane obrazy są rozdzielone, lecz część tekstów jest wspólna z train. Wielokrotne strojenie zwiększa winner’s curse.
3. **Test MSCOCO jest wielokrotnie oglądanym legacy/in-domain pomiarem**, z historycznymi decyzjami częściowo zależnymi od testu. Nie może udawać nowego, niezależnego cross-dataset sprawdzianu.
4. **Domain/data size confounding** w ablacji COCO pozostaje mimo wyrównania kroków i ekspozycji.
5. **Jednoseedowe małe różnice** nie rozstrzygają między bliskimi receptami. Bootstrap jednego val nie zastępuje seed replication ani niezależnego zbioru.
6. **Optymalizacja nie jest wyczerpana:** sprawdzono konkretne recepty; cosine/warmup połączono, część runów zatrzymano przed końcem schedulera; aligned head pochodzi z A1 epoch 9. Nie sprawdzono wszystkich faz pośrednich, augmentacji i regularizacji.
7. **Nieznane false negatives:** maska rozpoznaje tylko udokumentowane positives. Prosty sigmoid z dużą liczbą ujemnych par, duplikatami i niewykorzystanymi kontrastami wymaga oceny.
8. **Geometryczny bottleneck:** frozen przestrzeń PL ma służyć także obrazom, ale nie trenowano jej multimodalnie. Niezmienność text↔text nie implikuje optymalnego target space dla całego zakresu zadań wizualnych. To hipoteza, nie wykazany powód porażek.
9. **Jakość przekładów i modelowych etykiet** nie została szeroko zweryfikowana przez ludzi. Schema validity/recheck nie są semantycznym gold. Nie wolno utożsamiać „gotowe PL” z „poprawny i wizualnie rozstrzygalny query”.
10. **Filtrowanie VisualNews zmienia rozkład zadania** i może utracić rzadkie trudne przypadki. Trzeba zdefiniować, czy celem jest visual caption retrieval, czy news retrieval wykorzystujące też wiedzę o osobach i wydarzeniach.
11. **Near-duplicate guard ma granice**; brak wykrytych overlapów nie certyfikuje wszystkich transformacji ani pretrainingu.
12. **Natywne modele referencyjne różnią się językiem, wielkością i adapterami.** Niski wynik wybranego modelu na PL/local protocol nie jest sam w sobie dowodem słabości modelu we wszystkich zadaniach. Nie każda różnica między backbone’ami izoluje architekturę.

## 17. Konkretne pytania do niezależnego modelu

1. Czy cel „zachować dokładnie przestrzeń PolDense i mapować do niej vision” jest właściwym kompromisem? Jak odróżnić ograniczenie przestrzeni od małej liczby/doboru danych bez odmrażania text tower?
2. Czy protokół val na 419 obrazach COCO wystarcza do doboru tak wielu konfiguracji? Jak zaprojektować dodatkowy development holdout, zachowując obecny snapshot i blind XM?
3. Jak interpretować źródłowe splity ARO/M-BEIR nazwane `test`, obecne guardy, source domain overlap i powtarzające się captiony train/val? Co jest największym problemem, a co tylko ograniczeniem opisu?
4. Czy sigmoid/multi-positive/maska negatywów i skale lossu są prawidłowe dla tej wielokrotności podpisów? Jak przetestować false negatives, sampling po unique image i wykorzystanie prawdziwych contrastive negatives?
5. Czy sekwencja FT może jeszcze być zasadniczo niewłaściwa? Zaproponuj **najwyżej 2–3** diagnostyczne eksperymenty pozwalające odróżnić optymalizację od overfittingu/danych, z matching controls, a nie szeroki grid.
6. Czy opisana KD przekazuje dostatecznie mocny i właściwie skalibrowany sygnał? Jak ocenić temperaturę, λ, teacher/student marginesy i ewentualne łączenie z ograniczoną adaptacją vision?
7. Czy podstawą kolejnego etapu powinny być dane, lepszy backbone, batching/negatives, czy projector? Uzasadnij hierarchię z liczb powyżej, nie z ogólnej intuicji.
8. Jak zweryfikować VisualNews judge/recycle niezależnie od generatora i uniknąć hallucinations oraz selection bias? Jaką małą próbkę ludzką oznaczyć i jakie source/mix ablations zrobić po jej przygotowaniu?
9. Które dotychczasowe porównania są metodologicznie wiarygodne, a które wymagają ponownego wykonania lub słabszych twierdzeń? Co należy poprawić przed publikacją wyników?
10. Jaki powinien być warunek przejścia do final XM3600 i kiedy można uznać mały gain za wart większej liczby parametrów/compute? Zaproponuj regułę z góry, bez strojenia na wyniku XM.

Prosimy o końcową odpowiedź konsultanta w formie: **diagnoza → dowody i kontrargumenty → najważniejsze braki informacji → krótki plan eksperymentów z oczekiwanym rozstrzygnięciem**. Potencjalne błędy implementacji prosimy wskazywać jako hipotezy z testem, dopóki nie są wykazane.

## 18. Reprodukcja, integralność i mapa źródeł

Środowisko obliczeń treningowych: WSL, istniejący Python env `pl-mmb-benchmark`, **RTX 5070 Ti 16 GB**. Endpoint tłumaczeń/judge jest osobny. Cache modeli/obrazów/teacherów i wyniki per epoch mają manifesty. Dwa scalar params lossu nie są liczone jako inference model. Peak VRAM w logach to PyTorch allocated, nie cały proces/system.

SHA256 podstawowych niezmiennych artefaktów:

```text
FULL_CLEAN.jsonl:
2834cb60a308be03685602e1fb4b6137d2c9968e249c81ce80892dae1b63908c
validation.jsonl:
257145fc0c2cded5e67bb1e3eb0036d65af357099a86e909103a078cc73bd3a6
e1-full/data.jsonl:
b3720c5fdf6a9a62a8a4adc8568083302617a386b140420388fda3eb32c6f2f9
e1-full/blacklist.json:
971ddaa1de16966e7f97fd52be1d6b69915bb7c5ea0e28ddd79b0fce0338d6e6
validation retrieval graph:
474052b8757cbb38ff2cdcfce58781ed379de1392270e04d48dbf096f37af993
```

Kontrole: dotychczasowe 29 testów treningowych oraz testy suffix/gradient scope/LoRA/KD; osobne 4 testy schedulera. Sprawdzono hash wejść, chosen epoch, checksum checkpointów i zgodność wybranego T4 po ponownym odtworzeniu walidacji. Wykonanie tej dokumentacji nie uruchamiało treningu ani nowych benchmark scores.

Najważniejsze pliki względem root repozytorium:

| Obszar | Źródła |
|---|---|
| Ten dokument + datowany snapshot faktów | `reports/project_consultation_brief.md`, `reports/project_consultation_snapshot.json` |
| Dane i konfiguracja | `training/runs/e1-full/config.json`, `data.jsonl`, `data_audit.json`, `blacklist.json`, `cache.json` |
| Guard i split | `src/pl_mmb_training/leakage.py`, `full_data.py`, `cleanroom.py` |
| Protokół / metryki | `benchmark/protocol_v2.md`, `src/pl_mmb_training/retrieval.py`, `evaluation.py` |
| E1 / E2 | `reports/training_e0_e1*.md`, `reports/training_p0_e2.md` i odpowiednie JSON/CSV |
| Backbone studies | `reports/training_vision_four.md`, `reports/training_vision_commercial_five.md` |
| CLEAN-v1 | `reports/clean_v1.md`, `clean_v1_matched_budget.md`, `clean_v1_legacy_reproduction.json`, `clean_v1_visual_review.json` |
| A–F | `reports/tinyclip_adaptation.md`, `tinyclip_adaptation_loss.csv`, `tinyclip_adaptation_finalist.json`, `tinyclip_adaptation_verification.json` |
| LR/warmup/LP→FT | `reports/tinyclip_optimization.md`, `.csv`, `.json`, `tinyclip_optimization_epochs.jsonl`, `tinyclip_optimization_curves.png` |
| Uncertainty / weryfikacja LR | `reports/tinyclip_optimization_validation_diagnostics.json`, `tinyclip_optimization_verification.json`, `tinyclip_optimization_extended_verification.json` |
| Kod nowszych eksperymentów | `src/pl_mmb_training/tinyclip_adaptation.py`, `tinyclip_followups.py`, `tinyclip_optimization.py`; launchery w `training/` |
| VisualNews judge i pipeline | `reports/visualnews_judge_pilot/REPORT.md`, `reports/visualnews_pipeline/README.md`, `v1/manifest.json`, `v1/checkpoint.sqlite3` |
| Jakość tłumaczeń | `reports/quality_v2_review.md`, `reports/review_ab/summary.json`, `reports/aro_manual_fallback.json` |
| Blind XM3600 | `reports/xm3600_pl_preparation.json`, `reports/xm3600_pl_exclusions.json` |
| Historyczny plan | `training/PLAN.md` — część statusów jest nieaktualna; wykonanie/plan rozdzielono w tym briefie |

**Ważne przy interpretacji starych planów:** instrukcje „stop po czterech/pięciu modelach”, „TinyCLIP-61M przyszły” i „MLP/FT/KD dopiero plan” opisują wcześniejszy etap. W tej chwili opisane A–F oraz LR followup są już wykonane; powyższy status ma pierwszeństwo przed tymi starszymi opisami.

## 19. Załącznik: gotowe modele referencyjne w benchmarku PL

To **ewaluacje istniejących modeli**, nie kolejne treningi naszego projectora. 20 modeli × 4 configi × 1000 query = 80 pełnych przebiegów w protokole v2. Tabela pochodzi z `reports/benchmark_twenty_models.md`. Parametry liczą używane komponenty gotowego modelu, zwykle obie wieże; nie są porównywalne z kolumną „Vision M” w sekcji 8 bez doliczenia PolDense/projectora.

| Model / wariant | Parametry M | CIRR P@1 | FashionIQ P@1 | MSCOCO t2i P@1 | RefCOCO P@1 |
|---|---:|---:|---:|---:|---:|
| google/embeddinggemma-2, tekst+wizja bez audio | 438,76 | 19,40 | 15,40 | 57,00 | 64,60 |
| timm/ViT-B-16-SigLIP-i18n-256 | 370,63 | 15,90 | 16,10 | 54,80 | 66,80 |
| timm/ViT-B-32-SigLIP2-256 | 376,86 | 17,40 | 16,70 | 53,10 | 65,40 |
| alphaedge-ai/siglip2-base-patch16-512-pol-32768 | 204,38 | 17,40 | 18,40 | 45,20 | 70,20 |
| google/siglip2-base-patch16-512 | 375,82 | 17,30 | 18,70 | 45,00 | 70,10 |
| alphaedge-ai/siglip2-base-patch16-512-pol-16384 | 191,80 | 17,30 | 18,10 | 45,10 | 70,30 |
| google/siglip2-base-patch16-384 | 375,48 | 15,70 | 17,80 | 44,20 | 68,40 |
| google/siglip2-base-patch16-256, native text+vision | 375,23 | 16,30 | 16,90 | 43,70 | 68,00 |
| M-CLIP/XLM-Roberta-Large-Vit-B-16Plus + timm/vit_base_patch16_plus_clip_240.laion400m_e31 | 677,76 | 16,00 | 12,70 | **60,00** | 54,50 |
| google/siglip2-base-patch16-naflex | 375,23 | 15,90 | 18,90 | 42,00 | 64,40 |
| alphaedge-ai/metaclip-2-worldwide-m16-384-pol-16384 | 85,30 | 10,90 | 10,70 | 34,60 | 58,70 |
| Marqo/marqo-fashionSigLIP | 203,16 | 12,10 | 12,20 | 24,30 | 64,30 |
| alphaedge-ai/metaclip-2-worldwide-s16-384-pol-16384 | 49,72 | 11,50 | 9,70 | 30,00 | 57,20 |
| google/tipsv2-l14 | 487,94 | 8,00 | 7,90 | 3,20 | 68,60 |
| alphaedge-ai/clip-ViT-B-32-multilingual-v1-pl-32768 | 156,33 | 8,50 | 4,20 | 24,90 | 49,80 |
| google/tipsv2-b14 | 195,95 | 9,30 | 6,60 | 3,40 | 66,60 |
| alphaedge-ai/clip-ViT-B-32-multilingual-v1-pl-16384 | 143,75 | 8,40 | 4,00 | 23,30 | 49,60 |
| facebook/PE-Core-B16-224 | 447,66 | 9,50 | 8,80 | 6,90 | 58,30 |
| apple/MobileCLIP2-S2, native text+vision | 99,24 | 19,10 | 8,80 | 27,20 | 22,00 |
| BAAI/BGE-VL-base | 149,62 | 7,40 | 0,70 | 1,40 | 17,00 |

Native MobileCLIP2 27,20% i vision MobileCLIP2 + nasz projector/PolDense 49,10% to różne modele tekstowe/układy, nie sprzeczność. Analogicznie native SigLIP2 43,70% i vision SigLIP2→PolDense 49,70%. M-CLIP 60,00% jest użytecznym jakościowym punktem odniesienia, ale jest prawie dziesięciokrotnie większy niż nasz TinyCLIP+PolDense (~70,67M).

Adaptery mają przypięte rewizje i kontrole ładowania wag/preprocessingu. TIPS używa native 448, PE-Core 224, pozostałe rozdzielczości są zgodne z checkpointem. Bardzo słabe wyniki PL nie są niezależnym audytem wszystkich zdolności tych modeli; przy ocenie jakości harnessu warto rozważyć kontrolę EN/PL i native task protocol. Tu nie dopisujemy nieistniejących wyników takich nowych kontroli.
