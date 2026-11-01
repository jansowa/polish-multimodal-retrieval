# Polski enkoder tekstowo-wizyjny <100M — kontekst do niezależnej konsultacji

Stan odczytu: **31.10.2026**. Dokument jest samodzielnym kontekstem dla modelu, który nie ma dostępu do repozytorium. Powstał z lokalnych raportów, manifestów, kodu ewaluacji i plików statusu. Nie uruchamiano na jego potrzeby treningu ani nowych ocen modeli. Ścieżki źródeł na końcu umożliwiają audyt, ale nie są potrzebne do zrozumienia dokumentu.

## 1. Cel i najważniejsze ustalenia

Chcemy stworzyć **możliwie najmocniejszy polski model tekst ↔ obraz poniżej 100 mln parametrów całego modelu inferencyjnego**: wizja + tekst + projektory i inne niezbędne komponenty. Główny przypadek użycia to polskie zapytanie → wyszukanie obrazu; obraz → tekst jest drugim kierunkiem retrievalu. Model ma kodować obie modalności do wspólnej przestrzeni, bez uruchamiania dużego VLM przy każdym wyszukiwaniu.

Dotychczasowa strategia: zachować niezmieniony polski encoder tekstowy **PolDense-32M**, a reprezentację obrazu mapować do jego przestrzeni. Nasza główna mała architektura ma **70 667 520 parametrów wraz z projektorem**. Zamrożony tekst zachowuje oryginalną funkcję text→embedding; nie oznacza to jednak, że przeprowadziliśmy szeroki nowy audyt polskich zadań tekstowych.

Najważniejsze fakty dla konsultanta:

- Początkowe 70 289 wierszy train zawierało zaledwie **10 645 różnych obrazów**. Liczba podpisów znacząco zawyża intuicyjnie rozumiany rozmiar danych.
- Zastąpienie regresji kosinusowej lossem kontrastowym sigmoid dało duży zysk. Większy projektor, proste FT i LoRA na małej starej puli nie przyniosły przekonującego postępu. KD dała mały zysk val bez poprawy historycznego testu.
- Dodanie VisualNews znacząco poprawiło wyniki. Nasze modele są mocne na VN, ale znacznie słabsze na Wiki, VizWiz i natywnym polskim XM3600 niż mocne obecne referencje <100M.
- Najmocniejszą z ocenionych tutaj małych referencji jest **MetaCLIP-M16 PL z wariantem słownika 32768, 93,69M**. Nie twierdzimy, że przeszukaliśmy wszystkie modele świata.
- Mamy już **zakończoną ocenę XM3600**. Dawne instrukcje FINAL BLIND zostały zniesione przez użytkownika 31.10.2026. Zbiór nadal jest bezwzględnie wykluczony z treningu.
- Generacja Wiki1 trenowała na zaledwie 28 nowych rekordach; Wiki2 na **3539 obrazach / 935 komponentach**. Wyników tych generacji nie wolno utożsamiać. Wiki2 przeszła próg retencji VN, ale nie ma tutaj potwierdzonej nowej oceny Flickr/VizWiz/XM dla jej modeli.
- Istnieje wspólna wersjonowana bramka dekontaminacji nowych danych. Nie usuwa ona wstecz znanych ekspozycji starych modeli, w szczególności nakładania z pełnym XTD10.

**Do oceny strategicznej:** zachowanie dokładnie przestrzeni PolDense było dotychczasową decyzją projektową. Konsultant powinien porównać kontynuację tej ścieżki z adaptacją mocniejszego gotowego małego modelu multimodalnego. Propozycje odejścia od PolDense mają jawnie wskazać utracone własności i sposób oceny tekstu; nie traktować historycznej decyzji jako udowodnionego optimum.

## 2. Architektura, reprezentacje i sposób treningu

| Element naszej architektury | Parametry | Działanie |
|---|---:|---|
| `OPI-PIB/PolDense-32M` | 31 883 136 | Zamrożony encoder tekstu, embedding 384D |
| Vision z `wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M` | 38 587 392 | Obraz 224×224, patch16, 12 bloków, hidden512, oryginalna projekcja obrazowa |
| Nowy `Linear(512,384)` z biasem | 196 992 | Mapowanie obrazu do przestrzeni tekstowej |
| **Razem inference** | **70 667 520** | Oryginalna tekstowa wieża TinyCLIP nie jest używana |

W tabelach leaderboardu **V+T = 70 470 528** i head jest podany osobno. Limit <100M trzeba stosować do sumy, a nie tylko kolumny V+T lub parametrów uczonych. Dwa skalary lossu nie są potrzebne do inferencji. Teacher treningowy nie wchodzi do limitu finalnego studenta.

Przypięte rewizje: PolDense `ca70494a25dad90d980db7c735658dc7bf88ed25`; TinyCLIP `07a4b0bc751cb64fecd2b661c048c1dd98d69444`.

```text
PL tekst → natywny tokenizer/pooling PolDense → t:384D
obraz → natywny preprocessing TinyCLIP → vision + oryginalna projekcja → v:512D
v → uczony Linear → z:384D
score(tekst, obraz) = cosine(t,z)
```

Obecna recepta PolDense zachowuje prefiks `[query]: ` i limit 128 tokenów. Tryby promptów benchmarku są raportowane osobno; „bez instrukcji zadania” nie oznacza usunięcia wymaganego natywnego prefiksu modelu. Trzeba mierzyć truncation, szczególnie przy dłuższych opisach.

Główny loss jest wielopozytywnym wariantem sigmoid/SigLIP:

```text
S_ij = exp(a) * cosine(t_i, z_j) + b
y_ij = +1 dla znanej pozytywnej relacji tekst–obraz, w przeciwnym razie -1
L = sum_ij softplus(-y_ij * S_ij) / B
```

Normalizacja jest przez B, **nie B²**. Uczone są skala i bias, początkowo 1/0,07 oraz −10. Log-skala jest ograniczana do [0, ln100], bias do [−30,30], bez weight decay tych skalarów. Znane pozytywy są wyznaczane w całym splicie; aliasy tego samego obrazu i powtarzające się teksty mogą tworzyć wiele pozytywów. **Nieznane semantyczne pozytywy pozostają negatywami** — maska nie rozwiązuje wszystkich false negatives.

Bazowa recepta: AdamW, batch256, projector LR1e−3, weight decay0,01; zwykle maks.20 epok. W późniejszych porównaniach źródeł wspólny budżet wynosił **5500 aktualizacji / 1 405 780 prezentacji**, z batch145 co275 kroków. Warianty z Wiki zachowywały 702 880 prezentacji FULL. Cechy zamrożonych encoderów są cache’owane; wagi FP32 i BF16 autocast przy ekstrakcji, cache/head/loss FP32. Bardzo krótki czas pętli uczenia heada nie obejmuje pobierania, generowania podpisów, audytu i ekstrakcji cech.

Przy FT odmrażano całe ostatnie 1–2 bloki transformera. W opisanej starszej implementacji końcowa LayerNorm i oryginalna projekcja CLIP pozostawały frozen. Przetestowano też rank8/alpha16 LoRA q/v w ostatnich dwóch blokach, LP→FT, warmup/cosine, anchor dryfu cech oraz KD.

## 3. Jak pozyskiwaliśmy i przygotowywaliśmy dane

### 3.1. Początkowe korpusy i tłumaczenia

Importowaliśmy ARO Visual Relation/Attribution, SugarCrepe i M-BEIR MSCOCO. Tłumaczenia EN→PL były łączone ze źródłami przez stabilne identyfikatory i zgodność pól, a nie przypadkową kolejność wierszy. M-BEIR caption→image wynikało z **dodatnich qrels**; jeden tekst mógł wskazywać kilka obrazów. Brak obrazu lub wiarygodnego mapowania nie dawał zgody na utworzenie pary.

| Źródło | Train | Validation |
|---|---:|---:|
| ARO Visual Relation | 20 883 | 2343 |
| ARO Visual Attribution | 25 434 | 2615 |
| SugarCrepe | 6103 | 644 |
| M-BEIR MSCOCO | 17 869 | 2093 |
| **Razem** | **70 289** | **7695** |

Train to 67 445 unikalnych par, 59 955 surowych tekstów PL i 10 645 obrazów po hashu pikseli. Validation ma 7415 unikalnych par i 1216 obrazów; obrazy między źródłami się powtarzają. Snapshot: `FULL_CLEAN.jsonl`. Nie wszystkie duplikaty wierszy usunięto; rozpoznawanie wspólnych pozytywów nie wyrównuje automatycznie ich wag w samplerze.

**Ważne pochodzenie:** ARO pochodziło ze źródłowych splitów nazwanych `test`, M-BEIR COCO z `query/test` i test candidate pool. Wydzieliliśmy własny train/val i chroniliśmy lokalny eval. Nie można przedstawiać pełnych oficjalnych testów tych korpusów jako niezależnych od naszego treningu. ARO i SugarCrepe używano jako **dodatnich par**; dostępne błędne podpisy nie uczestniczyły jawnie w opisanym lossie.

Tłumaczenia i oceny wykonywał m.in. endpoint o aliasie `qwen3.8-27b`; później wykorzystywano obraz i reasoning przy tłumaczeniu. Rewizje danych i prompty zapisujemy, ale alias endpointu nie identyfikuje w pełni rewizji serwowanych wag. Zachowujemy odpowiedzi, próby, błędy, cache i provenance w SQLite; reasoning text jest usuwany z zapisywanych odpowiedzi.

Kontrola jakości nie dowodzi pełnej poprawności semantycznej. Historyczny audyt 302 tłumaczeń COCO wykazał poprawy oraz regresje: mylenie aktówki z walizką, błędy relacji przestrzennych i naturalności. `reviewed=true`, poprawny JSON i brak flagi modelu nie są ludzkim zatwierdzeniem. Ten sam model w roli generatora i sędziego może powtarzać własne błędy.

### 3.2. VisualNews: od newsowego podpisu do opisu widocznego obrazu

Zweryfikowano **283 012 par obraz–podpis**. Dalsze 254 556 podpisów nie miało potwierdzonego obrazu w dostępnych zasobach i nie było wysyłane jako pary. Oryginalne podpisy newsowe często wymagają rozpoznania osoby/wydarzenia albo zawierają informacje niewidoczne na fotografii.

Pipeline ma osobno checkpointowane etapy:

1. **Judge:** obraz + źródłowy EN, grounding0–4 i mismatch; zamrożona taksonomia.
2. **Recycle:** dla grounding1–2 bez mismatch — przeredagowanie EN z zachowaniem wizualnie potwierdzalnej treści oraz zapisaniem, co zachowano/usunięto. Grounding0 jest odrzucany z puli wynikowej, mismatch trafia do review.
3. **Recheck:** osobne żądanie oceny nowego podpisu; wymagane grounding3–4 bez mismatch. Brak nieskończonej pętli przepisywania.
4. **Translate:** zaakceptowany EN→PL z obrazem i reasoningiem. Cache pasuje po dokładnym finalnym EN; stare tłumaczenie nie jest automatycznie przypisywane nowemu podpisowi.

To dane w znacznej części modelowe, a nie polskie ludzkie gold labels. Osobne żądanie recheck nie oznacza niezależnego modelu ani niezależnego audytu ludzkiego. Filtr zmienia rozkład zadania z news retrieval w stronę visual caption retrieval.

W pierwszych treningach VN użyto **33 906 rekordów / 33 851 obrazów**. Połączenie z FULL dało 104 195 rekordów / 44 496 obrazów. Późniejszy pilot dopuścił dodatkowe **19 888 rekordów / 19 876 obrazów / 19 845 komponentów**; rozszerzony audyt nowych VN nie znalazł dalszych odrzuceń ani niezweryfikowanych rekordów w tej już wybranej puli.

### 3.3. Wikimedia/Wikidata i polska kultura

Własny kolektor korzysta z oficjalnych API. Odkrywa encje, P18, Commons `depicts`/P180, kategorie i ograniczone rozwijanie grafu. Zakres obejmuje m.in. architekturę, dzieła sztuki, twórców, historię, tradycje, jedzenie, krajobrazy i transport. Szeroka kategoria nie dowodzi, że obraz przedstawia konkretną encję. Sam kolektor zbiera istniejące teksty i metadane; późniejszy pipeline modelowy jest osobnym etapem.

Każdy obraz zachowuje Commons file ID/SHA1, URL oryginału/strony, encje/powiązania, teksty PL/EN i ich źródła, autora, attribution, licencję oraz surowe odpowiedzi API. Deduplikacja scala powiązania, zamiast udawać nowe niezależne obrazy. Oryginały są pobierane osobno od metadanych.

Jakość źródła A–D: A = caption konkretnego pliku, B = opis/tytuł, C = same identyfikatory depicts, D = kontekst encji/kategorii. To **inna skala niż modelowy grounding0–4**. Nie generujemy opisu tylko z QID ani nie uznajemy kontekstu kategorii za pewny caption. Próbowanych jest do8 tekstowych kandydatów na obraz.

Wymagane są zgodne jawne nazwy i URL licencji z allowlisty CC0/CC BY/CC BY-SA. Braki, NC, ND i nieznane warianty są wykluczane. Modelowy pipeline ocenia/recykluje tekst, ponownie sprawdza grounding, zachowuje zaakceptowany PL albo tłumaczy EN. Eksport rozróżnia **native original, native recycled, translated**. Recyklowany po polsku tekst jest syntetyczny mimo polskiego źródła.

### 3.4. Zebrane, gotowe i wytrenowane to różne liczby

Odczyt plików statusowych z 31.10.2026 (ostatnie aktualizacje ok.21:04/21:15 czasu Warszawy):

| Strumień | Pula w statusie | Gotowe pary wg pipeline’u | Status zapisany w pliku |
|---|---:|---:|---|
| VisualNews | 283 012 | 165 316 | `stopped_by_request` |
| Wikimedia | 417 353 | 46 968 | `stopped_by_request` |

To datowana obserwacja plików, nie sprawdzenie żywych procesów. **Ready nie oznacza zgody na trening**: potrzebne są dostępne obrazy, fingerprinty, audyt scalonej puli i release. Te liczby nie są liczebnością danych użytych w dotychczasowych modelach. Zamrożony eksport pilota z 30.10 miał 132 875 VN i24 125 Wiki; nie jest identyczny z późniejszym stanem produkcji.

## 4. Co trenowaliśmy i jakie uzyskaliśmy efekty

### 4.1. Starsze eksperymenty: loss i adaptacja

Wyniki poniżej dotyczą dawnych lokalnych protokołów COCO; nie należy porównywać ich wprost z procentami innych galerii.

| Eksperyment | Wynik i interpretacja |
|---|---|
| SigLIP2 vision→PolDense, regresja cosinusowa | Pilot ARO6k:13,07±0,71%; pełna pula:41,70±0,20% lokalnego COCO. Zmieniały się rozmiar i skład danych. |
| Ten sam pełny trening, sigmoid | 49,70±0,70% lokalnego COCO, 3 seedy; duża poprawa względem regresji. InfoNCE testowano na val, bez nowego testu COCO. |
| Mały TinyCLIP, frozen linear | Historyczna referencja FULL_CLEAN:45,20% COCO; val około58%. |
| MLP / proste FT / LoRA na FULL | MLP val54,31%; FT1/2bloki57,56/56,46%; LoRA56,41%, wobec linear58,23%. Jeden seed. |
| KD od SigLIP2 | Val58,76%, lecz test44,50%. Teacher image/EN vs student image/PL, symetryczny KL rozkładów podobieństwa, T0,07, λ0,1/0,3. Bez dowodu poprawy generalizacji. |
| LP→FT i LR/warmup/cosine | Najlepsze partial FT58,71% vs dopasowane frozen58,56%; +0,14p.p., warunkowe CI[−0,43;+0,72]. Jeden seed, bez nowego testu. |

Większa vision SigLIP2 z PolDense to około125M z headem, więc przekracza cel. MobileCLIP2 vision+PolDense dawał historycznie49,10% COCO przy około67,90M z headem, ale nowe galerie pokazały słabą generalizację tej starej konfiguracji. Wysoki wynik jednego COCO nie rozstrzyga wyboru architektury.

### 4.2. Dodanie VN i ablacje źródeł

W starszej adaptacji na pełnym dodatku VN, 3 seedy: frozen osiągnął COCO val62,20±0,13% i VNdev59,04±0,35%; ostatnie2 bloki+anchor osiągnęły62,87±0,00% i63,04±1,53%. To wspólny efekt konkretnej recepty warm start/augmentacja/anchor/FT, bez izolacji każdego składnika. Wybór checkpointu był po COCO val.

Późniejsza seria źródeł startowała od świeżego heada, z frozen encoderami, bez augmentacji, i wybierała checkpointy **po VN dev**:

| Wariant, seed42 | VN dev t2i R@1 | Lokalny historyczny COCO t2i R@1 po zamknięciu selekcji |
|---|---:|---:|
| VN_only | 78,08% | 45,00% |
| VN + strict nonCOCO ARO | 69,40% | 42,40% |
| VN_FULL, 50% FULL /50% VN | 74,09% | 52,30% |

W tej recepcie duża waga ARO pogarszała VN i nie wykazała korzyści COCO; M-BEIR COCO dawał wyraźny zysk COCO. Nie dowodzi to, że ARO szkodzi przy innym samplingu lub jawnym lossie na trudnych negatywach. „Bez COCO” dotyczy naszego nadzorowanego treningu i pochodzenia heada, nie nieznanego pretreningu encoderów.

### 4.3. R0/R1/R2/R3 oraz dwie generacje Wiki

Wspólny budżet5500 kroków, frozen TinyCLIP/PolDense, świeży linear+sigmoid. Checkpoint: maks. VNdev t2iR@1, potem średnia obu kierunków, potem wcześniejszy krok, ocena co275 i w kroku0.

| Wariant | Udziały prezentacji FULL /VN /Wiki | Pula |
|---|---|---|
| R0 | 50/50/0 | Stary VN_FULL, reuse zweryfikowanych modeli |
| R1 | 50/50/0 | VN rozszerzone o19 888 rekordów |
| R2 | 50/47/3 | Jak R1 +Wiki |
| R3 | 50/40/10 | Jak R1 +Wiki |

R2/R3 wybierano po Wiki dev dla seed42, potem wybrany wariant powtarzano na43/44. Dopuszczalny spadek VNdev wobec R1:≤2p.p. Reprezentant to mediana VNdev między seedami, przy remisie niższy seed; nie wybieramy najlepszego seeda po teście.

**Generacja1:** brak lokalnych oryginałów Wiki ograniczył train do28 obrazów/16 komponentów. R3 osiągnął Wiki17,221±1,047% vs R1 15,106±0,604%, lecz VN spadł o2,030 p.p.; guardrail niezaliczony. R3 G1 pozostał diagnostyką.

**Generacja2:** odzyskano10 214 obrazów w kolejce, a końcowy audyt dopuścił **3539 rekordów/obrazów w935 komponentach**. Fingerprinty oryginałów liczono przed zmniejszeniem. Archiwum obejmuje cały kadr do1536px JPEG92 i osobne bezstratne wejście TinyCLIP224 zgodne bitowo z preprocessingiem. R0/R1 odziedziczono po kontroli zgodności; Wiki trenowano ponownie.

| Model | VN dev t2i R@1, mean±SD | Wiki dev t2i R@1, mean±SD | XTD897 t2i R@1, mean±SD |
|---|---:|---:|---:|
| R0 | 73,922±0,175% | 15,710±0,799% | 51,394±0,000% |
| R1 | 74,472±0,485% | 15,106±0,604% | 50,910±0,451% |
| R3 generacja2 | 72,957±0,325% | 17,120±0,923% | 50,130±1,036% |

R3 G2 przeszedł guardrail VN: spadek1,514 p.p. Poprawa Wiki jest niewielka wobec luki do referencji. Wiki dev po treningu na Wiki jest źródłowo ID i służył selekcji udziału Wiki. **Nie przenosić na G2 wyników OOD ani XM3600 zapisanych dla R3 G1**; raport G2 wymaga nowego audytu OOD względem jego uniwersum.

## 5. Jaki benchmark stworzyliśmy i jak oceniamy modele

### 5.1. Zakres i pochodzenie zapytań

Zbudowaliśmy lokalny, wspólny benchmark **retrieval tekst↔obraz**, runner, import historycznych wyników oraz leaderboard Gradio. To zbiór lokalnych galerii i protokołów, nie deklaracja oficjalnej ewaluacji wszystkich źródłowych datasetów.

| Galeria | Obrazy /unikalne teksty PL | Pochodzenie i rola |
|---|---|---|
| Flickr30k PL local test | 952/4746, z4760 captionów | MT NLLB-200-1.3B; core test, lokalna oczyszczona galeria, nie oficjalny Flickr30k1K |
| VizWiz PL local validation | 1000/4312, z4329 captionów | Ludzkie EN→MT `opus-mt-en-zlw`; u nas core test, zmiana profilu zdjęć |
| XM3600 PL | 3597/7111, z7133 captionów | Natywne ludzkie PL, core test; trzy wcześniejsze wykluczenia zachowane |
| Wiki native dev v2 | 397/331 | Natywne źródłowe PL; diagnostyka i selekcja Wiki |
| VN dev | 2000/2003 | Opisy generowane/recyklowane/tłumaczone; diagnostyka i selekcja VN |
| XTD10 full | 1000/1000 | Ludzkie tłumaczenia PL, obrazy COCO; kontrola z ujawnionym overlapem |
| XTD10 no overlap | 897/897 | Wariant bez znanych nakładań z train/val/wcześniejszym eval; nadal domena COCO |

Flickr/VizWiz przygotowano przed scoringiem modeli, z kontrolą oryginałów, QA tekstów oraz komponentów. Zachowano wyłącznie źródłowe pozytywy; pHash nie tworzy qrels. MT nie jest rodzimym polskim goldem. Nie tworzono nowych ręcznych adnotacji.

DOCCI PL pozostaje przygotowaniem; brak potwierdzonej kompletnej galerii retrieval. Winoground/ImageCoDe mają ślady adapterów/rejestrów, ale nie gotowe lokalne polskie wyniki. CIRR, FashionIQ i RefCOCO są historycznymi zadaniami spoza obecnego celu tekst↔obraz.

### 5.2. Scorer, metryki i integralność

- Każdy model ma przypiętą rewizję i natywny preprocessing/tokenizację. Używamy wspólnego scorera `full_gallery.py`: normalizacja i iloczyn skalarny FP32, pełna galeria, jawne multi-positive qrels, oba kierunki.
- R@1/5/10 oznacza **co najmniej jeden znany pozytyw w top-k**. Raportujemy MRR i NDCG@10. Nie imputujemy brakujących wyników zerami.
- Galerie, kolejność query, qrels, prompty, implementacja i profil numeryczny mają identyfikatory/SHA. Tryby promptów i różne profile precyzji pozostają oddzielne.
- Zachowujemy wyniki per query, checkpoint SHA i logi umożliwiające reload. Cache musi odtworzyć adapter; wykryto m.in. zależność BF16 od batch8/batch32 i odtworzono oryginalne batche, zamiast osłabiać próg zgodności.
- Paired bootstrap:10 000 losowań całych komponentów, seed20261023, stała galeria i wspólne qrels. CI różnicy jest **warunkowe**; nie obejmuje wyboru recepty ani nieznanego pretreningu. Mean±SD3 seedów jest osobną informacją, nie tym samym przedziałem.
- Stary MMEB MSCOCO_t2i miał1000 zapyt. i pule po1000 kandydatów. Osobny fixed-gallery COCO ma1002 obrazy/1000 tekstów, i2t dla201 uprawnionych obrazów. COCO val:419 obrazów/2090 tekstów. Nie mieszać tych protokołów z nowymi galeriami ani oficjalnym COCO5k.

Leaderboard liczy wynik zadania jako średnią t2i/i2t, a średnią zadań z równymi wagami. Domyślnie pełne XTD nie dubluje oczyszczonego; po XM jest6 zadań w średniej. To **średnia mieszanych testów, dev i kontroli**, nie czysty wynik OOD ani bezpośrednio cel optymalizacji głównego t2i. Pochodzenie opisów native/human-translated/synthetic/MT jest oddzielnie oznaczone.

### 5.3. Wyniki aktualnych referencji i naszych reprezentantów

Pierwsza tabela pochodzi z ukończenia benchmarku31.10, **przed dodaniem XM do średniej**. Komórki to średnia obu kierunków R@1 w%. Parametry V+T, nasz head+0,197M dodatkowo. R0/R1/R3 to wcześniej zamrożone reprezentanty seed44; **R3 oznacza G1**.

| Model | V+T M | Flickr | VizWiz | Wiki | VN | XTD897 | Średnia5 zadań |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nasz R0 | 70,471 | 48,956 | 20,797 | 15,135 | 71,545 | 48,384 | 40,963 |
| Nasz R1 | 70,471 | 50,240 | 21,596 | 14,480 | 72,069 | 48,606 | 41,398 |
| Nasz R3 G1 | 70,471 | 48,546 | 20,901 | 15,991 | 69,996 | 48,551 | 40,797 |
| MetaCLIP-S16 PL,16384 | 49,722 | 48,060 | 26,325 | 30,144 | 58,033 | 40,580 | 40,628 |
| MetaCLIP-S16 PL,32768 | 56,014 | 55,487 | 33,963 | 38,303 | 68,649 | 49,220 | 49,124 |
| MetaCLIP-M16 PL,16384 | 85,302 | 53,427 | 32,488 | 33,291 | 65,702 | 44,426 | 45,867 |
| **MetaCLIP-M16 PL,32768** | **93,690** | **61,568** | **39,244** | **43,717** | **74,869** | **54,181** | **54,716** |
| EmbeddingGemma-2,tekst+wizja | 438,760 | 82,748 | 65,782 | 57,089 | 92,280 | 71,126 | 73,805 |

Model >100M jest punktem odniesienia jakości, nie kandydatem do finalnego produktu w tym limicie. Silniejsze modele mogą być nauczycielami, jeśli pozwalają na to warunki użycia.

**XM3600 — ukończone wyniki 31.10.2026:**7 modeli,14 wyników kierunkowych,140zweryfikowanych metryk. To ważniejsza aktualizacja niż starsze raporty z napisem „XM blind/nieuruchomione”.

| Model | t2i R@1% | i2t R@1% | Średnia kierunków% |
|---|---:|---:|---:|
| R0 seed44 | 38,53 | 35,84 | 37,18 |
| R1 seed44 | 38,63 | 36,98 | 37,80 |
| R3 G1 seed44 | 37,35 | 36,67 | 37,01 |
| MetaCLIP-S16 PL32768 | 45,97 | 58,19 | 52,08 |
| **MetaCLIP-M16 PL32768** | **50,29** | **62,66** | **56,48** |
| MetaCLIP-B16 PL32768,141,39M | 51,23 | 63,83 | 57,53 |
| EmbeddingGemma-2,438,76M | 68,18 | 76,62 | 72,40 |

Dla R1 luka do MetaCLIP-M16 PL32768 wynosi około**11,66 p.p.t2i** i**25,69 p.p.i2t** na XM. Sama przewaga historycznego COCO nad starszymi małymi referencjami nie uzasadnia tezy, że mamy najmocniejszy mały polski model multimodalny.

Dla głównego t2i, OOD B w3 seedach: R1 Flickr48,848±0,583%, VizWiz21,699±0,259%; R0 odpowiednio47,598±0,462% i21,374±0,801%. Źródłowe OOD dotyczy nowej kolekcji względem naszego uczenia projektora, nie całego pretreningu encoderów. Flickr/VizWiz z tej serii nie służyły jej selekcji; dalsze dobieranie recept po ich wynikach zmieniłoby ich rolę w development.

### 5.4. Diagnostyka, której jeszcze brakuje

7695 rekordów validation obejmuje2093COCO oraz**5602 pary ARO/SugarCrepe**. Pozytywne i negatywne podpisy PL/EN są dostępne i spięte z ID walidacji. Można mierzyć `score(image, positive)>score(image, negative)`, osobno dla relacji, atrybutów i SugarCrepe, z jawną obsługą remisów. **Nie ma tutaj potwierdzonego ukończonego panelu takich wyników.** Stara funkcja `tinyclip_diagnostics.hard_negatives` mierzy podobieństwo tekstów TRAIN i nie zastępuje tej oceny. Cały ARO/SugarCrepe nie może być niezależnym testem, ponieważ część służyła treningowi.

## 6. Dekontaminacja: jak wykluczamy dane testowe z treningu

### 6.1. Co chronimy i dlaczego sam ID nie wystarcza

Chronimy wszystkie obrazy galerii, **także kandydatów bez własnych zapytań**, podpisy EN/PL, walidacyjne trudne negatywy, stare holdouty i całe zarezerwowane komponenty Wiki, obejmujące1159 obrazów. Nowy podpis, tłumaczenie albo crop obrazu nie czyni go niezależnym przykładem treningowym.

Jeden obraz może występować w COCO, Visual Genome, Flickr, Commons, pod różnymi URL i jako crop. Historyczny guard mapował VG→COCO przez oficjalne metadane. Dla RefCOCO numer `image_<ann_id>_crop.jpg` oznacza ID adnotacji, nie ID obrazu; rodzica wyznacza się z oficjalnego mapowania. Wykluczenie rodzica obejmuje rodzinę cropów.

### 6.2. Etapy dopuszczenia nowej puli

1. **Zarejestrować holdouty przed treningiem.** Aktualne źródła i obrazy/teksty muszą znaleźć się w wersjonowanym rejestrze; samo dopisanie nazwy zbioru do katalogu nie daje ochrony.
2. **Zachować lineage:** ID źródeł, Commons file ID/SHA1, COCO/Flickr/OpenImages, URL, artykuły/strony i rzeczywiste relacje encji. Recykling, tłumaczenie i augmentacja dziedziczą pochodzenie. Wspólna licencja, autor lub ogólna strona kolekcji nie są tożsamością obrazu.
3. **Fingerprinty oryginału, przed resize/crop/JPEG:** SHA256 bajtów, SHA256 zdekodowanych pikseli RGB po EXIF oraz wspólny pHash DCT 32×32/blok 8×8/mediana z DC. Konserwatywne wykluczenie przy odległości Hamminga≤6; inny algorytm pHash nie jest wymienny.
4. **Teksty:** porównać źródłowe EN i finalne PL po normalizacji/casefold i scaleniu białych znaków; po tłumaczeniu wykonać kontrolę ponownie. Exact matching nie wykrywa wszystkich parafraz.
5. **Komponenty przechodnie:** graf powiązań budujemy z całych metadanych puli, również niepobranych aliasów. Jeżeli A łączy się z B, B z C, a C z holdoutem, odrzucamy cały komponentA/B/C z nowych kandydatów train. Nie wystarczy sprawdzić osobno wiersze już pobrane.
6. **Braki blokują:** niedostępny oryginał, fingerprint lub niezweryfikowane pochodzenie nie są świadectwem czystości. Potrzebny jest audyt scalonej puli, a potem ponowna kontrola po augmentacji/recyklingu.
7. **Zamrozić manifest i release:** SHA manifestów, rejestru, katalogu, kodu guarda/runnera i wejść. Runner sprawdza dowód przed ekstrakcją/treningiem i ponownie po uzyskaniu blokady GPU. Zmiana wejść/holdoutu unieważnia stary audyt i wymaga nowej generacji.

Operacyjnie **„usuwanie z treningu” oznacza wykluczenie z wersjonowanego manifestu dopuszczonych kandydatów**, z zachowaniem źródeł i powodów odrzucenia. Nie kasujemy danych ani plików i nie poprawiamy wycieku cichym usuwaniem testów ze starej galerii. Historycznych checkpointów taka operacja nie oducza.

### 6.3. Bieżący interfejs i zgodność zamrożonych eksperymentów

Źródło zasad: `training/DATA_SPLITS.md`. Nowe audyty: `src/pl_mmb_training/current_holdouts.py`, domyślnie `registry-v4.json`, z bieżącym `catalog-current.json`.

```bash
# Tylko kontrola, bez GPU; uruchamiać w środowisku projektu.
PYTHONPATH=src python -m pl_mmb_training.current_holdouts
PYTHONPATH=src python -m pl_mmb_training.current_holdouts --manifest new_pairs.jsonl
```

Kod wyjścia2 oznacza blokadę. Same te polecenia nie zastępują integracji release’u z runnerem ani pełnego grafu metadanych. Obecność modułu nie zabezpiecza automatycznie wszystkich starszych entrypointów.

Wiki G2 ma przypięte dokładne SHA **rejestru v2, bazowego `holdouts.py` i starego `catalog.json`**. Tych plików nie wolno zmieniać. Wersja v4 służy nowym audytom; zamrożony G2 poprawnie zachowuje v2. Stare napisy FINAL_BLIND w jego snapshotach nie zmieniają obecnej decyzji o ocenianiu XM ani ochrony XM przed train. Rozbudowa rejestru wymaga nowej wersji i nowego audytu, bez nadpisywania starego bindingu.

Wykryto wcześniej brak jawnego uwzględnienia później zbudowanych Flickr/VizWiz w kolejce nowych danych Wiki. Dodano rejestr do pełnego grafu **przed wyborem Wiki_train**, do zamrożonych SHA i do sprawdzeń runnera. Częściowy `queue_snapshot.json` jest obserwacją kandydatów, nie końcowym release’em.

### 6.4. Czego dekontaminacja nie dowodzi

- Pełny XTD1000 ma**47 obrazów powiązanych z historycznym FULL train**. Wariant897 wyklucza także inne znane związki z val/eval. Pełny wariant pozostaje ujawnioną kontrolą; nie wolno przedstawiać go jako czystego OOD. Oczyszczony również ma domenę COCO.
- Rozdzielenie obrazów nie usuwa zależności źródłowych i stylistycznych. W starych splitach występowały wspólne teksty train/val, a COCO było wielokrotnie oglądane przy rozwoju projektu.
- pHash≤6 nie dowodzi braku wszystkich cropów i transformacji; lineage trzeba zachować osobno. pHash służy wykluczeniom, nie dopisywaniu pozytywów do testu.
- Audyt naszych nowych danych nie dowodzi czystości oryginalnego pretreningu TinyCLIP, PolDense ani konkurencji. Ten zakres ekspozycji jest nieustalony.
- Wielokrotne strojenie na teście to osobny problem od duplikatów train/test. Zmiana etykiety testu nie przywraca utraconej niezależności.

## 7. Zasoby, efektywność i ograniczenia praktyczne

Treningi/ewaluacje: WSL, istniejące środowisko Python, **RTX5070Ti16 GB**. API tłumaczeń/judge jest osobnym zasobem. Wiele procesów i sesji współdzieli komputer; GPU ma wspólną blokadę. Nie zatrzymywać cudzych procesów, nie aktualizować środowisk w trakcie runów, nie zmieniać zamrożonych generacji i nie usuwać plików. Nowe pomysły realizować w nowych wersjonowanych eksperymentach.

Historyczne pomiary efektywności (mediany, CPU FP32 batch1,2 wątki; GPU BF16 autocast batch64,10 warmup/100 pomiarów): nasz model tekst14,50 ms i obraz77,11 ms CPU; tekst0,277 ms i obraz2,103 ms/element GPU. MetaCLIP-M16 starszy wariant:29,59/212,99 ms CPU i0,243/3,800 ms GPU. To pomiary określonych adapterów i precyzji, nie całej usługi ani nowych wariantów32768. Liczba parametrów nie zastępuje pomiaru latencji, RAM/VRAM i pamięci indeksu.

Lokalny audyt licencyjny rozdziela wagi, dane, nauczyciela, outputy i sposób publikacji. TinyCLIP deklaruje MIT, SigLIP2 Apache2.0, PolDense warunki Gemma; zakres obowiązków projektora/destylacji pozostaje do oceny. Wikimedia zachowuje deklaracje praw per rekord; prawa do newsowych obrazów i tekstów VN wymagają osobnej oceny. To **status lokalnego audytu**, nie aktualna porada prawna ani zgoda na publikację. Samo dopuszczenie do eksperymentu nie oznacza gotowości do redystrybucji.

## 8. Otwarte decyzje, które ma rozstrzygnąć konsultacja

Nie prosimy tylko o pomysły na kolejny grid. Potrzebujemy kolejności działań, które najszybciej zwiększą szansę na model mocniejszy od MetaCLIP-M16 PL32768 przy <100M:

1. Czy utrzymywać frozen PolDense jako twardy wymóg, czy rozwijać gotowy mały model multimodalny? Jak oddzielić ograniczenie przestrzeni tekstowej od jakości vision, danych i lossu?
2. Czy priorytetem jest dopuszczenie większej części gotowych danych, ich zróżnicowanie i ludzki QA, czy lepsza architektura/nauczyciel? Jak odróżnić zysk jakości danych od samego wzrostu liczby obrazów?
3. Jak zmienić mieszanki i sampler, aby poprawić naturalne polskie zapytania i trudne zdjęcia, zachowując retencję? Czy duplikaty/caption multiplicity i false negatives istotnie psują obecny loss?
4. Jak wykorzystać prawdziwe negatywne podpisy ARO/SugarCrepe bez ponownego mieszania train i validation? Czy ich wpływ mierzyć osobnym zadaniem kompozycyjnym?
5. Jaka KD lub adaptacja vision/text jest uzasadniona po nieudanej małej starej KD? Jakie kontrolowane doświadczenie odróżni brak sygnału nauczyciela od bariery frozen PolDense?
6. Jak ustalić nową regułę selekcji wielodomenowej, skoro VN i Wiki były już selekcyjne, COCO historycznie oglądane, a XM jest już ujawniony? Czy potrzebujemy nowego niezależnego holdoutu i jak go zbudować?
7. Co mierzyć jako główną miarę „najmocniejszego”: t2i, oba kierunki, domeny, kompozycyjność, tekst, latencję? Jak z góry ustalić sukces i regresje zamiast wybierać wygodną średnią po wynikach?

Każda rekomendacja powinna odwołać się do dowodów, wskazać kontrargument, minimalny rozstrzygający eksperyment, koszt, wymagany audyt danych i warunek zatrzymania. Nowych wyników, gotowych zbiorów ani budżetów nie wolno dopowiadać.

## 9. Mapa dowodów w repozytorium

Ścieżki względem root; dokument można czytać bez ich otwierania. Starsze raporty mogą zawierać nieaktualne instrukcje blind i stany kolejek — powyżej rozstrzygnięto je nowszymi artefaktami.

| Zakres | Główne źródła |
|---|---|
| Polityka i ochrona danych | `AGENTS.md`; `training/DATA_SPLITS.md`; `retrieval_benchmark/holdouts/README.md`; `src/pl_mmb_training/current_holdouts.py` |
| Początkowe dane/model/loss/eksperymenty | `reports/project_consultation_brief.md` — brief20.10, użyty tylko jako źródło historyczne; `reports/tinyclip_adaptation.md`; `reports/tinyclip_optimization.md` |
| Pozyskiwanie i jakość danych | `data_collection/README.md`; `data_collection/FAST.md`; `reports/visualnews_pipeline/README.md`; `reports/wikimedia_pipeline/README.md`; `reports/quality_v2_review.md` |
| Datowane liczniki produkcji | `reports/visualnews_pipeline_state.json`; `reports/wikimedia_pipeline_state.json` |
| VN i źródła | `reports/tinyclip_visualnews.md`; `reports/tinyclip_source_ablation.md`; `reports/tinyclip_source_ablation_findings.md` |
| R0/R1/WikiG1 | `reports/data_study_20261030/REPORT.md`; `experiments/data_study_20261030/preflight.py` |
| Wiki G2 | `reports/data_study_20261030/wiki_recovery_20261031/REPORT.md`; `training_report.json`; `global_holdout_training_release.json` w tym katalogu |
| OOD Flickr/VizWiz | `reports/ood_eval_20261030/REPORT.md` i `summary.json` |
| Benchmark i aktualne referencje | `retrieval_benchmark/README.md`; `catalog-current.json`; `COMPLETION_20261031.md`; `EVALUATION_COVERAGE.md`; `space/results.json` |
| Wykonany XM3600 | `reports/benchmark_decontamination_20261031/xm3600_completion.json` |
| Scorer i bootstrap | `src/pl_mmb_benchmark/full_gallery.py` |
| Licencje — status audytu30.10 | `reports/series_2026-10-23/LICENSING_DOSSIER_20261030.md` |

Kontrolne SHA256 historycznych snapshotów (nie są release’em nowej puli):

```text
FULL_CLEAN.jsonl: 2834cb60a308be03685602e1fb4b6137d2c9968e249c81ce80892dae1b63908c
validation.jsonl: 257145fc0c2cded5e67bb1e3eb0036d65af357099a86e909103a078cc73bd3a6
```

Załączony osobno prompt określa oczekiwany format niezależnej oceny. Ten dokument opisuje stan projektu i jego ograniczenia; nie stanowi autoryzacji do zmiany danych, procesów, modeli ani publikacji.
