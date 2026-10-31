# Stan projektu: encodery wizyjne dopasowane do polskich encoderów tekstowych

Eksport do niezależnej konsultacji, 21 października 2026. Dokładny czas wykonania, wersja repozytorium, źródła i sumy kontrolne są w `snapshot.json` oraz `MANIFEST.json`. Dokument zawiera kontekst potrzebny bez dostępu do repozytorium. Załączony ZIP dostarcza raporty liczbowe, konfiguracje i kod do głębszej kontroli.

## Cel konsultacji i hierarchia wymagań

**Aktualny cel właściciela projektu: przygotować jak najmocniejsze encodery wizyjne do gotowych polskich encoderów tekstowych.** Prosimy ocenić kierunek badań, a nie tylko zaproponować następny krok w dotychczasowym planie.

Dotychczasowy program koncentrował się na małym, użytecznym modelu **TinyCLIP-39M vision → Linear 512→384 → przestrzeń frozen PolDense-32M**. To jedna gałąź szerszego celu. Preferencja małego modelu, ograniczenia lokalnego GPU i wybór TinyCLIP nie są dowodem optymalności jakościowej ani bezwzględnym wymaganiem wynikającym z obecnego pytania. Należy osobno ocenić maksymalną jakość oraz kompromis jakość/rozmiar/koszt. Nie zakładać też, że recepta dla PolDense-32M przenosi się na wszystkie polskie encodery tekstowe; kontrolowanej serii między encoderami dotąd nie wykonano.

Historycznym założeniem produktu jest zachowanie oryginalnej funkcji i przestrzeni tekstowej: text tower frozen, ten sam prefiks, pooling i preprocessing, bez uczonego mapowania tekstu. Główną zdolnością jest PL text→image; image→text jest dodatkowym pomiarem. Composed retrieval i grounding mają odrębne znaczenie. Jeśli konsultant proponuje odejście od zachowania oryginalnej przestrzeni, powinien jawnie przedstawić korzyści, koszt kompatybilności i potrzebne kontrole tekstowe. Propozycja nie oznacza przyjęcia zmiany.

## Aktualny obraz wyników

Wartości w tabeli są procentami. Różne galerie i reguły wyboru checkpointów uniemożliwiają bezpośrednie porównywanie wszystkich wierszy jako jednego rankingu.

| Gałąź | Dane i selekcja | Najważniejszy wynik | Status dowodu |
|---|---|---|---|
| Pełny E1, SigLIP2 vision→PolDense, cosine regression | FULL_CLEAN, minimum val loss | lokalny COCO test R@1 41,70 ± 0,20 | 3 seedy |
| E2b, ten sam SigLIP2, sigmoid | te same dane/budżet, minimum val loss | COCO test 49,70 ± 0,70 | 3 seedy; +8,00 p.p. względem E1 |
| TinyCLIP frozen, A1 | FULL_CLEAN, COCO val retrieval | val 58,23; historyczny test 45,20 | wybrany seed 42; frozen replikacje val 57,67 ± 0,53 |
| TinyCLIP KD λ=0,3 | FULL_CLEAN; wybór po COCO val | val 58,76; test 44,50 | 1 seed; zysk val nie przeniósł się na test |
| G1 frozen + łagodna augmentacja | FULL_CLEAN; gęstsza selekcja | COCO val 59,47 | 1 seed; brak nowego testu |
| FULL_CLEAN + pełny dodatek VN, frozen | warm start A1; wybór po COCO val | COCO val 62,20 ± 0,13; VN dev 59,04 ± 0,35 | 3 seedy adaptacji, wspólny wcześniejszy head |
| Ta sama mieszanka, last2 + anchor | te same warunki adaptacji | COCO val 62,87 ± 0,00; VN dev 63,04 ± 1,53 | FT−frozen +0,67 p.p. COCO, +3,99 p.p. VN; brak testu tej serii |
| VN_only, oba encodery frozen | świeży projector; wybór wyłącznie po VN dev | VN dev 78,08; lokalny COCO test 45,00 | 1 seed; brak znanego COCO w naszym nadzorowanym treningu |
| VN_FULL, oba encodery frozen | świeży projector; wybór po VN dev | VN dev 74,09; COCO val post hoc 62,30; COCO test 52,30 | 1 seed; kontrola z COCO w train |
| VN + strict non-COCO ARO | świeży projector; wybór po VN dev | VN dev 69,40; COCO test 42,40 | 1 seed; dodatek nie pomaga w tej proporcji |

`±` oznacza odchylenie standardowe między wskazanymi seedami, nie przedział ufności. Wynik 62,87 ± 0,00 oznacza identyczne dyskretne R@1 na trzech wybranych checkpointach małej galerii, nie brak niepewności.

**Najważniejsza aktualizacja starszego briefu:** VisualNews zostało już użyte w treningach. Oryginalne FULL_CLEAN i jego validation pozostały niezmienione; nowe dane są osobnym zamrożonym dodatkiem. Nie należy utożsamiać „snapshot bez zmian” z „żadnych nowych danych w eksperymentach”. Wynik 52,30% jest najwyższym lokalnym COCO test R@1 w opisanych tu treningach TinyCLIP→PolDense; nie jest niezależnym ślepym potwierdzeniem ogólnej jakości ani wynikiem adaptacji last2 z tabeli wyżej.

## Dane faktycznie użyte, przygotowywane i chronione

- **FULL_CLEAN:** 70 289 train rows, 67 445 różnych par, 10 645 różnych obrazów; validation 7695 rows / 1216 obrazów. Train to ARO Attribution 25 434, ARO Relation 20 883, SugarCrepe 6103 i M-BEIR MSCOCO 17 869. Wybór checkpointów po COCO val obejmuje tylko 2090 unikalnych captionów i 419 obrazów. Nie wszystkie źródłowe wiersze są niezależnymi przykładami.
- **Zamrożone VN do wykonanych badań:** eksport 35 918 par z 21.10; po filtracji train 33 906 par / 33 851 obrazów, dev 2003 par / 2000 obrazów. Zagnieżdżone 10k ma 10 015 par. Train: Washington Post 25 953 par i USA Today 7953. 29 921/33 906 par train to opisy recycled lub reused-recycled (około 88,2%); należy jawnie uznawać je za modelowo przepisane, nie naturalne ludzkie podpisy. Dev ma ten sam pipeline i dwa źródła co train.
- **Kontrola nowych VN:** exact file/pixel/text, znane article paths i pHash≤6; grupowanie komponentów, wykluczanie powiązań z base/protected/XM. Odrzucono 9 wierszy komponentów. Nie gwarantuje to rozdzielenia wszystkich wydarzeń ani cropów. Kontrola 24 obrazów przez model ujawniła m.in. niepotwierdzony kontekst i szczegóły tracone przez native center crop. Brak niezależnego human gold.
- **Strict ARO bez znanego COCO:** 24 473 wiersze / 3037 obrazów, jawne rekordy VG bez COCO ID; kontrola transytywnych aliasów. To nie stwierdzenie o danych pretrainingowych TinyCLIP/PolDense.
- **Wikimedia:** kolektor treści związanych z polską kulturą oraz osobny grounding/recycle/translation pipeline. Allowlista metadanych CC0/CC BY/CC BY-SA, przechowywane autorstwo i atrybucje; brak NC/ND. Źródłowe klasy jakości A–D i grounding modelowy 0–4 to różne osie. Native PL zachowywane, EN tłumaczone, kandydaci opisów sprawdzani oddzielnie. Nie ma dotąd w odczytanych artefaktach zakończonego treningu ani wyniku retrievalu Wiki.
- **XM3600-PL:** przygotowane FINAL_BLIND, 3597 obrazów, 7133 PL captions / 7111 unikalnych tekstów po 3 wykluczeniach. Wyników nie obliczono. Użycie blacklisty do kontroli overlapu nie jest scoringiem. Nie uruchamiać benchmarku w ramach konsultacji/eksportu.
- **Dodatkowe zasoby tekstowe:** mFAQ PL 65 182 par, MAUPQA 3 660 140 wierszy z etykietą trafności; bez obrazów i bez użycia w dotychczasowych treningach multimodalnych. Winoground/ImageCoDe nie są gotowymi lokalnymi benchmarkami tylko dlatego, że istnieje adapter lub README.

Aktualne liczniki pipeline’ów i niedokończonego pilota zostały automatycznie zamrożone dalej w dokumencie. **Pipeline ready ≠ dane dopuszczone do treningu ≠ dane rzeczywiście użyte.** Liczniki z różnych odczytów nie powinny być sumowane ani zastępować audytu konkretnego runu.

## Rejestr hipotez: co odrzuciliśmy, co osłabiliśmy, co pozostaje otwarte

| Hipoteza / intuicja | Status i podstawa | Granica wniosku |
|---|---|---|
| Regresja wyłącznie pozytywnych par wystarcza jako najlepsza recepta | Odrzucona jako preferowana recepta na FULL: sigmoid daje +8,00 p.p. test w dopasowanej serii 3 seedów | Nie jest to twierdzenie o każdym zadaniu regresji |
| Brak zysku z FT wynika z błędnego suffix cache lub BF16 | Niewsparta przez audyt: cache/full forward i gradienty zgodne, BF16/FP32 gradient cosine 0,999960 | Kontrole dotyczą sprawdzonej implementacji i wejść |
| Frozen PolDense narzuca twardy sufit 62–65% val | Niewsparta: leave-one-caption-out centroid 70,53%; SigLIP2 historycznie wyżej niż TinyCLIP | Centroid korzysta z opisów val, nie jest osiągalnym modelem obrazu ani ścisłym sufitem |
| Liniowy projector nie może przeuczać, a wzrost val loss to tylko skala | Odrzucona taka interpretacja: wzrost pozostaje po stałych skalarach; rankingi osobno mogą się pogarszać | Sama strata nie dowodzi overfittingu rankingu |
| Większy head albo większy TinyCLIP automatycznie pomoże | Brak wsparcia: MLP 54,31 vs linear 58,23 val; TinyCLIP-61M 51,34 | Jeden wariant, seed i inne pretraining/patch; nie wyklucza innych architektur |
| Partial FT/LoRA na starych danych poprawia jakość | Niepotwierdzona w A–F/S/T/G; małe różnice mieszczą się w niepewności | Nie odrzuca FT z lepszymi danymi i innym budżetem |
| FT jest ogólnie bezużyteczne | Osłabiona przez nowe dane: VN + last2/anchor daje +3,99 p.p. VN dev średnio na 3 seedach | COCO +0,67 p.p. z CI obejmującym zero; brak niezależnego testu tej adaptacji |
| Osiągnięcie ~45% COCO wymaga COCO w naszym train | Odrzucona w zakresie lokalnego benchmarku: świeży VN_only daje 45,00% | Pretraining gotowych encoderów i nieznane kopie nie zostały wykluczone |
| Więcej ARO jako dodatnich par domyślnie pomaga | Niewsparta przy 50% mieszanki: A/R pogarszają VN dev i COCO val post hoc | Nie testuje małych dawek, innych zadań ani explicit hard negatives |
| Jawne ARO negatives są łatwym darmowym zyskiem | Ryzykowna: mediany cos(true,false) PolDense 0,99480 attribution / 0,98646 relation | Bliskość utrudnia margines, ale nie dowodzi nieuczalności; lossu z nimi nie uruchomiono |
| KD λ=0,3 to potwierdzona poprawa | Nie: +0,53 p.p. val, ale test 44,50 vs historyczne 45,20 | Gradienty sugerują słaby sygnał przy tej λ; KD + trainable vision i strojenia temperatur nie sprawdzono |
| Wysoki modelowy recheck i poprawny JSON potwierdzają etykiety | Nie: recheck to kolejna prośba do tego samego modelu/ekosystemu, nie niezależny sędzia ani human gold | Potrzebna odrębna kontrola semantyki i efektu danych |
| Brak wykrytego leakage oznacza całkowitą niezależność | Nie: guardy mają granice; val jest COCO-related, legacy test był oglądany wielokrotnie | Nie stwierdzono znanego image/parent overlapu objętego audytem |

Otwarte hipotezy obejmują różnorodność obrazów vs styl opisów, sampling i false negatives, geometrię różnych polskich encoderów, skalę/rodzaj teachera, FT po pełnym LP na nowej mieszance, korzyść Wiki i transfer poza dotychczasowe domeny. Nie należy utożsamiać kierunku obiecującego z wynikiem dowiedzionym.

## Najważniejsze ograniczenia metodologiczne

1. Lokalny COCO test to **1000 query z pulami po 1000 kandydatów**, osobna fixed gallery ma 1002 obrazy. Nie jest to oficjalny COCO 5k. Val ma tylko 419 obrazów. Wiele opisów jednego obrazu nie stanowi niezależnych obserwacji.
2. Legacy test był wielokrotnie oglądany, również podczas historycznego wyboru backbone’ów. Nowa selekcja po VN dev poprawia rozdzielenie etapów tej serii, ale nie czyni starego testu nowym blind holdoutem.
3. Bootstrap jest warunkowy względem galerii i wytrenowanych modeli. Nie obejmuje całej selekcji po wielu konfiguracjach, seedów, zmiany domen ani pełnej procedury treningowej. Trzy seedy adaptacji VN mają wspólny A1 head i nie są trzema powtórzeniami całego LP→FT.
4. Efekt dodania źródła łączy domenę, liczbę różnych obrazów, styl/liczbę podpisów i strukturę negatywów. W VN-adaptation 0→VN zmienia także ekspozycje FULL_CLEAN; 10k→all zachowuje proporcje, ale ma jeden seed.
5. Źródłowe ARO i M-BEIR MSCOCO używają splitów nazwanych `test`, z których wydzielono lokalne train/val i chronione eval. Nie wolno przedstawiać oryginalnych testów tych korpusów jako dziewiczo niezależnych od naszego treningu.
6. W train/val nie znaleziono wspólnych parentów i canonical pixels. Są identyczne znormalizowane teksty: w retrieval val 23 różne PL / 25 wierszy wspólne z train; w całym val 1012 różne PL / 1301 wierszy. To ograniczenie generalizacji do nowych tekstów, nie automatyczny dowód wspólnego obrazu.
7. Judge, generator/recycler i recheck korzystają z `qwen3.8-27b`; „independent recheck” w dokumentach technicznych oznacza osobne wywołanie, **nie inny niezależny model**. VN dev dziedziczy etykiety tego pipeline’u. Wysoki recheck pass rate może odzwierciedlać współdzielony błąd.
8. Native center crop w 224×224 może usuwać detale, na których oparto opis/judge większego obrazu. Dobry grounding pełnego zdjęcia nie gwarantuje zgodności z wejściem encodera.
9. Nie ma szerokiej oceny naturalnych polskich query, OCR, drobnych atrybutów, kultury polskiej i generalizacji domenowej z niezależnym human gold. CIRR/FashionIQ/RefCOCO raw-sum nie zastępują tych pomiarów.
10. Licencje raportowane przy checkpointach są zapisami wcześniejszego screeningu. Eksport nie weryfikuje obecnego stanu prawnego/licencji online. Rozróżniać warunki wag, danych, tekstowego PolDense i modeli nauczycieli; możliwość komercyjnego użycia lub destylacji nie wynika z samego napisu MIT/Apache na jednym komponencie.

## Plan i nierozstrzygnięte decyzje

**Prace widoczne w kodzie, ale bez ukończonych wyników:** pilot Wiki ma warianty 0/3/10% Wiki zastępujące sloty VN, frozen encodery, świeży wspólny projector, seed 42, po 5500 kroków; checkpoint wyłącznie po VN dev. COCO i XM nie są kryterium wyboru. Snapshot 1322 par zamrożono 21.10 o 21:24:43 Europe/Warsaw. Przy odczycie brak finalnego audit.json i wyników; status dotyczył pobierania. To plan/stan przygotowania, nie dowód działania Wiki. Nie uruchamiano go w ramach tego eksportu.

**Kierunki do uporządkowania przez konsultanta, nie automatycznie przyjęta kolejka:**

- Niezależny development holdout poza COCO i stylem generatora, naturalne krótkie polskie query oraz próbka ludzkiej kontroli etykiet. XM pozostaje final blind.
- Dopasowany LP→FT: najpierw head uczony na całej docelowej mieszance, potem FT, wobec frozen o tym samym dodatkowym budżecie; oddzielić dane od warm-startu A1.
- Kontrolowana krzywa liczby unikalnych obrazów i proporcji źródeł, w tym dawki ARO/Wiki, porównanie original/recycled, sampling po obrazach i jawne false negatives.
- Skala jakości: mocniejszy vision backbone/teacher oraz różne gotowe polskie encodery tekstowe przy porównywalnej procedurze. Trzeba ustalić, czy priorytetem jest absolutna jakość, Pareto jakość/rozmiar, czy rodzina kilku rozmiarów.
- KD + adaptowane vision, temperatura i waga dobierane po sygnale gradientu i kontrolach; nie kolejny szeroki grid bez hipotezy.
- Rozszerzenie ewaluacji o composed retrieval/grounding i koszt inference po zapewnieniu mocnego core. Nie wykonano końcowego deployment/latency/ONNX/quantization study.

Nie ma kontrolowanego porównania PolDense 17M/32M/68M, full vision FT ani szerokiego last4/LoRA, nauczyciela większego SigLIP2, treningu na nieoznaczonych VN przez distillation, jawnego hard-negative mining, treningu text tower ani końcowego scoringu XM. Część starszych planów mówi o nieuruchamianiu MLP/FT/VN; została wyprzedzona przez wykonane i opisane tutaj prace.

## Prośba do modelu analizującego

Oceń projekt krytycznie w odniesieniu do celu **jak najmocniejszych encoderów wizyjnych do istniejących polskich encoderów tekstowych**. Nie przyjmuj interpretacji autorów raportów jako faktów. Wskaż:

1. Czy dotychczas optymalizujemy właściwy problem, czy nadmiernie zawęziliśmy go do TinyCLIP i małego COCO val? Jak rozdzielić jakość absolutną od efektywności?
2. Co dane faktycznie rozstrzygają o bottlenecku: dane, przestrzeń tekstowa, backbone, loss/sampling czy procedura selekcji? Podaj alternatywne wyjaśnienia.
3. Które wcześniejsze twierdzenia trzeba wycofać, które są dobrze poparte, a które wymagają silniejszej kontroli? Oddziel błąd implementacji od hipotezy metodologicznej.
4. Jak zbudować wiarygodny development holdout i human audit, zachowując XM final blind? Jakie benchmarki i zdolności pomijamy?
5. Jakie najwyżej 3–5 następnych eksperymentów zapewni największą informację? Dla każdego: hipoteza, matched control, dane, koszt, kryterium sukcesu/stop, interpretacja obu możliwych wyników. Uwzględnij lokalny RTX 5070 Ti 16 GB, ale nazwij większy eksperyment, jeśli bez niego nie ocenimy celu.
6. Jaka recepta i rozmiar vision są obecnie uzasadnione do dalszej pracy, a jaki wynik potrzebny przed rekomendacją? Nie wybieraj „najlepszego” przez porównanie różnych galerii lub walidacji z testem.
7. Jakie informacje są nieobecne i co należy zebrać przed publikacją, większym treningiem lub finalnym XM? Nie zgaduj brakujących wyników.

Oczekiwany format: diagnoza → dowody i kontrargumenty → brakujące dane → krótki plan z warunkami decyzji. Osobno oceń kierunek długoterminowy i najbliższą serię. Konsultacja nie udziela zgody na zmianę snapshotów, uruchomienie final blind ani publikację danych.

## Jak czytać dalszą część

Dalsze sekcje dostarczają szczegółów architektury/lossu, wyników historycznych, nowych serii i przygotowania benchmarków. Starsze fragmenty są oznaczone chronologicznie. Ich ograniczenia typu „nie wykonano augmentacji/VN” dotyczą opisywanej wtedy serii, nie całego obecnego projektu. Aktualny stan ustala powyższa synteza, nowsze wyniki oraz snapshot eksportu.
