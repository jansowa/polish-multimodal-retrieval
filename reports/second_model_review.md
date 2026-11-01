# Ocena konsultacji drugiego modelu i wykonane sprawdzenia

Stan: 2026-10-21. Ten dokument uzupełnia `project_consultation_brief.md`; nie nadpisuje historycznego briefu.

Najbardziej użyteczne zalecenia to brakujące kontrole geometrii tekstu, wariancji seedów, wpływu skali lossu oraz finetuningu z augmentacją i ochroną pierwotnej reprezentacji. Nie przyjmuję natomiast jako faktów, że centroid wyznacza sufit, liniowy projektor nie może overfitować lub że KD przy frozen vision nie ma prawa pomagać.

## Wyniki diagnostyk D1

Wszystkie obliczenia używają istniejącego FULL_CLEAN i tej samej walidacji. Żaden test MSCOCO ani XM3600 nie został uruchomiony. Pliki manifestów sprawdzane SHA256 przed i po obliczeniach. Wyniki maszynowe: `tinyclip_diagnostics.json` i `tinyclip_frozen_replication.json`.

| Diagnostyka | Wynik | Wniosek i ograniczenie |
|---|---:|---|
| Centroid pozostałych captionów, PolDense PL | t2i R@1 70,53%; R@5 93,44%; R@10 97,32% | Nie potwierdza hipotezy sufitu 62–65%. To korzystający z opisów walidacyjnych test geometrii, a nie osiągalny model obrazu ani ścisłe ograniczenie górne. |
| Ten sam test, SigLIP2 EN | R@1 71,51%; R@5 91,87%; R@10 95,51% | Różnica około 0,98 p.p. przy R@1; zmieniają się jednocześnie język i encoder. 2092 unikalne EN vs 2090 PL, ta sama galeria 419 obrazów. |
| Ridge, najlepszy z czterech stałych poziomów regularizacji | t2i R@1 54,93%; i2t R@1 41,05% | Poniżej sigmoid/AdamW 58,23% / 69,69%. Inny loss i image-balanced weighting; nie dowodzi globalnej optymalności aktualnego projektora. |
| Val loss ze stałą skalą i biasem z epoki 9 | 4,80 w ep. 9 → 8,01 w ep. 20 | Wzrost nie znika po zamrożeniu skalarów podczas przeliczania. Zmiana embeddingów/projektora ma znaczenie; nie jest to sam efekt wyostrzania skali. |
| Val loss ze skalą/biasem z inicjalizacji | 6,61 w ep. 1 → 13,07 w ep. 20 | Także nie znika. Wartości bezwzględne silnie zależą od kalibracji; nie porównywać dowolnych skalarów jako osobnych modeli retrievalu. |
| Frozen TinyCLIP, val po wyborze checkpointu, seedy 42/43/44 | 58,23% / 57,18% / 57,61%; średnia 57,67%, sample std 0,53 p.p. | Brakujące seedy wykonane; dla seeda 42 wykorzystano istniejący A1. To nie jest wyznaczenie minimalnego wykrywalnego efektu ani test istotności po selekcji konfiguracji. |

Centroidy obejmują tylko unikalne captiony. Zapytanie usuwane jest ze wszystkich powiązanych z nim prototypów, także gdy ma kilka prawidłowych obrazów. Każdy obraz ma 4–5 unikalnych captionów. Ridge jest dopasowywany wyłącznie na 10 645 obrazach TRAIN, z interceptem bez kary, do średniej unikalnych znormalizowanych captionów PL. Sprawdzono natywne wejścia vision oraz ich normalizację do jednostkowej długości; cztery wartości kary to 0, 0.01, 0.1, 1 razy średnia przekątna macierzy Grama dla obu reprezentacji. Najlepszy wariant natywny ma R@1 54,74%; najlepszy jednostkowy 54,93%.

Historyczne SigLIP2→PolDense E2b osiągało już 61,56% ±1,05 p.p. val t2i przy tym samym zamrożonym PolDense. To także empiryczny argument, że około 58% TinyCLIP nie jest udowodnionym twardym ograniczeniem samej przestrzeni tekstowej.

Przeliczenie lossu ze stałymi skalarami nie jest samo w sobie testem overfittingu rankingu. Projektor może zmienić kalibrację podobieństw bez zmiany kolejności. O overfittingu retrievalu wnioskujemy z osobno mierzonych krzywych val retrieval, nie z samego rozdźwięku train/val loss.

## Hard negatives

Negatywne tłumaczenia pobrano wyłącznie z historycznych, zahashowanych snapshotów ARO i SugarCrepe i połączono po `translation_id` z rekordami TRAIN FULL_CLEAN. Nie dodano ich do lossu ani do manifestu. Encoder, rewizja, prefiks i pooling zgodne z istniejącym cache'em; minimalny cosine ponownie zakodowanych pozytywów do ich starych wektorów wyniósł 0,99999988.

| Źródło | Liczba par | Mediana cos(true,false) | Odsetek >0.95 | Mediana dla losowych par captionów z tego źródła |
|---|---:|---:|---:|---:|
| ARO attribution | 25 434 | 0,99480 | 99,33% | 0,70343 |
| ARO relation | 20 883 | 0,98646 | 97,74% | 0,72802 |
| SugarCrepe | 6 103 | 0,96365 | 67,49% | 0,70919 |

To silny argument za ostrożnością, zwłaszcza dla attribution. Wysoki cosine ogranicza dostępną różnicę wyników, ale sam próg 0.95 nie oznacza matematycznej nieuczalności. W tych parach nie ma identycznych stringów ani wektorów z cosine >0.99999. Zestaw losowy to deterministyczne przesunięcie captionów w ramach źródła; pojedyncze przypadki mogą opisywać ten sam obraz/obiekt, więc nie jest to czysty zbiór semantycznych negatywów.

Dla jednostkowego embeddingu obrazu maksymalna różnica cosinusów do dwóch jednostkowych tekstów wynosi `sqrt(2−2*cos(t_true,t_false))`. Przy cosine 0,995 pozostaje około 0,10 marginesu; duża bliskość jest utrudnieniem, nie dowodem sprzeczności etykiet.

## Co sprawdzono w implementacji FT

Osobny rzeczywisty audyt CUDA, `tinyclip_finetuning_audit.json`, porównał pełny forward TinyCLIP z oryginalnym cache'em wejść do ostatnich dwóch bloków. Cechy i gradienty były identyczne (relative L2 0, cosine 1). BF16 vs FP32 dało cosine gradientów 0,999960, co nie wskazuje na utratę sygnału przez precyzję. Cache sam w sobie nie psuje gradientu; dotychczasowy stały cache uniemożliwiał jednak nowe widoki obrazów.

Obecny sampler daje średnio 249,47 unikalnego obrazu w batchu 256; najczęstszy obraz ma 135 rekordów. Maski multi-positive uwzględniają powtórzenia. To inny problem niż przypadkowe traktowanie wszystkich innych captionów jako negatywnych — ten błąd nie został stwierdzony.

Siła wcześniejszego KD nie może być oceniana wyłącznie po wartości lossu: porównano normy gradientów TRAIN na ośmiu stałych batchach w checkpointcie T4. Lambda około 6 dałaby normę gradientu vision od KD równą połowie gold; wcześniejsze 0.3 było około 20 razy słabsze od tak zdefiniowanej kalibracji. To nie dowodzi, że większa lambda poprawi retrieval. Nie zmieniono temperatury nauczyciela ani nie uruchomiono nowego KD w tej serii.

Zamrożenie obu encoderów nie sprawia, że KD z definicji nie może pomóc: nauczyciel nadal może dostarczać inny sygnał dla uczonego projektora. Natywna skala lossu sigmoid SigLIP2 nie jest też automatycznie właściwą temperaturą dla warunkowych rozkładów softmax używanych w KL. Temperaturę i wagę należałoby porównać z dopasowanymi kontrolami, a nie uznać ich błędność na podstawie samych wartości lossu.

## Bezpośrednia próba finetuningu wynikająca z konsultacji

Kod: `src/pl_mmb_training/tinyclip_generalization.py`. Wyniki kolejno trafiają do `tinyclip_generalization.md` i `.json`.

Wspólne warunki: A1 epoka 9 jako start headu i skalarów, oryginalny TinyCLIP jako start vision, seed 42, head LR 1e-4, vision LR 1e-5, update-level warmup przez jedną epokę, cosine do 10% LR przez 12 pełnych epok. PolDense i oryginalna projekcja CLIP pozostają frozen. Mierzymy również środek pierwszych dwóch epok; zasada doboru po val retrieval i liczba pomiarów są jednakowe dla wszystkich kontroli. Pełny budżet usuwa wcześniejsze ucięcie ogona cosine przez early stopping.

| Eksperyment | Vision | Widoki TRAIN | Dodatkowa regularyzacja |
|---|---|---|---|
| G0 | frozen | native | brak |
| G1 | frozen | augmentacja | brak |
| G2 | ostatnie 2 bloki | augmentacja | brak |
| G3 | ostatnie 2 bloki | augmentacja | 50 × cosine anchor do oryginalnego TinyCLIP |

Augmentacja generowana na każdym kroku: zoom 1–1.08 i przesunięcie ±3%, bez flipów i zmian kolorów, które mogłyby naruszać opisy relacji i atrybutów. To świadomie łagodna próba, nie sprawdzenie wszystkich możliwych augmentacji. Te same seedy samplerów i widoków w kontrolach. Z cache'u pochodzi tylko natywny RGB; wejścia do ostatnich dwóch bloków liczone są na bieżąco dla nowego widoku przez frozen prefix. Walidacja zawsze natywna.

Anchor porównuje cechy natywnego TinyCLIP i jego adaptowanej kopii na dokładnie tym samym widoku, przed naszym projektorem. Współczynnik 50 zaokrąglono z 49,04 skalibrowanego wcześniej na gradientach TRAIN. Jest to ochrona geometrii tego samego modelu, nie kopiowanie surowych embeddingów innego nauczyciela do niezgodnej przestrzeni. Kontrola G2 rozdziela efekt augmentacji od anchoru; G1 rozdziela korzyść z treningu headu na nowych widokach od odmrożenia vision.

Mała poprawa po jednym seedzie pozostaje wynikiem eksploracyjnym. Do uznania korzyści z odmrożenia potrzebne jest porównanie z G1, nie wyłącznie ze starym A1. Lepsza selekcja checkpointu może pomóc także frozen baseline.

### Zakończone wyniki G0–G3

Każda próba ukończyła 3300 aktualizacji / 12 epok. Checkpointy ponownie wczytano; R@1/5/10 obu kierunków i val loss odtworzone. 14 testów zakresu odmrażania, KD, harmonogramu i nowych diagnostyk przeszło. Pełna tabela ośmiu wymaganych pól i wykresy: `tinyclip_generalization.md`; niezależna weryfikacja: `tinyclip_generalization_verification.json`.

| Wariant | Val t2i R@1 | Val i2t R@1 | Best epoch kontynuacji |
|---|---:|---:|---:|
| G0 frozen, native | 59,33% | 69,93% | 0,502 |
| G1 frozen, augmentacja | 59,47% | 69,69% | 0,502 |
| G2 ostatnie 2 bloki, augmentacja | 59,09% | 69,69% | 0,502 |
| G3 ostatnie 2 bloki, augmentacja + anchor | 59,09% | 69,69% | 0,502 |

G2−G1 wynosi −0,383 p.p., conditional paired 95% interval [−1,054; +0,287]. G3−G1 również −0,383 p.p., przedział [−1,006; +0,240]. G1−G0 to +0,144 p.p., przedział [−0,384; +0,714]. Bootstrap 10 000 razy po 416 komponentach galerii, 2090 zapytań. Przedziały nie uwzględniają selekcji konfiguracji/checkpointu ani zmienności seedów; nie ma potwierdzonej przewagi vision FT w tej próbie.

Anchor działał zgodnie z przeznaczeniem: końcowy dryf cech val spadł z 0,043271 do 0,001317 (około 33 razy), a końcowy t2i R@1 poprawił się z 55,65% do 57,42% względem G2. Nie poprawił jednak maksymalnego retrievalu osiągniętego wcześniej, ani nie przebił najlepszego frozen. To odróżnia działającą regularyzację od jej nieskuteczności dla głównego celu.

Najbardziej konkretnym przeoczonym szczegółem była częstotliwość walidacji: G0 osiągnął 59,33% w połowie pierwszej epoki kontynuacji, a 58,56% na jej końcu. Wcześniejszy T0 miał tylko pomiary całych epok. Ta obserwacja uzasadnia wspólną gęstszą selekcję dla wszystkich kolejnych porównań; sama nie stanowi jeszcze potwierdzonej na trzech seedach poprawy jakości produktu.

Wniosek z tej serii jest ograniczony do podanej recepty. Nie sprawdzono jeszcze połączenia similarity KD z uczonym vision ani startu finetuningu po krótszym linear probe, np. po ep. 2. Te dwie hipotezy są bardziej odrębne od już wykonanych prób niż kolejny niewielki ruch LR. Wszystkie bieżące zadania zakończone; snapshoty bez zmian, test MSCOCO i XM3600 nieuruchomione.

## Ocena pozostałych zaleceń

1. **D2, krzywa po liczbie obrazów:** sensowny kolejny eksperyment, ale wymaga osobnej definicji budżetu i wag źródeł. Nie wykonany w tej serii, która zachowuje pełny FULL_CLEAN. Nawet płaska krzywa na podzbiorach COCO/VG nie wyklucza korzyści z nowej domeny VisualNews.
2. **D3, nielabelowane VisualNews + regresja do teacher embeddings:** nie uruchomiony. Wykracza poza bieżącą serię niezależną od nowych VisualNews i zmienia wcześniejsze założenie KD po macierzach podobieństwa. Potrzebny byłby również niezależny audyt tych obrazów względem chronionych zbiorów przed destylacją. Brak captionów nie usuwa image leakage.
3. **MobileCLIP2-S2:** zysk historyczny jest duży, ale produktowo nie jest to darmowy zamiennik. Oficjalna licencja ogranicza użycie do niekomercyjnych badań i wyłącza product development. Nie zawiera wyraźnego wyjątku dla destylacji do produktu; nie zakładam, że użycie wyłącznie jako teachera omija warunki dotyczące użycia modelu. Nie rozstrzygam tu kwalifikacji prawnej konkretnego modelu pochodnego. [Oficjalna licencja Apple](https://huggingface.co/apple/MobileCLIP2-S2/blob/main/LICENSE).
4. **Większy SigLIP2 teacher:** potencjalna osobna kontrola jakości vision w tej samej przestrzeni; niewykonana. Nie jest konieczna do rozstrzygnięcia obecnego pytania o brak augmentacji/regularyzacji.
5. **VisualNews pipeline:** udział recycle należy jawnie raportować jako syntetyczne opisy wizualne. Niezależna ocena groundingu innym modelem i próbką ludzką jest rozsądna. W tej serii nie zmieniam procesu działającego na drugim serwerze ani nie przerywam jego pracy na podstawie ekstrapolacji z pilota.
6. **XTD10:** zbiór rzeczywiście obejmuje polski; autorzy opisują te captiony jako tłumaczenia wykonane przez ludzi. To przydatna kontrola tłumaczenia, ale nadal COCO i nadal tłumaczenia, a nie naturalne krótkie zapytania użytkowników. Nie został pobrany ani oceniony; wymaga sprawdzenia overlapu. [Repozytorium autorów](https://github.com/adobe-research/Cross-lingual-Test-Dataset-XTD10), [publikacja, sekcja 4.1.1](https://arxiv.org/html/2109.07622).
7. **Development holdout poza COCO:** merytorycznie przydatny; jego tworzenie jest oddzielnym protokołem danych. Nie zmieniono obecnej walidacji ani snapshotu. Przyszły holdout musi być wyłączony również z uczenia bez captionów.
8. **XM3600:** pozostaje blind. Zaproponowana przez drugi model reguła finalistów jest propozycją, nie zgodą użytkownika na uruchomienie ani przyjętym automatycznie progiem produktu. Osobna decyzja nadal wymagana.
9. **Osobna projekcja po stronie tekstu:** może zmienić geometrię retrievalu i zasady produktu, ale D1 nie wskazuje na postulowany niski sufit. PolDense i jego wyjściowa przestrzeń nie zostały zmienione.

## Odtworzenie

W środowisku WSL projektu:

```text
python training/tinyclip_diagnostics.py
python training/tinyclip_frozen_replication.py
python training/tinyclip_generalization.py
```

Nie uruchamiać równolegle kilku workerów treningowych na tej samej karcie. Generalization ma blokadę procesu i resume na granicy epoki. Testy diagnostyk sprawdzają usunięcie własnego captionu z centroidu, obrazy z jednym captionem, affine ridge, zgodność widoków między kontrolami i zerowy gradient anchoru w inicjalizacji. Testy nie zastępują weryfikacji rzeczywistych cache'ów/modeli, którą wykonują skrypty.
