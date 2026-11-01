# Plan eksperymentów E0–E11

> **Plan historyczny.** Aktualna kolejność i numeracja P0–P11 są w
> [PLAN.md](PLAN.md). E4 oznacza teraz porównanie backbone'ów; composed retrieval
> jest późną fazą P9. Bieżący zakres wykonania: P0 oraz E2a/E2b.

Główny cel użytkowy: **polski tekst → obraz**, bez obrazu w query.
W bieżącym zestawie bezpośrednim pomiarem tego celu jest `MSCOCO_t2i`.
CIRR, FashionIQ i RefCOCO w naszym harnessie mają query obraz+tekst;
ich wyniki oraz czterozadaniowe AVG zachowujemy jako diagnostykę, a nie
kryterium wyboru recepty T2I. Dobry wynik zadań obraz+tekst może częściowo
pochodzić z podobieństwa wizualnego mimo słabego dopasowania do PolDense.

Kolejne porównania powinny raportować przede wszystkim T2I R@1/R@5/R@10.
Obecny wynik MSCOCO dotyczy lokalnych pul MMEB; dodatkowym krokiem ewaluacji
będzie wyszukiwanie w pełnej, zamrożonej galerii testowej COCO/VisualNews,
po przygotowaniu obrazów, tłumaczeń i audytu leakage. Kontrola z przetasowanymi
query pozwoli sprawdzić, czy wynik zależy od dopasowania tekstu do obrazu.
Dla pomocniczych zadań obraz+tekst warto osobno mierzyć vision-only,
text-only i pełne query. Takich ablacjów jeszcze nie wykonano.
E4 pozostaje osobnym rozszerzeniem do composed retrieval; nie jest warunkiem
realizacji głównego celu T2I.

Status implementacji: E0/E1; E2–E11 są planem, nie wykonanymi wynikami.
Dotychczasowy wynik 5435 train / 565 validation jest pilotem 6k, a nie pełnym E1.
Domyślny E1 nie ma limitu par (`max_pairs=null`) i uwzględnia wszystkie skonfigurowane
źródła z ukończonymi tłumaczeniami. Snapshot, brakujące obrazy, brak zweryfikowanych
powiązań i wykluczenia leakage są raportowane oddzielnie dla każdego źródła.
Pełny run nie może być opisywany jako obejmujący wszystkie tłumaczenia, jeśli
część rekordów nie ma gotowych, zweryfikowanych obrazów.
Każdy krok zmienia jedną rzecz. Domyślny text encoder to frozen
`OPI-PIB/PolDense-32M`; pozostaje frozen w E0–E11 opisanych poniżej.
Ewentualne odmrożenie tekstu wymaga oddzielnej, ostatniej fazy po tych etapach.

| Etap | Zmiana | Kryterium / wynik |
|---|---|---|
| E0 | Zamrożony evaluator CIRR/FashionIQ/MSCOCO_t2i/RefCOCO, blacklista wszystkich eval queries/candidates, ID/hash/parent leakage, dwa odtworzone baseline'y | Każda ranga i metryka zgodna; 0 pozostałych overlapów train/eval. |
| E1 | Frozen PolDense-32M + frozen pretrained SigLIP2-256; linear image projector; cosine positive-pair regression | Najpierw MSCOCO_t2i; komplet czterech metryk i AVG; trzy seedy. |
| E2 | Tylko zmiana straty na multi-positive InfoNCE z in-batch negatives | Wszystkie wystąpienia tego samego targetu są pozytywne; bez false negatives. |
| E3 | Linear kontra 2-layer MLP; pozostałe ustawienia jak E2 | Wybór na train-validation, następnie trzy seedy i komplet benchmarków. |
| E4 | Composed query: normalize(v + αt), Linear(concat(v,t)), mały MLP([v,t,v*t,v-t]) | Osobne porównanie CIRR/FashionIQ/RefCOCO; zablokowane dane i projector z E3. |
| E5 | Wspólny multitask training na dostępnych, przefiltrowanych źródłach; balanced sampling per dataset | Log rzeczywistych proporcji i liczby par; bez zmiany architektury. |
| E6 | Najsilniejszy działający model harnessu jako teacher EN; student PL; contrastive + λ·ranking-KD | Cache teacher query/target EN, λ∈{0,0.1,0.3,1.0}; teacher wybierany przed student runami. |
| E7 | Teacher top-k hard negatives; k∈{16,32}; positive + hard + in-batch | Mining wyłącznie w puli train; blacklista eval i wszystkie pozytywy wykluczone z negatywów. |
| E8 | Dla najlepszego runu: ostatnie 1–2 bloki vision albo LoRA | Osobne warianty; PolDense nadal frozen. Cache wejść/teacher pozostaje offline, cech uczonej wizji nie wolno ponownie używać po zmianie wag. |
| E9 | Recepta zamrożona, porównanie PolDense-17M/32M/68M | Nowe rewizje i cache dla każdego modelu; identyczne źródła i splity. |
| E10 | Data scaling: 10k / 30k / pełna bezpieczna pula (~100k jako cel) | Zagnieżdżone podzbiory, stały eval/validation, rzeczywista liczebność po filtracji. Brak deklaracji 100k bez audytu. |
| E11 | Teacher-text-space → PolDense-space na dużym korpusie PL; najpierw linear | Najpierw osobny held-out test mappingu; dopiero potem teacher image embeddings jako dodatkowy sygnał destylacji wizji. |

Wspólne wymagania dla runów:

1. Pełny config, SHA rewizji modeli, wersje bibliotek, seed, hashe danych,
   jawne źródła/splity/ID i snapshot blacklisty. Równoległe EN i PL pozostają
   w artefaktach; teacher otrzymuje EN, student PL.
2. Żadne eval IDs, obrazy, cropy tych samych rodziców ani dane z innych
   datasetów współdzielące eval nie mogą wejść do train, validation, KD ani mining.
   Nieznana linia pochodzenia wymaga odrzucenia lub zweryfikowanego adaptera.
3. Cache wszystkich frozen embeddings offline; cache key obejmuje rewizję,
   preprocessing, pooling, prefiks, token limit, precyzję i zawartość obrazów.
   E8 wymaga odświeżania cech uczonego encodera; zamrożony cache byłby błędem.
4. Powtarzające się targety mają wspólny identyfikator, wszystkie są positives.
   Nie wolno stosować diagonal-only cross entropy dla takich batchy.
5. Wybór checkpointu i hiperparametrów na validation wydzielonym po parent IDs
   z danych treningowych. Eval nie służy do early stopping ani wyboru seeda.
6. Minimum trzy seedy dla obiecujących konfiguracji; publikować mean ± std,
   a nie najlepszy seed. E1 domyślnie już wykonuje trzy seedy.
7. Po każdym E1–E8 tabela `CIRR / FashionIQ / MSCOCO_t2i / RefCOCO / AVG`
   oraz Δ w punktach procentowych wobec poprzedniego eksperymentu.
   Dodatkowe NDCG/MRR/R@k zachować w JSON; nie mieszać metryk w AVG.
8. Testowe wyniki E1 z prostą sumą obrazu i tekstu traktować diagnostycznie
   do E4. Nie optymalizować naraz loss, projector, fusion i źródeł danych.

Plan E11 wymaga jawnego wyboru korpusu, deduplikacji względem tekstów eval
i niezależnego train/validation/test dla mappingu. To eksperyment oddzielny
od bieżącego E1, bez domyślnego pobierania dużego korpusu.
