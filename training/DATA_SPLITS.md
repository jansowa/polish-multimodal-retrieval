# Wspólna polityka splitów i dekontaminacji

Stan audytu: 31.10.2026. Obowiązuje nowe dane, tłumaczenia, recykling, augmentacje
i kolejne treningi. Bieżący interfejs: [current_holdouts.py](../src/pl_mmb_training/current_holdouts.py).
Rejestr nowych audytów: [registry-v4.json](../retrieval_benchmark/holdouts/registry-v4.json).
Zamrożona generacja Wiki2 zachowuje swój rejestr v2 i bazowy `holdouts.py`.

## Dev, test i zakres leaderboardu

| Zbiór lokalny | Liczba rekordów | Rola |
|---|---:|---|
| clean-v1 validation: COCO | 2093 | kontrola retencji i walidacja; źródło ID |
| clean-v1 validation: ARO relation | 2343 | walidacja relacji i trudnych negatywów |
| clean-v1 validation: ARO attribution | 2615 | walidacja atrybutów i trudnych negatywów |
| clean-v1 validation: SugarCrepe | 644 | walidacja kompozycyjna; źródło COCO/ID |
| VN dev | 2003 | selekcja checkpointów; 2000 obrazów w galerii |
| Wiki native dev v2 | 397 | 397 obrazów, 331 tekstów; selekcja wariantów Wiki |
| dawny Wiki pilot dev | 201 | historyczny holdout; nadal chroniony |
| Flickr30k PL local test | 4760 | 952 obrazy, 4746 unikalnych zapytań; MT, test |
| VizWiz PL local validation | 4329 | 1000 obrazów, 4312 zapytań; MT, u nas test |
| XTD10 full / clean | 1000 / 897 | kontrola COCO i tłumaczeń, nie dwie domeny |
| XM3600 PL local | 7133 | 3597 obrazów; zwykły test leaderboardu, 7111 zapytań PL |

Validation 7695 par jest w `training/runs/clean-v1/validation.jsonl`; nie mylić jej
z train `FULL_CLEAN.jsonl`. W leaderboardzie brakuje osobnych kontroli retencji
COCO oraz walidacyjnych ARO/SugarCrepe. Mamy ich dodatnie i ujemne tłumaczenia
w `training/runs/e1-full/translation_snapshots/`. Standardowa kontrola
`score(obraz, poprawny podpis) > score(obraz, trudny negatyw)` wymaga połączenia
**tylko ID walidacji** z tymi snapshotami. Dotychczasowe
`tinyclip_diagnostics.hard_negatives` bada podobieństwo tekstów z TRAIN; nie jest
gotowym wynikiem takiej walidacji obraz–tekst. Całych korpusów ARO/SugarCrepe
nie wolno przedstawiać jako niezależnego testu: używamy części w treningu.

MMEB MSCOCO_t2i to dodatkowy lokalny historyczny retrieval, z innym protokołem
kandydatów; jego wyników nie mieszać bezpośrednio z nowymi pełnymi galeriami.
CIRR, FashionIQ i RefCOCO są poza zakresem zamówionego benchmarku.

Dev służy do wyboru modeli; test do raportowania po zamknięciu wyboru. Wielokrotne
dobieranie recepty na podstawie Flickr/VizWiz również czyni je danymi selekcyjnymi,
nawet gdy ich obrazy nie trafiły do train. Użytkownik 31.10.2026 jawnie zniósł
FINAL BLIND dla XM3600 PL i zlecił zwykłą ocenę w leaderboardzie. Ochrona przed
treningiem nadal obowiązuje; role testu nie należy mylić z zakazem jego oceniania.

## Jedna kontrola przed dopuszczeniem danych

1. Rejestruj holdout **przed treningiem**: wszystkie obrazy galerii, również
   kandydatów bez pozytywnych par, teksty EN/PL i walidacyjne trudne negatywy.
   Rejestr obejmuje także 1159 obrazów całych zarezerwowanych komponentów Wiki.
2. Zachowuj źródłowe ID i relacje: Commons file ID/SHA1, OpenImages, Flickr, COCO,
   strony/artykuły oraz rzeczywiste powiązania encji. Tłumaczenie i recykling
   dziedziczą komponent; nowy tekst nie zeruje pochodzenia.
3. Dla oryginalnego obrazu licz SHA256 bajtów, SHA256 pikseli RGB po EXIF i pHash
   przed cropem/resize/JPEG. Wspólny algorytm: DCT 32x32, blok8x8, mediana z DC,
   odległość Hamminga <=6. Nie mieszaj go z innym wariantem pHash.
4. Buduj komponenty przechodnio, z całych metadanych kandydującej puli, również
   niepobranych aliasów. Jakikolwiek kontakt komponentu z holdoutem wyklucza cały
   komponent. Brak oryginału/fingerprintu nie oznacza czystego przykładu.
5. Zamroź manifest dopuszczonych danych, SHA rejestru, kodu i źródeł. Sprawdź
   ponownie przed treningiem i po oczekiwaniu na GPU. Zmiana któregokolwiek
   wejścia blokuje pracę i wymaga nowego audytu/generacji.
6. Powtarzaj kontrolę po scaleniu źródeł, augmentacji i recyklingu. Wykryty wyciek
   usuwa się z **nowych kandydatów treningowych**, nie po cichu z galerii testowej.

Kod rozpoznaje również aliasy Flickr w URL, ścieżki Windows/WSL i znormalizowane
teksty. Wspólna licencja, autor czy strona główna kolekcji nie są tożsamością obrazu.
pHash jest konserwatywnym filtrem podobieństwa, nie dowodem wszystkich możliwych
cropów ani podstawą tworzenia nowych qrels. Pochodzenie cropów i transformacji
musi być zachowane osobno. Czystość naszych danych nie dowodzi braku ekspozycji
w pretrainingu cudzych encoderów; raportuj ten zakres uczciwie.

## Aktualna kolejka Wiki, generacja 2

Audyt istniejącej kolejki wykazał ochronę starych train/dev/eval, Wiki i XM3600,
ale brak jawnego włączenia później utworzonych Flickr/VizWiz do filtra nowych
danych. Deklarowany ponowny audyt B po treningu nie zapewnia stałego testu.

Poprawka dodaje rejestr do pełnego grafu w `audit_generation.py`, przed wyborem
Wiki_train. Manifest rejestru, kod i binding SHA wchodzą do A_UNIVERSE_FROZEN.
`train_generation.py` ponownie sprawdza Wiki_train oraz utrzymane new_VN_train,
zapisuje `global_holdout_training_release.json` i kontroluje niezmienność wejść
po uzyskaniu blokady GPU. Nie zmienia downloadera, recepty, seedów ani generacji1.

Nowe VN: 19888 rekordów sprawdzono oddzielnie. Wiki wciąż jest pobierane;
`queue_snapshot.json` to obserwacja kandydatów, **nie zgoda na trening**.
Ostateczny status musi wynikać z przyszłego końcowego audytu i release'u.

## Kolejne wersje i inne sesje

Instrukcje użycia: [holdouts/README.md](../retrieval_benchmark/holdouts/README.md).
Generacja Wiki2 ma przypięte dokładne SHA rejestru v2, bazowego `holdouts.py`
i dawnego `catalog.json`; tych plików nie edytujemy. XM3600 jest tam w pełni
chroniony przed treningiem, niezależnie od dawnej etykiety w prywatnym snapshotcie.
Bieżący v4 obejmuje negatywne podpisy walidacji, pełny manifest XM3600 i aktualny
`catalog-current.json`. Nowe runnery używają `current_holdouts` i przypinają
rejestr oraz kod bazowego guarda i bieżącego interfejsu do swojej generacji.
Nie nadpisuj istniejących wersji. Dodanie gotowego
zbioru do katalogu celowo unieważnia stary audyt, dopóki nie powstanie nowy
rejestr uwzględniający jego obrazy i podpisy. Aktywacja nowej wersji jest jawna
w `current_holdouts.DEFAULT_REGISTRY`. Nie zmieniaj guarda ani katalogu
przypiętego do już zamrożonego treningu.

Każdy przyszły runner musi używać tego samego rejestru i wiązać dowód z manifestem
treningowym. Sama obecność modułu nie chroni dowolnego starszego skryptu.
Integracja została wykonana w aktualnej kolejce Wiki generacji2; pozostałe
historyczne entrypointy nie zostały automatycznie przebudowane.
