# Ewaluacje dostępne poza leaderboardem

Audyt 31.10.2026. Leaderboard ma sześć galerii, ale nie zastępuje całej walidacji
treningowej. Liczby poniżej pochodzą z obecnych manifestów, nie rozmiarów całych
źródłowych datasetów. Nie uruchamiano nowych ocen GPU podczas tego audytu.

| Co warto wykorzystać | Rekordy / unikalne obrazy | Stan i zastosowanie |
|---|---:|---|
| COCO validation | 2093 / 419 | Dane i ewaluacja lokalna gotowe; retencja i walidacja. 2090 unikalnych PL tekstów. Brak osobnej kolumny nowego leaderboardu. |
| ARO relation validation | 2343 / 527 | Obrazy, pozytywne i negatywne PL podpisy są lokalnie. Potrzebny walidacyjny odczyt image–caption accuracy. |
| ARO attribution validation | 2615 / 448 | Jak wyżej; przydatne do wykrywania utraty rozróżniania atrybutów. |
| SugarCrepe validation | 644 / 143 | Jak wyżej; trudne negatywy i kontrola relacji/obiektów. Źródło COCO. |
| Historyczny MMEB MSCOCO_t2i | 1000 zapytań | Mamy dane, loader, snapshot i wyniki. Osobno lokalne pule kandydatów i istniejąca pochodna pełna galeria; nie mieszać metryk. |
| Dawny Wiki pilot dev | 201 rekordów | Historyczny mały holdout. Nadal chronić, ale nie liczyć jako kolejnej niezależnej domeny obok Wiki native dev v2. |
| XM3600 PL | 3597 obrazów / 7133 par | Po decyzji użytkownika z 31.10.2026 zwykły test leaderboardu; siedem wskazanych modeli w nowej kolejce. |

Sumy obrazów między wierszami nie są addytywne: źródła mogą mieć wspólne obrazy.
Cała validation ma 7695 rekordów i 1216 unikalnych obrazów według pikseli.
Ścieżka: `training/runs/clean-v1/validation.jsonl`.

## Co jest już w leaderboardzie

Flickr30k PL, VizWiz PL, XM3600 PL, Wiki native dev v2, VN dev i oba warianty XTD10.
VN i Wiki są zbiorami selekcyjnymi; XTD-full i clean to warianty tego samego
zadania. Ich obecność nie daje sześciu niezależnych testów OOD.

## Najbardziej użyteczny następny krok przy treningach

1. Zachować VN dev jako obecną regułę wyboru checkpointu; nie zmieniać jej w trwającej kolejce.
2. Monitorować COCO validation jako kontrolę retencji ogólnych umiejętności.
3. Dodać osobny panel diagnostyczny dla 5602 walidacyjnych par ARO/SugarCrepe:
   odsetek `sim(image, positive) > sim(image, negative)`, jawna obsługa remisów,
   osobne wyniki dla trzech źródeł. Nie dodawać ich do głównej średniej retrievalu.
4. Wiki dev służy do selekcji udziału Wiki w obecnej recepcie; nie nazywać go
   końcowym testem OOD po treningu na tym źródle. Flickr/VizWiz raportować po
   zamknięciu wyboru. XM3600 PL jest zwykłym testem leaderboardu zgodnie z decyzją użytkownika.

Przy następnej serii można z góry zadeklarować dopuszczalny spadek COCO/ARO,
obok progu VN. Nie dobierać progu do już obejrzanych wyników i nie narzucać go
wstecz na zamkniętą lub działającą serię.

## Co dokładnie jest gotowe w ARO/SugarCrepe

Źródła dodatnich i ujemnych podpisów:
`training/runs/e1-full/translation_snapshots/{aro_visual_relation,aro_visual_attribution,sugarcrepe}.jsonl`.
Połączenie po `translation_id`, ograniczone do validation, sprawdzono dla wszystkich
5602 rekordów. Teksty obu języków objęto ochroną w rejestrze holdoutów;
lokalny manifest: `retrieval_benchmark/local/holdout-validation-caption-pairs-v1.jsonl`.

`src/pl_mmb_training/tinyclip_diagnostics.py:hard_negatives` to wcześniejsza
diagnostyka podobieństwa tekstów na TRAIN. Nie przedstawiać jej jako wyniku
odróżniania podpisów na walidacyjnych obrazach. Starszy loader ARO istnieje,
ale uruchomienie go na całym źródle obejmuje również przykłady użyte w treningu.

Winoground i ImageCoDe występują w rejestrze tłumaczeń, ale nie są gotowymi
lokalnymi polskimi ewaluacjami. mFAQ/MAUPQA mają dane tekstowe bez obrazów;
nie są testami multimodalnymi. Obecnie tekstowy encoder jest frozen, więc
ich okresowe mierzenie nie ocenia zmian samego projektora wizyjnego.
CIRR/FashionIQ/RefCOCO pozostają poza zamówionym zakresem.

Zasady ochrony wszystkich tych splitów: [DATA_SPLITS.md](../training/DATA_SPLITS.md).
