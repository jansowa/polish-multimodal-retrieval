# OOD tekst ↔ obraz — sesja B

Ukończono porównanie zamrożonych danych i modeli. Zero nowych ręcznych adnotacji. Selekcja A była zamknięta przed pierwszym wynikiem B; wyniki B nie służyły strojeniu tej serii.

## Galerie, pochodzenie i audyt

| Lokalna galeria | Obrazy | Unikatowe query PL | Oryginalne captiony | Komponenty |
|---|---:|---:|---:|---:|
| flickr30k_pl_local_test | 952 | 4746 | 4760 | 948 |
| vizwiz_pl_local_validation | 1000 | 4312 | 4329 | 982 |

Flickr: istniejące tłumaczenia NLLB-200-1.3B, nie native PL; 1000 źródłowych ID zweryfikowano z listą testową. To lokalna, oczyszczona galeria, bez deklaracji oficjalnego Flickr30k1K. Repack deklaruje CC BY 4.0, lecz zdjęcia zachowują prawa autorów i warunki oryginalnego dystrybutora (niekomercyjne badania/edukacja i warunki Flickr). Brak redystrybucji.

VizWiz: oryginalne zdjęcia validation pobrano zakresami ZIP, bez całego archiwum ani miniatur. Istniejące ludzkie EN odfiltrowano przez is_rejected/is_precanned i przetłumaczono wyłącznie tekst→tekst na CPU. Oryginalne źródło deklaruje CC BY 4.0. Dokładne modele, rewizje, SHA, parametry, cache i koszty są w summary.json.

Wykluczenia wobec całego uniwersum A i chronionych train/dev/val/eval objęły źródła/rodziców/URL, file i pixel SHA, pHash≤6 oraz identyczne znormalizowane EN/PL. Odrzucenia propagowano przez komponenty. Po MT powtórzono audyt PL. QA jest automatyczną kontrolą powierzchniową, nie ludzkim potwierdzeniem semantyki; przy niepewności pomijano cały obraz przed score, zachowując wszystkie oryginalne zaakceptowane captiony wybranych obrazów. Liczniki QA obrazów: `{"Flickr_images": 48, "digits_changed": 27, "images": 69, "language_uncertain": 26, "negation_changed": 27}`.

Qrels wynikają wyłącznie z oryginalnych par. Identyczne znormalizowane EN/PL łączą znane pozytywy; pHash nie dodaje pozytywów. Galerie dobierano hashem całych komponentów, seed 20261023, bez wyników modeli. DOCCI zachowano jako rezerwę (oryginalne archiwum przekracza budżet); AIDe zablokowane przez dostęp/prawa do oryginałów bez korespondencji.

Źródła: [Flickr repack](https://huggingface.co/datasets/aplominski/flickr30k-multilingual), [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/), [VizWiz](https://vizwiz.org/tasks-and-datasets/image-captioning/), [DOCCI](https://google.github.io/docci/), [AIDe](https://zil.ipipan.waw.pl/Scwad/AIDe).

## Wyniki podstawowe

R@1 w %. Nasze warianty: mean±sample SD trzech seedów; referencje: pojedynczy zamrożony model. Pełne R@1/5/10, MRR i NDCG@10 obu kierunków oraz obu promptów: metrics.csv i summary.json. Nie łączymy rankingów różnych galerii. R3 jest diagnostyką po niezaliczeniu guardraila A.

### flickr30k_pl_local_test

| Model | t2i R@1 [%] | i2t R@1 [%] |
|---|---:|---:|
| R0 | 47.598 ± 0.462 | 51.401 ± 0.947 |
| R1 | 48.848 ± 0.583 | 52.206 ± 0.458 |
| R3 (diagnostyczny) | 47.914 ± 0.739 | 51.120 ± 1.226 |
| metaclip-s16 | 38.243 | 57.878 |
| metaclip-m16 | 44.669 | 62.185 |
| embeddinggemma-2 | 76.001 | 89.496 |

### vizwiz_pl_local_validation

| Model | t2i R@1 [%] | i2t R@1 [%] |
|---|---:|---:|
| R0 | 21.374 ± 0.801 | 20.400 ± 1.179 |
| R1 | 21.699 ± 0.259 | 21.633 ± 0.306 |
| R3 (diagnostyczny) | 20.632 ± 0.163 | 20.600 ± 0.608 |
| metaclip-s16 | 24.351 | 28.300 |
| metaclip-m16 | 27.876 | 37.100 |
| embeddinggemma-2 | 58.465 | 73.100 |

## Niepewność i ograniczenia

Paired bootstrap komponentów: 10 000, seed 20261023, wspólne losowania, stała galeria i identyczne query/qrels dla każdej pary. Warunkowe CI różnic reprezentantów względem R0 i referencji są w paired_CI.csv; obejmują wszystkie metryki i oba tryby promptów. Są oddzielne od mean±SD seedów i nie opisują niepewności pretreningu ani nieznanych pozytywów.

Source-OOD oznacza nową kolekcję względem treningu projektora po audycie. Nie dowodzi braku ekspozycji w pretreningu enkoderów; dla konkurentów ekspozycja pozostaje nieustalona. Flickr zmienia kolekcję, VizWiz profil zdjęć, MT styl językowy. Wiki po uczeniu na Wiki jest ID dla R3. XM3600 pozostaje FINAL_BLIND.

## Parametry, tokenizacja i precyzja

| Model | V+T | V | T | Head osobno | Wymiar |
|---|---:|---:|---:|---:|---:|
| nasze R0/R1/R3 | 70470528 | 38587392 | 31883136 | 196992 | 384 |
| metaclip-s16 | 49722240 | 21959424 | 27762816 | 0 | 384 |
| metaclip-m16 | 85301760 | 38781952 | 46519808 | 0 | 512 |
| embeddinggemma-2 | 438760448 | 167757824 | 271002624 | 0 | 768 |

Własne encodery: FP32 wagi + BF16 autocast, cache/wynik FP32, native cosine wspólnego scorera. Bazowe cechy obrazów i tekstów współdzielono między dziewięcioma headami; sprawdzono zgodność ze stałym adapterem. Referencje ładowano pojedynczo z historyczną precyzją, Gemma bez audio. Tokeny i truncation każdego modelu/galerii/promptu: tokenization.json. Zgodne istniejące pomiary efektywności wskazano po SHA w summary.json; nie wykonywano ich ponownie.

Wszystkie zależności, source/revision/SHA, manifesty, qrels, komponenty, koszty MT i rejestracje: B_EVAL_FROZEN.json, B_RESULTS_READY.json i summary.json. Dane i modele nie zostały opublikowane na zewnątrz.

### Istniejące pomiary efektywności

Mediany ms/element. CPU batch1 FP32; GPU batch64 FP32+BF16 autocast, osobno od historycznej precyzji retrievalu.

| Model | CPU tekst | CPU obraz | GPU tekst | GPU obraz |
|---|---:|---:|---:|---:|
| metaclip-s16 | 18.898 | 137.686 | 0.211 | 3.632 |
| metaclip-m16 | 29.589 | 212.994 | 0.243 | 3.800 |
| embeddinggemma-2 | 101.640 | 5369.046 | 1.453 | 41.744 |
| ours | 14.498 | 77.105 | 0.277 | 2.103 |