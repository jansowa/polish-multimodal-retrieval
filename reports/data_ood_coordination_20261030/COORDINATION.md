# Kontrakt dwóch sesji: nowe dane i OOD, 30.10.2026

Status: przygotowany do wykonania w dwóch osobnych sesjach. Samo przygotowanie tych plików nie uruchamia treningów, pobierania danych ani ewaluacji.

## Cel i pierwszeństwo

A bada wpływ nowych VN/Wiki przy niezmienionych enkoderach i headzie. B rozszerza wyłącznie retrieval tekst↔obraz o dane spoza treningu projektora, bez nowych ręcznych adnotacji. W tej nowej serii prace te mają pierwszeństwo przed dawnymi P1.B/C/D. Nie uruchamiać ich równolegle ze scoutingiem, FT, LoRA ani destylacją. XM3600 pozostaje FINAL_BLIND.

Nie zmieniać zamrożonego FULL_CLEAN, validation, VN35918, starych snapshotów Wiki, Wiki dev397, dotychczasowych wyników, bramek ani alokacji działających pipeline'ów. Nie czytać aktywnych SQLite/WAL. Odłożenie checkpointu lub obrazu nie robi z COCO nowej domeny. Po uczeniu na Wiki jego dev jest dla danego modelu ID; baseline sprzed Wiki zachowuje dawny status.

## Własność plików

| Obszar | Wyłączny właściciel |
|---|---|
| `experiments/data_study_20261030/` | A: kod danych, cache i treningu |
| `reports/data_study_20261030/` | A: rejestracje i raporty treningu |
| `experiments/ood_eval_20261030/` | B: kod pobierania, tłumaczenia i ewaluacji |
| `reports/ood_eval_20261030/` | B: rejestracje i raporty nowych testów |
| `experiments/data_ood_common_20261030/` | A: wspólna biblioteka blokad, budżetów i publikacji; B tylko odczyt |
| `reports/data_ood_coordination_20261030/COORDINATION.md`, prompty | Materiały startowe; obie sesje tylko odczyt |
| `COMMON_READY.json`, `A_INPUTS_FROZEN.json`, `A_UNIVERSE_FROZEN.json`, `A_FINALISTS_FROZEN.json`, `A_GPU_DONE.json`, `A_STATUS.json` | A |
| `B_EVAL_FROZEN.json`, `B_RESULTS_READY.json`, `B_STATUS.json` | B |
| `DISK_RESERVATIONS.json` | A i B wyłącznie przez wspólny helper pod globalną blokadą |

Markery powyżej należą do `reports/data_ood_coordination_20261030/`. Nie nadpisywać kanonicznych `reports/series_2026-10-23/REPORT.md` i `summary.json`, `src/`, istniejących środowisk ani cudzych plików. Stare moduły można importować. Konieczne poprawki wspólne proponować w swoim raporcie, a nie wdrażać równolegle. Bez commitów/resetów/checkoutów/worktree, które mogłyby zabrać niezatwierdzone zmiany użytkownika.

Duże pliki trzymać pod `C:/Users/jsowa/benchmark-translations/data-ood-20261030/A/` i `/B/`. Na D tylko mały kod, metadane i raporty. Ścieżki Windows/WSL normalizować istniejącym `local_path`; nie tworzyć drugich kopii tych samych obrazów.

## Protokół przekazania

1. A jako pierwsze zadanie implementuje i sprawdza wspólny helper, publikuje `COMMON_READY.json` z SHA kodu i opisem API. B w tym czasie robi research i odczyty małych metadanych; większe pobieranie zaczyna po tym markerze.
2. `A_INPUTS_FROZEN.json`: dokładne ścieżki/SHA eksportów i starych wejść. Domyślny punkt startu: już zamrożone 132875 VN +24125 Wiki. Nowy eksport jest dopuszczalny tylko jako osobna, uprzednio zarejestrowana wersja przed zamrożeniem uniwersum. Nie podmieniać danych w toku.
3. A kończy audyt obrazów i publikuje `A_UNIVERSE_FROZEN.json`: suma **wszystkich obrazów i tekstów dopuszczonych do jakiegokolwiek wariantu A**, również historycznego train. Indeks ma zweryfikowane ścieżki/ID, rodziców, URL/tożsamość źródła, raw SHA, pixel SHA, pHash i znormalizowane teksty; obejmuje też chronione dev/val/eval. Nie publikować tego markera na podstawie samego starego audytu metadanych.
4. Po markerze A może tylko wybierać podzbiory, zmieniać próbkowanie zgodnie z prerejestracją i usuwać niekwalifikujące się przykłady z tego uniwersum. Nie dodaje nowych obrazów ani nowych wersji captionów. Zmiana puli oznacza nową generację i unieważnienie zależnych markerów B; nie wykonywać jej samowolnie.
5. B wybiera OOD wyłącznie spoza całego uniwersum A oraz chronionych zbiorów. Dopiero po pełnym audycie publikuje `B_EVAL_FROZEN.json`, z manifestami, SHA, wersjami tłumaczeń, galeriami, qrels i komponentami. Dzięki temu A nie musi czekać z treningiem na B.
6. A wybiera modele bez odczytywania wyników nowych benchmarków B. Publikuje `A_FINALISTS_FROZEN.json` z checkpointami/SHA i zamkniętą selekcją, a po zakończeniu wszystkich swoich zadań GPU — `A_GPU_DONE.json`.
7. B zaczyna obliczenia GPU po obu markerach A i swoim `B_EVAL_FROZEN.json`. Liczy ustaloną listę checkpointów i referencji. Wyniki B nie służą do dalszego strojenia w tej serii. B nie uruchamia nowych treningów A.

Markery są immutable: publikacja atomowa po zakończeniu zapisu wszystkich zależności, ze schematem, generacją i SHA każdego pliku. Używać formatu z `HANDOFF_FORMAT.md`. Konsument sprawdza hashe, nie tylko istnienie pliku. Własne `*_STATUS.json` są atomowo aktualizowane i mogą wskazywać oczekiwanie; nie stanowią zgody na pominięcie bramki. Przy oczekiwaniu kontynuować niezależne przygotowanie; nie uruchamiać własnych zamienników zadań drugiej sesji.

## GPU, dysk, CPU i sieć

- Jeden worker GPU. Wszystkie operacje CUDA, także ekstrakcja i translator, biorą ten sam POSIX `flock` na istniejącym `training/runs/tinyclip-visualnews-v1/worker.lock` w WSL. Nie używać innego pliku jako pozornej blokady. B dodatkowo czeka na `A_GPU_DONE`; nie zabijać ani nie przejmować cudzych procesów.
- A dostarcza helper rezerwacji dysku z krótką globalną blokadą i atomowym rejestrem pozostałych, jeszcze niezapisanych rezerwacji. Przed planowanym zapisem/pobraniem sprawdzić **fizyczne C i D**, rozmiar brakujących plików, bufor tymczasowy i rezerwacje drugiej sesji. Po zapisie rozliczać rezerwację; nie odejmować drugi raz już zapisanych bajtów.
- Twarda rezerwa: minimum 4 GiB na każdym fizycznym wolumenie. Odczyt `/` WSL nie zastępuje C. Aktywne pipeline'y nadal zużywają miejsce. Budżety nowych trwałych plików: A do 8 GiB, B do 3 GiB; wspólne tymczasowe do 4 GiB. To limity, nie gwarancja miejsca. Gdy brak miejsca, nie kasować; wybrać mniejszy pilot przed zamrożeniem albo zgłosić potrzebną pojemność.
- B na CPU maksymalnie 2 workery i 2 wątki Torch. A nie uruchamia wielu równoległych skanerów obrazów; maksymalnie 2 dekodery, strumieniowanie i ograniczony RAM. Unikać wielokrotnego dekodowania tych samych assetów.
- Nie zmieniać pracy pipeline'ów VN/Wiki ani zastanej alokacji (w odczycie30.10:20slotów,10+10). Nie przywracać tych liczb, jeśli użytkownik później je zmienił. Pobieranie Commons wyłącznie przez istniejący wspólny limiter. Tłumaczenia B: najpierw cache/gotowe wersje PL; nowy lokalny translator na CPU albo dopiero po `A_GPU_DONE`. Istniejący endpoint zdalny tylko gdy ma udokumentowaną bezkolizyjną kolejkę/limiter — nie dodawać potajemnie obciążenia i nie podszywać się pod kolejki VN/Wiki.
- Nie usuwać datasetów, cache'ów, starych runów ani folderów. B może usuwać **wyłącznie własne nowo pobrane wagi referencji/translatora**, po zapisaniu wszystkich zależnych wyników i sprawdzeniu ścieżki/SHA oraz braku obcych odwołań. TinyCLIP39M/PolDense są chronione. Wagi jednej referencji naraz; brak zgodności zatrzymuje cleanup, nie rozszerza jego zakresu. Xet i inne cache'e pozostają.

## Definicja OOD i metryk

OOD oznacza nowy zbiór/źródło względem treningu projektora po audycie rozłączności, nie udowodniony brak ekspozycji w pretreningu. Osobno opisywać przesunięcie semantyczne, długość tekstów i tłumaczenie. Flickr ma inną kolekcję, ale ogólna fotografia pozostaje podobna. VizWiz daje inny profil zdjęć; DOCCI — długie, precyzyjne opisy. WIT/Wiki po treningu Wiki i XTD/COCO nie są nowymi domenami w tym sensie.

Tylko query tekstowe→obraz oraz obraz→tekst. Bez CIRR, FashionIQ, RefCOCO, obraz+tekst, generowania obrazów, VQA ani ręcznych nowych etykiet. Qrels z oryginalnych istniejących powiązań, połączone dla identycznych tekstów; pHash służy komponentom/leakage, nie tworzy nowych pozytywnych etykiet.

Osobne galerie, native cosine, v2 bez dodatkowych instrukcji; `[query]: ` jest intrinsic dla naszych modeli. R@1/5/10, MRR, NDCG@10; paired bootstrap komponentów 10 000, seed20261023; trzy seedy mean±SD oddzielnie od warunkowych CI. Bez średniej/rankingu łączącego różne galerie. W tabelach V+T łącznie, V i T osobno; osobny head i pozostałe enkodery poza sumą; ID/OOD per checkpoint i benchmark, konkurencja bez audytu jako nieustalona.
