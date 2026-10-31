# Polski benchmark tekst ↔ obraz

To jest bieżący punkt wejścia dla **retrievalu tekst → obraz i obraz → tekst**.
Wspólna ochrona danych testowych przed treningiem: [polityka splitów](../training/DATA_SPLITS.md)
i [wersjonowany rejestr holdoutów](holdouts/README.md).
Starszy `benchmark.py` pozostaje historycznym runnerem MMEB/ARO, także z zadaniami,
których nie obejmuje ten benchmark. Nie mieszamy jego lokalnych pul kandydatów
z pełnymi galeriami nowego leaderboardu.

```
retrieval_benchmark/
  manage.py          # init, list, plan, run, sync, validate, compare
  catalog-current.json # bieżący zakres i statusy galerii
  runner.py          # jeden model, wspólny scorer, blokada GPU, przypięte rewizje
  ingest.py          # import istniejących wyników bez ponownej inferencji
  local/            # ignorowane: konfiguracja ścieżek i indeks surowych wyników
  publish.py         # plan wysyłki / wysyłka tylko pięciu plików Space
  space/             # niezależna aplikacja Gradio + pojedynczy results.json
```

## Aktualny zakres

Gotowe: Flickr30k PL (952 obrazy), VizWiz PL (1000), Wiki native dev (397),
VN dev (2000), XTD10 pełne (1000) i XTD10 bez znanych nakładań (897).
Core to Flickr/VizWiz/XM3600, Wiki/VN są diagnostyczne i wcześniej używane do selekcji,
XTD10 to kontrola. XTD pełne i oczyszczone nie stanowią dwóch niezależnych domen.
Native PL, ludzkie tłumaczenia, MT i etykiety generowane mają oddzielne oznaczenia.

**XM3600 PL jest zwykłym testem leaderboardu zgodnie z decyzją użytkownika z 31.10.2026.**
Lokalna galeria ma 3597 obrazów i 7111 unikalnych zapytań z 7133 natywnych podpisów;
trzy wcześniejsze wykluczenia są zachowane. `python retrieval_benchmark/xm3600.py run`
ocenia siedem zarejestrowanych modeli. Dane nadal są chronione przed treningiem.
DOCCI PL: trwa przygotowanie tłumaczeń, brak kompletnej galerii retrieval.

## Import i lokalny leaderboard (bez GPU)

Z katalogu głównego projektu, Python >=3.11:

```bash
python retrieval_benchmark/manage.py init
python retrieval_benchmark/manage.py sync
python retrieval_benchmark/manage.py validate
python retrieval_benchmark/manage.py list
python retrieval_benchmark/manage.py serve
```

`init` tworzy `local/config.json` z istniejącymi ścieżkami Windows/WSL, 20 referencjami
i dziewięcioma checkpointami projektu. Ścieżki można poprawić w tym jednym pliku;
`--config /inna/konfiguracja.json` podaje się **przed** podkomendą. `init` nie nadpisuje
istniejącej konfiguracji. Duże wyniki trafiają domyślnie na C poza repo.

`sync` czyta oryginalne wyniki serii i OOD, sprawdza zgodność galerii, zachowuje
metryki bez zaokrąglania oraz eksportuje jeden `space/results.json`.
Nie wczytuje modeli i nie pobiera danych. Lokalny indeks źródeł zawiera ścieżki
i SHA plików umożliwiające dalszy audyt/CI; publiczny JSON nie zawiera ścieżek,
podpisów, zdjęć, wag ani danych logowania.

Samodzielny podgląd, najlepiej w osobnym środowisku:

```bash
python -m venv retrieval_benchmark/.venv
# Linux/WSL:
retrieval_benchmark/.venv/bin/pip install -r retrieval_benchmark/space/requirements.txt
retrieval_benchmark/.venv/bin/python retrieval_benchmark/space/app.py
# Windows: użyj odpowiednio .venv/Scripts/pip.exe i python.exe
```

Adres: http://localhost:7860. Aplikacja nie wymaga GPU. Jedna tabela zawiera średni
wynik i osobne kolumny wszystkich gotowych zadań. Wynik zadania jest średnią T2I
i I2T dla wybranej metryki. Średni wynik to nieważona średnia wybranych zadań;
brak choć jednego pomiaru pozostawia średnią pustą. Domyślnie pełne XTD10 nie
wchodzi do średniej, żeby nie dublować oczyszczonego wariantu.
„Konfiguracja tabeli” pozwala wybierać kolumny i zadania do średniej, a także
prompty, profil pomiaru, próg parametrów, seedy i kontrole angielskie.
R3 jest diagnostyką po wcześniejszym niezaliczeniu guardraila, nie zwycięzcą.
Domyślnie pokazujemy wcześniej ustalonych reprezentantów, bez wybierania najlepszego
seeda po wyniku testowym. CSV odpowiada dokładnie bieżącemu widokowi. T2I, I2T,
organiczne, syntetyczne i MT to dodatkowe kolumny domyślnie ukryte.
Organiczne obejmują native PL i tłumaczenia ludzkie, syntetyczne — generowanie
i recykling, a MT jest osobną grupą. Średnie grup wymagają kompletu pomiarów
wybranych zadań danej grupy. Ukrywanie kolumn nie zmienia zbioru zadań w średniej.
Kolor wyniku przechodzi od czerwonego do zielonego względem innych wyników tej
samej kolumny. Parametry, nazwy i brakujące wartości pozostają bez kolorowania.
Czas, precyzja, rewizja i ekspozycja nie mają kolumn w interfejsie; audytowe
informacje pozostają w JSON. Ekspozycja oznacza znajomość danych/domeny z treningu.

`serve --port 7865` zmienia port. `serve --python /sciezka/do/python` wskazuje inne
środowisko Gradio. Na obecnym Windows automatycznie wykrywamy osobne środowisko
UI utworzone na C; nie instalujemy Gradio do środowisk treningowych.

## Nowy pomiar

Worker GPU działa w istniejących środowiskach **Linux/WSL**, bez instalacji pakietów
przez runner. Wymaga dotychczasowych zależności inferencji i PyArrow dla Flickr.

```bash
~/.venvs/pl-mmb-benchmark/bin/python retrieval_benchmark/manage.py plan --model metaclip-s16 --datasets core
~/.venvs/pl-mmb-benchmark/bin/python retrieval_benchmark/manage.py run --model metaclip-s16 --datasets core
python retrieval_benchmark/manage.py sync
```

Jeden model na wywołanie; wszystkie wybrane galerie kończą się przed następnym modelem.
W razie potrzeby worker przechodzi do istniejącego interpretera właściwego dla modelu.
Używa tej samej blokady `training/runs/tinyclip-visualnews-v1/worker.lock` co treningi;
gdy zajęta, kończy z błędem. Nie zmienia alokacji slotów pipeline'ów.
Nie kasuje plików, katalogów ani wcześniejszych wyników. Przy większej kolejce
trzeba uwzględnić rozmiar pobieranych wag i istniejące zasady ich osobnego sprzątania.

Nowy model CLIP/SigLIP Transformers: dopisz do `local/config.json` pod `models`
unikatowy alias, `repository`, niezmienną 40-znakową `revision`, `role` i dodatni
`download_budget_bytes` obejmujący wszystkie wymagane wagi. Preflight wymaga
tego budżetu i 4 GiB rezerwy; obecność wag może oznaczać konserwatywny nadmiar.
Niestandardowe architektury wymagają adaptera `encode(values, images=False)`;
nie deklarujemy zgodności dowolnego repozytorium HF. Model tekstowy bez sparowanej
części wizyjnej nie wystarcza do tego benchmarku.

Galerie są identyczne dla wszystkich modeli i weryfikowane po SHA manifestów,
protokołów i oryginalnych obrazów. Rekord `plan` obejmuje rewizję, implementację,
batch, galerie i prompty. Wyniki zapisujemy niezmiennie, a ukończone pomijamy.
Nowy runner ma własny profil numeryczny; nie miesza go automatycznie ze starszymi
wynikami. Brak pomiaru pozostaje brakiem, nigdy zerem. Próby wykonania niegotowego
DOCCI kończą się przed ładowaniem modelu. XM3600 jest normalnie dostępny.

## Metryki i audyt

Uzupełnianie według klas parametrów ≤500M: [raport](COMPLETION_20261031.md).
Zarejestrowane rewizje, kontrole zgodności i dowody sprzątania nowych wag są poza
repo, w `C:/Users/jsowa/benchmark-translations/retrieval-benchmark-v1/completion-20261031/`.
`completion.py` zachowuje pierwotny dobór dziewięciu referencji i uzupełnia tylko
braki; `own_legacy.py` osobno rejestruje dwa starsze własne warianty SigLIP2/MobileCLIP2
z PolDense-32M. Nie obejmuje XM3600 ani DOCCI. Własne R1/R3 importujemy z oryginalnych
wyników sesji A dopiero po kontroli R0 na tych samych galeriach; nie są liczone ponownie.

Wspólny scorer: `src/pl_mmb_benchmark/full_gallery.py`. Native cosine, FP32 cechy,
pełna galeria, jawne wiele-pozytywne qrels, oba kierunki, R@1/5/10, MRR, NDCG@10.
R@k oznacza trafienie co najmniej jednego znanego pozytywu w top-k.
Model używa natywnego preprocessingu i przypiętej rewizji; oba tryby promptów
mają osobne wyniki. Truncation każdego nowego pomiaru raportujemy, gdy backend
udostępnia tokenizer; brak raportu jest jawny, nie traktowany jako zero.

Parametry V+T to **wizja + tekst**, bez audio, dodatkowego headu i skal loss.
Osobno zapisujemy V, T, head i shared. Historyczne pomiary efektywności mają
osobną precyzję i procedurę; nie utożsamiamy ich z czasem całej ewaluacji.
Przedtreningowa ekspozycja modeli jest nieustalona. ID/OOD dotyczy znanej ekspozycji
treningowej, a nie samego braku identycznego zdjęcia w treningu.

Warunkowe CI różnic OOD są już w publicznym JSON. Nowe porównanie na surowych
wynikach: użyj `compare --model MODEL_ID --reference MODEL_ID --dataset DATASET
--direction t2i --mode v2_no_instructions --profile historical-native-v2 --output
NOWY_PLIK.json`. `MODEL_ID` znajdziesz w JSON. Bootstrap: 10 000 losowań komponentów,
seed 20261023; odmowa przy różnych galeriach/qrels. CI jest oddzielne od SD seedów.

## Hugging Face Spaces i Git

**Nie potrzeba submodułu.** Space jest osobnym repozytorium, którego korzeniem
może zostać zawartość `space/`. HF udostępnia `upload_folder` dla dowolnego lokalnego
folderu. Nie wysyłamy całego projektu ani jego historii Git.

```bash
# Tylko plan: nie łączy się z HF, nie wymaga tokena.
python retrieval_benchmark/publish.py --repo TWOJ_LOGIN/pl-image-text-leaderboard
# Po decyzji o publikacji, z wcześniej wykonanym hf auth login:
python retrieval_benchmark/publish.py --repo TWOJ_LOGIN/pl-image-text-leaderboard --upload
# --public tworzy NOWY publiczny Space; domyślnie nowy Space jest prywatny.
```

Wysyłane jest dokładnie pięć plików: `app.py`, `view.py`, `results.json`,
`requirements.txt`, `README.md`. Istniejący Space zachowuje widoczność.
Nie kasujemy plików zdalnych. Podmiana JSON aktualizuje wyniki bez uruchamiania
modeli w Space. App działa także po skopiowaniu katalogu poza ten projekt.

Jeśli później zechcesz niezależną historię aplikacji, możesz umieścić `space/`
w osobnym repo i dopiero wtedy dodać je jako submoduł. Push wykonuje się wtedy
z repo submodułu. Nie wprowadzamy teraz zagnieżdżonego `.git` ani zmian `.gitmodules`.

Dokumentacja: [Gradio Spaces](https://huggingface.co/docs/hub/spaces-sdks-gradio),
[upload_folder](https://huggingface.co/docs/huggingface_hub/guides/upload).

## Weryfikacja

```bash
python -m unittest discover -s retrieval_benchmark/tests -v
python retrieval_benchmark/manage.py validate
# Opcjonalny test HTTP, Python ze środowiska Gradio:
python retrieval_benchmark/tests/smoke_ui.py
```
