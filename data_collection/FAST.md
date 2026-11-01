# Szybkie zbieranie przez oficjalne API

## Bieżąca kontynuacja na dysku C

Od 27.10.2026 aktywny przebieg korzysta z konfiguracji `continued.yaml`
i katalogu `C:/Users/jsowa/datasets/polish_culture`. Przejął 239497 rekordów
z `outputs/polish_culture_expanded`, wszystkie dowody surowe, deduplikację
i cache. Poprzedni zbiór na D pozostaje zachowany.

Wznowienie z katalogu głównego repozytorium:

```bash
python -X utf8 -u collect.py collect --config data_collection/configs/continued.yaml --min-free-gb 2
```

Zakres: do 50 tys. encji, głębokość grafu 3, do 100 połączeń na encję,
3000 kandydatów na encję, do 200 kategorii o głębokości 3 i do 5000 członków
kategorii. Filtry akceptacji, wymagania licencji oraz wspólny limiter pozostają
takie same. Te limity określają zakres pracy, nie gwarantowaną liczbę wyników.

Parquet/JSONL są eksportowane po 5000 nowych rekordów lub po 15 minutach
przetwarzania i na koniec przebiegu. Przy większym zbiorze ogranicza to koszt
wielokrotnego przepisywania całego eksportu. SQLite i lokalna przeglądarka
udostępniają nowe dane od razu, niezależnie od eksportu.

`--min-free-gb` pozostawia minimum wskazane w GiB oraz zapas na atomową wymianę
eksportów. Kontrola odbywa się między partiami; przy zbyt małej ilości miejsca
proces kończy się ze stanem `storage_limit_reached` i zachowuje checkpoint.
Po zwolnieniu miejsca uruchom tę samą komendę. Flaga `-X utf8` zapewnia
poprawne kodowanie logów i wyników również po przekierowaniu do plików Windows.

Wskaźnik `outputs/collection_browser/active_dataset.json` wybiera katalog na C
dla identyfikatora `fast` w lokalnej przeglądarce. Adres pozostaje taki sam:
http://127.0.0.1:8766/?dataset=fast.

Poniżej opisano wcześniejsze, mniejsze konfiguracje i mechanizmy wspólne.

Uruchomienie i wznowienie z katalogu głównego repozytorium:

```bash
python collect.py collect --config data_collection/configs/fast.yaml
```

Przebieg zapisuje dane w `outputs/polish_culture_fast`. Polecenie można przerwać
przez Ctrl+C i uruchomić ponownie; checkpoint, cache, ukończone zadania oraz
terminy `Retry-After` przetrwają restart. Krótki pomiar:

```bash
python collect.py collect --config data_collection/configs/fast.yaml --max-seconds 120
```

Limit czasu jest miękki: bieżące żądanie/paczka i końcowy eksport mogą go wydłużyć.
`collect` zbiera metadane i adresy obrazów. Pobieranie oryginalnych plików to
osobny krok `download-images`, korzystający z tego samego limitera.

## Rozszerzony przebieg

Pierwszy zakres ukończył 605 encji i zapisał 8297 obrazów. Zakończenie wynikało
z wyczerpania kolejki, a nie z limitu pół godziny. Rozszerzona konfiguracja
`expanded.yaml` dopuszcza 20 tys. encji, do 1000 kandydatów na encję, głębokość
grafu 2, do 50 połączeń na encję i do 100 kategorii o głębokości 2. Te wartości
są granicami wyszukiwania, nie gwarantowaną liczbą zaakceptowanych obrazów.

Jednorazowe utworzenie większego przebiegu z zachowaniem już zebranych danych:

```bash
python data_collection/expand.py --from-output outputs/polish_culture_fast --config data_collection/configs/expanded.yaml
python collect.py collect --config data_collection/configs/expanded.yaml
```

Pierwszą komendę wykonuje się tylko raz; wymaga nowego `output_dir` i braku
aktywnego kolektora tego budżetu. Kopiuje SQLite, dowody surowe, cache i obrazy,
zachowuje identyfikatory, deduplikację, odrzucenia i odroczenia, a następnie
otwiera kolejkę do szerszego przeszukania. Pozwala wyłącznie zwiększyć granice
wyszukiwania przy identycznych filtrach i polityce pozyskiwania. Raport
`expansion.json` zapisuje źródło oraz zmianę limitów. Poprzedni zbiór pozostaje
bez zmian. Ponowne wznowienia wymagają już tylko drugiej komendy.

Nowe dane trafiają do `outputs/polish_culture_expanded`. Po restarcie lokalna
przeglądarka automatycznie wybiera ten zbiór pod dotychczasowym `dataset=fast`.
Przebieg nie ma ustawionego limitu czasu; kończy się po wyczerpaniu szerszej
kolejki lub po przerwaniu. Monitoruj `progress.json`, aby odróżnić zbieranie,
odroczenie przez API i ukończenie zakresu.

Podgląd całego zbioru ze stronicowaniem:

```bash
python data_collection/browse.py serve --port 8766
```

Adres: http://127.0.0.1:8766/?dataset=fast. Serwer czyta aktualny SQLite,
więc nie czeka na kolejny eksport Parquet. Opublikowana wcześniej strona Sites
jest osobną migawką i wymaga ponownej publikacji, aby pokazywać nowe dane.

## Co przyspiesza zbieranie

- Wikidata: do 50 encji w jednym wywołaniu `wbgetentities`.
- Commons: tanie sprawdzenie wymiarów/MIME do 50 plików, kosztowne
  `extmetadata` do 10 plików, MediaInfo do 50 plików. Filtry wstępne pomijają
  nieprzydatne pliki przed kolejnymi zapytaniami. Lokalny test 50 poprawnych
  obrazów wymaga 7 wywołań API zamiast 100 osobnych wywołań metadanych/MediaInfo;
  wyszukiwanie kandydatów jest dodatkowym kosztem.
- Jedna sesja HTTP utrzymuje połączenia TCP/TLS; odpowiedzi używają gzip.
- Cache odpowiedzi i pojedynczych plików działa również po zmianie składu
  paczki. Deduplikacja scala powiązania zamiast pobierać wspólny obraz ponownie.
- Kolejka encji i surowe odpowiedzi z poprzednich przebiegów są wykorzystywane
  ponownie. Odkrywanie i pobieranie metadanych przeplatają się małymi partiami.
- Tryb `fast.yaml` nie wymaga SPARQL. Używa znanych QID-ów, kategorii Commons,
  wyszukiwania `depicts` i ograniczonego rozwijania grafu. Opcjonalny WDQS ma
  osobną kolejkę odroczeń; jego niedostępność nie blokuje gotowych zadań Commons.
- Eksport Parquet/JSONL odbywa się okresowo, zamiast po każdym rekordzie.

Nie zmieniono wymagań licencyjnych, filtrów rozdzielczości ani oznaczania jakości
tekstów. Jakość D nadal oznacza kontekst encji/kategorii, który może nie opisywać
konkretnego zdjęcia. Wszystkie źródła i surowe odpowiedzi zostają w provenance.

## Budżet żądań i zasady Wikimedia

Konfiguracja identyfikuje klienta przez nazwę aplikacji i uzgodniony adres
`jan.sowa@bielik.ai` w User-Agent. Ma limit **180 żądań/min**, z 5% marginesem
czasowym, jedno żądanie naraz i wspólny budżet dla projektów Wikimedia.
To sufit, nie gwarantowane tempo: opóźnienia odpowiedzi i przetwarzanie zwykle
obniżają wykorzystanie. Każda próba ponowienia zużywa budżet.

Według [Wikimedia API rate limits](https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits)
klient z prawidłowym User-Agent i kontaktem ma obecnie limit 200 żądań/min;
bez takiej identyfikacji limit wynosi 10/min. Walidator konfiguracji nie pozwala
przekroczyć tych wartości. Dokumentacja źródła może się zmieniać; kod zawsze
respektuje odpowiedzi serwera niezależnie od skonfigurowanego sufitu.

`Retry-After`, 429, błędy 5xx i API `maxlag` powodują odroczenie. Po 429 klient
zwalnia, a potem stopniowo odzyskuje tempo. Terminy są zapisane na dysku;
restart ich nie kasuje. API Wikidata i Commons dzielą zakres odroczenia,
WDQS ma osobny. Nie ma rotacji IP/tożsamości ani obchodzenia blokad.

Wszystkie kolektory tej tożsamości/IP powinny używać jednego
`http.state_path` i działać kolejno. Blokada procesu odrzuca drugiego collectora
z tym samym plikiem stanu. Nie uruchamiaj równolegle innej konfiguracji z osobnym
budżetem. Zmienione filtry lub zakres wymagają nowego `output_dir`; wspólny
`state_path` i wcześniejsze katalogi cache można zachować.

Zasady i wskazówki źródła:
[API etiquette](https://www.mediawiki.org/wiki/API:Etiquette),
[koszt extmetadata](https://www.mediawiki.org/wiki/API:Imageinfo),
[dostęp do Wikidata](https://www.wikidata.org/wiki/Wikidata:Data_access).
Do ponownego wykorzystania materiałów nadal stosuje się warunki ich licencji,
w tym oznaczenie autorstwa i ewentualne share-alike; są zapisane w rekordach.

## Import lokalnych list lub dumpów

Przy większej skali można ominąć wyszukiwanie identyfikatorów przez import
wcześniej pozyskanej listy QID-ów albo wycinka oficjalnego dumpa Wikidata.
W **nowej** konfiguracji ustaw:

```yaml
discovery:
  entity_files:
    - data_collection/input/polish_entities.jsonl
  seed_checkpoints:
    - outputs/polish_culture/checkpoint.sqlite3
```

Plik wejściowy: jeden `Q123` na linię albo jeden obiekt encji Wikidata JSON na
linię (akceptowane są też nawiasy i końcowe przecinki liniowego dumpa).
Pełne encje z `claims` można przetworzyć bez wywołania API encji; same QID-y
są uzupełniane paczkami. Import zachowuje surowy obiekt i respektuje
`max_entities`. Pipeline nie pobiera automatycznie wielkiego dumpa i nie filtruje
go pod kątem Polski — przygotuj odpowiedni wycinek wcześniej.

## Rzeczywisty pomiar i monitoring

Pierwszy test 21.10.2026: **659 unikalnych zaakceptowanych rekordów w 122,9 s**,
181 żądań HTTP, 62 trafienia cache, 349 odrzuconych kandydatów, bez błędów HTTP
i odpowiedzi 429. Eksport Parquet przeszedł walidację.
To około **19,3 tys. rekordów/h** po przeliczeniu krótkiego pomiaru oraz
około **88 żądań/min**. Pomiar korzystał z wcześniej odkrytej kolejki i cache;
nie jest wynikiem od zera ani gwarancją tempa długiego przebiegu. Zbieranie
oryginalnych plików nie było częścią pomiaru.

Dowód pomiaru: `outputs/polish_culture_fast/fast_smoke_report.json`.
Stan bieżącego procesu:

- `progress.json`: liczba rekordów, nowe rekordy i tempo od startu procesu,
  kolejki oczekujące/odroczone, liczniki HTTP;
- `http_metrics.json`: rzeczywiste próby HTTP, hosty, błędy, cache i cooldowny;
- `checkpoint.sqlite3`: pełne rekordy, zadania i błędy;
- `metadata/records.parquet`, `metadata/records.jsonl`: okresowe eksporty;
- `logs/collection.log`: szczegóły przebiegu.

Liczniki tempa/HTTP dotyczą bieżącego uruchomienia, nie sumy wszystkich wznowień.
`requests` liczy żądania, a `records` unikalne obrazy: jedno żądanie może zwrócić
wiele rekordów lub nie dać żadnego zaakceptowanego obrazu.
