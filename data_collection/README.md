# Pozyskiwanie danych o polskiej kulturze

Czwarta, niezależna odnoga projektu. Zbiera istniejące pary obraz–tekst z
Wikimedia Commons i kontekst z Wikidata. Nie używa modeli do tworzenia captionów.
Nie importuje treningu, benchmarków ani klienta tłumaczeń. `collect` pobiera
wyłącznie metadane; obrazy są osobnym, jawnym krokiem.

Do wygodnego przeglądania całego zbioru służy [przeglądarka WWW](BROWSER.md)
z numerowanymi stronami, filtrami i szczegółami rekordów:
`python data_collection/browse.py serve --port 8766` → http://127.0.0.1:8766.

## Instalacja i pierwszy przebieg

Do większego zbierania użyj [trybu szybkiego](FAST.md):
`python collect.py collect --config data_collection/configs/fast.yaml`.
Łączy paczki API, trwały cache, import wcześniejszej kolejki i wspólny limiter;
nie czeka na dostępność SPARQL. Dokument zawiera pomiar i zasady wznowienia.

Z głównego katalogu repozytorium, Python >=3.11, bez GPU:

```bash
pip install -r data_collection/requirements.txt
python collect.py collect --config data_collection/configs/small_test.yaml
python collect.py validate --config data_collection/configs/small_test.yaml
python collect.py stats --config data_collection/configs/small_test.yaml
python collect.py preview --config data_collection/configs/small_test.yaml --sample-size 12 --seed 42
```

Alternatywnie `pip install -e '.[collection]'` udostępnia `pl-mmb-collect` oraz
`python -m pl_mmb_collection`. Bez instalacji pakietu działa `collect.py`.
Przed większym przebiegiem ustaw nazwę aplikacji w `http.user_agent` i kontakt
w `http.contact`. Konfiguracja `fast.yaml` zawiera uzgodniony kontakt projektu.

`small_test.yaml` ogranicza odkrywanie do 20 encji i maksymalnie dwóch kandydatów
na encję. Liczba **zaakceptowanych** plików może być mniejsza: limity dotyczą
kandydatów, a filtry licencji i jakości są stosowane później. Konfiguracja
`wikimedia.yaml` pokazuje szersze wyszukiwanie i kontrolowane rozszerzanie grafu.
YAML i JSON mają ten sam schemat, nieznane klucze są błędem. Wszystkie ścieżki
w konfiguracji są względem bieżącego katalogu roboczego.

Poszczególne etapy:

```bash
python collect.py discover-entities --config data_collection/configs/wikimedia.yaml
python collect.py collect-metadata --config data_collection/configs/wikimedia.yaml
python collect.py download-images --config data_collection/configs/small_test.yaml
python collect.py deduplicate --config data_collection/configs/small_test.yaml
python collect.py validate --config data_collection/configs/small_test.yaml
```

`--source wikimedia` i `--output outputs/inny_przebieg` mogą nadpisać konfigurację.
`collect-metadata` korzysta z encji już zapisanych przez `discover-entities`.
Kody zakończenia: 0 — sukces, 1 — błąd konfiguracji/walidacji, 2 — pozostały
nieudane zadania API/pobierania, 130 — przerwanie klawiaturą. Pusty zbiór może
być poprawny strukturalnie; sprawdź `records`, odrzucenia i zadania oczekujące.

## Struktura kodu

```text
collect.py                         # entrypoint bez instalacji
data_collection/
  configs/{small_test,wikimedia}.yaml
  requirements.txt
  README.md
src/pl_mmb_collection/
  cli.py, config.py, preview.py
  models/record.py                 # Entity, ImageCandidate, DatasetRecord
  sources/
    base.py                       # protokół DataSource
    http.py                       # rate limit, retry, raw cache
    wikidata.py                   # encje, claims, SPARQL
    wikimedia.py                  # P18, kategorie, depicts, MediaInfo
    normalization.py              # tekst, języki, provenance
  pipeline/
    discovery.py, collect.py, download.py
    filters.py, dedup.py, reporting.py
  storage/
    checkpoint.py                 # SQLite jako źródło prawdy
    parquet.py                    # jawny Arrow schema, atomowy eksport
tests/data_collection/
  fixtures/commons.json
  test_collection.py
```

## Odkrywanie i zakres

Źródła startowe to `discovery.seeds` (QID-y), `discovery.categories` i opcjonalne
`discovery.sparql_classes`. Dla klas zapytania WDQS ograniczają wyniki do
`country_qid` przez P17 (kraj), P27 (obywatelstwo) lub P495 (kraj pochodzenia).
Przykładowa konfiguracja obejmuje obiekty architektury, muzea, zamki, kościoły,
malarstwo, rzeźbę, literaturę, muzykę, naukowców, polityków i wydarzenia.
Kategorie obejmują też jedzenie, folklor, stroje, tradycje, II wojnę światową,
Solidarność, krajobrazy, Tatry, Bałtyk, transport i sport. Listy można rozszerzać.

Graf rozwija się po **jawnie wybranych wychodzących właściwościach**
(`properties`, np. P150/P527/P800), z ograniczeniami `max_depth`,
`max_links_per_entity` i `max_entities`. Cykl nie powoduje ponownego pobierania
tej samej encji. Nie jest to dowód, że każda pochodna encja dotyczy Polski;
seed i ścieżka pozyskania zostają w provenance. Zapytania krajowe nie obejmą
wszystkich historycznych powiązań ani wszystkich polskich twórców za granicą.
Przy małym `max_entities` wcześniejsze seedy i klasy mogą wypełnić limit.

Kandydaci obrazów pochodzą kolejno z P18, opcjonalnego wyszukiwania Commons
`haswbstatement:P180=QID`, a następnie kategorii P373/sitelinku Commons.
Kategorie mają własne limity plików, kategorii i głębokości. Związek przez
kategorię może być szeroki — np. zdjęcie motyla z Lublina dotyczy miejsca,
ale nie musi przedstawiać zabytku. Nie jest automatycznie traktowany jako
potwierdzone `depicts`. Włącz `use_depicts_search` dla dodatkowych kandydatów.

Używane są oficjalne API:
[Wikibase](https://www.mediawiki.org/wiki/Wikibase/API),
[Imageinfo](https://www.mediawiki.org/wiki/API:Imageinfo),
[MediaInfo/captions](https://commons.wikimedia.org/wiki/Commons:File_captions).
Kod nie scrapuje stron HTML; usuwa tylko markup z tekstów zwróconych przez API.

## Rekord, tekst i provenance

`DatasetRecord` ma jawny model dataclass i schema Arrow, także dla pustego
Parquet. Zawiera identyfikatory, URL pliku i strony źródłowej, QID, nazwy/opisy
PL/EN, captiony PL/EN, opis obrazu, `depicts`, kategorie, twórcę, attribution,
datę, lokalizację, licencję i URL licencji, rozdzielczość, MIME, czas pobrania
oraz pochodzenie tekstu i obrazu. Brakujące dane pozostają `null` lub pustą
listą. `width/height` dotyczą oryginału, `image_width/image_height` wskazanego
wariantu `image_url`. Pobieranie plików zawsze używa `image_original_url`.

`caption_source` i `caption_quality` są strukturami `{pl, en}` — każdy język
może mieć inne źródło. `text_candidates` zachowuje także alternatywne teksty:

| Jakość | Źródło | Znaczenie |
|---|---|---|
| A | `structured_caption` | caption konkretnego pliku z etykiety MediaInfo |
| B | `image_description`, `image_title` | opis pliku lub jego tytuł |
| C | `depicts` | identyfikatory obiektów P180, bez generowania zdania |
| D | `entity_description`, `entity_title` | ogólny kontekst encji |
| D | `category_description`, `collection_description`, `category_title` | kontekst kategorii/kolekcji |

Wybór captionu jest osobny dla PL i EN: A, następnie B, następnie dostępny
tekst kontekstowy D. QID-y P180 zostają w `depicts` i `text_candidates`; nie
udają naturalnego captionu. Nieoznaczone językowo opisy i tytuły mają `und`,
a identyfikatory `zxx`. Parametr języka interfejsu API nie jest dowodem języka
opisu. Kontekst kategorii nie jest opisem konkretnego zdjęcia. Adaptery mogą
przekazać jawny opis kategorii/kolekcji przez `ImageCandidate.context_description`;
obecny adapter Commons pobiera jej tytuł, nie ekstrahuje opisów z wikitextu.

Jeden rekord oznacza jeden unikalny obraz. `associations` przechowuje wszystkie
znalezione powiązania z encjami, seedami, metodami i kategoriami. Główne pola
encji/captionu pochodzą z pierwszego zaakceptowanego kontekstu. `raw_requests`
wskazuje pliki `raw/<hash>.json`, zawierające URL zapytania, czas i pełną
odpowiedź API. Zachowywane są rewizje encji i MediaInfo oraz SHA1 źródła.
Kontynuacja Imageinfo do historii wersji pliku jest pomijana; dalsze strony
kategorii są odczytywane. Cache jest snapshotem bez TTL: odświeżony dataset
powinien używać nowego katalogu przebiegu.

## Licencje i filtry

Domyślnie wymagane są **jednocześnie jawna nazwa licencji i zgodny URL**.
Allowlista obejmuje CC0 i wymienione wersje CC BY/CC BY-SA. Nieznane licencje,
niezgodność nazwy z URL, ogólne deklaracje public domain bez wspieranego
URL i odmiany krajowe są odrzucane. Nie jest odgadywany status pliku.
To celowo pomija także część plików, które mogłyby nadawać się do ponownego
użycia. Licencja obrazu nie zastępuje warunków ponownego użycia opisów i innych
metadanych; ich surowe źródła zostają do audytu.

`creator`, `attribution`, `license`, `license_url` i `source_page_url` zachowują
dane źródłowe. Sam fakt pobrania nie realizuje obowiązków attribution/share-alike
przy dalszej publikacji. Informacje o wymaganiach źródła:
[ponowne użycie treści Commons](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia).

Filtry obejmują minimalną szerokość, wysokość i obszar oryginału oraz miniatury,
MIME, SVG, obecność użytecznego tekstu i licencji. Opcjonalne regexy kategorii
pozwalają odrzucać mapy, flagi, herby, diagramy, skany/dokumenty i pliki techniczne.
Te heurystyki są domyślnie **wyłączone**. `require_license: false` oraz pusta
allowlista jawnie wyłączają ochronę licencyjną; domyślna konfiguracja jej wymaga.

## Output i wznowienie

```text
outputs/polish_culture_pilot/       # small_test.yaml
  manifest.json
  checkpoint.sqlite3
  raw/<request_sha256>.json
  metadata/
    records.parquet
    records.jsonl                 # jsonl: true
    stats.json
    validation.json
    deduplication.json            # po deduplicate
  images/                         # po download-images
  logs/collection.log
  preview.html                    # po preview
```

SQLite zapisuje zaakceptowane/odrzucone pliki, encje, tożsamości i błędy.
Powtórz identyczną komendę, aby wznowić. Ukończone encje i pliki są pomijane;
nieudane zadania pozostają do ponowienia. Wspólny obraz nie jest pobierany
ponownie dla kolejnej encji. Zapisy pliku i checkpointu są atomowe. Po etapie
metadanych, również po `Ctrl+C`, powstaje atomowy eksport Parquet. Po twardym
zabiciu procesu SQLite pozostaje autorytatywny; ponowienie `collect-metadata`
odtworzy eksport. Nie uruchamiaj dwóch writerów do tego samego outputu.

Manifest wiąże całą konfigurację i wersję pipeline’u. Zmiana filtrów, limitów
czy polityki wymaga nowego `output_dir`; zapobiega to mieszaniu polityk
odrzuceń/wznowienia. Cache można skopiować do `raw/` nowego przebiegu.

Żądania są sekwencyjne, z timeoutem, przerwą, exponential backoff, obsługą
`Retry-After`, 429, 5xx i API `maxlag`. Błąd pojedynczego pliku nie zatrzymuje
pozostałych. Log i tabela `failures` zachowują przyczynę. Przy odpowiedziach
429 zwiększ `http.interval` w konfiguracji nowego przebiegu; nie zwiększaj
równoległości. Zasady źródła: [User-Agent policy](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy).

## Deduplikacja i podgląd

Na etapie metadanych exact dedup używa ID źródła, URL oryginału i SHA1.
Alias zachowuje provenance, ale nie tworzy kolejnego wiersza. Po pobraniu
oryginału weryfikowany jest źródłowy SHA1, format obrazu i zapisywany SHA256.
Powtórne pobieranie pomija istniejące pliki o zgodnym SHA256.

Opcjonalny pHash na CPU:

```bash
pip install -e '.[collection,collection-phash]'
python collect.py download-images --config data_collection/configs/small_test.yaml
python collect.py deduplicate --config data_collection/configs/small_test.yaml --phash
```

`download.phash` włącza także zapis pHash podczas pobierania. Near dedup oznacza
`duplicate_of` i zapisuje raport kandydatów, **nie usuwa obrazów ani rekordów**.
Próg odległości Hamminga jest konfigurowalny. Aktualne porównywanie pHash jest
kwadratowe; dla dużego datasetu potrzebny będzie indeks. `PerceptualHasher`
jest miejscem rozszerzenia fingerprintów; wyszukiwanie po embeddingach wymaga
oddzielnego indeksu/kroku i nie instaluje obecnie CLIP/DINO.

`preview` tworzy samodzielny HTML z losowaniem reservoir sampling (stały seed),
szukaniem, filtrem jakości, licencjami i provenance. Dwuklik otwiera go lokalnie.
Obrazy ładuje z Commons na żądanie, więc podgląd wymaga internetu. Aby wybrać
inną próbkę, zmień `--seed`; przycisk w HTML tylko przestawia już wybrane pary.

## Testy i nowe źródła

```bash
PYTHONPATH=src python -m pytest tests/data_collection -q
```

PowerShell, także gdy systemowy katalog tymczasowy jest niedostępny:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/data_collection -q --basetemp outputs/collection_tests_new
```

`--basetemp` powinien wskazywać nowy, przeznaczony wyłącznie do testów katalog.
Testy nie korzystają z live API. Obejmują parsery, captiony/języki, filtry,
licencje, exact/near dedup, pobieranie, retry/cache, wznowienie, graf/kategorie,
stabilny Parquet, HTML oraz pełny lokalny przebieg 20 encji.

Aby dodać Europeana/NAC/Polona/SzukajWArchiwach:

1. Zaimplementuj `DataSource` (`discover`, `candidates`, `fetch`) w `sources/`.
2. Zwracaj `DatasetRecord` z własnym `source`, stabilnym ID, licencją i URL;
   zapisuj raw odpowiedzi oraz `associations`. Brak metadanych pozostawiaj pusty.
3. Dodaj wybór adaptera w `cli.make_source`, konfigurację i fixture odpowiedzi.
4. W razie innych licencji rozszerz jawne parowanie nazwa/URL w filtrze i testy;
   nie uznawaj automatycznie dowolnego URL licencji za dopuszczalny.

Pozostałe etapy, checkpoint i eksport są wspólne. Europeana i polskie archiwa
nie mają jeszcze gotowych adapterów. Pipeline nie ocenia prawdziwości opisu,
nie weryfikuje wizualnie `depicts`, nie generuje tekstu i nie tworzy splitów
treningowych ani benchmarkowych.
