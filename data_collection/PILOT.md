# Weryfikacja pierwszego przebiegu Wikimedia

Konfiguracja: `data_collection/configs/small_test.yaml`. Przebieg korzystał
z rzeczywistych API Wikidata i Wikimedia Commons, z zachowaniem surowych
odpowiedzi. Pobieranie pełnych plików graficznych nie było częścią pilota.

Wynik w `outputs/polish_culture_pilot/`:

- 20 odkrytych i przetworzonych encji; brak zadań oczekujących.
- 20 zaakceptowanych, unikalnych obrazów powiązanych z 15 różnymi QID-ami.
- 17 odrzuconych kandydatów; 16 miało licencję spoza jawnej allowlisty lub
  niewystarczające metadane licencyjne. Powody mogą się nakładać.
- 0 nieudanych zadań API po zakończeniu.
- 20 tekstów PL i 20 EN. To 40 wersji językowych, nie 40 obrazów.
- 7 tekstów jakości A, 12 jakości B i 21 jakości D.
- Licencje: CC BY-SA 4.0 — 9, CC BY-SA 3.0 — 4, CC0 — 3,
  CC BY 3.0 — 2, CC BY 2.0 — 1, CC BY-SA 2.0 — 1.
- 16 obrazów ma co najmniej 4 MP, pozostałe 4 mieszczą się w przedziale 1–4 MP.

`metadata/records.parquet` ma 20 wierszy i jawny schema Arrow, także dla pól
zagnieżdżonych. `validate` potwierdziło zgodność z checkpointem, polityką
filtrów i schematem. Ponowne wykonanie `collect` zachowało wynik 20/20 i nie
wykonało nowych żądań API. Przebieg respektował odpowiedzi 429/Retry-After.

Podgląd: `outputs/polish_culture_pilot/preview.html` — 12 losowych par, seed 42.
Pełne przykłady: `metadata/examples.json`; debug: `metadata/records.jsonl`.

## Przykłady z rzeczywistego wyniku

| ID | Obraz | Tekst PL | Źródło / jakość | Licencja |
|---|---|---|---|---|
| wikimedia:10010548 | Lech Walesa - 2009.jpg | Prezydent Rzeczypospolitej Polskiej Lech Wałęsa, 2009. | structured_caption / A | CC BY-SA 2.0 |
| wikimedia:110485696 | 11A 0044.jpg | Fragment ul. Nyskiej. | image_description / B | CC BY-SA 4.0 |
| wikimedia:110525695 | Zamek Królewski na Wawelu (1).jpg | miasto w Polsce, stolica województwa małopolskiego i historyczna stolica Małopolski | entity_description / D | CC BY-SA 4.0 |

Trzeci przykład pokazuje ograniczenie D: opis Krakowa nie opisuje dokładnie
zdjęcia Wawelu. EN tego samego pliku ma bezpośredni opis obrazu jakości B.
Dlatego źródło i jakość są przechowywane osobno dla każdego języka.

## Sprawdzenia

- Nowa odnoga: **34 testy offline przeszły** w Windows. Zawierają pełny
  przebieg 20 syntetycznych encji, eksport/walidację, wznowienie, filtry,
  deduplikację, pobieranie z mockiem oraz bezpieczeństwo HTML.
- Dostępna regresja w istniejącym środowisku benchmarku WSL: **100 testów
  przeszło** (obejmuje także testy scrapera oraz trening/benchmarki).
- Pozostałe testy w istniejącym środowisku tłumaczeń `.venv` w WSL:
  **49 testów przeszło**. Łącznie oba rozłączne zestawy obejmują wszystkie
  **149 testów repozytorium**.
- JavaScript podglądu: `node --check` zakończył się poprawnie.
- Kontrola `git diff --check` dla zmienionych README i pyproject: poprawna.

Systemowy Python nie ma zależności istniejących pipeline'ów (`openai`,
`datasets`, `torch`), a środowisko benchmarku nie ma klienta `openai`.
Nie zmieniano tych środowisk. Runner pytest zainstalowano wyłącznie w
ignorowanym `outputs/collection_test_deps`.

## Ograniczenia i kolejna iteracja

- Pilot potwierdza przepływ P18/kategorie, normalizację i checkpointy.
  Duży run z krajowymi zapytaniami SPARQL i wyszukiwaniem `depicts` wymaga
  oddzielnego audytu pokrycia tematycznego. Paginacja/graf są pokryte fixture.
- Kategorie bywają szerokie: w próbce są również motyl z Lublina, wydarzenie
  w Krakowie i skan aktu urodzenia. Filtry kategorii są opcjonalne i domyślnie
  wyłączone; pipeline nie udaje semantycznej weryfikacji zdjęcia.
- Konserwatywna allowlista odrzuca m.in. wiele historycznych plików z ogólnym
  oznaczeniem public domain bez wspieranego URL. Nie zgaduje ich statusu.
- Pobieranie obrazów i SHA256 sprawdzono lokalnie na fixture. Opcjonalny
  pHash ma adapter i test z kontrolowanym hasherem, ale nie uruchamiano
  produkcyjnego porównania pHash na tym pilocie. Near dedup pozostawia rekordy
  i oznacza kandydatów; nie usuwa danych.
- Europeana/NAC/Polona/SzukajWArchiwach pozostają kolejnymi adapterami.
  Nie dodawano generowania captionów ani deduplikacji po embeddingach.
