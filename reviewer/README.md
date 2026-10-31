# Lokalny przegląd EN / PL / obraz

Po uruchomieniu otwórz **http://localhost:8765** w przeglądarce na tym komputerze.

Wybierz zbiór i przebieg w polu „Przebieg”. Przeglądarka wykrywa checkpointy
wszystkich zbiorów w `outputs/` oraz katalogach `output_dir` zapisanych w
`reports/*plan.json`, w tym `C:\Users\jsowa\benchmark-translations`.
Przebiegi pełne i pilotaże są pogrupowane osobno.

- MSCOCO t2i: oryginał EN, tłumaczenie PL i obraz docelowy z kandydata 0.
- CIRR, FashionIQ i RefCOCO: zapytanie EN/PL, obraz wejściowy oraz obraz docelowy
  (w RefCOCO: wycinek obiektu). Obrazy pochodzą z przygotowanych źródeł
  `.cache/absence-v3/*.jsonl`; zgodność tekstu sprawdza SHA-256.
- ARO i SugarCrepe: oba opisy pary, pozytywny i negatywny, w obu językach.
- M-BEIR MSCOCO i VisualNews: przetłumaczone podpisy z nowych katalogów wyników.

Obrazy ARO, SugarCrepe i M-BEIR nie są jeszcze pobrane lokalnie; dla tych zbiorów
strona pokazuje teksty i komunikat o braku obrazu. Nie przypisuje im zastępczych
obrazów z innych zbiorów.

- Szukaj po ID albo po słowach w obu językach.
- Oznacz przykład jako poprawny, do poprawy lub z niejasnym źródłem.
- Dodaj notatkę i opcjonalną propozycję polskiego tekstu.
- Oceny zapisują się automatycznie w `reviewer/annotations.sqlite3`.
- „Eksport ocen” pobiera JSON z ocenami wybranego przebiegu.
- Strzałki na klawiaturze przechodzą między przykładami; 1, 2, 3 ustawiają ocenę.

Oceny nie zmieniają checkpointów tłumaczeń. Są związane z konkretnym przebiegiem,
ID oraz treścią EN–PL (dla par: obu opisów). Dane z ukończonych rekordów są
odświeżane co 30 sekund. Strona pobiera po 100 rekordów; wyszukiwanie, filtry
ocen i losowanie obejmują cały wybrany przebieg.
Strona jest lokalna; nie wysyła ocen ani obrazów do modelu lub usługi hostingowej.

Start w WSL, z katalogu projektu:

```bash
.venv/bin/python -u reviewer/server.py --detach --build-index
```

Po zmianie kodu przeglądarki dodaj `--restart`. Restartuje to tylko serwer
przeglądarki i zachowuje oceny; procesy tłumaczeń pracują dalej.

Proces działa w tle niezależnie od czatu i terminala. Port domyślny to 8765
(zmiana: `--port 8766`). WSL i komputer muszą pozostać uruchomione.

Przy pierwszym uruchomieniu powstaje `reviewer/image_index.json`. Przygotowanie
mapowania korzysta z danych MMEB/Hugging Face i istniejącego cache obrazów;
nie wykonuje zapytań do modelu. Interfejs jest dostępny od razu, a obrazy
pojawią się po zakończeniu indeksowania. Logi:

```text
reports/reviewer_server.log
reports/reviewer_index.log
```

Po restarcie komputera powtórz komendę startową. Można też uruchomić sam serwer
w Windows poleceniem `python reviewer/server.py`, jeśli indeks obrazów już istnieje.
Backend korzysta wyłącznie ze standardowej biblioteki Pythona.
