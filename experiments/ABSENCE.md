# Pełne tłumaczenia podczas nieobecności

Kolejka obejmuje SugarCrepe (7511), CIRR/MMEB (1000), FashionIQ/MMEB (1000),
RefCOCO/MMEB (1000), ARO Visual Relation (23937) i ARO Visual Attribution (28748).
Łącznie 63196 rekordów. W przypadku MMEB są to kompletne wersje ewaluacyjne
tego benchmarku, a nie pełne oryginalne zbiory treningowe.

W katalogu projektu, w WSL:

```bash
# Jednorazowe przygotowanie pełnych źródeł i sprawdzenie liczebności/obrazów:
OPENBLAS_NUM_THREADS=1 .venv/bin/python -u experiments/absence_suite.py prepare

# Osobne pilotaże po 40 rekordów (losowanie ze stratyfikacją):
OPENBLAS_NUM_THREADS=1 .venv/bin/python -u experiments/absence_suite.py pilot --detach

# Po ocenie pilotaży i zapisaniu raportu oceny:
OPENBLAS_NUM_THREADS=1 .venv/bin/python -u experiments/absence_suite.py full --detach

# Zwięzły status, również po ponownym otwarciu terminala:
.venv/bin/python experiments/absence_suite.py status
```

Ponowienie `full --detach` wznawia zapisane wyniki. Blokada procesu zapobiega
uruchomieniu dwóch kolejek jednocześnie. Zamknięcie terminala lub czatu nie
zatrzymuje odłączonego procesu. Restart Windows/WSL wymaga ponowienia komendy.

Teksty źródłowe i wybrane ID znajdują się w `.cache/absence-v3/` oraz
`reports/absence_suite_plan.json`. Plan zawiera rewizje źródeł, sumy SHA256,
parametry modelu i hash polityki tłumaczenia, bez klucza API. Nie edytuj planu
ani plików źródłowych podczas pracy. Klucz jest odczytywany z `.env` przy starcie.

Wyniki zatwierdzonego przebiegu: `outputs/qwen3.8-27b-absence-v3b-single/<zbiór>/<tryb>/`.
Osobne próbki mają nazwę przebiegu z końcówką `-pilot`.
Pary kontrastowe tłumaczymy wspólnie, tekstowo, zachowując celowo fałszywe opisy;
obrazy oryginalnego benchmarku pozostają bez zmian. CIRR/FashionIQ/RefCOCO używają
obrazów jako kontekstu. Jeden przebieg, bez automatycznego review; audyt zapisuje
tekst wejściowy modelu, odpowiedź i zużycie tokenów.

Produkcja wymaga ukończonego pilotażu oraz wpisu w
`reports/absence_pilot_assessment.json`, związanego z konkretnym źródłem i polityką.
To zapis dokonanej oceny, nie automatyczna ocena jakości modelu.

Kolejka sprawdza dokładny komplet ID, błędy i nadmiarowe rekordy. Po błędzie
przechodzi do kolejnego zbioru; brakujące rekordy ponawia w maksymalnie trzech
obiegach. Proces bez postępów przez 30 minut jest zatrzymywany i później
wznawiany. Przy mniej niż 1,5 GiB wolnego miejsca kolejka się zatrzymuje.
Niekompletne zadania pozostają wyraźnie oznaczone. Brak limitu 24 godzin.

Pełny stan i brakujące ID: `reports/absence_suite_state.json`.
Stan pilotaży: `reports/absence_pilot_state.json`.
Logi: `reports/absence-suite.log` oraz `reports/absence-full-<zbiór>.log`.

`experiments/keep_awake_windows.py`, uruchomiony przez Windows Python, tymczasowo
blokuje automatyczne uśpienie komputera na czas pracy pełnej kolejki. Monitor
może się wyłączyć. Po ukończeniu/zatrzymaniu kolejki helper przywraca zwykłe
zachowanie uśpienia; nie zmienia planu zasilania.

Pipeline v3 wymaga nowej nazwy przebiegu przy wynikach v2. Istniejące tłumaczenia
MSCOCO i oceny w przeglądarce pozostają w swoich katalogach.

Aktualna kontynuacja, lokalizacje wyników na C: i komendy wznawiania:
[reports/absence_work_status.md](../reports/absence_work_status.md).
