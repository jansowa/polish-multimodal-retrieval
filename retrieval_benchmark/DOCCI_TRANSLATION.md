# DOCCI PL — tłumaczenia testu

Użytkownik zlecił przygotowanie tłumaczeń tym samym modelem co recykling.
Zachowany manifest `reports/visualnews_pipeline/v1/manifest.json` wskazuje
**qwen3.8-27b** i używany serwer inferencyjny. Nie pobieramy jego wag lokalnie.

Runner: `retrieval_benchmark/docci_translation.py`. Oficjalne lokalne adnotacje:
`C:/Users/jsowa/benchmark-translations/data-ood-20261030/B/docci/descriptions.jsonl`.
Wybrano cały split `test` — 5000 opisów, przed oceną jakiegokolwiek modelu na DOCCI.
Nie tłumaczymy train ani nie wybieramy przykładów według rezultatów modeli.

Tekst→tekst, bez obrazów, bez recyklingu/generowania nowych podpisów i bez
streszczania długich opisów. Wykorzystany benchmarkowy prompt wiernego tłumaczenia
zachowuje liczby, nazwy, negacje i relacje. Temperature 0 i limit 8192 są zgodne
z konfiguracją tłumaczeń dawnego pipeline'u. Na polecenie użytkownika wyłączono
reasoning przez `chat_template_kwargs.enable_thinking=false` i ustawiono 16
równoległych żądań. Serwer potwierdził zero tokenów reasoningu w próbie API.
Oryginalne EN pozostają zachowane. Automatyczne QA nie daje ludzkiej gwarancji
semantycznej poprawności; różnice w tokenach liczbowych zapisujemy jako flagi.

Rejestr, odpowiedzi, osobne trwałe receipts i końcowy eksport:
`C:/Users/jsowa/benchmark-translations/retrieval-benchmark-v1/docci-qwen3.8-27b-20261031-nothinking-c16/`.
Osiem wcześniejszych opisów zachowano z oryginalnym pochodzeniem i oznaczeniem
wcześniejszego reasoningu. Nowe żądania używają trybu bez reasoningu; poprzedni
katalog i jego kod pozostają zachowane.
Sekrety API są czytane lokalnie z konfiguracji środowiska; nie zapisujemy ich
w rejestrze, odpowiedziach ani logach. Ukończonych rekordów runner nie nadpisuje.

```powershell
python retrieval_benchmark/docci_translation.py smoke
python retrieval_benchmark/docci_translation.py run
```

Nie uruchamiać dwóch procesów na tym samym katalogu jednocześnie. Wznowienie
pomija zapisane rekordy po sprawdzeniu identyfikatorów, SHA źródła i rejestru.
Tłumaczenie nie wykorzystuje lokalnego GPU i nie zmienia aktywnej kolejki Wiki.

Po tłumaczeniu nadal potrzebne są obrazy, fingerprinty, dekontaminacja,
wersjonowany protokół i qrels. DOCCI nie otrzymuje statusu gotowego benchmarku
na podstawie samej obecności polskich opisów. Dodanie gotowej galerii wymaga
następnej wersji rejestru holdoutów; obecnego v4 nie nadpisujemy.
