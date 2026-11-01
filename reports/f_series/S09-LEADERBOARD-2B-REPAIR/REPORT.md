# S09-LEADERBOARD-2B-REPAIR — naprawa 2B i filtr licencji

Na jawne zlecenie właściciela naprawiono i uruchomiono wyłącznie
Qwen/Qwen3-VL-Embedding-2B oraz Alibaba-NLP/gme-Qwen2-VL-2B-Instruct.
Qwen 8B nie należy do tej kolejki. Po starcie dodano żądany filtr licencji.
Brak treningu, selekcji F, instalacji zależności i zmian zamrożonych generacji.

Przyczyny: Qwen używa nowego zapisu pooling_mode=lasttoken, GME starej flagi
pooling_mode_lasttoken=true. Adapter waliduje oba formaty, odrzuca inne i złożone
poolingi. Pooling wybiera ostatni rzeczywisty token według maski, niezależnie
od strony paddingu. Dla 2B cały model jest jawnie przenoszony na CUDA.
Przypięte źródła konfiguracji i SHA: preflight.json, qwen3/gme-pooling.json.

Utworzono nową, odrębną rejestrację i katalog joba; poprzedniej rejestracji,
wyników i logów nie nadpisano. Źródło zakresu: registration.json, SHA256
b0ebed1652b9eeb8d26512a719a703fcaf7dde1dd152bb8da9c4f3ddf4e4461c.
Po 14 brakujących komórek galeria/prompt dla każdego modelu; istniejące komórki
dowolnego profilu są pomijane. Ta sama kolejność galerii, rewizje i scorer.

Odłączony run: training/runs/f_series/leaderboard-2b-repair-20261101/JOB.json.
Oba adaptery przeszły rzeczywisty test GPU: tekst i obraz w obu trybach promptów,
zgodne wymiary, skończone wektory i wszystkie parametry na CUDA. Syntetyczne
testy nie są wynikami benchmarku. Smokes poprzedzają pełne obliczenia;
GME testowano jako pierwszy, jego wagi usunięto, potem testowano Qwen.
Pełny Qwen jest już liczony, GME następny. W danej chwili tylko jeden model.

Snapshot przy domknięciu: launch-receipt.json, SHA256
a6b4eb0815ceee895bef574e077a5cbe51657e6bf2f39341f6481c6d2ad662b9.
Potwierdza oba smokes i 6 zapisanych komórek z pierwszych galerii Qwena z SHA.
Wyniki pozostają w dotychczasowym output poza repo na C:. Wagi w prywatnym
~/.cache/pl_mmb_requested_2b_20261101/hub, usuwane po zakończeniu procesu modelu;
wspólny cache i dane chronione. Blokada GPU jest dotychczasowa.
Po modelu eksport i Windows publisher wykonają commit/push do origin/main.
Publisher uruchomiono dla nowego RunId/Stage; PUBLISH_STATUS.json potwierdzi push.

Filtr Licencje: wszystkie, komercyjne (w tym własne), niekomercyjne, nieustalone.
Własne checkpointy kwalifikowane zgodnie z deklaracją właściciela; deklaracje
źródłowe pozostają niezmienione. Brak deklaracji nie jest etykietą NC.
Filtr wspólny dla tabeli, punktów, obu granic i CSV; współdziała z resztą filtrów.
Wdrożono na http://localhost:7865/ i sprawdzono rzeczywiste API:
filter-live-validation.json, SHA256
e0ac5ad33640bd4fdae70711ebd1d47165b2bef7ccef21e3d613acd5526eaf84.

Testy CPU WSL: unit-tests.log — 44 passed; UI HTTP: http-smoke.log — passed.
validation.json zawiera SHA danych, kodu, ticket i pełne wyniki testów adapterów.
SHA256 39c6c5985e9e36aa577a39632967b917008947494243b1b9fe3aa00e05a30a51.
Odbiór końcowych wyników to odrębny etap; wznowienie: requested_2b_retry.py start.
