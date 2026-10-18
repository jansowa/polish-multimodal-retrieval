# Plan tłumaczeń na nieobecność — 2026-10-17

Ustalenia użytkownika: priorytety dobiera agent; tłumaczymy pełne wybrane wersje
zbiorów; praca może trwać przez całą nieobecność i dłużej niż dobę. Celem jest
użyteczny polski benchmark embedderów multimodalnych, z naciskiem na zachowanie
znaczenia i warunków ewaluacji.

## Proponowana kolejność

| Kolejność | Zbiór i wersja | Zakres | Wartość dla benchmarku |
|---|---|---:|---|
| 1 | SugarCrepe, mteb/SUGARCREPE_fmt | 7511 par | Rozróżnianie poprawnego opisu od trudnego negatywu |
| 2 | CIRR, MMEB-eval | 1000 rekordów | Wyszukiwanie na podstawie obrazu referencyjnego i tekstowej zmiany |
| 3 | FashionIQ, MMEB-eval | 1000 rekordów | Atrybuty ubrań i tekstowa modyfikacja zapytania obrazowego |
| 4 | RefCOCO, MMEB-eval | 1000 rekordów | Rozróżnianie wskazywanego obiektu i szczegółów opisu |
| 5 | ARO Visual Relation, mteb/ARO-Visual-Relation | 23 937 par | Role podmiotu i obiektu oraz relacje przestrzenne |
| 6 | ARO Visual Attribution, mteb/ARO-Visual-Attribution | 28 748 par | Przypisanie właściwego atrybutu właściwemu obiektowi |

Łącznie plan podstawowy obejmuje około 63 tys. rekordów/par. Liczebności ARO
pochodzą z kart danych i wymagają jeszcze sprawdzenia dla przypiętych rewizji.
Nie traktujemy jednostki MMEB z listą kandydatów jak pojedynczej pary opisów.

MSCOCO_t2i w wersji MMEB jest już gotowe: 1000/1000 rekordów. CIRR, FashionIQ i
RefCOCO w tej kolejce oznaczają kompletne warianty MMEB, nie pełne oryginalne
repozytoria tych zbiorów. SugarCrepe ma w używanym repozytorium jeden split
nazwany `train`; plan obejmuje cały ten materiał benchmarkowy.

Winoground jest dodatkiem warunkowym po potwierdzeniu dostępu do repozytorium.
Nie powinien blokować całej kolejki. OCR i dokumenty oraz M-BEIR nie należą do
podstawowej kolejki na tę nieobecność. Tłumaczenie tekstu w dokumentach wymaga
osobnego ustalenia, jak traktujemy angielski tekst pozostający na obrazach.

## Warunki jakości przed pełnym przebiegiem

1. Sprawdzić dostęp, faktyczne pola, rewizje i kompletność źródeł oraz obrazów.
2. Uruchomić osobne pilotaże po około 30–50 przykładów z każdego zadania;
   dobierać różne typy negatywów i relacji, zamiast samych pierwszych rekordów.
3. Ocenić zachowanie obiektów, atrybutów, liczb, negacji i ról. W parach
   kontrastowych sprawdzić oba opisy łącznie, w tym możliwe zlanie się ich sensu.
4. Podjąć na pilotażu decyzję o obrazach jako kontekście: obraz może rozstrzygać
   znaczenie słowa, ale nie może naprawiać celowo fałszywego opisu.
5. Utrwalić prompt, model, ustawienia, rewizje i listę zadań przed startem.
   Produkcja obejmuje wszystkie rekordy; limity dotyczą tylko oznaczonych pilotów.

Domyślnie jeden przebieg tłumaczenia. Obecny automatyczny review tym samym modelem
nie jest bezwarunkowo przyjmowany jako poprawa: w eksperymencie potrafił cofnąć
poprawną sygnalizację do błędnych świateł stopu. Dodatkowa kontrola semantyczna
powinna wskazywać podejrzane rekordy, a poprawki wymagają ponownego porównania
z oryginałem. Nie optymalizujemy samej płynności polszczyzny kosztem kontrastu.

Nie uznajemy poprawnego JSON ani braku flag modelu za dowód poprawnego znaczenia.
Źródłowe niejasności i wady benchmarku zachowujemy w raporcie; nie tworzymy
samowolnie nowych etykiet ani nie usuwamy po cichu trudnych rekordów.

## Harmonogram przygotowania

- Najbliższe 20–30 minut: przywrócenie działania WSL i kontrola źródeł.
- Następne około 40–50 minut: pilotaże i ocena jakości na różnych zadaniach.
- Ostatnie 20–30 minut przed wyjściem: zamrożenie konfiguracji, start kolejki,
  sprawdzenie pierwszych zapisanych rekordów oraz wznawiania po błędzie.

To budżet przygotowania, a nie gwarancja czasu pobierania danych. Zbiory
niesprawdzone lub niedostępne nie są oznaczane jako gotowe do produkcji.

## Wymagania dla kolejki

- Niezależny proces w tle, nieuzależniony od otwartego czatu.
- Osobne checkpointy i manifesty każdego zbioru; zachowanie istniejących wyników.
- Błędy przejściowe ponawiane z odstępem; trwały błąd jednego rekordu lub zbioru
  nie może zablokować pozostałych zadań na cały wyjazd.
- Podsumowanie `expected / completed / failed` oraz lista brakujących ID.
  Zbiór jest kompletny dopiero po uzgodnieniu wszystkich rekordów źródłowych.
- Brak samowolnego limitu 24 godzin i brak ukrytego przycinania zbiorów.
- Kontrola wolnego miejsca, serwera i pobranych obrazów przed wyjściem;
  komputer ma pozostać aktywny podczas pracy.

## Stan sprawdzony w tej sesji

- Pełny MSCOCO_t2i: 1000 ukończonych rekordów, 0 błędów końcowych.
- Eksperyment review: 99/100 ukończonych rekordów; 1 rekord zakończył się pustą
  odpowiedzią przy limicie wyjścia. 25 z 99 tłumaczeń zmieniło się po review;
  sama zmiana nie jest miarą poprawy. Dostępny raport
  `trip_review_experiment_summary.json` rozdziela grupę diagnostyczną i losową.
- Skonfigurowany serwer udostępnia wyłącznie model `qwen3.8-27b`.
- `HF_TOKEN` w konfiguracji nie jest ustawiony; dostępu do Winoground nie
  potwierdzono (możliwa jest też autoryzacja przez lokalne konto HF).
- Kontrola danych w WSL przerwała się po odczytaniu listy modeli. Ubuntu jest
  zatrzymane; również prosty test `/bin/echo` kończył się `E_UNEXPECTED` po
  `wsl --shutdown`. Dostępność źródeł i pilotaże pozostają niezweryfikowane.
- Nowa kolejka wielozbiorowa nie została jeszcze uruchomiona.

Źródła: registry.py projektu oraz karty danych:
https://huggingface.co/datasets/mteb/SUGARCREPE_fmt,
https://huggingface.co/datasets/mteb/ARO-Visual-Relation,
https://huggingface.co/datasets/mteb/ARO-Visual-Attribution,
https://huggingface.co/datasets/TIGER-Lab/MMEB-eval,
https://huggingface.co/datasets/facebook/winoground.
