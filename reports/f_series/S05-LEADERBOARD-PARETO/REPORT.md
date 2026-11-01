# S05-LEADERBOARD-PARETO — wykres i filtry

Jawne zlecenie właściciela rozszerza etap prezentacji leaderboardu; nie jest to
odbiór kolejki ani etap F01. Dodano SVG jakości względem liczby parametrów V+T:
punkty modeli, linię wszystkich referencji i linię modeli komercyjnych.
Własne checkpointy należą do grupy komercyjnej zgodnie z instrukcją właściciela;
deklaracje licencyjne w danych pozostają niezmienione.

Po doprecyzowaniu właściciela tabela i wykres otrzymują ten sam przefiltrowany
widok. Dotyczy to punktów i obu granic Pareto, metryki/zadań, promptów/profilu,
seedów/kontroli EN, minimum/maksimum parametrów oraz organizacji.
Granice rozmiaru są wliczane do zakresu; maksimum zero wyłącza górny limit.
Checkboxy organizacji pochodzą z danych, domyślnie wszystkie zaznaczone.
Własne modele są pod jansowa; bez własnych modeli nie tworzymy tej opcji.
Odznaczenie wszystkich daje pusty widok. Nowa organizacja przy odświeżeniu jest
domyślnie włączana tylko gdy wcześniej zaznaczono wszystkie organizacje.

Frontier pomija wyniki zdominowane; remisy zachowują najmniejszy model.
Wiersze bez pełnej średniej mają osobny pas bez wartości osi Y.
Opis punktu zawiera nazwę, rozmiar i wynik; opcjonalna skala logarytmiczna.
Bez nowych zależności, pobierania danych, inferencji ani modyfikacji kolejki.
SVG pozostaje w view.py, więc standalone Space nadal obejmuje dotychczasowe pliki.

Dowody: validation.json, SHA256
7640f274453c3cdfde16501836bdb581145d2e69a073fee85bdd1c4d3f4ce504.
Zawiera SHA snapshotu wyników, kodu i chronionych plików, punkty obu granic
oraz zestawy modeli dla kombinacji filtrów. Wszystkie chronione SHA zgodne.
Snapshot obejmuje 49 widocznych modeli: 32 pełne średnie i 17 niepełnych.
Źródłem tych liczb jest validation.json; aktywna kolejka zmienia późniejsze wyniki.

Testy: unit-tests.log — 37 passed (WSL, dotychczasowe środowisko benchmarku).
http-smoke.log — rzeczywisty serwer Gradio: tabela/Pareto, min/max, organizacje,
jansowa, pusty wybór, skala logarytmiczna i CSV — passed w istniejącym UI env.
Podgląd preview.html i screenshot preview.png z osobnego Edge headless;
kontrola wyglądu początkowego wykresu wykonana. Nie instalowano pakietów.

Implementacja w głównym checkoutcie; commit/push do origin/main.
Kolejka referencji i publisher pozostają odrębnymi procesami.
Następny etap: odbiór wg NEXT; odłożony F01-b nadal nie został rozpoczęty.
