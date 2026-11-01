# S07-LEADERBOARD-LABELS — stałe nazwy modeli

Na jawne zlecenie właściciela dodano stale widoczne, pełne nazwy wszystkich
modeli aktualnego widoku Pareto. Domyślnie przełącznik nazw jest włączony;
wyłączenie pozostawia dotychczasowy tryb opisów po najechaniu.
Filtrowanie tabeli i wykresu pozostaje wspólne, a obie granice Pareto bez zmian.

Nazwy są zawijane do osobnych pól w dwóch bocznych kolumnach. Każde pole ma
linię prowadzącą do właściwego punktu; wysokość całego wykresu wynika z liczby
wierszy tekstu. Pakowanie uwzględnia odstępy i skrajne położenia punktów.
Nakładające się punkty otrzymują oddzielne nazwy. Stały czytelny rozmiar pisma
i przewijanie poziome przy wąskim oknie zapobiegają ściskaniu tekstu.
Najechanie lub fokus podkreśla nazwę i wzmacnia linię prowadzącą.

Nie dodano pakietów. Zmieniono wyłącznie warstwę prezentacji i testy;
nie wykonywano inferencji ani nie modyfikowano kodu pracującej kolejki GPU.
Chronione SHA zgodne z rejestracją kolejki.

Dowód geometrii, źródłowy SHA wyników i SHA kodu: validation.json, SHA256
5a9347529e5a2535f8e5e8733baac6cbf3fe2ca86ecf4b30d2b9deff8e60af32.
Snapshot: pełny widok 49 etykiet; filtr do 100M — 21 etykiet, także w skali
logarytmicznej. Żadne pola nazw nie nachodzą na siebie.
Rzeczywiste pomiary glifów przez Edge getBBox: browser-validation.json,
SHA256 6d35a5a1d0f5b23710444f65f3e66e2cf1ab6b3ea57861e1f4eb16d552de39ec;
brak wychodzenia tekstów poza pola we wszystkich wariantach. W pierwszej próbie
wykryto szerokość fontu większą od szacowanej; zawijanie poprawiono przed wdrożeniem.
Podglądy HTML/PNG zachowano; wariant do 100M sprawdzono wizualnie.

unit-tests.log: 39 passed w istniejącym WSL benchmark env, w tym test gęstych
identycznych współrzędnych z pełnymi długimi nazwami. http-smoke.log potwierdza
przełącznik, filtrowanie i zgodność liczby etykiet z wierszami tabeli.
Obliczenia podglądu nie ładują modeli.

Przeładowano wyłącznie wskazany serwer http://localhost:7865/; restart.json
identyfikuje proces. Live API potwierdziło domyślne włączenie nazw i działanie
trybu hover-only: live-validation.json, SHA256
31204fe41712a4d652e10bd01a89b1c68b6b6a051810b1b4c253325996bb3fc9.
Kolejka i publisher nie były zatrzymywane. Commit i push na origin/main.
