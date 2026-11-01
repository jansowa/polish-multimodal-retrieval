# S08-LEADERBOARD-COMPACT — lokalne etykiety i wybór modeli

Na jawne zlecenie właściciela zastąpiono boczne kolumny nazw układem szukającym
wolnego miejsca blisko każdego punktu. Nazwy są pełne, zawijane w pola o rozmiarze
zależnym od długości tekstu. Kandydaci zaczynają się tuż obok punktu; dalsze
pozycje są brane pod uwagę dopiero przy kolizji. Pola nie zakrywają innych nazw,
kropek ani opisu osi X. Linie do punktów kończą się na najbliższej krawędzi pola.
Przełącznik stałych nazw/hover-only pozostaje dostępny.

Nad wykresem dodano wyszukiwany wybór konkretnych modeli. Wybrany zestaw
ogranicza wspólnie tabelę, punkty, obie granice Pareto i CSV. Puste pole pokazuje
wszystkie modele spełniające pozostałe filtry. Wybór łączy się z organizacjami,
zakresem parametrów, seedami i kontrolami; lokalne modele opisano prefiksem
jansowa w opcjach. Zmiana wyświetlanych kolumn nie zmienia identyfikacji modeli.

Dowód geometrii, SHA wejściowych wyników, kodu i chronionych plików:
validation.json, SHA256
c0c904c1c071fe3568707b71361bfc81232a00cee26d76e6c8e73fd5c1e1c5a8.
Przykład wybranych 5 modeli: maksymalna linia do etykiety 11 px SVG; gęsta grupa
4 własnych modeli: 50,35 px. Te wartości pochodzą z przypadków selected/cluster
w validation.json. Pola są rozłączne także dla pełnego i ograniczonego widoku.
Pełny liczny zestaw nadal bywa gęsty: bez zmiany współrzędnych nie da się
pomieścić wszystkich długich nazw tuż obok skupiska; wybór małego zestawu jest
podstawowym sposobem przygotowania czytelnego porównania.

Pomiary rzeczywistych glifów Edge: browser-validation.json, SHA256
2da78503a786e79098299b1dffc9b1b74397fab4050831e00e1bf53c34920a2d.
Brak wychodzenia tekstów poza pola dla wybranych modeli, gęstej grupy własnych
i limitu parametrów. Podglądy selected.png/cluster.png sprawdzono wizualnie.
unit-tests.log: 41 passed w istniejącym WSL benchmark env. http-smoke.log:
wybór modeli, wspólne filtry, etykiety, przełącznik i CSV — passed w UI env.

Przeładowano właściwy serwer http://localhost:7865/; restart.json wskazuje proces.
Live API potwierdziło dokładny zestaw wybranych modeli i przywrócenie wszystkich
po wyczyszczeniu pola: live-validation.json, SHA256
4c422c17372256f27d6254179b8be222b9e4777c5263104b84448bd974f9bf54.
Bez instalacji pakietów, nowych obliczeń benchmarku lub zmian kolejki GPU.
Chronione SHA zgodne. Commit i push do origin/main; odbiór kolejki osobno.
