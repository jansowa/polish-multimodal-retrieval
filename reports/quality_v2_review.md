# Porównanie jakości tłumaczeń MSCOCO t2i: v1 i v2

Data: 2026-10-17. Stały wycinek bazy odczytany o 11:43:09 UTC.

**Ocena: ujednolicenie instrukcji działa, a jakość opisów jest mieszana. Są konkretne poprawy, ale pozostają błędy znaczeniowe i językowe oraz występują regresje. Obecna kontrola tym samym modelem nie daje podstaw do automatycznego zatwierdzania opisów.**

## Zakres i sposób oceny

Porównałem wszystkie 302 ukończone rekordy dostępne w chwili odczytu nowego przebiegu `qwen3.8-27b-v2-review` ze starym przebiegiem `qwen3.8-27b`. W obu przebiegach tekst źródłowy tych rekordów jest identyczny. Stary przebieg zawierał 770 ukończonych rekordów. Oceniałem tekst EN–PL bez oglądania obrazów.

Zachowany plik `quality_v2_snapshot.json` zawiera oryginał, obie wersje opisu, flagi i podsumowanie kontroli struktury. Bazy otworzyłem tylko do odczytu, w transakcjach zapewniających spójny wycinek. Wyniki bieżącego tłumaczenia mogą już zawierać więcej rekordów.

| Właściwość | Wynik w badanym wycinku |
|---|---:|
| Nowe ukończone rekordy | 302 / 1000 |
| Rekordy z metadanymi `reviewed=true` | 302 |
| Rekordy z flagą do ręcznej oceny | 25 (8,3%) |
| Wpisy w tabeli `failures` | 0 |
| Błędy kluczy, typów, długości list, tokenów obrazów i pustych wartości w sprawdzonych danych | 0 |
| Rekordy z odtworzoną listą 1000 kandydatów | 302 |
| Warianty instrukcji obrazu / prefiksu zapytania | 1 / 1 |
| Opisy identyczne po usunięciu prefiksu i skrajnych białych znaków | 129 |
| Opisy różniące się tekstowo | 173 |

Zmiana tekstu nie jest automatycznie poprawą jakości. Flaga modelu nie oznacza automatycznie błędu, a brak flagi nie oznacza poprawności.

## Co się poprawiło

Numery oznaczają końcówki identyfikatorów `MSCOCO_t2i`.

| ID | v1 | v2 | Ocena |
|---|---|---|---|
| 19 | „Szkła i butelki” | „Szklanki i butelki” | Poprawny dobór znaczenia `glasses`. |
| 24 | „łyżki są ustawione na stole” | „łyżki są ustawione w rzędzie na stole” | Przywrócone znaczenie `lined up`. |
| 66 | „Stos motocykli” | „Gromada motocykli” | Usunięte błędne piętrzenie obiektów; nadal lepsze stylistycznie byłoby „kilka motocykli”. |
| 88 | „przed winnicą” | „przed winiarnią” | Trafniejsze tłumaczenie `winery`. |
| 115 | „Para sygnalizacji świetlnych” | „Dwa sygnalizatory świetlne” | Poprawa naturalności. |
| 183 | „ze sbowboarderem” | „ze snowboardzistą” | Poprawnie rozpoznana literówka źródła. |
| 247 | „z stawem” | „ze stawem” | Poprawna forma przyimka. |
| 255 | „na kufra podróżnego” | „na kufrze podróżnym” | Poprawna odmiana. |
| 261, 265, 269 | `briefcase` jako „walizka” | `briefcase` jako „aktówka” | Poprawa rozróżnienia obiektów w tych trzech rekordach. |
| 287 | „w skafandrze nurkowym” | „w kombinezonie neoprenowym” | Trafniejszy odpowiednik `wetsuit`. |
| 295 | „na zebra pasącego się” | „Zbliżenie zebry pasącej się” | Poprawa odmiany i składni. |

Stałe instrukcje są jednolite we wszystkich 302 rekordach: `Przedstaw podany obraz.` i `Znajdź obraz z życia codziennego pasujący do podanego opisu: …`. Jest to efekt szablonów w kodzie, niezależny od jakości modelu.

## Błędy, które pozostały

| ID | Problem v2 | Flaga modelu |
|---|---|---|
| 118 | `stop lights` na słupie nadal jako „światła stopu”; kontekst językowy wskazuje sygnalizację. | Tak, model uznał znaczenie za niejednoznaczne. |
| 120 | „z stołem” zamiast „ze stołem”. | Nie |
| 78 | „z złotą ramą” zamiast „ze złotą ramą”. | Nie |
| 214 | „z świątyniami” zamiast „ze świątyniami”. | Nie |
| 233 | „słoń […] w zacienionym czyszczeniu”. Źródło ma `cleaning`, prawdopodobnie zamiast `clearing`. | Tak |
| 169 | „Dwoje pływało, pływając […]” z uszkodzonego `Two swam swimming…`. Tłumaczenie nie nadaje się do użycia bez rozstrzygnięcia źródła. | Tak |
| 159 | „ptak siedzi w kwiatowym chłopcu” z uszkodzonego `flower boy`. | Tak |
| 266 | `briefcase` nadal jako „walizka”. | Nie |
| 290 | `some cliffs` jako „niektórych klifów” zamiast naturalnego „klifów” lub „kilku klifów”. | Nie |

W grupie pięciu źródeł zawierających `briefcase` / `brief case` (261, 265, 266, 268, 269) poprawne rozróżnienie aktówki występuje w v1 w 1/5, a w v2 w 3/5 rekordów. To lokalna kontrola tego konkretnego słowa, nie miara jakości całego zbioru.

W 280 zniknęło nieuzasadnione „zapuszczony”, ale `shaggy` jako „o szorstkiej sierści” nadal jest mniej trafne od „kudłaty”: długość i bujność sierści nie oznacza jej szorstkości.

## Regresje i ryzykowne zmiany

| ID | v1 → v2 | Ocena |
|---|---|---|
| 268 | „na aktówce” → „na walizce” | Wyraźna regresja znaczeniowa; bez flagi. |
| 64 | „na światłach” → „na sygnalizatorze” | Zmiana relacji przestrzennej: `at` nie oznacza siedzenia na urządzeniu; bez flagi. |
| 137 | „Stado ptaków lecących” → „Stado ptaków leciące” | Nowy błąd językowy; bez flagi. |
| 96 | „zza okien” → „z za okien” | Nowy błąd pisowni; bez flagi. |
| 107, 130 | „na plaży/polu” → „na wierzchu plaży/pola” | Pogorszenie naturalności przez dosłowność; 107 ma flagę, 130 nie. |
| 239 | „kajaki […] puste” → „na pustym brzegu rzeki” | Zmieniono obiekt, któremu przypisano pustkę. Źródło ma niezgrabną składnię; wymaga oceny, a flaga nie rozwiązuje problemu. |

## Interpretacja i dalsze kroki

Model zgłasza realne uszkodzenia źródła, co pomaga skierować je do ręcznej oceny. Jednocześnie bywa nadmiernie dosłowny i zachowuje nonsens jako „najbliższe tłumaczenie”, zamiast bezpiecznie rozpoznać znaczenie z kontekstu językowego. Część flag dotyczy zwykłej wieloznaczności składniowej i nie wymaga ingerencji.

Brak flagi nie powinien być warunkiem automatycznego zatwierdzenia. Przykłady 64, 96, 137 i 268 pokazują błędy przepuszczone przez oba etapy.

Nie można przypisać zmian osobno nowemu promptowi i drugiemu przebiegowi: obecny checkpoint przechowuje wynik końcowy, ale nie zapisuje draftu sprzed przeglądu. Następny eksperyment powinien zachować oba teksty i porównać je na tej samej, wcześniej ustalonej próbce, obejmującej również zwykłe, poprawne opisy. Pozwoli to ocenić, czy drugi przebieg pomaga, czy tylko zwiększa koszt i liczbę przekształceń.

Nie ma jeszcze wystarczającej próbki problematycznych form liczby mnogiej „zebry”, ocenianych poprzednio: rekordy 305 i dalsze nie należą do tego wycinka. Nie oceniam ich poprawy na podstawie kilku dostępnych opisów jednej zebry.

Nie zmieniałem kodu pipeline'u, manifestów ani wyników tłumaczenia podczas tej oceny.
