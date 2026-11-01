# Prompt do wysłania razem z plikiem kontekstu

Jesteś głównym badaczem odpowiedzialnym za stworzenie możliwie najmocniejszego polskiego enkodera tekstowo-wizyjnego **poniżej 100 mln wszystkich parametrów używanych w inferencji**. Otrzymujesz załącznik `KONTEKST_PROJEKTU_DLA_MODELU_2026-10-31.md`, który opisuje dane, modele, treningi, wyniki, benchmark i dekontaminację istniejącego projektu.

Przeczytaj cały załącznik i przeprowadź niezależny, krytyczny przegląd. Chcę wiedzieć, **jakie działania są teraz najpilniejsze i mają największą szansę doprowadzić nas do najlepszego małego polskiego modelu**, a nie tylko poprawić wygodnie wybraną metrykę.

Nie zakładaj, że dotychczasowa architektura, interpretacje ani cel zachowania niezmienionej przestrzeni PolDense są optymalne. Porównaj co najmniej dwie ścieżki: rozwijanie TinyCLIP→PolDense oraz adaptację mocnego gotowego modelu multimodalnego <100M. Zachowanie obecnych zdolności tekstowych jest wartościowe, ale wskaż jego koszt i powiedz wprost, jeżeli uważasz je za istotną barierę. Trzecia ścieżka ma sens tylko wtedy, gdy masz konkretny argument, że może wygrać z pierwszymi dwiema.

Uwzględnij następujące zasady:

- Fakty z załącznika odróżniaj od swoich hipotez i wiedzy zewnętrznej. Nie wymyślaj wyników ani zawartości plików, których nie dostałeś. Brak dostępu do repozytorium nie powinien blokować analizy; wskaż minimalne dane potrzebne do rozstrzygnięcia niepewności.
- Nie mieszaj starych wyników COCO z nowymi pełnymi galeriami, średnich t2i/i2t z samym t2i, wybranych seedów ze średnią seedów ani R3 generacji1 z generacją2. XM3600 został już oceniony i nie jest blind. Żaden przyszły test nie może być strojoną wstecz miarą sukcesu.
- Uczciwie rozlicz różnicę między wierszami, unikalnymi obrazami, komponentami i prezentacjami. „Ready” w pipeline nie jest zgodą na trening. Uwzględnij syntetyczność opisów, błędy MT, samopotwierdzanie sędziego i bias filtracji.
- Limit parametrów obejmuje oba encodery i head; duży teacher może istnieć wyłącznie podczas treningu. Rozważ też latencję, pamięć i istniejący zasób RTX5070Ti16 GB; nie zakładaj nieograniczonego compute ani niepodanego budżetu API.
- Zachowaj ochronę wszystkich holdoutów, wersjonowanie i audyt nowych danych. Nie proponuj cichego usuwania przypadków z benchmarku ani traktowania nowego guarda jako oczyszczenia historycznych modeli. Uwzględnij stan licencji jako warunek wykonalności publikacji, bez zgadywania rozstrzygnięć prawnych.
- Jeżeli masz przeglądarkę, zweryfikuj aktualne modele <100M i metody istotne dla decyzji wyłącznie w źródłach pierwotnych: artykułach, oficjalnym kodzie, kartach modeli/danych. Podaj linki, datę i sposób liczenia parametrów. Jeśli nie masz dostępu, jawnie oznacz niezweryfikowane propozycje. Nie zastępuj analizy załącznika ogólnym przeglądem literatury.

Oczekuję odpowiedzi w następującym układzie:

1. **Diagnoza:** maksymalnie5 najważniejszych wniosków z konkretnymi liczbami. Co jest najbardziej prawdopodobnym ograniczeniem: dane/różnorodność, frozen przestrzeń tekstowa, vision, loss/sampler, optymalizacja czy selekcja/ewaluacja? Dla każdej tezy podaj kontrargument i poziom pewności. Rozdziel problemy wykazane od podejrzeń.
2. **Decyzja strategiczna:** którą ścieżkę wybrać jako główną, którą jako kontrolę i dlaczego? Wprost oceń lukę do MetaCLIP-M16 PL32768 oraz sens pozostania przy PolDense. Wskaż, co mogłoby zmienić Twoją decyzję.
3. **Priorytety P0/P1/P2:** najwyżej8 działań łącznie. Dla każdego: problem, spodziewana wartość, dowód, zależności, koszt względny i jasny warunek ukończenia. Ułóż je w kolejności; nie proponuj równoległego wielkiego gridu. Oszacowania zysku podawaj jako ostrożne hipotezy, nie gwarancje.
4. **Pierwsze3–5 eksperymentów rozstrzygających:** dla każdego hipoteza, dokładna różnica treatment/control, dane i reguła dekontaminacji, stały budżet/prezentacje, seedy, checkpoint selection, metryka główna i guardraile, kryterium sukcesu/przerwania oraz dalsza decyzja dla wyniku dodatniego i ujemnego. Oddziel mały screening od potwierdzenia na seedach. Uzasadnij, dlaczego te doświadczenia dadzą więcej informacji niż kolejny tuning LR.
5. **Plan danych i ewaluacji:** co dopuścić z już przygotowanych pul, co zbierać dalej, jak wybrać mały ludzki QA, jak potraktować trudne negatywy i składniki kompozycyjne, czy i jak stworzyć nowy niezależny holdout. Zaproponuj z góry kryterium „najmocniejszy <100M”, oddzielając główne t2i, i2t, retencję tekstu, domeny i koszt inferencji. Wyjaśnij, jak uniknąć dalszego dopasowywania do ujawnionych testów.
6. **Plan realizacji:** pierwsze48 godzin, następne2 tygodnie i warunki przejścia do większego treningu. To etapy/zależności, nie pozorna gwarancja czasu bez znajomości budżetu.
7. **Czego teraz nie robić:**3–5 kuszących działań o słabym stosunku wartości do kosztu. Na końcu podaj najwyżej5 brakujących informacji, które rzeczywiście mogą zmienić główną rekomendację.

Pisz po polsku, konkretnie. Najpierw podaj rekomendację i najpilniejszy krok. Możesz zdecydowanie zakwestionować dotychczasową strategię, jeśli wynika to z dowodów. Potrzebuję planu podejmowania decyzji i eksperymentów, który da się wykonać, a nie listy wszystkich możliwych technik.
