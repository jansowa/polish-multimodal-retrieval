# Handoff S08-LEADERBOARD-COMPACT — 2026-11-01

## Zrobione
Na zlecenie właściciela nazwy przeniesiono z bocznych kolumn do wolnych miejsc
przy punktach. Pola o zmiennej szerokości, krótkie linie, brak kolizji z tekstem,
punktami i osią. Duże skupiska nadal wymagają ograniczenia zestawu.
Dodano wyszukiwany wybór konkretnych modeli nad wykresem: wspólny dla tabeli,
punktów, obu granic i CSV. Puste pole przywraca wszystkie w ramach innych filtrów.
Stałe nazwy/hover-only oraz istniejące filtry pozostają dostępne.

## Jak zweryfikowano
S08-LEADERBOARD-COMPACT/REPORT.md, validation.json i browser-validation.json:
SHA danych/kodu/chronionych plików, rozłączność pól, odległości i pomiary glifów.
Unit-tests i HTTP smoke passed; podglądy małego zestawu i gęstej grupy sprawdzone.
Live API sprawdziło dokładny wybór modeli i wyczyszczenie filtra;
live-validation.json SHA256:
4c422c17372256f27d6254179b8be222b9e4777c5263104b84448bd974f9bf54.

## Działa w tle
UI localhost:7865 przeładowano; proces/logi w restart.json i server.*.log.
Nie uruchamiano nowych jobów ani nie zmieniano wcześniejszej kolejki GPU.

## Bramki / następny etap
Brak. S09-LEADERBOARD-RECEIVE — odbiór kolejki referencji.
Nie rozpoczęto odłożonego F01-b.

## Zmienione pliki
space/app.py/view.py/README, testy, raport/podglądy/dowody, .gitignore,
STATUS/handoff/NEXT/SESSIONS. Commit [S08-LEADERBOARD-COMPACT].
