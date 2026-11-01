# Handoff S05-LEADERBOARD-PARETO — 2026-11-01

## Zrobione
SVG Pareto z punktami i liniami wszystkich modeli oraz komercyjnych + własnych.
Po doprecyzowaniu właściciela wykres i obie linie korzystają dokładnie z widoku
tabeli po wszystkich filtrach. Dodano minimum/maksimum parametrów (włącznie),
checkboxy organizacji z domyślnym pełnym wyborem; własne modele pod jansowa,
wyłącznie gdy obecne w danych. Brak średniej ma osobny pas, nie wartość zero.
Opcjonalna skala logarytmiczna; bez nowych zależności.

## Jak zweryfikowano
Raport: reports/f_series/S05-LEADERBOARD-PARETO/REPORT.md.
validation.json z SHA wejścia/kodu i zestawami modeli po filtrach, SHA256
7640f274453c3cdfde16501836bdb581145d2e69a073fee85bdd1c4d3f4ce504.
Unit-tests i rzeczywisty HTTP smoke Gradio passed; logi w katalogu raportu.
Podgląd HTML/PNG, niezależny Edge headless. Chronione SHA zgodne.

## Działa w tle
Nie uruchamiano nowych jobów. Wcześniejsza kolejka:
training/runs/f_series/leaderboard-requested-20261101/JOB.json;
stan STATE.json, publikacja PUBLISH_STATUS.json. Nie zatrzymywano procesów.

## Bramki / następny etap
Brak bramki. S06-LEADERBOARD-RECEIVE: odbiór kolejki i końcowego pushu.
Nie rozpoczęto F01-b.

## Zmienione pliki
space/app.py, view.py, README; testy Pareto/HTTP/zakresu; raport, dowody,
STATUS/handoff/NEXT/SESSIONS i ignorowanie diagnostycznego logu przeglądarki.
