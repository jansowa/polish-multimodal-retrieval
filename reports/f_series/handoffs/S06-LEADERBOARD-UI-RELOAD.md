# Handoff S06-LEADERBOARD-UI-RELOAD — 2026-11-01

## Zrobione
Właściciel zgłosił brak wykresu na http://localhost:7865/.
Zmiany były w Git, lecz istniejący serwer Gradio nadal miał stary kod.
Zweryfikowano proces przypisany do tego portu i uruchomiono nowy serwer
z istniejącego środowiska UI na tym samym adresie. Bez zmian w kodzie ML.

## Jak zweryfikowano
Rzeczywiste /config zawiera wykres i filtry min/max/organizacji.
Wywołanie /leaderboard z wyborem jansowa: zgodne modele w tabeli i SVG.
Dowody: S06-LEADERBOARD-UI-RELOAD/restart.json, verify.py, live-validation.json.
SHA256 live-validation.json:
78adff59aa3dd4abdd936d4ef9bd2e4ea3fd310412e27dcedfc62914f75cecad.

## Działa w tle
Nowy UI: PID w restart.json, logi server.stdout/stderr.log w katalogu etapu.
Kolejka leaderboard-requested-20261101 i publisher nie były zatrzymywane.

## Bramki / następny etap
Brak bramki. Kolejny S07-LEADERBOARD-RECEIVE — odbiór kolejki.
Przy przyszłych zmianach app.py/view.py potwierdzić wdrożenie na właściwym
serwerze, nie tylko testowy serwer HTTP i commit kodu.

## Zmienione pliki
Dowody live, STATUS/handoff/NEXT/SESSIONS. Bez nowych obliczeń benchmarku.
