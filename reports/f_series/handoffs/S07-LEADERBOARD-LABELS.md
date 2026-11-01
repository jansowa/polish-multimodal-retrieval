# Handoff S07-LEADERBOARD-LABELS — 2026-11-01

## Zrobione
Pełne nazwy modeli stale widoczne domyślnie; przełącznik pozwala pozostawić
je tylko po najechaniu. Oddzielne wiersze po obu stronach, zawijanie nazw,
dynamiczna wysokość i linie do punktów. Hover/fokus podkreśla nazwę.
Wykres nadal odpowiada dokładnie filtrowanej tabeli; bez nowych zależności.

## Jak zweryfikowano
Raport: S07-LEADERBOARD-LABELS/REPORT.md, źródła i SHA w validation.json.
Logi testów WSL i HTTP passed, w tym gęste wspólne współrzędne.
Edge getBBox potwierdził brak wychodzenia tekstu poza pola; pola rozłączne
dla całego widoku, limitu parametrów i skali logarytmicznej.
Podglądy HTML/PNG. Chronione SHA zgodne.
Serwer localhost:7865 przeładowano; live-validation.json potwierdza stałe
etykiety i przełącznik, SHA256
31204fe41712a4d652e10bd01a89b1c68b6b6a051810b1b4c253325996bb3fc9.

## Działa w tle
UI PID/logi: restart.json i server.stdout/stderr.log w katalogu etapu.
Istniejąca kolejka leaderboard-requested-20261101 i publisher nietknięte.
Nie uruchamiano jobów GPU ani nowych pomiarów benchmarku.

## Bramki / następny etap
Brak. S08-LEADERBOARD-RECEIVE — odbiór kolejki; F01-b nadal odłożony.

## Zmienione pliki
space/app.py/view.py/README, testy, raport/podglądy/dowody, .gitignore,
STATUS/handoff/NEXT/SESSIONS. Commit z prefiksem [S07-LEADERBOARD-LABELS].
