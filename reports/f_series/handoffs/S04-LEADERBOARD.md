# Handoff S04-LEADERBOARD — 2026-11-01

## Zrobione
Na jawne zlecenie właściciela uruchomiono kolejkę referencji do leaderboardu,
bez treningu i selekcji F. Zatwierdzono aclip→clip i GitHub
jansowa/polish-multimodal-retrieval. Rewizje, braki i SHA są w
reports/leaderboard_requested_20261101/registration.json; liczebności i dowody
w summary.json i REPORT.md w tym samym katalogu.

## Jak zweryfikowano
Testy WSL deduplikacji i ograniczonego czyszczenia passed.
Smoke: pełne wyniki XM3600 w podkatalogach joba, smoke/complete.json z SHA.
Końcowy snapshot: reports/leaderboard_requested_20261101/launch-receipt.json,
SHA256 14a08d5042b20c568dbfc7257b63a3cda2fda90be928546628d8fe3491f5ab20.
Potwierdza ukończone modele, cleanup i status publishera pushed.
Chronione pliki weryfikowane przed pracą; bez instalowania zależności.
Pierwszy preflight bez inferencji zachowano; poprawiony przed właściwym startem.

## Działa w tle
run_id: leaderboard-requested-20261101; ticket:
training/runs/f_series/leaderboard-requested-20261101/JOB.json.
Stan i log: STATE.json, supervisor.log, events.jsonl. Szacunek: ticket.
Po każdym modelu czyszczenie prywatnych wag, dopisanie wyników i zlecenie pushu.
Publisher Windows: PUBLISH_STATUS.json; zdalny main, bez force push.
Wznowienie obu procesów: retrieval_benchmark/REQUESTED_QUEUE.md.

## Bramki / następny etap
Brak wymaganej decyzji. Następny: odbiór braków, błędów i końcowego pushu.
F01-b odłożony; F GUARD/TEST bez odczytów selekcyjnych.

## Zmienione pliki
Adaptery, kolejka/publisher, testy, raport/rejestracja, eksport wyników,
STATUS/handoff/NEXT/SESSIONS. Commit z prefiksem [S04-LEADERBOARD].
