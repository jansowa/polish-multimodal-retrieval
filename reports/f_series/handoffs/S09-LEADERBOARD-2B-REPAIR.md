# Handoff S09-LEADERBOARD-2B-REPAIR — 2026-11-01

## Zrobione
Nowa kolejka tylko Qwen3-VL-Embedding-2B i GME-Qwen2-VL-2B. Naprawiono
obsługę obu formatów poolingu i umieszczenie całych modeli 2B na CUDA.
Starego rejestru i wyników nie zmieniano; nowe SHA w odrębnej registration.json.
Po starcie dodano filtr licencji do tabeli/wykresu/CSV: komercyjne, NC,
nieustalone i wszystkie. Własne komercyjne wg właściciela. Live UI wdrożone.

## Jak zweryfikowano
Oba adaptery: tekst/obraz, oba prompty, CUDA, finite i zgodne wymiary passed.
CPU unit-tests i rzeczywisty HTTP/UI API passed. Źródła i SHA w REPORT.md,
preflight.json, validation.json, filter-live-validation.json.
launch-receipt.json zawiera ticket, smokes i zapisane komórki z SHA:
a6b4eb0815ceee895bef574e077a5cbe51657e6bf2f39341f6481c6d2ad662b9.

## Działa w tle
Run_id leaderboard-2b-repair-20261101; JOB.json w training/runs/f_series/.
Qwen liczy pełne galerie; GME kolejny. Stan STATE.json, log supervisor.log;
podkatalogi modeli: adapter-smoke.json, worker.log, complete/failure/cleanup.
Szacunek w ticket; nie jest pomiarem. Po każdym modelu czyszczenie prywatnych
wag, eksport i Windows commit/push (PUBLISH_STATUS.json).
Wznowienie WSL: ~/.venvs/pl-mmb-benchmark/bin/python retrieval_benchmark/requested_2b_retry.py start.
Publisher: requested_publish.ps1 -RepoRoot <repo> -RunId leaderboard-2b-repair-20261101 -Stage S09-LEADERBOARD-2B-REPAIR.

## Bramki / następny etap
Brak. S10-LEADERBOARD-2B-RECEIVE: odbiór i potwierdzenie kompletów/pushu.
Qwen 8B pozostaje poza tą naprawą, z pierwotnym błędem miejsca na C:.

## Zmienione pliki
Adapter, kolejka retry i parametry publishera, filtr UI/testy, dowody/ticket,
STATUS/handoff/NEXT/SESSIONS; commit [S09-LEADERBOARD-2B-REPAIR].
