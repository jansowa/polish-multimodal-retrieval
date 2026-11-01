# S04-LEADERBOARD — 2026-11-01

Jawne zlecenie właściciela zastępuje planowy F01-b: zwykła ocena referencji,
bez treningu i selekcji F. Zatwierdzono aclip→clip, sekwencyjne usuwanie własnych
wag i GitHub jansowa/polish-multimodal-retrieval. Lokalny master jest wypychany
fast-forward do zdalnego main, bez force push.

Źródło zakresu: summary.json, SHA256
e04c84586deb9397ce7857fb7a650f5fb94d1be8a1017ba3e2a1ebc9244149c1.
Modele: 15; istniejące komórki: 52; brakujące: 158 (galeria × prompt).
Istniejące komórki dowolnego profilu nie są liczone ponownie.
Każda ocena obejmuje oba kierunki i metryki wspólnego scorera.
Rewizje, wagi LFS, galerie, źródła istniejących wyników, kod i chronione SHA:
registration.json, SHA256
c6394bce1059a1bf100106f2a6471f1cb86f343aef2a63c9a87d50db9b927707.

Testy WSL: deduplikacja, wybór formatu wag, czyszczenie tylko zarejestrowanych
prywatnych blobów, odmowa usuwania zmienionych wag — passed.
Polecenie: python -m unittest retrieval_benchmark.tests.test_requested_queue -v.
Nie instalowano zależności. Używane są istniejące środowiska WSL.

Odłączony job: training/runs/f_series/leaderboard-requested-20261101/JOB.json.
Stan: STATE.json; zdarzenia: events.jsonl; log: supervisor.log w tym katalogu.
Podkatalogi modeli zawierają worker.log, smoke.json, complete/failure.json oraz
cleanup.json. Pierwsze pełne wyniki XM3600 stanowią smoke test.
Wagi są w prywatnym cache Linux ~/.cache/pl_mmb_requested_20261101/hub;
po modelu usuwane są tylko zarejestrowane bloby po kontroli SHA/rozmiaru.
Wspólny cache i dane pozostają chronione. Duże wyniki są poza repo na C:.

Eksport dopisuje wyłącznie własne artefakty ze sprawdzonym SHA, zachowując stare
rekordy. Windows publisher robi commit/push kolejnych wyników; dowód:
PUBLISH_STATUS.json. Błąd modelu nie zatrzymuje pozostałych; błąd czyszczenia
zatrzymuje kolejkę, aby nie kumulować wag. Szacunek zakończenia jest w JOB.json,
nie jest pomiarem; model 8B może korzystać z CPU offload.

Pierwsza próba nie przeszła preflightu przed inferencją (rezerwa D oraz git.exe
z WSL). Dowody: failed-launch-ticket.json i registration-launch-preflight.json.
Poprawiono kontrolę D do małych logów, oddzielono publisher Windows i zastąpiono
historyczny sync eksportem dopisującym. Rejestrację poprawiono przed inferencją.
Wznowienie i kontrola: retrieval_benchmark/REQUESTED_QUEUE.md.
Odbiór i końcowe potwierdzenie kompletu oraz pushu są osobnym etapem.
