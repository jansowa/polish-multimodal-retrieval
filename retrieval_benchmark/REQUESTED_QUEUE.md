# Kolejka modeli z 2026-11-01

Zakres i SHA: reports/leaderboard_requested_20261101/.
Uzupełnia tylko braki model/revision/dataset/prompt; dowolny istniejący profil
wyklucza ponowną ocenę tej samej komórki. Nie zmieniać przypiętego kodu w trakcie.

Stan i wznowienie w WSL, z repo:

```bash
~/.venvs/pl-mmb-benchmark/bin/python retrieval_benchmark/requested_queue.py status
~/.venvs/pl-mmb-benchmark/bin/python retrieval_benchmark/requested_queue.py start
```

Publikacja działa osobno w Windows, bo WSL nie uruchamia lokalnego git.exe.
W PowerShell z repo uruchomić odłączony proces:

```powershell
$root = (Get-Location).Path
$script = Join-Path $root 'retrieval_benchmark/requested_publish.ps1'
Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"' + $script + '"'), '-RepoRoot', ('"' + $root + '"'))
```

Logi i status są w training/runs/f_series/leaderboard-requested-20261101/:
JOB.json, STATE.json, supervisor.log, events.jsonl, PUBLISH_STATUS.json.
GitHub: origin, zdalny main, bez force push i bez dodawania cudzych zmian.
Po restarcie komputera wznowić oba procesy; zamknięcie czatu ich nie zatrzymuje.
Błąd modelu nie zatrzymuje pozostałych, szczegóły w jego worker.log.
Końcowy odbiór: reports/f_series/NEXT.md.
