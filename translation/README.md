# Tłumaczenie danych

Kod: `src/pl_mmb_translate/`. Punkt wejścia: `translate.py` lub `pl-mmb-translate`.
Kolejki i eksperymenty: `experiments/`. Przegląd danych: `reviewer/` (port 8765).
Stan oraz wznowienie: [../reports/absence_work_status.md](../reports/absence_work_status.md).

Obecne ścieżki pozostają stabilne, aby działające procesy i checkpointy mogły
kontynuować pracę. Tłumaczenie używa zewnętrznego API i własnego środowiska `.venv`.
GPU, PyTorch i zależności benchmarku są instalowane w osobnym środowisku WSL.
