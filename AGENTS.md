# Dane treningowe i holdouty

Przed dopuszczeniem **nowych** danych do treningu przeczytaj
[training/DATA_SPLITS.md](training/DATA_SPLITS.md). Wspólny rejestr i kontrola są
w `src/pl_mmb_training/current_holdouts.py`; zweryfikuj rejestr i uruchom audyt nowych
manifestów przed ekstrakcją cech/treningiem. Sam blacklist historyczny nie wystarcza.

Chroń wszystkie obrazy galerii (także bez zapytań), podpisy EN/PL, negatywne podpisy
walidacyjne i całe zarezerwowane komponenty Wiki. Wersjonuj manifesty i wiąż ich SHA
z treningiem. Brak fingerprintów lub zmiana holdoutu po audycie ma blokować trening.
Nie naprawiaj wycieku przez ciche usuwanie przykładów z istniejącego benchmarku.

Użytkownik 31.10.2026 zniósł FINAL BLIND dla XM3600 PL i zlecił zwykłą ocenę
w leaderboardzie. Nadal chroń XM3600 przed treningiem. VN dev i Wiki dev są selekcyjne. Historyczne ekspozycje COCO/XTD
pozostają jawnie oznaczone; nowa bramka nie oczyszcza wstecz już wytrenowanych modeli.

Na tym komputerze pracują inne sesje. Nie zatrzymuj cudzych procesów ani nie
zmieniaj zamrożonych generacji. Użytkownik nie zgadza się na usuwanie danych/plików.

Generacja Wiki2 ma zamrożone dokładne SHA `holdouts.py` i starego `catalog.json`.
Nie edytuj tych dwóch plików. Bieżący katalog leaderboardu to
`retrieval_benchmark/catalog-current.json`, a nowe audyty używają rejestru v4
przez `current_holdouts`. Generacja2 nadal używa prawidłowo swojego rejestru v2.

## Protokół sesji

1. **Jedna sesja = jeden etap** (pakiet, krok pakietu albo ramię eksperymentu). Zakres etapu wynika z `09 §4` (plan sesji) albo z `reports/f_series/NEXT.md` zapisanego przez poprzednią sesję. Nic spoza zakresu — pomysły i zauważone problemy trafiają do `reports/f_series/BACKLOG.md`.
2. **Etap wykonuje się w całości**, do kryterium „gotowe” z `04`/`05`. Nie zaczynaj następnego etapu w tej samej sesji, nawet jeśli bieżący skończył się szybko.
3. **Czytaj minimalnie:** na starcie `reports/f_series/STATUS.md`, `reports/f_series/NEXT.md`, najnowszy plik w `reports/f_series/handoffs/`, `docs/agent_spec_f/REPO_MAP.md`, sekcję pakietu w `04_PAKIETY_PRACY.md`, sekcję eksperymentu w `05_EKSPERYMENTY.md` (jeśli dotyczy) oraz pliki z tabeli „kolejność czytania” w `00` dla danego typu akcji. Nie czytaj całej specyfikacji ani dużych plików (`results.json`, manifesty, logi) w całości — używaj skryptów, `head`, `jq`; podsumowania zapisuj do plików.
4. **Bez gałęzi, worktree i PR.** Praca na bieżącej gałęzi głównego checkoutu; `git status` czysty na starcie; na koniec sesji jeden commit (lub kilka) z prefiksem `[<etap>]`. Zamrożonych plików (`src/pl_mmb_training/holdouts.py`, stary `retrieval_benchmark/catalog.json`, zamrożone generacje) nie dotykaj.
5. **Zadania GPU są uruchamiane, nie pilnowane.** Job musi być wznawialny i mieć ticket `training/runs/f_series/<run_id>/JOB.json` (polecenie startu i wznowienia, pid, start, szacowany koniec, ścieżki wyników). Po starcie i smoke teście sesja czeka najwyżej 15 minut, potem kończy się handoffem; odbiór wyników to osobny etap („odbiór”), który sam staje w `NEXT.md`.
6. **Bramka ludzka = koniec sesji.** Eskalacja wg `07 §8` zapisana w handoffie i w `NEXT.md` (etap „oczekuje na H…”). Nie obchodź bramki, nie zgaduj decyzji.
7. **Koniec sesji (obowiązkowo, w tej kolejności):** `STATUS.md` → handoff `reports/f_series/handoffs/<SESSION_ID>.md` (≤ 1 strona, §3) → `NEXT.md` (następny etap, §3) → wiersz w `reports/f_series/SESSIONS.md` → commit → odpowiedź ≤ 10 linii.
8. **Logi w czacie ≤ 50 linii;** pełne logi do plików. **Każda liczba w raporcie ma źródło** (ścieżka, SHA); nie przepisuj liczb z pamięci rozmowy.

## Środowiska wykonawcze

- Kod ML uruchamia się wyłącznie w WSL. Interpretery:
  - `~/.venvs/pl-mmb-benchmark/bin/python` — trening, benchmark oraz PyTorch/CUDA.
  - `~/.venvs/pl-mmb-legacy-cu130/bin/python` — starsze adaptery oraz zadania CPU wymagające SciPy.
  - `~/.venvs/pl-mmb-modern/bin/python` — EmbeddingGemma-2.
- Repozytoryjny `.venv` służy tylko pipeline’owi tłumaczeń, bez GPU i bez SciPy. Nie używać go do treningu ani ewaluacji.
- Nie instalować pakietów do istniejących środowisk. Nowe zależności dopuszcza się wyłącznie w nowym, wersjonowanym środowisku po zgodzie właściciela.
- Polecenia GPU i skrypty treningowe wywoływać z katalogu repozytorium w WSL; jeżeli sesja działa pod Windows, używać `wsl.exe` z katalogiem roboczym ustawionym na repozytorium w WSL.

Ścieżki i role interpreterów zweryfikowano w `benchmark/README.md`, `training/README.md` oraz `src/pl_mmb_benchmark/models.py` (`worker_python`).
