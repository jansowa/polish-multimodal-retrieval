# Backlog serii F

- Zweryfikować brak seedów R3 Wiki2 42/44 w leaderboardzie bez zmieniania zamrożonej generacji (S00-BOOT-audit.md).
- Ustalić dowód liczby VN 53794 par; nie przepisywać niepotwierdzonej liczby (S00-BOOT-audit.md).
- Doprecyzować opis latencji PE-Core: B16 ma pomiar, S16/T16 nie (S00-BOOT-evidence.json).
- Historycznie w S00 źródłowe specyfikacje i część danych/kodu pozostawały nieśledzone (rozwiązano w S02-REPO-HYGIENE); nie dołączać cudzych zmian do commitów sesji. Po domknięciu S00 artefakty F są w głównym checkoutcie.
- F01: sprawdzić rozbieżność total parametrów leaderboardu względem sumy vision/text/head; bez poprawiania w S00.

Rozbieżności S00 pozostają otwarte: R3 G2 tylko seed43 w leaderboardzie; PE-Core-B16 ma latencję; VN 53 794 niepotwierdzone. Szczegóły i dowody: STATUS.md / „Rozbieżności do wyjaśnienia” oraz S00-BOOT-audit.md.

- H-OPS-S01 — rozwiązane w S02-REPO-HYGIENE przez tematyczne commity i zatwierdzone reguły ignorowania; historyczny problem: rozstrzygnąć konflikt nieczystego głównego checkoutu z warunkiem czystego startu; bez usuwania, stashowania ani commitowania cudzych zmian. Dowody: S01-F01-a-evidence.json; eskalacja: handoffs/S01-F01-a.md.

- Sprawdzić przyczynę zatrzymania niezależnej kolejki Wiki (download_compact.py): `reports/data_study_20261030/wiki_recovery_20261031/FAILED.md`; komunikat i SHA zapisano w `S02-REPO-HYGIENE/observed-wiki-failure.json`. Poza zakresem porządkowania i F01 kroków 1–2; nie uruchamiano napraw ani restartów.
- Historyczne whitespace zapisane w `S02-REPO-HYGIENE/whitespace-notes.json`; ewentualne porządki wyłącznie poza zamrożonymi plikami i z uwzględnieniem wiązań SHA.
