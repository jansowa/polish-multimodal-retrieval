# STATUS — seria F / M

Aktualizacja: 2026-11-01; S02-REPO-HYGIENE zakończono: zastane zmiany zapisano tematycznie, dane robocze i kopie wyłączono z Git bez usuwania. H-OPS-S01 rozwiązano; F01 nadal nie rozpoczęto.

| Pakiet | Stan | Blokery | Oczekujące bramki | Ostatnia aktualizacja | Linki (plan / raport / decyzja) |
|---|---|---|---|---|---|
| F01 | todo | Nie rozpoczęto. Blokadę checkoutu usunięto w S02-REPO-HYGIENE; ponowić kroki 1–2. | — | 2026-11-01 | — / [porządkowanie](S02-REPO-HYGIENE/REPORT.md) / [handoff](handoffs/S02-REPO-HYGIENE.md) |
| F02 | todo | Nie rozpoczęto; zależności sprawdzi kolejna sesja w 04. | H1 | 2026-11-01 | — / — / — |
| F03 | todo | Nie rozpoczęto; zależności sprawdzi kolejna sesja w 04. | — | 2026-11-01 | — / — / — |
| F04 | todo | Nie rozpoczęto; zależności sprawdzi kolejna sesja w 04. | — | 2026-11-01 | — / — / — |
| F05 | todo | Nie rozpoczęto; zależności sprawdzi kolejna sesja w 04. | H3 (udział ludzi) | 2026-11-01 | — / — / — |
| F06 | todo | Nie rozpoczęto; zależności sprawdzi kolejna sesja w 04. | H2 | 2026-11-01 | — / — / — |
| F07 | todo | Nie rozpoczęto; zależności sprawdzi kolejna sesja w 04. | — | 2026-11-01 | — / — / — |
| F08 | todo | Nie rozpoczęto; zależności sprawdzi kolejna sesja w 04. | — | 2026-11-01 | — / — / — |
| F09 | todo | Nie rozpoczęto; zależności sprawdzi kolejna sesja w 04. | — | 2026-11-01 | — / — / — |
| F10 | todo | Nie rozpoczęto; zależności sprawdzi kolejna sesja w 04. | H3 | 2026-11-01 | — / — / — |
| F11 | todo | Nie rozpoczęto; zależności sprawdzi kolejna sesja w 04. | H2, H5 (gdy wymagane) | 2026-11-01 | — / — / — |
| F12 | todo | Nie rozpoczęto; zależności sprawdzi kolejna sesja w 04. | H4, H5 (gdy wymagane), H6 | 2026-11-01 | — / — / — |
| M01 | todo | Faza 2: dopiero po DF3 i dostarczeniu danych tekstowych przez właściciela. | H1 (dane) | 2026-11-01 | — / — / — |
| M02 | todo | Faza 2: dopiero po DF3 i dostarczeniu danych tekstowych przez właściciela. | — | 2026-11-01 | — / — / — |
| M03 | todo | Faza 2: dopiero po DF3 i dostarczeniu danych tekstowych przez właściciela. | H5 (gdy wymagane) | 2026-11-01 | — / — / — |

Liczniki nowych odczytów: **GUARD=0**, **TEST=0**. S00 nie obliczał nowych wyników GUARD/TEST.
Wpis historyczny **GUARD #0 — 31.10.2026 — XM3600 PL**: ekspozycja istniejących wyników z `retrieval_benchmark/space/results.json`; dowód `reports/benchmark_decontamination_20261031/xm3600_completion.json`. Datowanie historyczne zgodne z 00/09; bez rekonstruowania nieznanych timestampów i SHA checkpointów. Nie zwiększa licznika nowych odczytów. Maszynowy log/wrapper pozostają zadaniem S03-F01-c; pełne SHA źródeł: [evidence](S00-BOOT-evidence.json).

Aktywny rejestr: **v4** dla nowych audytów (`src/pl_mmb_training/current_holdouts.py`); Wiki2 zachowuje **v2**. v5/v6 nie utworzono.
Aktywny poziom: **WEIGHTS — domyślny wg 01 §4, bez decyzji H1**; rejestr licencji F02 nie istnieje, brak nowego dopuszczenia danych/składników.

Otwarte decyzje właściciela wg 00 (nie są bramką bootstrapu):
- H1: WEIGHTS/STRICT, obowiązki PolDense, nauczyciel EG2, zakres REF NC, generator i korpusy mostu/replay.
- Przed FE3: budżet latencji / rozdzielczość obrazu.
- H2: manifesty nowych danych; H3: annotacja i QA; H5: budżety ponad limity.
- H4: przyszły TEST; H6: zgodność i publikacja.
- Faza M: wymóg wspólnej przestrzeni / dwóch trybów i dostarczenie danych replay.

Audyt: [każdy wiersz 00](S00-BOOT-audit.md), [źródła i SHA](S00-BOOT-evidence.json).
Następna sesja: **S03-F01-a**, wyłącznie protokół v3-F, VizWiz A/B i nowy katalog; czytać sekcję F01 w 04 oraz 02 zgodnie z 09 §6.

## Rozbieżności do wyjaśnienia

- R3 G2: w leaderboardzie jest tylko `R3_wiki2_seed43`; deklaracja trzech seedów pozostaje do wyjaśnienia.
- PE-Core-B16 ma pomiar latencji; ogólna deklaracja braku pomiarów PE-Core wymaga doprecyzowania.
- VN 53 794 par: liczba niepotwierdzona w audycie S00; wymaga wskazania dowodu.

Źródło rozbieżności i zakres sprawdzenia: [audyt S00](S00-BOOT-audit.md), [agregaty i SHA źródeł](S00-BOOT-evidence.json). W domknięciu ich nie rozstrzygano.

Domknięcie S00-BOOT (2026-11-01): artefakty scalono fast-forward do `master`; worktree i gałąź `f/boot` usunięto. Dalsza praca na bieżącej gałęzi głównego checkoutu, bez nowych gałęzi, worktree i PR. Wszystkie pakiety pozostają `todo`, GUARD=0, TEST=0; historyczny GUARD #0 pozostaje zapisany. Następny etap: [NEXT.md](NEXT.md).

Historycznie S01-F01-a: warunek czystego startu niespełniony; pełny stan i SHA w [dowodach](S01-F01-a-evidence.json). GUARD/TEST bez nowych odczytów; brak jobów tej sesji. Blokadę H-OPS-S01 rozwiązano w S02-REPO-HYGIENE; ponowienie F01 kroków 1–2 zapisano jako S03-F01-a. Nie przechodzono do F01-b.

S02-REPO-HYGIENE (2026-11-01): etap organizacyjny na jawne zlecenie użytkownika, bez realizacji F01. [Raport i SHA dowodów](S02-REPO-HYGIENE/REPORT.md), [commity](S02-REPO-HYGIENE/commits-before-handoff.json), [handoff](handoffs/S02-REPO-HYGIENE.md). H-OPS-S01 rozwiązano przez uporządkowanie Git; wymóg czystego startu pozostaje. Liczniki GUARD/TEST bez zmian. Następna sesja: **S03-F01-a**, wyłącznie ponowienie F01 kroków 1–2. Zaobserwowany komunikat zatrzymania odrębnej kolejki Wiki zapisano w BACKLOG; brak nowych jobów tej sesji.
