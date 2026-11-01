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

1. Jedna sesja obejmuje tylko pakiet, krok lub ramię wskazane w prompcie; inne pomysły zapisuj po jednej linii w reports/f_series/BACKLOG.md.
2. Nie zaczynaj kolejnego pakietu w tej samej sesji; zakończ handoffem.
3. Czytaj minimalnie: STATUS.md, najnowszy handoff, właściwe sekcje 04/05 i dokumenty wymagane przez 00; duże pliki przeglądaj selektywnie lub agreguj.
4. Przed przeszukiwaniem całego repo sprawdź docs/agent_spec_f/REPO_MAP.md.
5. Zaczynaj od czystego git status, pracuj na f/<pakiet> i rób małe commity [ID]; nie zmieniaj holdouts.py, starego catalog.json ani zamrożonych generacji.
6. Job GPU musi być wznawialny i mieć training/runs/f_series/<run_id>/JOB.json z poleceniami, PID, czasami i ścieżkami; po smoke teście czekaj najwyżej 15 minut, potem handoff i osobna sesja odbioru.
7. Kończ kolejno: STATUS.md → handoffs/<SESSION_ID>.md (szablon 09 §3, najwyżej jedna strona) → SESSIONS.md → commit → odpowiedź najwyżej 10 linii.
8. Bramka ludzka kończy sesję: przygotuj eskalację wg 07 §8, zapisz ją w handoffie; nie obchodź bramki ani nie zgaduj decyzji.
9. Logi w czacie ogranicz do 50 linii; pełne logi zapisuj w plikach, podawaj ścieżkę i wniosek.
10. Każda liczba raportowa musi mieć źródło: ścieżkę pliku i SHA; nie przepisuj liczb z pamięci rozmowy.
