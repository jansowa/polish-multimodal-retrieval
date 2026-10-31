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
