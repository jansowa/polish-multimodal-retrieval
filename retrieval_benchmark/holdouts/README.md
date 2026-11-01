# Rejestr ochrony danych ewaluacyjnych

Zasady: [training/DATA_SPLITS.md](../../training/DATA_SPLITS.md).
Nowe audyty używają `pl_mmb_training.current_holdouts.DEFAULT_REGISTRY` (v4).
Zamrożona generacja Wiki2 nadal używa rejestru v2 przez bazowy `holdouts.py`.
Nie zawiera wyników ani wag. Odwołuje się do lokalnych audytowanych manifestów;
nie jest częścią pięciu plików publikowanych do Space.

Sprawdzenie integralności i kompletności fingerprintów, bez GPU:

```bash
PYTHONPATH=src python -m pl_mmb_training.current_holdouts
```

Audyt **nowych dopuszczanych rekordów**, nie całego historycznego FULL:

```bash
PYTHONPATH=src python -m pl_mmb_training.current_holdouts --manifest new_pairs.jsonl
```

Kod wyjścia2 oznacza blokadę. Rekordy muszą zachować źródłowe ID, lineage_verified,
SHA256 oryginału, pixels_sha256 i phash. Kontrola jest przechodnia; sprawdzaj
scaloną pulę, nie każdy rekord osobno. W pełnym procesie metadane wszystkich
aliasów muszą uczestniczyć w grafie jeszcze przed finalnym wyborem.

`freeze.py` tworzy snapshot z bieżących lokalnych galerii, walidacji, całych
komponentów Wiki i historycznych blacklist. Dołącza źródła negatywnych podpisów
walidacyjnych i podpisuje katalog benchmarku. Ponowne uruchomienie akceptuje tylko
identyczne dane; nigdy nie nadpisuje rejestru z innymi SHA.

```bash
python retrieval_benchmark/holdouts/freeze.py --output retrieval_benchmark/holdouts/registry-v5.json
```

Przed aktywacją nowej wersji sprawdź, że konfiguracja lokalna obejmuje wszystkie
gotowe galerie z katalogu, a nowe formaty mają pełne tożsamości obrazów i tekstów.
Następnie jawnie zmień current_holdouts.DEFAULT_REGISTRY i utwórz nowy audyt/generację; nie zmieniaj
bindingu istniejącego treningu. Dodanie datasetu do samego JSON katalogu nie daje
mu ochrony: bieżący audyt blokuje trening zmianą SHA `catalog-current.json`.
Historyczny `catalog.json` jest zamrożonym wejściem Wiki2, nie katalogiem live leaderboardu.

`audit_queue_snapshot.py` czyta zamknięte JSON receipts aktywnego downloadera,
nie dotyka SQLite/WAL ani kolejek. Wynik jest częściową obserwacją kandydatów.
Nie sprawdza ponownie wszystkich payloadów obrazów i nie wydaje release'u.

Cache fingerprintów generacji1 jest użyty tylko jako dowód oryginalnego pHash
dla faktycznie zarejestrowanych holdoutów. Jego rekordy treningowe **nie** stają
się przez to holdoutami. Znane historyczne ekspozycje XTD-full pozostają oznaczone.
