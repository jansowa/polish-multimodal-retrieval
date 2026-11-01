# Porządki w training — 2026-10-21

Odzyskano **2,331 GiB** (2 503 110 254 bajty). Na D było **4,19 GiB wolnego** przy odczycie o 20:20:54 czasu warszawskiego. To odczyt chwilowy; osobny pipeline nadal zapisuje dane.

| Operacja | Efekt |
|---|---|
| Starsze zakończone serie: 16 zbędnych stanów wznowienia + 4 kopie snapshotów zastąpione hardlinkami | 1,589 GiB |
| Zakończone i zweryfikowane treningi VN: 8 stanów wznowienia | 0,310 GiB |
| Bezstratna kompresja NTFS cache’y | 0,432 GiB |

Usunięte `last.pt` zawierały optimizer/RNG potrzebne do wznawiania zakończonych przebiegów. Wybrane `best.pt`, historie treningu, konfiguracje, oryginalny FULL_CLEAN, walidacja i zamrożone manifesty VN zostały zachowane. Każdy wybrany checkpoint zweryfikowano przed usunięciem odpowiadającego mu stanu wznowienia.

Kompresja zachowała SHA256 i ścieżki wszystkich trzech cache’y. RGB COCO zajmuje fizycznie 1 734 397 952 bajty, a RGB VN 5 000 663 040 bajtów. Cache prefixów vision (4 803 143 808 bajtów) nie poddał się tej kompresji. Te duże pliki umożliwiają kolejne treningi i odtworzenie metryk bez ponownego przygotowywania cache’u.

Hardlinki zachowują wszystkie dotychczasowe ścieżki do niezmiennych snapshotów ewaluacyjnych, współdzieląc ich dane na dysku. Zwykłe sumowanie rozmiarów plików po ścieżkach liczy je wielokrotnie i nie uwzględnia fizycznego rozmiaru kompresji NTFS.

Dokładne ścieżki, SHA i liczby bajtów:

- [Zbiorczy audyt](training_storage_summary.json)
- [Starsze serie i duplikaty](training_storage_cleanup.json)
- [Zakończona seria VN](training_storage_cleanup_visualnews.json)
- [Kompresja cache’y](training_cache_compression.json)

Wyniki treningów: [TinyCLIP z VisualNews](tinyclip_visualnews.md). Wszystkie 10 wybranych checkpointów ponownie wczytano z dysku i odtworzono R@1/5/10 obu kierunków oraz val loss. FULL_CLEAN i walidacja zachowały SHA. Test MSCOCO i retrieval XM3600 nie były uruchamiane.
