# Audyt i pilot nowych danych — sesja A, 30.10.2026

Wykonano wyłącznie zarejestrowane R0/R1/R2/R3. Nie uruchamiano sesji B, nowego legacy COCO, XM3600, FT/LoRA/KD ani nowych backbone’ów.

Zamrożony eksport: 132 875 VN i 24 125 Wiki. Pilot: 19,888 nowych rekordów VN (19,876 obrazów według pixel SHA; 19,845 komponentów) i 28 Wiki (28 obrazów; 16 komponentów).
Wiki: 23,562 rekordów bez lokalnego oryginału; 401 dostępnych miało już bezpośrednią chronioną tożsamość. Spośród pozostałych 162 pełne komponenty dopuściły 28.
VN: oficjalne caption/DID/path i artykuły sprawdzone bez SQLite; z 20 000 deterministycznie wybranych oryginałów odrzucono 112. Powody wielokrotne: `{"COCO_val": 3, "FULL_train": 1, "VN_dev": 3, "VN_train": 103, "XM_blacklist_only": 3, "legacy_eval": 2}`.

Stare FULL/VN zachowują oryginalne podpisy. Nowe kopie starych komponentów wykluczono. pHash≤6 służy wykluczeniom, nie tworzy nowych pozytywnych qrels.

Każdy nowy run: 5500 kroków, 1 405 780 prezentacji, w tym 702 880 FULL; batch256 i ostatni145 co275. R0/R1 używają dokładnie historycznego samplera. Wiki zastępuje tylko sloty VN: 42 173 prezentacje dla3% lub140 578 dla10%.
Enkodery frozen, świeży Linear512→384 i loss SigLIP; FP32 head/loss/cache, BF16 autocast przy ekstrakcji enkoderów. Wybór checkpointu: VNdev t2iR1, potem meanR1, potem wcześniejszy krok, co275 i krok0.

| Wariant | Seed42 VNdev t2i R@1 | Seed42 Wiki dev t2i R@1 | Wybrany krok |
|---|---:|---:|---:|
| R0 | 74.089% | 16.616% | 2475 |
| R1 | 74.888% | 15.710% | 2475 |
| R2 | 74.189% | 16.314% | 2475 |
| R3 | 72.941% | 16.616% | 2475 |

Wybrano R3 według Wiki dev seed42; reguła i niekorzystne wyniki zachowane.
Spadek VN względem R1 po trzech seedach: 2.030 p.p.; guardrail≤2p.p.: niezaliczony — brak rekomendacji zastąpienia R1.

| Wariant | Galeria | Status | t2i R@1 mean±SD | i2t R@1 mean±SD |
|---|---|---|---:|---:|
| R0 | VN_dev | ID | 73.922±0.175% | 69.967±0.708% |
| R0 | wiki_native_dev | źródłowo OOD | 15.710±0.799% | 14.274±1.018% |
| R0 | xtd10_full_1k | kontrola COCO/tłumaczenia | 49.633±0.306% | 44.667±0.929% |
| R0 | xtd10_no_overlap | kontrola COCO/tłumaczenia | 51.394±0.000% | 46.302±0.948% |
| R1 | VN_dev | ID | 74.472±0.485% | 69.800±0.218% |
| R1 | wiki_native_dev | źródłowo OOD | 15.106±0.604% | 13.686±0.524% |
| R1 | xtd10_full_1k | kontrola COCO/tłumaczenia | 49.633±0.643% | 45.333±0.252% |
| R1 | xtd10_no_overlap | kontrola COCO/tłumaczenia | 50.910±0.451% | 47.083±0.341% |
| R3 | VN_dev | ID | 72.441±0.435% | 68.317±0.513% |
| R3 | wiki_native_dev | ID | 17.221±1.047% | 15.785±0.727% |
| R3 | xtd10_full_1k | kontrola COCO/tłumaczenia | 49.600±0.529% | 45.400±0.500% |
| R3 | xtd10_no_overlap | kontrola COCO/tłumaczenia | 51.096±0.232% | 46.637±0.672% |

Wszystkie warianty, seed42 (po zamknięciu selekcji, galerie posthoc):

| Wariant | Galeria | t2i R@1 | i2t R@1 |
|---|---|---:|---:|
| R0 | VN_dev | 74.089% | 70.400% |
| R0 | wiki_native_dev | 16.616% | 13.098% |
| R0 | xtd10_full_1k | 49.700% | 45.300% |
| R0 | xtd10_no_overlap | 51.394% | 47.269% |
| R1 | VN_dev | 74.888% | 69.950% |
| R1 | wiki_native_dev | 15.710% | 14.106% |
| R1 | xtd10_full_1k | 50.100% | 45.100% |
| R1 | xtd10_no_overlap | 51.394% | 47.157% |
| R2 | VN_dev | 74.189% | 70.200% |
| R2 | wiki_native_dev | 16.314% | 14.610% |
| R2 | xtd10_full_1k | 50.500% | 45.100% |
| R2 | xtd10_no_overlap | 52.062% | 47.046% |
| R3 | VN_dev | 72.941% | 68.750% |
| R3 | wiki_native_dev | 16.616% | 15.365% |
| R3 | xtd10_full_1k | 49.800% | 45.400% |
| R3 | xtd10_no_overlap | 51.282% | 47.269% |

Pozostałe R@5/10, MRR i NDCG@10 oraz wartości seedów: `training_report.json`. Warunkowe CI różnic reprezentantów: `conditional_CI.json`, paired bootstrap komponentów 10 000, seed20261023; osobno od mean±SD seedów. OOD B otrzyma dokładnie trzech reprezentantów; dziewięć checkpointów pozostaje analizą seedów. R0 pozostaje seed44; pozostali to mediana VNdev, remis→niższy seed.

V+T:70 470 528; V:38 587 392; T:31 883 136. Head196 992 liczony osobno.
Nowe runy: 7; suma zmierzonego czasu pętli treningowych: 115.13 s. Nowe pliki: C: 0.954 GiB, D: 4.87 MiB. Czas audytu: 369.87 s. Bieżące wolne miejsce i szczegóły: `final_accounting.json`. Czas od prerejestracji obejmuje przerwę na wyłączenie komputera, nie jest czasem aktywnych obliczeń.

Ograniczenia: tylko28 dostępnych i rozłącznych Wiki; wielokrotne prezentacje tej małej puli ograniczają wnioskowanie o całym eksporcie. Nie zbierano nowych ręcznych adnotacji. Wiki zachowuje native/recycled/translated i metadane źródeł; nie jest w całości ludzkim goldem.
Licencje/atrybucja: zachowane source_page_url, creator/attribution, license i license_url Wiki oraz rewizje i oryginalne źródła VN. Metadane Wiki deklarują CC BY/CC BY-SA/CC0; podpisy recycled/translated pochodzą z pipeline’u. Nie podjęto decyzji o publikacji ani nie uznano licencji zbioru za automatyczne zezwolenie na publikację każdego obrazu VN.

Historyczny baseline sprzed Wiki i wszystkie osiem preservation SHA pozostały identyczne. Nowy release znajduje się w wersjonowanym pliku `Wiki_training_release*.json`; historyczny raise pozostaje bez zmian. Kod w src nie był zmieniany. Test cache wymagał odtworzenia batchów BF16; obrazy i teksty są wtedy bitowo zgodne z adapterem (`parity_resolution.md`, `adapter_cache_parity.json`).

SHA wszystkich dziewięciu checkpointów, manifestów treningowych i rewizje enkoderów: `../data_ood_coordination_20261030/A_FINALISTS_FROZEN.json`. Reprezentanci do OOD: R0_seed44, R1_seed44, R3_seed44. Dodatkowy R2_seed42 służy wyłącznie zamkniętej analizie pilota i nie rozszerza listy OOD.

Handoff: `HANDOFF.md`, markery w `reports/data_ood_coordination_20261030/`. GPU A zakończone; lock zwolniony. Nie odczytywano wyników B.
