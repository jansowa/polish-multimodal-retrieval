# S00-BOOT — weryfikacja statusu wyjściowego

Data: 2026-11-01. Zakres: każdy wiersz tabeli z 00; bez nowych ocen, treningu i pobierania.

Źródło: zastany brudny checkout `D:\work\translate-vision-embedder-benchmark\pl-multimodal-benchmark-translator\pl-multimodal-benchmark-translator`, HEAD `ba8901712837716d2f783e85596a0f0f3433acd9` (inny niż snapshot e1b9676 w 00). Nie utożsamiamy plików roboczych z zawartością tego commitu. Czysty worktree wynikowy: `.s00-boot-worktree`, gałąź `f/boot`. Dokumenty/specyfikacje oraz część dowodów są nieśledzone w źródłowym checkoutcie; nie są automatycznie dołączone do commitu S00. Ścieżki w tabeli odnoszą się do źródła.

| Wiersz 00 | Weryfikacja / rozbieżność | Dowód lub miejsce sprawdzenia |
|---|---|---|
| Benchmark v2, scorer, bootstrap, leaderboard | Potwierdzone: 50 modeli, 584 wyniki; 7 ready + DOCCI planned. | retrieval_benchmark/space/results.json; retrieval_benchmark/catalog-current.json; src/pl_mmb_benchmark/full_gallery.py (per_query, paired_bootstrap); retrieval_benchmark/space/ |
| XM3600 PL | Potwierdzone wpisy wymienionych modeli; także R3_wiki2_seed43. Raport completion dotyczy wcześniejszego podzbioru, nie pełnego obecnego leaderboardu. | retrieval_benchmark/space/results.json; reports/benchmark_decontamination_20261031/xm3600_completion.json |
| Holdout v4 / Wiki2 v2 | Potwierdzone bieżące DEFAULT_REGISTRY=v4; polityka zachowania v2 dla Wiki2. Nie przeprowadzano ponownego audytu danych. | src/pl_mmb_training/current_holdouts.py; retrieval_benchmark/holdouts/registry-v4.json; retrieval_benchmark/holdouts/registry-v2.json; training/DATA_SPLITS.md |
| Holdout v5 / v6 | brak w repo: registry-v5.json, registry-v6.json; zgodne z todo. | retrieval_benchmark/holdouts/ (lista v1–v4) |
| R0/R1/R3 G1, R3 G2, VN_only, A1 | R0/R1/R3 G1: seedy 42/43/44 obecne; VN_only i historical_A1 obecne. R3 G2: tylko R3_wiki2_seed43; seedy 42 i 44 w leaderboardzie: brak w repo. Nie rozstrzyga to istnienia checkpointów poza repo. | retrieval_benchmark/space/results.json → models; S00-BOOT-evidence.json → models |
| Latencja | R0_seed44 CPU obraz 77,105 ms, tekst 14,498 ms potwierdzone. PE-Core-B16 ma efficiency; brak dla PE-Core-S16/T16, natywnych TinyCLIP, ViTamin i OpenVision. Zapis o całej rodzinie PE-Core jest za szeroki. | retrieval_benchmark/space/results.json → models.*.efficiency; src/pl_mmb_benchmark/efficiency.py |
| Kandydaci wizji | Rewizje, deklaracje licencji i wyniki referencji potwierdzone; także ViTamin-S-LTT. Dowód kompletnego cache/sondy F z PolDense: brak w repo w nowym obszarze F. Nie dowodzi to braku historycznych cache TinyCLIP. | retrieval_benchmark/space/results.json → models/results; training/VISION_STUDY.md; training/VISION_COMMERCIAL_STUDY.md (ścieżki istnieją; treści nie czytano) |
| Nauczyciel embeddinggemma-2 | Rejestracja, wyniki i CUDA obraz 41,744 ms potwierdzone. Cache treningowy F04: brak w repo w obszarze F. | retrieval_benchmark/space/results.json → models.*.efficiency/results |
| Pule VN / Wiki | Potwierdzone VN ready=165316 / total=283012; Wiki ready=46968 / total=417353; oba stopped_by_request. To nie dopuszczenie do treningu. | reports/visualnews_pipeline_state.json; reports/wikimedia_pipeline_state.json |
| Dane dotychczasowe FULL / VN / Wiki G2 | FULL 70289 wierszy / 10645 obrazów i Wiki2 3539 obrazów mają dowody dokumentacyjne. VN 53794 par: brak w repo potwierdzenia w sprawdzonych wskazanych niżej źródłach; liczby nie uznano za zweryfikowaną. | training/CLEAN_V1.md; reports/KONTEKST_PROJEKTU_DLA_MODELU_2026-10-31.md; reports/tinyclip_visualnews_data_audit.json |
| Dossier licencyjne | Dossier istnieje; commercial_register-v1.json: brak w repo; zgodne z częściowym stanem. Nie wydawano decyzji prawnej. | reports/series_2026-10-23/LICENSING_DOSSIER_20261030.md; licensing/ |
| Generator qwen3.8-27b | Alias widoczny w konfiguracji źródeł; dowód tożsamości wag i rozstrzygnięcie zakresu DATA-GEN: brak w repo w rejestrze F02. | training/configs/e1-pilot6k.json; licensing/ |
| Oracle / hubness / głowica tekstowa | Artefakty F03/F08: brak w repo w reports/f_series/ i experiments/f_series/; zgodne z todo. Nie twierdzimy, że żaden historyczny kod nie implementuje podobnych obliczeń. | reports/f_series/; experiments/f_series/ |
| Panel ARO/SugarCrepe | Dane walidacyjne istnieją; zliczono 5602 par po polu dataset. Snapshoty tłumaczeń istnieją; ukończony panel F01: brak w repo. | training/runs/clean-v1/validation.jsonl; training/runs/e1-full/translation_snapshots/; training/DATA_SPLITS.md |
| Nowy holdout / ludzki QA | Nowy rejestr v6 i artefakty F10: brak w repo; istniejące reviewer/ nie oznacza gotowego nowego holdoutu. | retrieval_benchmark/holdouts/; reports/f_series/; reviewer/ |
| Retencja PIRB / replay | Harness i dostarczony manifest M01: brak w repo w obszarze serii F/M; stan todo, dane ma dostarczyć właściciel. | reports/f_series/; experiments/f_series/; training/runs/f_series/ |

„Brak w repo” oznacza brak wskazanego artefaktu w sprawdzonym obszarze; nie jest dowodem braku plików na dyskach zewnętrznych ani niedokończonych prac innych sesji. Surowych manifestów i results.json nie wypisywano w całości; JSON agregowano, walidację czytano strumieniowo.

Źródła liczb i ich pełne SHA256: [S00-BOOT-evidence.json](S00-BOOT-evidence.json). Wpis GUARD #0 jest historyczną adnotacją, nie nową oceną.
