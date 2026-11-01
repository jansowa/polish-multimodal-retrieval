# Mapa repo — S00-BOOT

Ścieżki źródłowe względem `D:\work\translate-vision-embedder-benchmark\pl-multimodal-benchmark-translator\pl-multimodal-benchmark-translator`; po domknięciu S00 wszystkie artefakty F są w głównym checkoutcie na bieżącej gałęzi; `.s00-boot-worktree` usunięto. Kod, testy, specyfikacje oraz wybrane dowody zapisano w Git w S02-REPO-HYGIENE; dane robocze i kopie pozostają lokalne, ignorowane. Raport: reports/f_series/S02-REPO-HYGIENE/REPORT.md. Dowody i SHA: reports/f_series/S00-BOOT-evidence.json. Jedna linia na pozycję.

- Scorer / bootstrap: src/pl_mmb_benchmark/full_gallery.py — per_query(), paired_bootstrap().
- Loss: src/pl_mmb_training/losses.py.
- Backbones: src/pl_mmb_training/backbones.py; src/pl_mmb_training/commercial_backbones.py; retrieval_benchmark/commercial_models.py; retrieval_benchmark/openvision_commercial.py.
- Features: src/pl_mmb_training/features.py — encode_cached(..., cache_root); training/cache_commercial_backbone.py.
- Holdout zamrożony: src/pl_mmb_training/holdouts.py; retrieval_benchmark/holdouts/registry-v2.json — nie edytować.
- Holdout bieżący: src/pl_mmb_training/current_holdouts.py — DEFAULT_REGISTRY=registry-v4.json; training/DATA_SPLITS.md.
- Coordination API: experiments/data_ood_common_20261030/common.py — gpu_lock(), reserve(); reports/data_ood_coordination_20261030/; blokada training/runs/tinyclip-visualnews-v1/worker.lock.
- Efficiency: src/pl_mmb_benchmark/efficiency.py; training/run_series_our_efficiency.py; retrieval_benchmark/space/results.json → models.*.efficiency.
- Reviewer: reviewer/server.py; reviewer/index.html; reviewer/app.js; reviewer/README.md.
- Benchmark: retrieval_benchmark/space/ — frontend i results.json; retrieval_benchmark/local/ — konfiguracja lokalna; retrieval_benchmark/holdouts/ — rejestry.
- Katalog bieżący: retrieval_benchmark/catalog-current.json; retrieval_benchmark/catalog.json — zamrożony, nie edytować.
- Dane konfiguracji: retrieval_benchmark/local/config.json → datasets.*.image_manifest/protocol; raporty lokalne reports/series_2026-10-23/; brak sprawdzania plików zewnętrznych.
- Cache domyślny: src/pl_mmb_translate/config.py → HF_CACHE_DIR=./.cache/huggingface; OUTPUT_DIR=./outputs; features.py przyjmuje cache_root od wywołującego; .env nie czytano.
- Wiki źródło: reports/wikimedia_pipeline/runtime_config.json → C:/Users/jsowa/datasets/polish_culture.
- Wiki2 magazyn: experiments/data_study_20261030/wiki_recovery_20261031/config.py → C:/Users/jsowa/benchmark-translations/data-ood-20261030/A/wiki-recovery-20261031; EXPORT=.../series-2026-10-23/expanded-inputs-20261029/wikimedia_final_pairs.jsonl.
- Cache tekstu Wiki2: reports/data_study_20261030/wiki_recovery_20261031/feature_identity.json → new_text_cache / reused_text_cache_sha256; nie zmieniać generacji.
- Walidacja / negatywy: training/runs/clean-v1/validation.jsonl; training/runs/e1-full/translation_snapshots/ — tylko ID walidacji.
- Pipeline stany: reports/visualnews_pipeline_state.json; reports/wikimedia_pipeline_state.json; run_dir/checkpoint w tych plikach; stopped_by_request.
- Seria F: reports/f_series/; protocols/; licensing/; experiments/f_series/; training/runs/f_series/ — szkielet S00, bez wyników nowych eksperymentów.
- Konfiguracja data_study_results: retrieval_benchmark/local/config.json → C:/Users/jsowa/benchmark-translations/data-ood-20261030/A/evaluation.
- Konfiguracja ood_frozen: retrieval_benchmark/local/config.json → C:/Users/jsowa/benchmark-translations/data-ood-20261030/B/frozen.
- Konfiguracja ood_results: retrieval_benchmark/local/config.json → C:/Users/jsowa/benchmark-translations/data-ood-20261030/B/results.
- Konfiguracja output: retrieval_benchmark/local/config.json → C:/Users/jsowa/benchmark-translations/retrieval-benchmark-v1/runs.
- Konfiguracja series_results: retrieval_benchmark/local/config.json → C:/Users/jsowa/benchmark-translations/series-2026-10-23/evaluation.
- Konfiguracja flickr_parquet: retrieval_benchmark/local/config.json → C:/Users/jsowa/benchmark-translations/data-ood-20261030/B/flickr30k/test.parquet.
