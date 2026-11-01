# Wikimedia: source captions → grounding → recycling → Polish training pairs

The worker reads `C:/Users/jsowa/datasets/polish_culture/checkpoint.sqlite3`, the expanded image collection, without modifying source records. New records are discovered every two minutes. The current inference cap is 30, with priority shares Wiki 24 and VisualNews 6 when both queues have model-ready requests waiting. `work_conserving: true` lets either dataset use capacity for which the other has no ready request. Wiki fetches and normalizes images before acquiring an inference slot; prepared bytes are pinned across LRU eviction to avoid a second download. The existing shared Wikimedia HTTP pacing, cooldowns, license checks and original hash checks are retained. Thus image-fetch waits do not reserve inference slots, and VisualNews can use them. The hard combined inference cap always applies, including while returning borrowed capacity. Configuration is in `reports/balanced_inference_config.json`; actual active requests, model-ready waiters and priority shares are in `reports/balanced_inference_state.json`.

Start the coordinated worker with `.venv-windows/Scripts/python.exe -B -X utf8 -u experiments/run_balanced_inference.py`. Do not start separate workers while it is active; it owns both worker locks. Create `reports/balanced_inference.stop` to drain and stop both. Allocation and actual active-slot counts are updated every 10 seconds in `reports/inference_allocation.json` and `reports/balanced_inference_state.json`. Combined logs are `reports/balanced_inference.log` and `reports/balanced_inference_error.log`.

The source expansion verified the image identities (file ID, original URL and SHA1) for all 8,235 existing pipeline items before updating source provenance. Previous results, requests, candidates and grades remain in the same database. A full pre-expansion SQLite backup, previous manifest and migration audit are retained in `v1/`; completed items are not requested again.

Images are scheduled by their initial selected source-caption grade: A, then B, C, D. The grade remains fixed when trying fallback captions. A lower class does not start while a higher class has pending or running work, including delayed retries. Within the active class, finishing translation/recheck/recycling takes priority over starting new judges. Permanent errors are reported separately and do not block all subsequent classes. Class C depicts IDs alone remain ineligible without a PL/EN textual caption.

Admission requires a matching explicit license name and canonical Creative Commons URL, using the collector's existing validator. Allowed: CC0, CC BY and CC BY-SA versions 2.0, 2.5, 3.0 and 4.0. Missing, inconsistent, NC, ND and unknown licenses are excluded before image download or model requests. Authors, attribution, license URL, original URLs and full source metadata are retained. Source quality A–D remains distinct from model grounding 0–4.

Per-image structured captions and image descriptions take priority over entity/category context. PL/EN candidate captions are judged separately, with at most eight distinct candidates per image. A failed caption does not automatically reject an image: the next source candidate is tried. Unlabelled-language file titles are not assumed to be English.

1. Judge: image and caption, grounding 0–4 and mismatch assessment.
2. Recycle only grounding 1–2 without mismatch: image, original caption, explicit language, complete initial assessment. Output stays in the input language.
3. Independent recheck: recycled caption and image. Requires grounding 3–4 without mismatch.
4. Accepted Polish is retained directly. Accepted English is translated by the existing image-aware reasoning-enabled translator; exact-source prior translations can be reused.

All model attempts, including failures and unsuccessful candidate captions, are stored without reasoning in the `requests` table. `outputs` contains the current candidate's stage results. Frozen prompts and settings are in `manifest.json`; the original records are in `items.record_json`. Final exports include selected candidate provenance and full source records. Rejected candidates are not silently converted to unrelated image-only captions.

## Live preview of ready training pairs

Verified compact images are now retained in the shared recovery archive. See [IMAGE_ARCHIVE.md](IMAGE_ARCHIVE.md) for paths, original/native verification, immutable publication, recovery-queue ownership and audit metadata. The in-memory LRU is no longer the only retained image representation.

Open **http://127.0.0.1:8767/** on this computer. This dedicated gallery reads only completed ready pairs with Polish text and an allowed open license. It shows the final Polish text, original selected caption, recycled caption, independent grounding scores, author/license metadata and downloadable record JSON. Source grade and recycling filters, text search, random ordering and pagination are available. It refreshes every 30 seconds; the detail dialog stays stable while open.

The original collector browser at port 8766 shows source records. The training gallery at 8767 shows processed training outputs. Images are loaded from Wikimedia, so viewing images needs internet. The service binds only to 127.0.0.1 and does not publish the dataset.

Restart the gallery with `.venv-windows/Scripts/python.exe -B -X utf8 -u experiments/wiki_training_browser.py`. Service metadata and logs are under `reports/wiki_training_browser/`. Four backend tests and a real Chrome desktop/mobile check passed; evidence is saved there as `ui_verification.json`, `desktop.png` and `mobile.png`.

## Output files

- `v1/checkpoint.sqlite3`: authoritative output database; continuously committed.
- `v1/final_pairs.jsonl`: ready Polish pairs, exported every minute and at shutdown.
- `v1/manifest.json`: prompts, schemas, model and processing settings.
- `v1/state.json` and `../wikimedia_pipeline_state.json`: live status.
- `v1/errors.jsonl`: image/model processing errors and retries.

The active `v1` directory is a Windows junction to `C:/Users/jsowa/benchmark-translations/pipeline-storage/wikimedia-v1`. The VisualNews active directory similarly points to `.../visualnews-v1`. Previous D: copies were retained as `v1.before_storage_migration_*`; they are historical backups, not live results. This keeps existing project paths usable while reserving space for continuing work.

Images are fetched serially using the scraper's shared request budget and server cooldowns. Original hashes are checked when available. Model images are normalized to JPEG, maximum side 1280; an LRU cache holds at most approximately 64 MiB in RAM. No bulk original-image download is created. Source and request image hashes are saved with each request. Both workers stop below a 4 GiB reserve on their output volume.

## Run / stop

```powershell
.venv-windows/Scripts/python.exe -B -X utf8 -u experiments/run_balanced_inference.py
```

Create `reports/wikimedia_pipeline.stop` for a graceful stop; remove it to resume. The worker lock prevents duplicate instances for the same run directory. Source license revocations found during refresh exclude affected items, including previously ready records. A record lacking an eligible PL/EN source caption is logged as excluded rather than generating a fresh image-only caption.

## Validation

Recovery on 2026-10-23: the Windows shared-budget lock fix is loaded by a fresh worker, permission errors and wrapped server timeouts are retried, and a worker with only failed records reports `blocked_errors` instead of `running`. 4,208 failed records were requeued with audit entries in `recovery_events` and `recovery_20261023.json`; original outputs and request histories were retained. Six oversized images remain outside the retry queue. A malformed translation JSON received one additional bounded attempt. 83 tests passed before restarting both workers with concurrency 16 each.

73 tests passed across the new adapter, VisualNews routing and collector suites. A real two-image smoke test produced one unchanged Polish pair and one independently rechecked recycled English caption translated to Polish, with five saved model requests. These two results and their request histories were adopted into production without repeating their requests.
