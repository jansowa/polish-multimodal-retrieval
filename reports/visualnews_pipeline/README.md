# VisualNews: grounding → recycling → Polish

The continuous worker processes all **283,012 verified image/caption pairs**. The other 254,556 captions lack a verified image mapping in the current assets and are not sent without an image. Original translation and judge checkpoints remain intact; their former workers were drained before this pipeline started.

The selected model is `qwen3.8-27b`. The four steps are separate requests and independently checkpointed:

1. **Judge** — original English caption plus verified image, frozen `B_taxonomy_v3`, temperature 0.2, thinking off. Previously completed compatible judge outputs are imported rather than requested again.
2. **Recycle** — only grounding 1–2 without mismatch. A separate prompt receives the image, original English caption and complete first-stage assessment. It returns a new English caption, retained visual information, removed information and an explicit `usable` decision.
3. **Recheck** — the new English caption plus the same image is independently judged. Grounding 3–4 without mismatch is required before translation. Failed rescues go to review; there is no unlimited rewrite loop.
4. **Translate** — the final accepted English caption is translated by the project's existing `ThinkingImageTranslator`, with the image and `enable_thinking=true`, temperature 0, maximum 8192 output tokens. Existing exact-source translations are reused. Thinking token counts are retained, but reasoning text is removed from saved responses.

Grounding 0 goes to `rejected_grounding`; an original mismatch goes to `review_mismatch`. These decisions select the new output dataset; source data is never deleted. `review_recycling_failed` and stage errors are excluded from ready training pairs.

## Existing Polish translations

`existing_translations.sqlite3` contains **44,955 unique English texts**, collected from 44,998 completed rows in the previous VisualNews checkpoints. Matching uses the exact English string after removing only the fixed MMEB task instruction and edge whitespace. It does not use approximate text matching or an ID alone.

Preference among duplicate historical translations is image+reasoning, then image, then recency. An original accepted caption can use its old Polish translation. A recycled caption must match its **new** English text; the original's translation is never silently assigned to the new text. Current-run translations are also reused, and identical English texts share an asynchronous lock to prevent duplicate simultaneous translation requests.

A small terminology guard detects the observed mistranslation of `riot gear` as `antyradziecki`. It requests one explicit correction through the existing translation path; both responses are retained. This guard is a targeted check, not a claim that all semantic translation errors are automatically detected.

## 16 versus 32 requests

Four end-to-end runs used the **same 64 source pairs**, in order **16, 32, 32, 16**, with fresh initial judging each round. The sample contained 8 initially strong captions and 56 recycling candidates. Existing-translation cache contents were held fixed across rounds. Some initial scores changed between draws, so stage counts are reported explicitly in the comparison.

| Global concurrency | Valid prompt outputs/s | Ready EN/PL pairs/h | Valid outputs |
|---|---:|---:|---:|
| 16 | 3.252 | 3163 | 470/470 |
| 32 | 3.566 | 3486 | 464/464 |

**32 was selected:** 10.2% more final bilingual pairs/hour. This is a small matched local measurement, not a universal server capacity claim. Individual requests were slower at 32; total useful throughput was higher. The bound is **32 requests total across all stages**, not 32 per stage.

On 2026-10-21 the user requested a reduction to **8 total concurrent requests**. The active runtime override is saved in `runtime_config.json` and takes precedence over the historical benchmark selection on restarts. The launcher also accepts `--concurrency 8`. This runtime change preserves the same prompts, dataset and checkpoint.

[Full comparison](benchmark/comparison.json), [protocol](benchmark/protocol.json), [fixed IDs](benchmark/ids.json).

The terminology guard was added after inspecting the integration smoke test; none of the four benchmark runs contained its trigger error, so those runs required no extra correction calls. The corrected smoke test recorded the initial bad response and its successful correction. One completed benchmark round and the corrected smoke records were adopted into production rather than computed again; the other benchmark outputs remain preserved separately.

## Storage and exports

- [Live status](../visualnews_pipeline_state.json)
- [Production checkpoint](v1/checkpoint.sqlite3)
- [Frozen settings and prompt texts](v1/manifest.json)
- [Recycling prompt](v1/recycle_prompt.txt)
- [Grounding prompt](v1/judge_prompt.txt)
- [Ready bilingual examples](v1/final_pairs.jsonl)
- [Production log](../visualnews_pipeline_production.log)
- [Existing-translation provenance](translation_cache_manifest.json)

SQLite is the durable source of truth:

- `items`: original caption, verified image reference, next stage, routing decision, final EN and PL.
- `outputs`: every successful stage output, including reused outputs and their source checkpoint/record.
- `requests`: every new prompt attempt, exact text inputs, image hashes/references, prompt hash, model response without reasoning, timings and validation status. Failures and corrective responses are preserved.
- `imports`: benchmark and smoke records adopted into production.

Imported first-stage raw HTTP responses remain in the previous judge checkpoint's `attempts` table; their exact location and image hashes are recorded in output provenance. Historical Polish translations retain their original source/target JSON and checkpoint ID. No legacy output was discarded.

`final_pairs.jsonl` is refreshed every ten minutes and on graceful shutdown. It can also be exported at any time from a consistent read-only snapshot:

```powershell
.venv-windows/Scripts/python.exe experiments/visualnews_pipeline_export.py --kind pairs
.venv-windows/Scripts/python.exe experiments/visualnews_pipeline_export.py --kind stages
```

The second command produces `v1/stage_outputs.jsonl`, including failed/retried model responses and references to reused legacy outputs. `valid=1` describes technical output validation; it does not turn generated labels into human ground truth. Image references point into the existing verified Parquet cache; images are not duplicated for each stage.

To request a graceful stop, create `reports/visualnews_pipeline.stop`. In-flight requests finish and their outputs are committed. After inspecting and removing that stop request, resume the same frozen run with:

```powershell
.venv-windows/Scripts/python.exe -B -X utf8 -u experiments/visualnews_pipeline_launch.py
```

A new prompt/configuration requires a new run directory, protecting the current run's provenance. The active worker prevents automatic Windows sleep while it runs.
