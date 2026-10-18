# Draft vs review experiment

This experiment keeps the v2 prompt, templates, image context and sampling
temperature unchanged. It records both stages of the same request sequence to
measure whether the extra review improves captions enough to justify its cost.

`review_ab_ids.json` selects 100 MSCOCO t2i examples: 20 previously diagnosed cases
and 80 seeded random examples from the 770 records completed in the v1 run.
`review_ab_sample.json` records the seed, source pool and group membership.
The diagnostic group is deliberately biased toward errors. Evaluate the two
groups separately; neither represents a random sample of the full benchmark.

Run from the project directory in WSL:

```bash
source .venv/bin/activate
RUN_NAME=qwen3.8-27b-review-ab100 QUALITY_REVIEW=true SAVE_DRAFTS=true \
  TEMPERATURE=0 BATCH_SIZE=4 \
  python translate.py mscoco_t2i true --ids-file experiments/review_ab_ids.json
```

Use the identical command to resume. The sample IDs and draft-capture mode are
checked in the manifest so that the same run cannot silently change its scope.
`--ids-file` accepts a JSON list of unique exact unit IDs, requires a single
dataset, and cannot be combined with `--limit`.

The output directory is:

```text
outputs/qwen3.8-27b-review-ab100/mscoco_t2i/with_image/
```

In addition to `translations.jsonl` and the live SQLite checkpoint, the runner
exports `review_comparison.jsonl` at completion/interruption. Each comparison
contains `source_compact`, `draft_compact`, `reviewed_compact`, `quality_flags`,
`review_changed` and per-stage request times, attempts and reported token usage.
The same information is stored under `metadata.translation_audit` in completed
checkpoint rows. Fully local template-only records need no API calls and have no
stage comparison. Repeated fixed candidate instructions are omitted from this
small audit export.

To inspect a running or completed experiment without interrupting it:

```bash
python experiments/summarize_review_ab.py
```

This opens SQLite read-only and writes `reports/review_ab/summary.json`,
`pairs.jsonl` and a CSV with blank assessment columns. The summary reports
completion, changes, flags and stage costs separately for the two groups; it does
not score translation quality automatically.

Assess changes with the practical retrieval use case in mind: changes to objects,
counts, colors, negation and relations matter; harmless stylistic differences do
not. A changed caption is not automatically better. Record semantic improvements
and regressions separately from language-only changes and unresolved source
defects. No-review performance is represented by the saved draft from the same
example; a second translation run is unnecessary.

`SAVE_DRAFTS` defaults to false. Ordinary v2 runs retain the same prompt hash and
manifest configuration as before this instrumentation change.

## Unattended full run after the experiment

Start the queue once from the project directory in WSL:

```bash
.venv/bin/python -u experiments/queue_full_translation.py --detach
```

The detached worker waits for the current `qwen3.8-27b-review-ab100` translation
process to exit, then runs all 1000 MSCOCO t2i examples with images, temperature 0,
concurrency 4, and no review or draft capture. Results are separate under
`outputs/qwen3.8-27b-full-v2-single/mscoco_t2i/with_image/`.
The worker can resume an interrupted experiment once, and retries/resumes the
full job up to three passes if records remain incomplete. Its lock prevents a
second queue from launching another full run. It requires no chat interaction.

Status: `reports/queued_full_translation.json`. Log:

```bash
tail -f reports/queued_full_translation.log
```

The computer and WSL must remain running and the model endpoint must remain
reachable. A reboot stops this local worker; rerun the same detach command to
resume from checkpoints. This is a local process, not a Windows startup task.
