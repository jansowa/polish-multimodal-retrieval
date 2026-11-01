# Session B — CPU preparation

This directory belongs exclusively to B. No model loading or CUDA is permitted
until verified A handoffs and B's frozen evaluation exist, and A's common GPU
lock has been acquired. No alternative lock or disk-reservation implementation.

`python experiments/ood_eval_20261030/prepare.py` performs bounded metadata-only
research with two CPU workers, pins the three reference registrations, and
updates B's status. It never downloads model weights or image archives.

Large downloads wait for verified COMMON_READY and its documented reservation
API. Dataset freezing waits for the verified complete A universe. Missing A
markers produce a blockedByA report, never fabricated frozen/results markers.

Implemented stages:

- `vizwiz_assets.py`: original validation ZIP members via bounded HTTP ranges,
  source ETag and member CRC, at most two workers; resumable image/fingerprint pairs.
- `audit_candidates.py`: SHA-verified A universe and protected text/image indices,
  source/parent/URL identities, raw/pixel SHA, pHash ≤ 6 and normalized EN/PL;
  rerun after final translations.
- `translate_cpu.py`: pinned Marian EN→PL, CPU only, two private instances with
  sequential initialization and one Torch thread each. Existing JSONL is reused.
- `qa_flickr_cpu.py`: the same registered surface/language QA for supplied NLLB PL.
- `freeze_cpu.py`: component-based selection, all original captions for retained
  images, explicit multi-positive qrels and immutable B_EVAL_FROZEN.
- `evaluate.py`: unchanged shared scorer/ProjectorEncoder geometry; native base
  features reused across all nine heads; three fixed references sequentially.
  Every CUDA operation is after verified A/B handoffs and inside the common lock.
- `report_results.py`: every metric in JSON/CSV, three-seed mean/sample SD,
  representative paired component bootstrap (10,000; seed 20261023), report and
  immutable B_RESULTS_READY.
- `run.py`: sequential orchestration after the current CPU translation completes.

Use existing WSL `/home/jsowa/.venvs/pl-mmb-legacy-cu130/bin/python` for CPU
image/translation stages (the repository `.venv` lacks SciPy). Gemma uses the
existing `pl-mmb-modern` interpreter. Never install into those environments.
The small vendored language detector lives only inside B's C-drive asset root.

Flickr originals remain embedded in their single pinned Parquet file. GPU
loaders expose them through process-local memory descriptors, without storing
a second image copy. All large B data/results/features remain on C; D holds
only code, metadata and compact report tables.
