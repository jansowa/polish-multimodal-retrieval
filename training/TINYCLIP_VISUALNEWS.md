# TinyCLIP × VisualNews: registered data-volume comparison

This study keeps `clean-v1/FULL_CLEAN.jsonl` and `clean-v1/validation.jsonl` byte-for-byte unchanged. It adds a separate immutable VisualNews manifest under `training/runs/tinyclip-visualnews-v1/data`.

## Data

- Published pipeline export frozen on 2026-10-21, 15:37:51 Europe/Warsaw: 35,918 pairs; SHA256 `e8e010ca5ff14182a9b8b0981f9d3ed4b65a44b875a256c6aa03ba010da9f59b`.
- Read only the published JSONL; never access or modify the live producer SQLite/WAL. Continuing labels do not enter this experiment.
- Exact official caption DID → image DID/path join through the previously verified static asset index. Original metadata supplies all known article paths.
- Verify every used Parquet shard against the pinned asset manifest; decode original image bytes and compute file SHA256, EXIF-normalized pixel SHA256 and original DCT pHash.
- Compare against all existing train/validation images and protected benchmark query/candidate images. Merge the XM3600 blacklist for overlap detection only. No XM3600 retrieval scores.
- Exclude whole components on exact hash/text or pHash Hamming distance ≤6 to protected/base images. The seven disjoint bit bands guarantee no missed pHash match within that radius. This does not detect every crop or every related news event.
- Split connected groups by canonical pixels, known article paths, original and final captions, and pHash ≤6. Dev images and captions never enter either training pool.
- Result: 33,851 train images / 33,906 pairs; 2,000 VN dev images / 2,003 pairs. Nine rows excluded. Largest component contains five images.
- Small pool: 10,000 train images / 10,015 pairs, nested within the full pool and proportional to Washington Post / USA Today source counts.
- VN dev uses the pipeline's model rewrites/translations, not independent human gold. A 24-example model spot check and its caveats are in `reports/tinyclip_visualnews_quality_notes.md`.

## Models and budget

- Pinned `wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M`, standard 512D projected embedding; original CLIP projection frozen.
- Frozen `OPI-PIB/PolDense-32M`, native 384D, original query prefix/pooling/max length.
- Linear 512→384; existing multi-positive sigmoid gold loss. No ARO/SugarCrepe explicit negative captions added.
- Frozen control vs trainable last two blocks; FT uses cosine feature anchor λ=50 to the original TinyCLIP on the same augmented view. No SigLIP teacher KD in this series.
- Shared initialization: A1 epoch 9 projector and learned loss scalars; native vision weights. Seed 42.
- Head/scalar LR 1e-4, vision LR 1e-5; AdamW weight decay .01 (loss scalars zero); warmup 275 updates then cosine to 10% over 3,300 updates.
- All image views use zoom 1–1.08 and translation ±3%, no flips or color edits. Reference prefix computed online for each view; stored RGB is an exact native-preprocessing roundtrip.
- Same 843,468 total record presentations as the historical 12-window controls: 274 batches of 256 + one batch of 145 in every 275-update window.
- Added-data sampling: half original FULL_CLEAN rows, half uniform canonical VN images, one available caption drawn per selected image. Source proportion fixed across 10k/all; original FULL_CLEAN row weighting preserved within its half.
- Consequently, mixed runs have half as many FULL_CLEAN exposures as no-VN controls. The 10k/all comparison holds compute and source ratio fixed; the 0/add comparison also changes this source balance. Report this limitation rather than attributing every gain solely to unique-image count.
- Existing G1/G3 controls without VN are reused, their checkpoints/SHA and COCO metrics reproduced. Best checkpoints and manifests retained after storage cleanup; obsolete resume/optimizer states are unnecessary.

## Selection and reporting

Checkpoint selection uses only the original 419-image COCO validation gallery: t2i R@1, then mean t2i/i2t R@1, then earliest update. Monitor the same quarter-window points in the first two windows and whole windows thereafter. VN dev is evaluated only after selecting the checkpoint; it does not select epochs.

Keep train gold/anchor/total loss, original validation loss, validation loss at fixed initial scalars, R@1/5/10, group gradient norms, unique VN images seen, feature drift, trainable parameter count and peak PyTorch allocated VRAM in JSONL. Report whole-gallery COCO and VN dev separately; their absolute recalls are not directly comparable across different galleries.

The scout has six rows: two reused no-VN controls and four new 10k/all × frozen/last2 runs. Fine-tuning seed repetitions reuse the same initial A1 head and vary the adaptation sampling/augmentation seed; this is not variance of the entire original linear-probe stage. Any followup decision is recorded separately before executing it.

## Commands (WSL project environment)

```text
python training/prepare_tinyclip_visualnews.py
python training/tinyclip_visualnews.py
python training/repeat_tinyclip_visualnews.py
python training/verify_tinyclip_visualnews.py
```

Preparation reuses the frozen snapshot on subsequent invocations. Training has a process lock and exact sampler/optimizer resume at window boundaries. Run GPU workers sequentially. Verification reloads each selected checkpoint and reproduces COCO/VN retrieval and COCO loss, checks immutable input hashes, and produces conditional paired bootstrap intervals and curves. MSCOCO test and XM3600 retrieval are not invoked.

## Registered followup

The six-row scout completed. Adding the full VisualNews pool improved the primary frozen control by 2.78 percentage points over the reused no-VN frozen control. Before starting any repeats, `replication_decision.json` recorded the decision to prioritize replication of this fixed data recipe over another hyperparameter scout. The initial plan's automatic trigger (a >=1pp FT-versus-frozen gain) was not met: the scout's FT effect was +0.62pp. The followup is instead justified by the larger primary data effect, and tests both full-data frozen and FT variants on adaptation seeds 43 and 44, without changing the recipe.

The three-seed tables report sample standard deviations. Paired seed differences are separate from development-query uncertainty. The component-bootstrap interval after averaging the three fitted seeds is conditional on those seeds and excludes configuration/checkpoint-selection uncertainty; it is not a significance test over training seeds. No no-VN seed-43/44 controls are implied.

## Storage

`reports/training_storage_cleanup.json` records the completed cleanup: 16 obsolete optimizer/RNG resume checkpoints from completed older studies removed, and four byte-identical immutable evaluation copies replaced by NTFS hardlinks. Every selected best checkpoint was retained and verified. This recovered about 1.59 GiB. The original FULL_CLEAN and validation snapshots, active VisualNews files, RGB arrays and vision-prefix cache were retained. These caches are still needed for the running study and its reproducibility.

After all eight new adaptation fits completed and all ten selected checkpoints passed independent metric reproduction, `cleanup_completed_visualnews.ps1 -Apply` removed the eight new optimizer/RNG resume states (0.31 GiB). Selected weights, histories and source/data identities remain. `compress_training_caches.ps1 -Apply` then used transparent NTFS compression on the three large caches: RGB compression recovered another 0.432 GiB; the prefix cache did not compress. SHA256 before/after compression matched for all three files. No paths or logical file contents changed.

The final combined recovery is **2,503,110,254 bytes = 2.331 GiB**. The audit `reports/training_storage_summary.json` records 24 removed resume states, four hardlinked duplicate files, the compression audit, and 4,502,220,800 free bytes on D at 2026-10-21 18:20:54 UTC. Free space can change as the separate live pipeline continues. Ordinary directory size totals count each hardlink path and report logical rather than compressed cache sizes, so they overstate physical disk use.
