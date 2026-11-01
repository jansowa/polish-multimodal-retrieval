# E0 + E1

Primary task: Polish text query → image retrieval. MSCOCO_t2i is the primary metric; it uses the pinned local MMEB candidate pools, not a global gallery. CIRR/FashionIQ/RefCOCO and AVG are diagnostic until E4.

Scores: Recall@1 (%); deltas: percentage points. AVG requires all four tasks.

Data selection and hyperparameters were fixed before evaluation. Checkpoints use only grouped training-validation loss.

| Experiment | CIRR | FashionIQ | MSCOCO_t2i | RefCOCO | AVG |
|---|---:|---:|---:|---:|---:|
| E0 replay: google/siglip2-base-patch16-256 | 16.30 | 16.90 | 43.70 | 68.00 | 36.23 |
| E0 replay: BAAI/BGE-VL-base | 7.40 | 0.70 | 1.40 | 17.00 | 6.62 |
| E0 untrained seed 42 | 8.00 | 3.10 | 0.40 | 48.10 | 14.90 |
| E1 seed 42 | 8.00 | 4.70 | 41.50 | 35.80 | 22.50 |
| Δ vs training/runs/e1-aro6k/summary.json | +2.47 | +3.53 | +28.43 | +19.00 | +13.36 |
| E0 untrained seed 43 | 7.30 | 3.20 | 0.10 | 47.80 | 14.60 |
| E1 seed 43 | 8.70 | 4.40 | 41.90 | 36.10 | 22.77 |
| Δ vs training/runs/e1-aro6k/summary.json | +3.17 | +3.23 | +28.83 | +19.30 | +13.63 |
| E0 untrained seed 44 | 8.10 | 2.40 | 0.00 | 44.90 | 13.85 |
| E1 seed 44 | 8.40 | 4.50 | 41.70 | 35.20 | 22.45 |
| Δ vs training/runs/e1-aro6k/summary.json | +2.87 | +3.33 | +28.63 | +18.40 | +13.31 |
| E1 mean | 8.37 | 4.53 | 41.70 | 35.70 | 22.57 |
| E1 std across seeds | 0.35 | 0.15 | 0.20 | 0.46 | 0.18 |
| E1 mean Δ vs previous | +2.83 | +3.37 | +28.63 | +18.90 | +13.43 |

E1 trains only the image projector with positive-pair cosine regression. Composed queries use the existing raw sum as a diagnostic until E4.

Train pairs: 70289; validation: 7695; excluded during data preparation: 17286. See the per-source audit for leakage, missing images and missing positive mappings.

ARO is now training data for these checkpoints and must not be reported as an independent ARO evaluation. The eval blacklist covers IDs, bytes, RGB pixels, English/Polish captions and VG→COCO parent IDs; arbitrary transformed duplicates without recoverable source lineage are outside what exact hashes can certify.

Reproduce with `train.py` and `training/configs/e1.json`; see [training instructions](../training/README.md) and [E2–E11 plan](../training/EXPERIMENTS.md). Full local artifacts: `/mnt/d/work/translate-vision-embedder-benchmark/pl-multimodal-benchmark-translator/pl-multimodal-benchmark-translator/training/runs/e1-full`.

E0 cross-dataset eval parent groups: 0; pixel groups: 0. Post-filter train/eval overlap: 0.

## Training losses

| Seed | Validation before training | Validation epoch 1 | Train loss epoch 1 | Train loss final epoch | Best validation loss | Selected epoch |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0.95714 | 0.13165 | 0.16125 | 0.12016 | 0.12923 | 17 |
| 43 | 0.99189 | 0.13172 | 0.16235 | 0.12015 | 0.12917 | 19 |
| 44 | 0.95401 | 0.13202 | 0.15999 | 0.12014 | 0.12918 | 19 |

Checkpoints were selected using validation loss, not benchmark scores. Full epoch history is exported to `training_e0_e1_full_loss.csv` and the JSON report.

Translated source records: 95088; retained pairs: 77984; distinct target images by decoded pixels: 11861; pair limit: None.

## Source coverage

| Source | Translated records | Usable before selection | Excluded |
|---|---:|---:|---:|
| aro_visual_relation | 23937 | 23226 | 711 |
| aro_visual_attribution | 28748 | 28049 | 699 |
| sugarcrepe | 7511 | 6747 | 764 |
| mbeir_mscoco | 24809 | 19962 | 5027 |
| mbeir_visualnews | 10083 | 0 | 10085 |

Exclusion reasons: `{"eval_parent_image": 7182, "eval_text": 1167, "missing_image_asset": 1451, "no_verified_positive_image_mapping": 8634}`.

A translated record can map to several positive images; usable/excluded counts refer to pairs (or records without any verified mapping). Multiple exclusion reasons can apply to one entry.

The generic Sentence Transformers warning about `is_causal=False` was checked against the installed ModernBERT SDPA implementation and an actual future-token perturbation test. The first-token representation changed (L2=3.90442), confirming bidirectional attention for the pinned model. No encoder or library was changed.
