# E0 + E1 — pilot 6000 par (5435 train)

To historyczny pilot ograniczony do dwóch shardów ARO Visual Relation.
Nie jest wynikiem pełnego E1. Ograniczenie zostało usunięte z konfiguracji
domyślnej; oryginalny config zachowano w `training/configs/e1-pilot6k.json`.

Scores: Recall@1 (%); deltas: percentage points. AVG requires all four tasks.

Data selection and hyperparameters were fixed before evaluation. Checkpoints use only grouped training-validation loss.

| Experiment | CIRR | FashionIQ | MSCOCO_t2i | RefCOCO | AVG |
|---|---:|---:|---:|---:|---:|
| E0 replay: google/siglip2-base-patch16-256 | 16.30 | 16.90 | 43.70 | 68.00 | 36.23 |
| E0 replay: BAAI/BGE-VL-base | 7.40 | 0.70 | 1.40 | 17.00 | 6.62 |
| E0 untrained seed 42 | 8.00 | 3.10 | 0.40 | 48.10 | 14.90 |
| E1 seed 42 | 5.60 | 1.30 | 12.30 | 17.60 | 9.20 |
| Δ vs E0 untrained projector (same seed) | -2.40 | -1.80 | +11.90 | -30.50 | -5.70 |
| E0 untrained seed 43 | 7.30 | 3.20 | 0.10 | 47.80 | 14.60 |
| E1 seed 43 | 5.80 | 1.00 | 13.70 | 17.00 | 9.38 |
| Δ vs E0 untrained projector (same seed) | -1.50 | -2.20 | +13.60 | -30.80 | -5.22 |
| E0 untrained seed 44 | 8.10 | 2.40 | 0.00 | 44.90 | 13.85 |
| E1 seed 44 | 5.20 | 1.20 | 13.20 | 15.80 | 8.85 |
| Δ vs E0 untrained projector (same seed) | -2.90 | -1.20 | +13.20 | -29.10 | -5.00 |
| E1 mean | 5.53 | 1.17 | 13.07 | 16.80 | 9.14 |
| E1 std across seeds | 0.31 | 0.15 | 0.71 | 0.92 | 0.27 |
| E1 mean Δ vs previous | -2.27 | -1.73 | +12.90 | -30.13 | -5.31 |

E1 trains only the image projector with positive-pair cosine regression. Composed queries use the existing raw sum as a diagnostic until E4.

Train pairs: 5435; validation: 565; excluded for leakage/unknown lineage: 173.

ARO is now training data for these checkpoints and must not be reported as an independent ARO evaluation. The eval blacklist covers IDs, bytes, RGB pixels, English/Polish captions and VG→COCO parent IDs; arbitrary transformed duplicates without recoverable source lineage are outside what exact hashes can certify.

Reproduce with `train.py` and `training/configs/e1.json`; see [training instructions](../training/README.md) and [E2–E11 plan](../training/EXPERIMENTS.md). Full local artifacts: `training/runs/e1-aro6k/`.

E0 cross-dataset eval parent groups: 0; pixel groups: 0. Post-filter train/eval overlap: 0.

## Training losses

| Seed | Train loss epoch 1 | Train loss epoch 20 | Best validation loss | Selected epoch |
|---|---:|---:|---:|---:|
| 42 | 0.37033 | 0.08945 | 0.12388 | 7 |
| 43 | 0.38338 | 0.08956 | 0.12365 | 10 |
| 44 | 0.35700 | 0.08953 | 0.12354 | 8 |

Validation loss begins rising after its minimum while training loss keeps decreasing. Checkpoints were selected using validation loss, not benchmark scores. Full epoch history is exported to `training_e0_e1_loss.csv` and the JSON report.

The generic Sentence Transformers warning about `is_causal=False` was checked against the installed ModernBERT SDPA implementation and an actual future-token perturbation test. The first-token representation changed (L2=3.90442), confirming bidirectional attention for the pinned model. No encoder or library was changed.
