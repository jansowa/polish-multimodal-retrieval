# CLEAN-v1: matched steps and examples

Conditional control after large natural-size ablation gaps. Same model, loss, optimizer, batch-size schedule and fixed validation.
FULL_CLEAN is reused from the first ablation; smaller datasets are repeatedly shuffled.

| Variant | Records | Unique images | Steps | Examples seen | MSCOCO R@1 | Val t2i R@1 | Val i2t R@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FULL_CLEAN | 70289 | 10645 | 5500 | 1405780 | 45.20 | 58.23 | 69.69 |
| NO_MBEIR_MSCOCO | 52420 | 7102 | 5500 | 1405780 | 32.40 | 47.51 | 55.37 |
| NON_COCO_CLEANROOM | 24473 | 3037 | 5500 | 1405780 | 22.00 | 34.11 | 44.87 |

MSCOCO deltas vs FULL_CLEAN: NO_MBEIR -12.80 pp; NON_COCO -23.20 pp.
The gap persists under the same available optimization budget. Distinct image count and domain/source diversity still differ.
Reported steps/examples are completed training caps. Checkpoints are selected by validation and can consume different prefixes of those caps; CSV/JSON record selected_checkpoint_steps/examples separately.
These are domain/data-dependence results, not evidence of instance leakage. One seed; no uncertainty estimate.

Recommendation: retain MSCOCO as in-domain/legacy and add XM3600-PL for independent cross-dataset t2i.
