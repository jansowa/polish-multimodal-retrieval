# CLEAN-v1: leakage guard audit and COCO exposure ablation

Historical results remain LEGACY / pre-audit. No historical checkpoint or score was changed.

Known retained source/parent/hash overlaps: **0**.

## Existing guard exclusions (train + validation preparation)

| Source | Rejected | Parent | Content hash | Text | Unknown lineage | Retained |
|---|---:|---:|---:|---:|---:|---:|
| aro_visual_relation | 711 | 711 | 0 | 0 | 0 | 23226 |
| aro_visual_attribution | 699 | 699 | 0 | 0 | 0 | 28049 |
| sugarcrepe | 764 | 762 | 0 | 143 | 0 | 6747 |
| mbeir_mscoco | 5027 | 5010 | 0 | 1024 | 0 | 19962 |

Counts are non-exclusive across reasons and benchmarks. Retained counts repeat per benchmark. Unknown lineage cannot be assigned to a benchmark.

See clean_v1_leakage.csv for attribution to each protected benchmark. Reasons can overlap.

## Natural-size data ablation, seed 42

| Variant | Records | Unique pairs | Unique images | COCO R@1 | R@5 | R@10 | Val t2i | Val i2t | CIRR | FashionIQ | RefCOCO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FULL_CLEAN | 70289 | 67445 | 10645 | 45.20 | 79.10 | 88.30 | 58.23 | 69.69 | 9.40 | 3.30 | 48.90 |
| NO_MBEIR_MSCOCO | 52420 | 49601 | 7102 | 35.20 | 67.40 | 80.70 | 49.09 | 56.09 | 10.70 | 4.90 | 46.80 |
| NON_COCO_CLEANROOM | 24473 | 23601 | 3037 | 22.30 | 49.10 | 66.00 | 35.79 | 44.63 | 11.00 | 3.90 | 45.30 |

Metrics are percentages. Same native TinyCLIP features, frozen PolDense, linear projector, sigmoid loss, batch 256, AdamW and 20 epochs.
Validation gallery is fixed across variants; checkpoints use val t2i R@1, mean t2i/i2t R@1 breaks ties. Test is read only after selection.

## Near duplicates

Exhaustive pHash comparisons: 202886780; pairs at distance <= 6: 0.
Candidates are a review queue, not confirmed leakage. No perceptual filtering was applied.
pHash can miss crops and significant transformations. TinyCLIP cosine annotates pHash candidates; no exhaustive embedding search is performed.

## Interpretation

Removing M-BEIR changes MSCOCO R@1 by -10.00 pp; removing all known COCO lineage changes it by -22.90 pp.
These changes measure data/domain dependence together with reduced training size and budget. They do not establish instance leakage.
A matched-steps/examples control is needed to isolate domain exposure from the smaller dataset if these gaps are substantial.

Keep MSCOCO as IN-DOMAIN / LEGACY. Add independent XM3600-PL; do not treat composition/grounding as substitutes for independent t2i.

## Limits

- Single seed, natural dataset sizes and unequal examples seen.
- CLEAN-v1 uses validation t2i checkpoint selection; historical runs used minimum validation loss.
- Near-duplicate candidates are not automatically filtered or confirmed.
- Excludes known COCO lineage and pixel aliases of known COCO images. Missing VG-to-COCO links are not a proof of absence of all possible COCO ancestry.

M-BEIR source metadata: query/test + task3 test candidate pool in frozen original loader; internal train/validation split is separate

Instance leakage, COCO domain exposure, and repeated test inspection are separate issues.
