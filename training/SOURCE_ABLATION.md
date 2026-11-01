# Frozen-projector source ablation, v1

User decisions: only the projector is trained, one seed for the initial screen, no wiki data. COCO must not select checkpoints or configurations. Original FULL_CLEAN/validation and the published VisualNews snapshot remain byte-for-byte unchanged.

## Fixed recipe

Original frozen TinyCLIP-39M projected 512D features and frozen PolDense-32M text features, Linear 512→384. Fresh random projector (seed 42) and fresh sigmoid scale/bias for every condition. No A1 or any other previously trained projector/loss state is loaded. Same 5,500 updates and 1,405,780 example presentations per condition, batch 256 with a batch of 145 every 275th step. AdamW LR .001, weight decay .01 for the projector and zero for loss scalars. No scheduler or augmentation, as in the original linear probe. No explicit false-caption negatives beyond the existing multi-positive sigmoid loss.

Checkpoint selection: VN dev t2i R@1, mean t2i/i2t R@1 breaks ties, then earliest update. Monitor every 275 updates. No COCO metrics are computed during training or used for any followup choice. The previously frozen 2,000-image VN dev remains unchanged; its model-generated captions are a limitation.

## Fourteen conditions

- VN only.
- VN + verified non-COCO ARO Attribution, Relation, or both (three conditions).
- VN + Attribution, Relation, SugarCrepe, or M-BEIR COCO separately (four conditions).
- VN + FULL_CLEAN, and four leave-one-source-out versions (five conditions).
- FULL_CLEAN only, as a domain-exposed reference.

Single-source additions draw 50% VN / 50% added-source rows. VN is sampled uniformly over canonical images, then one available caption. Added-source rows retain original row weighting. Full mix draws 50% VN / 50% FULL_CLEAN rows. In leave-one-out conditions, replace each removed source slot by VN, leaving every retained original-source draw unchanged. This estimates replacement by VN at fixed compute, rather than renormalizing all remaining sources. VN only uses all slots for VN. FULL only uses all slots for the original FULL_CLEAN.

These source additions/omissions can change image diversity, caption style, positive multiplicity and batch-negative composition together. Results describe these available pools and the stated sampling proportions, not universal dataset quality or intrinsic per-example value.

## COCO ancestry

Recheck the pinned official 108,077-row Visual Genome metadata. Build transitive VG/Flickr/COCO components, also joining known file/pixel aliases across original training, validation and protected benchmark images. Strict ARO must have an explicit official VG entry, no COCO ID, and no transitive link to a known COCO source. Unknown VG ancestry is excluded. Any known COCO alias in VN train/dev aborts preparation.

All SugarCrepe and M-BEIR COCO are excluded from the non-COCO branch. Full ARO variants may contain COCO-derived images and are clearly labeled as domain-exposed controls. This tests absence of known COCO-derived supervised examples in our training. It cannot establish absence of unidentified copies or COCO exposure in the original pretrained encoders.

## Evaluation freeze

Close every training run and write `selection_registry.json` with checkpoint SHA and selected updates before opening COCO evaluation. Fixed benchmark finalists: VN only, VN + verified non-COCO ARO both, VN + FULL_CLEAN as a domain-exposed control, plus the best strict condition chosen solely by VN dev if distinct. This caps the final benchmark at four selected conditions; no test-driven configuration changes follow. COCO validation may be reported post hoc for all 14 conditions, without feeding it back into training or selection. Report the existing local MMEB COCO protocol separately from whole-gallery retrieval; neither is the official COCO 5k benchmark.

No automatic three-seed grid. Small effects remain inconclusive in this first screen; any seed followup must be decided without COCO test scores. XM3600 remains blind. Wiki is not loaded.

## Commands

```text
PYTHONPATH=src python -m unittest discover -s tests -p test_source_ablation.py -v
python -u training/source_ablation.py
python -u training/verify_source_ablation.py
python training/summarize_source_ablation.py
```

Native feature/text caches and all frozen image assets are reused. No RGB/prefix cache is duplicated. Only small head checkpoints, samplers, histories and data-index manifests are written.
