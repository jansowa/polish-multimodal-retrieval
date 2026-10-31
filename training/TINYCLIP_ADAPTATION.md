# TinyCLIP adaptation on FULL_CLEAN

This study reads the existing `clean-v1/FULL_CLEAN.jsonl` and
`clean-v1/validation.jsonl`. Both must match the original E1 snapshot exactly.
It verifies the existing audit input checksums and keeps the same MSCOCO
validation retrieval graph. It never loads MSCOCO test or XM3600.

Run in the existing WSL environment:

```bash
~/.venvs/pl-mmb-benchmark/bin/python -u training/tinyclip_adaptation.py all
```

Stages `cache`, `A`, `B`, `C`, and `report` can also run separately. Output is
`training/runs/tinyclip-adaptation/`; the small report is
`reports/tinyclip_adaptation.md` and JSON. Each completed experiment is reusable
only with identical source, inputs, cache checksums and checkpoint checksum.

- A1: native final TinyCLIP projected image embedding; linear to 384.
- A2: native post-layernorm pooled CLS before original CLIP projection; linear to 384.
- B1: exact reuse of the winning A linear run.
- B2: 768 hidden units, GELU, linear to 384; no additional normalization/residual.
- C0: exact reuse of the winning frozen A/B run.
- C1/C2: only the last one/two transformer blocks train; final native LayerNorm
  and original projection stay frozen. Projector trains from the same seeded
  initialization as C0; this isolates the effect of trainable vision blocks.

Seed 42, batch 256, 20 epoch ceiling, AdamW, projector LR 0.001, vision LR 1e-5,
weight decay 0.01, unchanged multi-positive sigmoid loss and learned scale/bias.
Checkpoint selection uses validation t2i R@1, then mean bidirectional R@1, then
earliest epoch. C uses early stopping after five epochs without improvement by
this rule. A/B use the original full 20 epoch budget.

The cache contains train/validation images only. Vision extraction uses the
established FP32 weights with BF16 autocast. Native A1 features are checked
against the existing TinyCLIP cache (minimum cosine >= 0.999). C caches the
unchanged prefix input to the final two blocks as FP32 tokens, then recomputes
the suffix with autograd during training and fresh validation features each
epoch. This avoids a stale cache of an adapted encoder.

Every epoch logs train/validation loss, val t2i and i2t R@1/5/10, mean/max global
gradient norm (before any clipping; no clipping is applied), elapsed time,
peak allocated CUDA memory and trainable parameters. Trainable parameter count
includes the learned sigmoid scale and bias. Checkpoints include the projector,
adapted suffix and loss state. Base model revisions and input hashes are in
`inputs.json`. Peak VRAM is PyTorch allocated memory, not total device usage.

D is conditional on C overfitting or failing to help. E and F follow A–D in that
order if GPU time remains before VisualNews is ready. Neither is part of the
`all` command above. XM3600 requires a separate user decision.

```bash
~/.venvs/pl-mmb-benchmark/bin/python -u training/tinyclip_followups.py all
~/.venvs/pl-mmb-benchmark/bin/python -u training/tinyclip_finalists.py
~/.venvs/pl-mmb-benchmark/bin/python training/export_tinyclip_adaptation.py
~/.venvs/pl-mmb-benchmark/bin/python training/verify_tinyclip_adaptation.py
```

Followups also accept `D`, `E`, or `F` separately. D runs only after a >=1 p.p.
best-to-final validation t2i decline in C, or failure to improve over C0.
It uses q/v LoRA in the final two blocks, rank 8, alpha 16, LR 1e-4.
E caches native SigLIP2 English text features and reuses verified frozen image
features. KD is symmetric KL between teacher and student cosine similarity
distributions at temperature 0.07, added to gold sigmoid loss with lambda 0.1
or 0.3. Teacher embeddings are never regression targets. F pins its model
revision and repeats the best frozen A/B recipe once.

The finalist script requires A–F decisions, commits the overall validation
winner, and only then evaluates that checkpoint on local MSCOCO. Historical
FULL_CLEAN is retained as the reference. The exporter records B1/C0 reuse
explicitly and writes the comparison and epoch diagnostics as CSV. All runs
are single-seed scouts; small gains are not evidence of a robust improvement.
