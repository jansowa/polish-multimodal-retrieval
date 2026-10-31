# TinyCLIP: learning rate, schedule and head initialization

This followup tests whether the earlier negative partial-finetuning result was
specific to a constant LR and training the head from random initialization.
It does not change the original A–F results or their selected checkpoints.

All experiments use the existing FULL_CLEAN (70,289 train rows), original
7,695-row validation snapshot and identical validation retrieval graph.
TinyCLIP-39M native projected features, linear image projector to 384D,
frozen PolDense-32M and existing multi-positive sigmoid loss remain fixed.
There is no KD, test evaluation, XM3600 run, new data or data filtering.

The precommitted seed-42 experiments are:

| Run | Initialization | Trainable vision | Projector LR | Vision LR | Schedule |
|---|---|---|---:|---:|---|
| S0_frozen_schedule | Random head | None | 1e-3 | 0 | 2 epochs warmup + cosine |
| S1_last1_low_constant | Random head | Last block | 1e-3 | 3e-6 | Constant |
| S2_last1_schedule | Random head | Last block | 1e-3 | 1e-5 | 2 epochs warmup + cosine |
| S3_last1_low_schedule | Random head | Last block | 1e-3 | 3e-6 | 2 epochs warmup + cosine |
| S4_last2_low_schedule | Random head | Last 2 blocks | 1e-3 | 3e-6 | 2 epochs warmup + cosine |
| T0_frozen_aligned | A1 epoch 9 | None | 1e-4 | 0 | 1 epoch warmup + cosine |
| T1_last1_aligned | A1 epoch 9 | Last block | 1e-4 | 1e-6 | 1 epoch warmup + cosine |
| T2_last2_aligned | A1 epoch 9 | Last 2 blocks | 1e-4 | 1e-6 | 1 epoch warmup + cosine |

The T runs load exactly the same trained projector and learned loss scalars;
the native vision tower is initially unchanged. Every T run starts a fresh
AdamW optimizer. T0 controls for additional head training and the changed LR.
This probes the linear-probe-then-finetune idea described by
[Kumar et al.](https://arxiv.org/abs/2202.10054); it is a hypothesis for this
retrieval setting, not a claim that the paper predicts the outcome here.

S1 vs old C1 isolates the smaller constant vision LR (shared prefix of the
20-epoch run budget; stopping differs). S2 vs old C1 tests a schedule bundle,
warmup plus cosine; it cannot attribute an improvement to warmup alone.
S2/S3/S4 vs S0 measure adaptation with matching head/loss optimization.
T1/T2 vs T0 measure adaptation from an already aligned head.

AdamW, batch 256, weight decay 0.01 on projector/vision and zero on loss
scalars remain unchanged. All three groups follow the same schedule multiplier.
No gradient clipping or additional regularizer is introduced. The native
post-LayerNorm and CLIP projection stay frozen. Warmup increases the LR every
optimizer update from peak/warmup_steps to peak, then cosine decreases it to
10% of peak at the final planned update. Unit tests cover the first actual
optimizer update, both schedule boundaries and matched parameter-group ratios.

S runs have a 20-epoch ceiling, T runs have 12 additional epochs after the
original A1 9 epochs. Early stopping uses the unchanged val t2i R@1, then mean
t2i/i2t R@1, then earliest epoch; patience is 6, with at least 10 S epochs or
8 T epochs. Epoch zero is eligible, so adaptation cannot silently discard the
starting checkpoint. The T and S budgets/initialization differ intentionally;
use their matched controls when interpreting gains.

The verified original FP32 frozen-prefix cache is reused. Adapted blocks run
with FP32 weights/BF16 autocast; validation features are recomputed each epoch.
Besides the requested loss/retrieval/VRAM/parameter logs, diagnostics include:

- Actual first/last/min/max LR for each parameter group each epoch.
- Mean and maximum gradient norm per parameter group, before the optimizer.
- Actual AdamW step norm divided by parameter norm, sampled every 25 updates.
- Parameter drift relative to initial group weights and validation feature
  cosine drift relative to native TinyCLIP features.

`last.pt` stores epoch-boundary optimizer/model/RNG state for resumption;
`best.pt` stores the checkpoint selected solely by validation. Inputs, code,
cache and initialization checkpoints are protected by SHA256 provenance.

```bash
PY=~/.venvs/pl-mmb-benchmark/bin/python
$PY -u training/tinyclip_optimization.py all
$PY training/tinyclip_optimization.py verify
$PY -u training/tinyclip_optimization_midpoint.py
$PY training/tinyclip_optimization.py report
$PY training/plot_tinyclip_optimization.py
$PY training/analyze_tinyclip_optimization.py
$PY training/summarize_tinyclip_optimization.py
```

`scout` and `aligned` can run the two groups separately. Output is
`training/runs/tinyclip-optimization/`; compact results and full epoch logs are
`reports/tinyclip_optimization.*`. This study deliberately has no test command.

Conditional followup: if both aligned-head 1e-6 runs improve by less than
0.5 p.p. over their matched frozen control, `tinyclip_optimization_midpoint.py`
tests only the 3e-6 midpoint for one/two blocks, keeping all other T settings.
The triggering decision and supplemental provenance verification are saved.
The original eight-run selection is preserved separately before updating the
validation-only selection. This is an adaptive followup, not a prespecified
independent confirmation.
