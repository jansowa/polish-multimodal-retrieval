# TinyCLIP adaptation — FULL_CLEAN, seed 42

Fixed 70,289 train rows / 7,695 validation rows; unchanged snapshot and validation retrieval gallery.
Selection: validation t2i R@1, then mean t2i/i2t R@1, then earliest epoch. XM3600 remains blind.

| experiment | feature source | projector | vision trainable blocks | trainable params | val t2i R@1 | val i2t R@1 | best epoch | test MSCOCO |
|---|---|---|---:|---:|---:|---:|---:|---|
| A1 | projected | linear | 0 | 196,994 | 58.23% | 69.69% | 9 | — |
| A2 | pooled | linear | 0 | 196,994 | 57.85% | 63.01% | 2 | — |
| B1 | projected | linear | 0 | 196,994 | 58.23% | 69.69% | 9 | — |
| B2 | projected | MLP 768 GELU | 0 | 689,282 | 54.31% | 63.72% | 3 | — |
| C0 | projected | linear | 0 | 196,994 | 58.23% | 69.69% | 9 | — |
| C1 | projected | linear | 1 | 3,349,378 | 57.56% | 68.74% | 4 | — |
| C2 | projected | linear | 2 | 6,501,762 | 56.46% | 65.63% | 2 | — |
| D_LoRA | projected | linear | 2 (LoRA) | 229,762 | 56.41% | 68.50% | 9 | — |
| E_KD_0.1 | projected | linear | 0 | 196,994 | 58.37% | 69.93% | 9 | — |
| E_KD_0.3 | projected | linear | 0 | 196,994 | 58.76% | 70.41% | 9 | 44.50% |
| F_TinyCLIP61 | projected | linear | 0 | 196,994 | 51.34% | 66.35% | 10 | — |

B1 and C0 reuse the corresponding completed run exactly. Trainable params include sigmoid scale/bias.
A2 is native pooled CLS after final native LayerNorm, before CLIP visual_projection.
Projector LR 0.001; C vision LR 1e-5. D uses rank-8, alpha-16 q/v LoRA in the final two blocks, LR 1e-4.
E uses frozen SigLIP2 Base/16 256 image/EN-caption similarity distributions, symmetric KL at temperature 0.07;
student uses image/PL-caption similarities. No raw embedding regression. Lambda 0.1 and 0.3 only.
F uses the same best frozen projector recipe, with native projected TinyCLIP-61M image features.
Peak VRAM is peak PyTorch allocated CUDA memory; full losses, retrieval and gradients are in the CSV/JSON logs.

Finalist selected before test: **E_KD_0.3**.
Local MSCOCO t2i R@1: **44.50%**. Historical FULL_CLEAN frozen reference: 45.20%.
This is the existing local MSCOCO protocol, not independent cross-dataset validation.
All configurations have one seed. The small KD gain needs replication before a strong conclusion.
