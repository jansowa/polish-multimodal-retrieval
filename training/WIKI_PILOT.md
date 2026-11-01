# Wiki addition pilot

Run `training/fetch_wiki_pilot_images.py` using Windows Python, which shares the
producer's Windows HTTP reservation lock and cooldown state. The script freezes
the published wiki JSONL once. It never opens the producer SQLite database.
Then run `PYTHONPATH=src python training/run_wiki_pilot.py` in the WSL ML environment.
The latter waits for verified image acquisition, audits and encodes data, trains
three configurations, and reproduces checkpoint metrics before exporting results.

The study uses the existing frozen VN train/dev snapshot and a new, separately
versioned wiki addendum under `training/runs/tinyclip-wiki-pilot-v1`. FULL_CLEAN,
COCO validation/test and XM3600 retrieval are not training or selection inputs.
Protected-image identity checks do not compute benchmark retrieval scores.

Wiki train/dev components join actual associated Wikidata entity IDs (not generic
depicts classes or collection seed entities), original source SHA1, decoded pixels,
normalized captions and pHash distance <=6. Exclusions propagate through complete
components. Dev targets 200 images, allows 150–250, requires >=15 components, and
keeps components over 100 images in train. These rules are fixed before fitting.
Official Wikimedia renders at 1280 px are accepted only after API confirmation
of the original source SHA1 against the frozen export; downloaded/render identities
are recorded separately from original identities. The first original downloads
are retained as such. Native CLIP input preprocessing is checked against the
pinned Hugging Face processor on retained original/render samples.

All fits start from the same random Linear(512,384) and fresh sigmoid scalars;
TinyCLIP-39M and PolDense-32M are frozen. No trained COCO head is loaded. Seed42,
AdamW LR0.001, projector decay0.01, scalar decay0, no scheduler or augmentation.
Each run has 5500 updates: 20 windows of 274 batches of256 and one batch of145,
exactly 1,405,780 example presentations. Evaluate every275 updates and at zero.
Select only by VN-dev t2i R@1, then bidirectional mean R@1, then earliest update.

Wiki replaces 0%, 3%, or 10% of baseline VN slots at fixed compute. The exact
cumulative quotas are 0, 42,173 and 140,578 wiki presentations. Remaining VN slots
are identical, and 3% replacements are nested in the 10% replacements. A separate
image-uniform wiki sampler uses a shuffled bag and samples one caption per image.
The zero-wiki sampler reproduces the prior VN-only source-study sampler.
Multi-positive labels use only training rows eligible in that configuration.

Wiki dev and a deterministic 200-image wiki train probe are diagnostics only;
they never select checkpoints. The final step5500 is also retained as a fixed
budget diagnostic. Reports include selected and final checkpoints, R@1/5/10,
i2t R@1, losses, gradients, peak VRAM, trainable parameter counts, source exposure,
and conditional paired component-bootstrap intervals. These intervals exclude
training-seed and checkpoint-selection uncertainty.

The prespecified exploratory recommendation is the lowest wiki share with a
positive wiki-dev point gain and at most0.5pp VN-dev degradation, otherwise0%.
This is a one-seed screen. It cannot establish large-corpus scaling or separate
the effects of wiki content, caption style and changed batch negatives. Once its
scores are viewed, this wiki dev must not be described as blind in future work.
