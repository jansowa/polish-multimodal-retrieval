# Durable Wiki image archive

The inference pipeline now retains compact verified images in:

`C:/Users/jsowa/benchmark-translations/data-ood-20261030/A/wiki-recovery-20261031/`

- `masters1536/<original-SHA1>.jpg`: full frame after EXIF orientation, maximum side 1536, aspect ratio retained, no upscaling, JPEG quality 92, 4:4:4 (`subsampling=0`).
- `tinyclip224/<original-SHA1>.png`: lossless RGB 224×224 generated directly from the decoded original, using the existing pinned TinyCLIP processor and its historical RGB/EXIF policy. The normalized tensor is compared bit-for-bit with original-image preprocessing.
- `compact_receipts/<original-SHA1>.json`: the existing ready-receipt schema, original and derivative hashes, dimensions, verification evidence and producer metadata.

Original source SHA1, raw SHA256, canonical full-size pixel SHA256 and reference pHash are computed before creating derivatives. Source URL, source page, original dimensions, license, creator and attribution are retained. New receipts include source metadata. When reusing someone else's valid receipt, metadata is appended under `compact_receipts/source_metadata/<SHA1>/<metadata-SHA256>.json`; the valid receipt and image files are not rewritten.

`experiments/wiki_compact_archive.py` runs as one CPU-only WSL helper through `wiki_archive_client.py`, using the already-installed processor environment. It performs no network operations and uses no GPU. Downloads still use the inference pipeline's existing shared HttpClient and limiter. No request allocation or limiter settings are changed.

The helper holds a per-SHA1 `flock` from the initial lookup through publication, so competing integrated producers cannot download or publish the same original concurrently. Files use unique same-directory staging files, fsync and atomic hard-link create-if-absent commits; the ready receipt is published last. Existing different/corrupt artifacts are reported and preserved. Only the helper's own private staging files are cleaned up.

The separate running recovery downloader predates these per-SHA1 locks. To avoid racing that unmodified process, the pipeline never writes a missing/non-ready SHA1 belonging to its frozen download catalogs; it waits for and validates the other queue's receipt. Permanent failed receipts are preserved and reported for review. All other new originals are archived by the inference pipeline. The separate queue is not restarted or modified.

Before downloading an original, the helper validates an existing receipt and both file hashes/dimensions. Correct archived images are reused without downloading or regenerating. Qwen then receives its normal maximum-1280 JPEG derived from the authenticated full-frame master. Each new model attempt records `input.image_archive`, including original hashes, master/native hashes, receipt path, source-metadata path and the fact that the model image comes from `master1536`. No original stream is falsely represented as retained.

The reference behavior is `experiments/data_study_20261030/wiki_recovery_20261031/image_assets.py` and `download_compact.py`. The source/configuration files of that other queue remain unchanged. The encoder includes sufficient JPEG optimizer workspace for high-entropy 4:4:4 images without changing quality or sampling.

Validation: 12 WSL archive tests cover binary agreement with reference master/native artifacts, exact TinyCLIP tensors, EXIF/no-upscale behavior, cross-process locking, catalog ownership, immutable writes, corrupt/failed-file preservation and recovery after a partial write. 33 existing/shared inference tests passed, including archive reuse without HTTP. `archive_integration_probe.json` records a live existing-asset reuse check with no network download and unchanged receipt/image hashes and modification times. `archive_deployment_guard.json` records allocation and shared HTTP source hashes for deployment verification. `archive_verification.json` records verified new production artifacts and their corresponding model requests.
