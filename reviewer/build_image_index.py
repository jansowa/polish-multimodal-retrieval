"""Build the exact MMEB caption -> ground-truth image mapping (no model calls)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_dataset
from pl_mmb_translate.adapters.mmeb import MMEBAdapter, MMEB_PATH, MMEB_REVISION
from pl_mmb_translate.config import Settings
from pl_mmb_translate.image_utils import _safe_local_path


def main() -> None:
    settings = Settings.from_env(ROOT / ".env")
    adapter = MMEBAdapter("MSCOCO_t2i")
    image_root = Path(adapter._prepare_images(settings)).resolve()
    # Store paths relative to the workspace so the index works in Windows and WSL.
    image_root_relative = image_root.relative_to(ROOT).as_posix()
    ds = load_dataset(MMEB_PATH, "MSCOCO_t2i", split="test", revision=MMEB_REVISION,
                      cache_dir=str(settings.hf_cache_dir), token=settings.hf_token, streaming=True)
    records = {}
    for row_index, row in enumerate(ds):
        paths = row.get("tgt_img_path") or []
        if isinstance(paths, str):
            paths = [paths]
        if not paths or not paths[0]:
            continue
        image = _safe_local_path(str(paths[0]), str(image_root)).resolve()
        source = row.get("qry_text") or ""
        records[f"MSCOCO_t2i:{row_index}"] = {
            "path": image.relative_to(image_root).as_posix(),
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        }
        if len(records) % 100 == 0:
            print(f"Indexed {len(records)} captions", flush=True)
    result = {"dataset": "MSCOCO_t2i", "revision": MMEB_REVISION,
              "image_root": image_root_relative, "records": records}
    target = ROOT / "reviewer/image_index.json"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)
    print(f"Ready: {len(records)} captions, ground-truth candidate 0", flush=True)


if __name__ == "__main__":
    main()
