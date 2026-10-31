from __future__ import annotations

import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from ..config import Settings
from ..models import ImageRef, TranslationUnit
from .base import DatasetAdapter

MMEB_PATH = "TIGER-Lab/MMEB-eval"
# This revision contains per-dataset image ZIPs under images_zip/, avoiding a 7+ GB all-images archive.
MMEB_REVISION = "eb3a8af230a66be5c9308cd2be5779b9d6de4cb3"


def _has_text(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_has_text(v) for v in value)
    if isinstance(value, dict):
        return any(_has_text(v) for v in value.values())
    return False


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    return value if isinstance(value, list) else [value]


class MMEBAdapter(DatasetAdapter):
    def __init__(self, config_name: str, prompt_profile: str = "faithful"):
        self.config_name = config_name
        self.prompt_profile = prompt_profile

    @staticmethod
    def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
        dest_resolved = dest.resolve()
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if dest_resolved not in target.parents and target != dest_resolved:
                raise RuntimeError(f"Unsafe path in ZIP: {member.filename}")
        zf.extractall(dest)

    def _prepare_images(self, settings: Settings) -> str:
        image_root = settings.hf_cache_dir / "pl_mmb_images" / "mmeb" / self.config_name
        marker = image_root / ".extracted"
        if marker.exists():
            return str(image_root)

        image_root.mkdir(parents=True, exist_ok=True)
        filename = f"images_zip/{self.config_name}.zip"
        zip_path = hf_hub_download(
            repo_id=MMEB_PATH,
            filename=filename,
            repo_type="dataset",
            revision=MMEB_REVISION,
            token=settings.hf_token,
            cache_dir=str(settings.hf_cache_dir),
        )
        with zipfile.ZipFile(zip_path) as zf:
            self._safe_extract(zf, image_root)
        marker.write_text("ok\n", encoding="utf-8")
        return str(image_root)

    def iter_units(self, settings: Settings, with_image: bool) -> Iterator[TranslationUnit]:
        image_root = self._prepare_images(settings) if with_image else None
        ds = load_dataset(
            MMEB_PATH,
            self.config_name,
            split="test",
            revision=MMEB_REVISION,
            cache_dir=str(settings.hf_cache_dir),
            token=settings.hf_token,
            streaming=settings.hf_streaming,
        )

        for row_index, row in enumerate(ds):
            # One MMEB row is one benchmark example and therefore one TranslationUnit/request.
            # Keeping query + candidate texts together gives the translator the full local context
            # and avoids cross-example contamination from unrelated records.
            texts: dict[str, Any] = {}
            qry_text = row.get("qry_text")
            tgt_text = row.get("tgt_text")
            if _has_text(qry_text):
                texts["qry_text"] = qry_text
            if _has_text(tgt_text):
                texts["tgt_text"] = tgt_text
            if not texts:
                continue

            images: list[ImageRef] = []
            if with_image:
                qry_images = _as_list(row.get("qry_img_path"))
                tgt_images = _as_list(row.get("tgt_img_path"))

                # Prioritize query-side visual context, then the positive/first target image.
                # MMEB documents candidate 0 as the ground-truth target. We intentionally do not
                # attach every candidate image: images are context for translation, not data to rank.
                for pos, value in enumerate(qry_images):
                    if value:
                        images.append(
                            ImageRef(f"query_image[{pos}]", value, root=image_root)
                        )
                    if len(images) >= settings.max_images_per_record:
                        break
                if len(images) < settings.max_images_per_record:
                    for pos, value in enumerate(tgt_images):
                        if value:
                            label = (
                                "groundtruth_target_image"
                                if pos == 0
                                else f"target_image[{pos}]"
                            )
                            images.append(ImageRef(label, value, root=image_root))
                        if len(images) >= settings.max_images_per_record:
                            break

            yield TranslationUnit(
                unit_id=f"{self.config_name}:{row_index}",
                texts=texts,
                images=images,
                metadata={
                    "dataset": self.config_name,
                    "hf_path": MMEB_PATH,
                    "hf_config": self.config_name,
                    "split": "test",
                    "row_index": row_index,
                    "kind": "benchmark_example",
                },
                prompt_profile=self.prompt_profile,
            )
