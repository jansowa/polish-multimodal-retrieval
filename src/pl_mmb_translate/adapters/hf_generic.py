from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from datasets import load_dataset

from ..config import Settings
from ..models import ImageRef, TranslationUnit
from .base import DatasetAdapter


def _is_nonempty_text(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_is_nonempty_text(v) for v in value)
    if isinstance(value, dict):
        return any(_is_nonempty_text(v) for v in value.values())
    return False


def _sanitize_id(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._:-]+", "_", str(value))


def _image_values(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [v for v in value if v is not None and v != ""]
    return [value]


class HFGenericAdapter(DatasetAdapter):
    def __init__(
        self,
        *,
        dataset_name: str,
        path: str,
        split: str,
        text_fields: Sequence[str],
        image_fields: Sequence[str] = (),
        config: str | None = None,
        revision: str | None = None,
        id_field: str | None = None,
        prompt_profile: str = "faithful",
        image_root: str | Path | None = None,
        image_root_env: str | None = None,
        streaming: bool | None = None,
    ):
        self.dataset_name = dataset_name
        self.path = path
        self.config = config
        self.split = split
        self.text_fields = tuple(text_fields)
        self.image_fields = tuple(image_fields)
        self.revision = revision
        self.id_field = id_field
        self.prompt_profile = prompt_profile
        self.image_root = str(image_root) if image_root else None
        self.image_root_env = image_root_env
        self.streaming = streaming

    def prepare(self, settings: Settings, with_image: bool) -> str | None:
        root = self.image_root
        if self.image_root_env:
            import os

            value = os.getenv(self.image_root_env, "").strip()
            if value:
                root = str(Path(value).expanduser().resolve())
        return root

    def iter_units(self, settings: Settings, with_image: bool) -> Iterator[TranslationUnit]:
        root = self.prepare(settings, with_image)
        if with_image and self.image_fields and self.image_root_env and not root:
            raise RuntimeError(
                f"{self.dataset_name}: WITH_IMAGE=true requires {self.image_root_env} in .env"
            )

        kwargs: dict[str, Any] = {
            "path": self.path,
            "split": self.split,
            "cache_dir": str(settings.hf_cache_dir),
            "token": settings.hf_token,
            "streaming": settings.hf_streaming if self.streaming is None else self.streaming,
        }
        if self.config:
            kwargs["name"] = self.config
        if self.revision:
            kwargs["revision"] = self.revision
        ds = load_dataset(**kwargs)

        for index, row in enumerate(ds):
            texts = {
                field: row[field]
                for field in self.text_fields
                if field in row and _is_nonempty_text(row[field])
            }
            if not texts:
                continue
            raw_id = row.get(self.id_field) if self.id_field and self.id_field in row else index
            unit_id = f"{self.dataset_name}:{_sanitize_id(raw_id)}"
            images: list[ImageRef] = []
            if with_image:
                for field in self.image_fields:
                    if field not in row:
                        continue
                    for pos, value in enumerate(_image_values(row[field])):
                        images.append(ImageRef(label=f"{field}[{pos}]", value=value, root=root))
                        if len(images) >= settings.max_images_per_record:
                            break
                    if len(images) >= settings.max_images_per_record:
                        break
            yield TranslationUnit(
                unit_id=unit_id,
                texts=texts,
                images=images,
                metadata={
                    "dataset": self.dataset_name,
                    "hf_path": self.path,
                    "hf_config": self.config,
                    "split": self.split,
                    "row_index": index,
                },
                prompt_profile=self.prompt_profile,
            )
