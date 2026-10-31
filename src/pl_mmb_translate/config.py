from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _slug(value: str) -> str:
    value = value.strip().lower().replace("/", "-")
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    return value.strip("-") or "run"


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    model: str
    batch_size: int
    request_timeout: float
    max_retries: int
    max_output_tokens: int
    temperature: float
    output_dir: Path
    hf_cache_dir: Path
    hf_token: str | None
    hf_streaming: bool
    image_max_side: int
    image_jpeg_quality: int
    max_images_per_record: int
    max_images_per_request: int
    mbeir_image_root: Path | None
    run_name: str
    quality_review: bool = False
    save_drafts: bool = False

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> "Settings":
        load_dotenv(env_file, override=False)
        model = os.getenv("OPENAI_MODEL", "").strip()
        if not model:
            raise ValueError("OPENAI_MODEL is required in .env")

        batch_size = int(os.getenv("BATCH_SIZE", "8"))
        if batch_size < 1:
            raise ValueError("BATCH_SIZE must be >= 1")

        image_quality = int(os.getenv("IMAGE_JPEG_QUALITY", "88"))
        if not 1 <= image_quality <= 100:
            raise ValueError("IMAGE_JPEG_QUALITY must be in 1..100")

        mbeir_root = os.getenv("MBEIR_IMAGE_ROOT", "").strip()
        run_name = os.getenv("RUN_NAME", "").strip() or _slug(model)

        return cls(
            base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/"),
            api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
            model=model,
            batch_size=batch_size,
            request_timeout=float(os.getenv("REQUEST_TIMEOUT", "180")),
            max_retries=int(os.getenv("MAX_RETRIES", "4")),
            max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "8192")),
            temperature=float(os.getenv("TEMPERATURE", "0")),
            output_dir=Path(os.getenv("OUTPUT_DIR", "./outputs")).expanduser().resolve(),
            hf_cache_dir=Path(os.getenv("HF_CACHE_DIR", "./.cache/huggingface")).expanduser().resolve(),
            hf_token=os.getenv("HF_TOKEN") or None,
            hf_streaming=_env_bool("HF_STREAMING", True),
            image_max_side=int(os.getenv("IMAGE_MAX_SIDE", "1280")),
            image_jpeg_quality=image_quality,
            max_images_per_record=int(os.getenv("MAX_IMAGES_PER_RECORD", "2")),
            max_images_per_request=int(os.getenv("MAX_IMAGES_PER_REQUEST", "4")),
            mbeir_image_root=Path(mbeir_root).expanduser().resolve() if mbeir_root else None,
            run_name=_slug(run_name),
            quality_review=_env_bool("QUALITY_REVIEW", False),
            save_drafts=_env_bool("SAVE_DRAFTS", False),
        )
