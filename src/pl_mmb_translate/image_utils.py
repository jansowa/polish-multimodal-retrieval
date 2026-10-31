from __future__ import annotations

import base64
import io
import mimetypes
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

from .config import Settings
from .models import ImageRef


def _safe_local_path(value: str, root: str | None) -> Path:
    p = Path(value)
    if p.is_file():
        return p
    if root:
        root_path = Path(root)
        candidate = root_path / value
        if candidate.is_file():
            return candidate
        candidate = root_path / Path(value).name
        if candidate.is_file():
            return candidate
        # Last-resort basename lookup. This is intentionally lazy and only used if needed.
        matches = list(root_path.rglob(Path(value).name))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise FileNotFoundError(f"Ambiguous image basename {Path(value).name!r} under {root_path}")
    raise FileNotFoundError(f"Cannot resolve image path: {value!r}; root={root!r}")


def _open_image(value: Any, root: str | None) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.copy()

    if isinstance(value, dict):
        raw = value.get("bytes")
        path = value.get("path")
        if raw:
            return Image.open(io.BytesIO(raw))
        if path:
            return Image.open(_safe_local_path(str(path), root))

    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(value)))

    if isinstance(value, str):
        if value.startswith("http://") or value.startswith("https://"):
            with urllib.request.urlopen(value, timeout=60) as response:  # noqa: S310 - dataset-controlled URL
                return Image.open(io.BytesIO(response.read()))
        return Image.open(_safe_local_path(value, root))

    raise TypeError(f"Unsupported image value type: {type(value)!r}")


def image_to_data_uri(ref: ImageRef, settings: Settings) -> str:
    image = _open_image(ref.value, ref.root)
    try:
        image.load()
        image.thumbnail((settings.image_max_side, settings.image_max_side), Image.Resampling.LANCZOS)
        if image.mode not in {"RGB", "L"}:
            background = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode in {"RGBA", "LA"}:
                alpha = image.getchannel("A")
                background.paste(image.convert("RGB"), mask=alpha)
                image = background
            else:
                image = image.convert("RGB")
        elif image.mode == "L":
            image = image.convert("RGB")

        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=settings.image_jpeg_quality, optimize=True)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    finally:
        image.close()
