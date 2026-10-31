from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

JsonText = str | list["JsonText"] | dict[str, "JsonText"]


@dataclass
class ImageRef:
    label: str
    value: Any
    root: str | None = None


@dataclass
class TranslationUnit:
    unit_id: str
    texts: dict[str, JsonText]
    images: list[ImageRef] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    prompt_profile: str = "faithful"
