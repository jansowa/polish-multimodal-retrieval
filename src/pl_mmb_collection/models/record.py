from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Entity:
    qid: str | None
    seed: str
    depth: int = 0
    label_pl: str | None = None
    label_en: str | None = None
    description_pl: str | None = None
    description_en: str | None = None
    categories: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    revision: int | None = None
    raw_request: str | None = None


@dataclass(frozen=True)
class ImageCandidate:
    title: str
    method: str
    category: str | None = None
    category_depth: int | None = None
    raw_requests: tuple[str, ...] = ()
    # Optional source-provided context. Adapters must never invent this text.
    context_description: dict[str, str] = field(default_factory=dict)
    context_source: str = "category_description"


@dataclass
class DatasetRecord:
    id: str
    source: str
    source_page_url: str
    image_url: str
    image_original_url: str
    image_filename: str
    wikidata_qid: str | None = None
    entity_label_pl: str | None = None
    entity_label_en: str | None = None
    entity_description_pl: str | None = None
    entity_description_en: str | None = None
    caption_pl: str | None = None
    caption_en: str | None = None
    description: str | None = None
    depicts: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    creator: str | None = None
    attribution: str | None = None
    date: str | None = None
    location: str | None = None
    license: str | None = None
    license_url: str | None = None
    width: int | None = None
    height: int | None = None
    image_width: int | None = None
    image_height: int | None = None
    mime_type: str | None = None
    retrieved_at: str | None = None
    caption_source: dict[str, str | None] = field(default_factory=lambda: {"pl": None, "en": None})
    caption_quality: dict[str, str | None] = field(default_factory=lambda: {"pl": None, "en": None})
    retrieval_method: str | None = None
    seed_entity: str | None = None
    source_file_id: str | None = None
    source_sha1: str | None = None
    structured_revision: int | None = None
    raw_requests: list[str] = field(default_factory=list)
    text_candidates: list[dict[str, str | None]] = field(default_factory=list)
    associations: list[dict[str, Any]] = field(default_factory=list)
    local_path: str | None = None
    sha256: str | None = None
    phash: str | None = None
    duplicate_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
