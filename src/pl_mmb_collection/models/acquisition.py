from __future__ import annotations

from dataclasses import dataclass

from .record import DatasetRecord, Entity, ImageCandidate


@dataclass
class FetchResult:
    candidate: ImageCandidate
    entity: Entity
    record: DatasetRecord | None = None
    error: Exception | None = None
