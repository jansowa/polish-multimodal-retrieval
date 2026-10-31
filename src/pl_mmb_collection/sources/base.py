from __future__ import annotations

from typing import Any, Iterable, Protocol

from ..models import DatasetRecord, Entity, ImageCandidate


class DataSource(Protocol):
    """Adapters return normalized records and preserve their raw API evidence."""

    def discover(self, store: Any) -> None: ...
    def candidates(self, entity: Entity) -> Iterable[ImageCandidate]: ...
    def fetch(self, candidate: ImageCandidate, entity: Entity) -> DatasetRecord: ...
