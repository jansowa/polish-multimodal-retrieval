from __future__ import annotations

import json
from pathlib import Path

from ..storage.checkpoint import atomic_json


class RecordLimitReached(RuntimeError):
    pass


def record_limit(directory: Path, requested: int | None = None) -> int | None:
    """Persist an explicit operator ceiling, including across plain resumptions."""
    path = directory / "collection_control.json"
    if requested is None and path.exists():
        requested = json.loads(path.read_text(encoding="utf-8-sig")).get("max_records")
    if requested is not None:
        if type(requested) is not int or requested <= 0:
            raise ValueError("max-records must be a positive integer")
        if not path.exists() or json.loads(path.read_text(encoding="utf-8-sig")).get("max_records") != requested:
            atomic_json(path, {"max_records": requested})
    return requested
