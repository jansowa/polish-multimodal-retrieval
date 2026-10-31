from __future__ import annotations

from typing import Iterable, Iterator
from urllib.parse import quote


def batches(values: Iterable[str], size: int = 50, encoded_limit: int = 5000) -> Iterator[list[str]]:
    """Bound item count and encoded GET length (file titles can be very long)."""
    batch: list[str] = []
    length = 0
    for value in values:
        extra = len(quote(value, safe="")) + 3
        if batch and (len(batch) >= size or length + extra > encoded_limit):
            yield batch
            batch, length = [], 0
        batch.append(value)
        length += extra
    if batch:
        yield batch
