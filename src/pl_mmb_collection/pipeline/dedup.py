from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from ..storage.checkpoint import CheckpointStore, atomic_json


class PerceptualHasher(Protocol):
    """Extension point for perceptual fingerprints; candidates never delete data."""

    def fingerprint(self, path: Path) -> str: ...
    def distance(self, left: str, right: str) -> float: ...


class PHash:
    def fingerprint(self, path: Path) -> str:
        import imagehash
        from PIL import Image
        with Image.open(path) as image:
            return str(imagehash.phash(image.convert("RGB"), hash_size=8))

    def distance(self, left: str, right: str) -> float:
        return (int(left, 16) ^ int(right, 16)).bit_count()


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def deduplicate(store: CheckpointStore, distance: int = 6, use_phash: bool = False,
                hasher: PerceptualHasher | None = None) -> dict:
    hasher = hasher or PHash()
    hashes: dict[str, str] = {}
    perceptual: list[tuple[str, str]] = []
    matches = []
    for record in store.records():
        if not record.local_path:
            continue
        path = store.directory / record.local_path
        if not path.exists():
            continue
        record.sha256 = file_hash(path)
        record.duplicate_of = None
        if record.sha256 in hashes:
            record.duplicate_of = hashes[record.sha256]
            matches.append(dict(id=record.id, duplicate_of=record.duplicate_of, method="sha256", distance=0))
        else:
            if use_phash:
                record.phash = hasher.fingerprint(path)
                for canonical, fingerprint in perceptual:
                    delta = hasher.distance(record.phash, fingerprint)
                    if delta <= distance:
                        record.duplicate_of = canonical
                        matches.append(dict(id=record.id, duplicate_of=canonical, method="phash", distance=delta))
                        break
                if not record.duplicate_of:
                    perceptual.append((record.id, record.phash))
            hashes[record.sha256] = record.duplicate_of or record.id
        store.update_record(record)
    report = {"duplicate_candidates": matches, "count": len(matches), "phash_enabled": use_phash}
    atomic_json(store.directory / "metadata/deduplication.json", report)
    return report
