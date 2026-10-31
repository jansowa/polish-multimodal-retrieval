from __future__ import annotations

import hashlib
import logging
from urllib.parse import urlparse

from ..config import Config
from ..sources.http import HttpClient
from ..storage.checkpoint import CheckpointStore
from ..storage.parquet import export
from .dedup import PHash, file_hash

LOG = logging.getLogger(__name__)


def download_images(config: Config, store: CheckpointStore, client: HttpClient) -> None:
    directory = store.directory / "images"
    directory.mkdir(parents=True, exist_ok=True)
    if config.download.phash:
        try:
            import imagehash  # noqa: F401
        except ImportError as exc:
            raise ValueError("pHash needs the optional collection-phash extra") from exc
    try:
        for record in store.records():
            key = "download:" + record.id
            try:
                url = urlparse(record.image_original_url)
                if url.scheme != "https" or (record.source == "wikimedia" and url.hostname != "upload.wikimedia.org"):
                    raise ValueError("Unexpected original image URL")
                suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
                          "image/svg+xml": ".svg", "image/tiff": ".tif"}.get(record.mime_type, ".bin")
                path = directory / (hashlib.sha256(record.id.encode()).hexdigest() + suffix)
                if path.exists() and record.sha256 and file_hash(path) == record.sha256:
                    if config.download.phash and not record.phash:
                        record.phash = PHash().fingerprint(path)
                        store.update_record(record)
                    store.clear_failure(key)
                    continue
                if not path.exists() or record.sha256:
                    client.download(record.image_original_url, path, config.download.max_bytes)
                if record.source_sha1:
                    expected = record.source_sha1.lower()
                    actual = file_hash(path, "sha1")
                    # MediaWiki returns base-36 SHA1; some sources return hexadecimal.
                    valid = actual == expected or int(actual, 16) == int(expected, 36)
                    if not valid:
                        path.unlink(missing_ok=True)
                        raise ValueError("Source SHA1 mismatch; file version may have changed")
                if record.mime_type != "image/svg+xml":
                    from PIL import Image
                    try:
                        with Image.open(path) as image:
                            image.verify()
                    except Exception:
                        path.unlink(missing_ok=True)
                        raise
                record.local_path = str(path.relative_to(store.directory)).replace("\\", "/")
                record.sha256 = file_hash(path)
                if config.download.phash:
                    record.phash = PHash().fingerprint(path)
                store.update_record(record)
                store.clear_failure(key)
                LOG.info("Downloaded %s", record.id)
            except Exception as exc:
                LOG.exception("Image download failed: %s", record.id)
                store.failure(key, exc)
    finally:
        export(store, config.jsonl)
