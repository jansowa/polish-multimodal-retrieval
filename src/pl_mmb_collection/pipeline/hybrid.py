from __future__ import annotations

import logging
import shutil
import time

from ..config import Config
from ..sources.http import HttpClient
from ..sources.wikimedia import WikimediaSource
from ..storage.checkpoint import CheckpointStore, atomic_json
from ..storage.parquet import export
from .collect import collect
from .discovery import discover, seed
from .imports import import_seeds
from .limits import RecordLimitReached, record_limit

LOG = logging.getLogger(__name__)


class StorageLimitReached(RuntimeError):
    pass


def run(config: Config, store: CheckpointStore, source: WikimediaSource,
        client: HttpClient, max_seconds: float | None = None, min_free_bytes: int = 0,
        max_records: int | None = None) -> None:
    started = time.monotonic()
    initial = store.counts()["records"]
    max_records = record_limit(store.directory, max_records)
    seed(config, store)
    import_seeds(config, store, client)

    def progress(status: str = "collecting") -> None:
        elapsed = max(.001, time.monotonic() - started)
        counts = store.counts()
        client.save_metrics()
        atomic_json(store.directory / "progress.json", {
            **counts, "status": status, "elapsed_seconds": elapsed,
            "max_records": max_records,
            "new_records": counts["records"] - initial,
            "records_per_hour": (counts["records"] - initial) * 3600 / elapsed,
            "http": client.metrics, "next_retry_at": store.next_retry()})

    def check_space() -> None:
        # Leave enough space to export both the old and replacement snapshots.
        export_dir = store.directory / "metadata"
        snapshot_bytes = sum(p.stat().st_size for p in export_dir.glob("records.*") if p.is_file() and not p.name.endswith(".tmp"))
        reserve = max(min_free_bytes, snapshot_bytes + 512 * 1024 ** 2)
        if min_free_bytes and shutil.disk_usage(store.directory).free < reserve:
            raise StorageLimitReached(f"Free disk space below export reserve ({reserve} bytes); rerun after freeing space")

    def checked_progress() -> None:
        progress()
        check_space()

    status = "complete"
    try:
        while True:
            if max_records is not None and store.conn.execute("SELECT count(*) FROM records").fetchone()[0] >= max_records:
                raise RecordLimitReached(f"Reached {max_records} unique records")
            check_space()
            before = store.counts()
            # Each round hydrates one small entity batch and immediately harvests
            # ready images. SPARQL never stands between known IDs and Commons.
            discover(config, store, source.wikidata, config.pipeline.discovery_batches_per_round, allow_sparql=False)
            collect(config, store, source, config.pipeline.collect_entities_per_round,
                    final_export=False, on_progress=checked_progress,
                    deadline=started + max_seconds if max_seconds is not None else None,
                    max_records=max_records)
            counts = store.counts()
            progress()
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                status = "time_budget_reached"
                break
            if not counts["pending_discovery"] and not counts["pending_collection"]:
                if config.discovery.enable_sparql:
                    discover(config, store, source.wikidata, max_batches=0, allow_sparql=True)
                    if store.counts()["pending_discovery"]:
                        continue
                status = "waiting" if store.next_retry() else "complete"
                if status == "complete":
                    break
            if counts == before:
                retry_at = store.next_retry()
                if retry_at:
                    progress("waiting")
                    delay = min(30.0, max(1.0, retry_at - time.time()))
                    LOG.info("Waiting %.1fs for service cooldown; completed files remain available", delay)
                    time.sleep(delay)
                else:
                    # Permanent individual errors remain resumable; do not hammer
                    # the same failed entities/files in an unbounded retry loop.
                    status = "partial"
                    break
    except RecordLimitReached as exc:
        LOG.info("%s; stopping collection and exporting", exc)
        status = "record_limit_reached"
    except StorageLimitReached as exc:
        LOG.warning("%s", exc)
        status = "storage_limit_reached"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception:
        status = "failed"
        raise
    finally:
        export(store, config.jsonl)
        progress(status)
