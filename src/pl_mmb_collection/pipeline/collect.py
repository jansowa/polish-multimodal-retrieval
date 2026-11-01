from __future__ import annotations

import logging
import time
from typing import Callable

from ..config import Config
from ..sources.base import DataSource
from ..sources.rate_limit import RetryLater
from ..models.acquisition import FetchResult
from ..storage.checkpoint import CheckpointStore
from ..storage.parquet import export
from .filters import rejection_reasons
from .limits import RecordLimitReached, record_limit

LOG = logging.getLogger(__name__)


def collect(config: Config, store: CheckpointStore, source: DataSource,
            entity_limit: int | None = None, final_export: bool = True,
            on_progress: Callable[[], None] | None = None, deadline: float | None = None,
            max_records: int | None = None) -> None:
    max_records = record_limit(store.directory, max_records)

    def check_record_limit() -> None:
        if max_records is not None and store.conn.execute("SELECT count(*) FROM records").fetchone()[0] >= max_records:
            raise RecordLimitReached(f"Reached {max_records} unique records")

    states: dict[str, dict] = {}
    queue = []
    last_export = 0.0
    last_count = store.counts()["records"]

    def finish(key: str) -> None:
        state = states[key]
        if state["finished"] and not state["pending"] and not state["failed"]:
            store.complete_entity(key)
            store.clear_failure("collect:" + key)

    def flush() -> None:
        nonlocal last_export, last_count
        if not queue:
            return
        check_record_limit()
        work = [(candidate, entity) for _, candidate, entity in queue]
        if hasattr(source, "fetch_many"):
            try:
                results = source.fetch_many(work)
                if len(results) != len(work):
                    raise ValueError("Source returned an incomplete metadata batch")
            except Exception as exc:
                results = [FetchResult(c, e, error=exc) for c, e in work]
        else:
            results = []
            for candidate, entity in work:
                try:
                    results.append(FetchResult(candidate, entity, record=source.fetch(candidate, entity)))
                except Exception as exc:
                    results.append(FetchResult(candidate, entity, error=exc))
        for (key, candidate, entity), result in zip(queue, results):
            state = states[key]
            state["pending"] -= 1
            task = "file:" + candidate.title
            if result.error:
                state["failed"] = True
                if isinstance(result.error, RetryLater):
                    store.defer(task, result.error)
                    store.defer("collect:" + key, result.error)
                else:
                    LOG.warning("Metadata failed for %s: %s", candidate.title, result.error,
                                exc_info=(type(result.error), result.error, result.error.__traceback__))
                    store.failure(task, result.error)
            else:
                record = result.record
                reasons = rejection_reasons(record, config.filters)
                if reasons:
                    store.reject(candidate.title, reasons)
                    LOG.info("Rejected %s: %s", candidate.title, ", ".join(reasons))
                else:
                    canonical = store.accept(record, candidate.title)
                    LOG.info("Accepted %s (%s)", candidate.title, canonical)
                store.clear_failure(task)
            finish(key)
            check_record_limit()
        queue.clear()
        count = store.counts()["records"]
        if (count - last_count >= config.pipeline.export_every_records
                or count != last_count and time.monotonic() - last_export >= config.pipeline.export_interval):
            export(store, config.jsonl)
            last_export, last_count = time.monotonic(), count
        if on_progress:
            on_progress()

    try:
        check_record_limit()
        pending = [(k, e) for k, e in store.entities("collect") if store.ready("collect:" + k)]
        for key, entity in pending[:entity_limit]:
            if deadline is not None and time.monotonic() >= deadline:
                break
            states[key] = dict(pending=0, failed=False, finished=False)
            exhausted = True
            try:
                for candidate in source.candidates(entity):
                    if store.link_existing(candidate.title, entity, candidate):
                        continue
                    if not store.ready("file:" + candidate.title):
                        states[key]["failed"] = True
                        continue
                    states[key]["pending"] += 1
                    queue.append((key, candidate, entity))
                    if len(queue) >= config.wikimedia.structured_batch_size:
                        flush()
                    if deadline is not None and time.monotonic() >= deadline:
                        exhausted = False
                        break
                states[key]["finished"] = exhausted
                finish(key)
                if not exhausted:
                    break
            except RecordLimitReached:
                raise
            except RetryLater as exc:
                states[key]["failed"] = True
                store.defer("collect:" + key, exc)
            except Exception as exc:
                states[key]["failed"] = True
                LOG.exception("Candidate discovery failed for %s", key)
                store.failure("collect:" + key, exc)
        flush()
    finally:
        # Ctrl+C and individual failures retain both the checkpoint and an export.
        if final_export:
            export(store, config.jsonl)
