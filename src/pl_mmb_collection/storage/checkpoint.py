from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from ..config import Config
from ..models import DatasetRecord, Entity, ImageCandidate


def atomic_replace(temporary: Path, target: Path) -> None:
    # Windows readers/virus scanners can briefly hold the destination open.
    # Retry the local replacement, without repeating the HTTP request.
    for attempt in range(4):
        try:
            temporary.replace(target)
            return
        except PermissionError:
            if attempt == 3:
                raise
            time.sleep(.1 * 2 ** attempt)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    atomic_replace(tmp, path)


def association(entity: Entity, candidate: ImageCandidate) -> dict[str, Any]:
    return {"wikidata_qid": entity.qid, "seed_entity": entity.seed,
            "entity_label_pl": entity.label_pl, "entity_label_en": entity.label_en,
            "retrieval_method": candidate.method, "category": candidate.category,
            "candidate_title": candidate.title,
            "category_depth": candidate.category_depth,
            "entity_revision": entity.revision, "entity_raw_request": entity.raw_request}


class CheckpointStore:
    """SQLite is authoritative. Completion and record writes are transactional."""

    def __init__(self, directory: Path, config: Config):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(directory / "checkpoint.sqlite3")
        self.conn.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS manifest (id INTEGER PRIMARY KEY, json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS entities (
                key TEXT PRIMARY KEY, json TEXT NOT NULL, discovered INTEGER DEFAULT 0,
                collected INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS identities (identity TEXT PRIMARY KEY, record_id TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS files (
                title TEXT PRIMARY KEY, record_id TEXT, rejection TEXT);
            CREATE TABLE IF NOT EXISTS failures (
                key TEXT PRIMARY KEY, error TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 1);
            CREATE TABLE IF NOT EXISTS deferred (
                key TEXT PRIMARY KEY, retry_at REAL NOT NULL, reason TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS discovery_tasks (
                key TEXT PRIMARY KEY, completed INTEGER NOT NULL DEFAULT 0);
        """)
        manifest = json.dumps(config.manifest(), sort_keys=True)
        row = self.conn.execute("SELECT json FROM manifest WHERE id=1").fetchone()
        if row and row[0] != manifest:
            self.conn.close()
            raise ValueError("Configuration/version differs from checkpoint. Use a new output_dir.")
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO manifest VALUES (1, ?)", (manifest,))
        atomic_json(directory / "manifest.json", config.manifest())

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> CheckpointStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def entity_count(self) -> int:
        return self.conn.execute("SELECT count(*) FROM entities").fetchone()[0]

    def add_entity(self, entity: Entity, discovered: bool = False) -> None:
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO entities(key,json,discovered) VALUES (?,?,?)",
                              (entity.qid or entity.seed, json.dumps(asdict(entity)), int(discovered)))

    def save_entity(self, key: str, entity: Entity) -> None:
        with self.conn:
            self.conn.execute("UPDATE entities SET json=?, discovered=1 WHERE key=?",
                              (json.dumps(asdict(entity)), key))
            self.conn.execute("DELETE FROM failures WHERE key=?", ("discover:" + key,))

    def entities(self, stage: str | None = None) -> list[tuple[str, Entity]]:
        where = {None: "", "discover": " WHERE discovered=0",
                 "collect": " WHERE discovered=1 AND collected=0"}[stage]
        return [(key, Entity(**json.loads(raw))) for key, raw in self.conn.execute(
            "SELECT key,json FROM entities" + where + " ORDER BY rowid")]

    def complete_entity(self, key: str) -> None:
        with self.conn:
            self.conn.execute("UPDATE entities SET collected=1 WHERE key=?", (key,))
            self.conn.execute("DELETE FROM failures WHERE key=?", ("collect:" + key,))

    def failure(self, key: str, error: Exception) -> None:
        with self.conn:
            self.conn.execute("""INSERT INTO failures(key,error) VALUES (?,?)
                ON CONFLICT(key) DO UPDATE SET error=excluded.error, attempts=attempts+1""",
                              (key, str(error)[:10000]))

    def clear_failure(self, key: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM failures WHERE key=?", (key,))
            self.conn.execute("DELETE FROM deferred WHERE key=?", (key,))

    def defer(self, key: str, error: Any) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO deferred VALUES (?,?,?)", (key, error.retry_at, str(error)))

    def ready(self, key: str) -> bool:
        row = self.conn.execute("SELECT retry_at FROM deferred WHERE key=?", (key,)).fetchone()
        return row is None or row[0] <= time.time()

    def next_retry(self) -> float | None:
        row = self.conn.execute("SELECT min(retry_at) FROM deferred WHERE retry_at>?", (time.time(),)).fetchone()
        return row[0]

    def task_completed(self, key: str) -> bool:
        return self.conn.execute("SELECT 1 FROM discovery_tasks WHERE key=? AND completed=1", (key,)).fetchone() is not None

    def complete_task(self, key: str) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO discovery_tasks VALUES (?,1)", (key,))
        self.clear_failure(key)

    def file_result(self, title: str) -> tuple[str | None, str | None] | None:
        return self.conn.execute("SELECT record_id,rejection FROM files WHERE title=?", (title,)).fetchone()

    def link_existing(self, title: str, entity: Entity, candidate: ImageCandidate) -> bool:
        result = self.file_result(title)
        if result is None:
            return False
        if result[0]:
            record = self.get_record(result[0])
            link = association(entity, candidate)
            if link not in record.associations:
                record.associations.append(link)
                self.update_record(record)
        return True

    def reject(self, title: str, reasons: list[str]) -> None:
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO files VALUES (?,NULL,?)",
                              (title, json.dumps(reasons)))
            self.conn.execute("DELETE FROM failures WHERE key=?", ("file:" + title,))

    @staticmethod
    def identities(record: DatasetRecord) -> list[str]:
        values = [f"{record.source}:page:{record.source_file_id}" if record.source_file_id else None,
                  "url:" + record.image_original_url,
                  "sha1:" + record.source_sha1 if record.source_sha1 else None]
        return [v for v in values if v]

    def accept(self, record: DatasetRecord, title: str) -> str:
        identities = self.identities(record)
        with self.conn:
            canonical = None
            for identity in identities:
                row = self.conn.execute("SELECT record_id FROM identities WHERE identity=?", (identity,)).fetchone()
                if row:
                    canonical = row[0]
                    break
            if canonical:
                current = self.get_record(canonical)
                for link in record.associations:
                    if link not in current.associations:
                        current.associations.append(link)
                # Evidence from aliases remains available without creating another image row.
                current.raw_requests = sorted(set(current.raw_requests + record.raw_requests))
                self._write_record(current)
            else:
                canonical = record.id
                self._write_record(record)
            for identity in identities:
                self.conn.execute("INSERT OR IGNORE INTO identities VALUES (?,?)", (identity, canonical))
            self.conn.execute("INSERT OR REPLACE INTO files VALUES (?,?,NULL)", (title, canonical))
            self.conn.execute("DELETE FROM failures WHERE key=?", ("file:" + title,))
        return canonical

    def _write_record(self, record: DatasetRecord) -> None:
        self.conn.execute("INSERT OR REPLACE INTO records VALUES (?,?)", (record.id, json.dumps(record.to_dict())))

    def update_record(self, record: DatasetRecord) -> None:
        with self.conn:
            self._write_record(record)

    def get_record(self, record_id: str) -> DatasetRecord:
        row = self.conn.execute("SELECT json FROM records WHERE id=?", (record_id,)).fetchone()
        if row is None:
            raise KeyError(record_id)
        return DatasetRecord(**json.loads(row[0]))

    def records(self) -> Iterator[DatasetRecord]:
        for (raw,) in self.conn.execute("SELECT json FROM records ORDER BY id"):
            yield DatasetRecord(**json.loads(raw))

    def counts(self) -> dict[str, int]:
        return {"records": self.conn.execute("SELECT count(*) FROM records").fetchone()[0],
                "entities": self.entity_count(),
                "pending_discovery": len(self.entities("discover")),
                "pending_collection": len(self.entities("collect")),
                "rejected_files": self.conn.execute("SELECT count(*) FROM files WHERE rejection IS NOT NULL").fetchone()[0],
                "deferred_tasks": self.conn.execute("SELECT count(*) FROM deferred WHERE retry_at>?", (time.time(),)).fetchone()[0],
                "failures": self.conn.execute("SELECT count(*) FROM failures").fetchone()[0]}
