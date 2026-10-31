from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import TranslationUnit


class CheckpointStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db_path = directory / "checkpoint.sqlite3"
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS completed (
                unit_id TEXT PRIMARY KEY,
                source_json TEXT NOT NULL,
                translated_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                model TEXT NOT NULL,
                with_image INTEGER NOT NULL,
                completed_at TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS failures (
                unit_id TEXT PRIMARY KEY,
                error TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def is_completed(self, unit_id: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM completed WHERE unit_id=?", (unit_id,)).fetchone()
        return row is not None

    def put_completed(
        self,
        unit: TranslationUnit,
        translated: dict[str, Any],
        *,
        model: str,
        with_image: bool,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """
            INSERT OR REPLACE INTO completed
            (unit_id, source_json, translated_json, metadata_json, model, with_image, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                unit.unit_id,
                json.dumps(unit.texts, ensure_ascii=False),
                json.dumps(translated, ensure_ascii=False),
                json.dumps(unit.metadata, ensure_ascii=False, default=str),
                model,
                int(with_image),
                now,
            ),
        )
        self.conn.execute("DELETE FROM failures WHERE unit_id=?", (unit.unit_id,))
        self.conn.commit()

    def put_failure(self, unit_id: str, error: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """
            INSERT INTO failures(unit_id, error, attempts, updated_at)
            VALUES (?, ?, 1, ?)
            ON CONFLICT(unit_id) DO UPDATE SET
                error=excluded.error,
                attempts=failures.attempts+1,
                updated_at=excluded.updated_at
            """,
            (unit_id, error[:10000], now),
        )
        self.conn.commit()

    def counts(self) -> tuple[int, int]:
        completed = self.conn.execute("SELECT COUNT(*) FROM completed").fetchone()[0]
        failures = self.conn.execute("SELECT COUNT(*) FROM failures").fetchone()[0]
        return int(completed), int(failures)

    def export_jsonl(self) -> Path:
        path = self.directory / "translations.jsonl"
        tmp = self.directory / ".translations.jsonl.tmp"
        with tmp.open("w", encoding="utf-8") as f:
            cursor = self.conn.execute(
                "SELECT unit_id, source_json, translated_json, metadata_json, model, with_image, completed_at "
                "FROM completed ORDER BY rowid"
            )
            for unit_id, source, translated, metadata, model, with_image, completed_at in cursor:
                row = {
                    "id": unit_id,
                    "source": json.loads(source),
                    "translated": json.loads(translated),
                    "metadata": json.loads(metadata),
                    "model": model,
                    "with_image": bool(with_image),
                    "completed_at": completed_at,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(path)
        return path

    def export_review_comparison(self) -> Path:
        """Small paired audit export, without expanded image-candidate lists."""
        path = self.directory / "review_comparison.jsonl"
        tmp = self.directory / ".review_comparison.jsonl.tmp"
        with tmp.open("w", encoding="utf-8") as f:
            for unit_id, raw_metadata, model, with_image, completed_at in self.conn.execute(
                "SELECT unit_id, metadata_json, model, with_image, completed_at "
                "FROM completed ORDER BY unit_id"
            ):
                metadata = json.loads(raw_metadata)
                audit = metadata.get("translation_audit")
                if not audit:
                    continue
                row = {
                    "id": unit_id, **audit,
                    "quality_flags": metadata.get("translation_quality", {}).get("flags", []),
                    "model": model, "with_image": bool(with_image), "completed_at": completed_at,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(path)
        return path
