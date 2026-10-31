from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from ..config import Config
from ..models import Entity
from ..sources.http import HttpClient
from ..sources.wikidata import parse_entity
from ..storage.checkpoint import CheckpointStore, atomic_json

LOG = logging.getLogger(__name__)


def import_seeds(config: Config, store: CheckpointStore, client: HttpClient) -> None:
    """Reuse previous QIDs or consume local JSONL dumps; never downloads a full dump."""
    def add(entity: Entity, discovered: bool = False) -> None:
        if store.entity_count() < config.discovery.max_entities:
            store.add_entity(entity, discovered)
    for source in config.discovery.seed_checkpoints:
        path = Path(source).resolve()
        if path.is_dir():
            path = path / "checkpoint.sqlite3"
        task = "import-checkpoint:" + str(path)
        if store.task_completed(task):
            continue
        # Read a bounded, consistent snapshot from an existing checkpoint.
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as previous:
            for raw, discovered in previous.execute("SELECT json,discovered FROM entities ORDER BY rowid"):
                entity = Entity(**json.loads(raw))
                if entity.raw_request:
                    src = path.parent / "raw" / (entity.raw_request + ".json")
                    if src.exists():
                        atomic_json(client.raw_dir / src.name, json.loads(src.read_text(encoding="utf-8")))
                    else:
                        discovered = 0
                add(entity, bool(discovered))
        store.complete_task(task)
        LOG.info("Imported entity queue from %s", path)
    for source in config.discovery.entity_files:
        path = Path(source).resolve()
        task = "import-file:" + str(path)
        if store.task_completed(task):
            continue
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if store.entity_count() >= config.discovery.max_entities:
                    break
                text = line.strip().rstrip(",")
                if not text or text in {"[", "]"}:
                    continue
                item = {"qid": text} if re.fullmatch(r"Q[1-9][0-9]*", text) else json.loads(text)
                if isinstance(item, str):
                    item = {"qid": item}
                qid = item.get("qid") or item.get("id")
                if not isinstance(qid, str) or not re.fullmatch(r"Q[1-9][0-9]*", qid):
                    raise ValueError(f"Invalid QID in {path.name}")
                if "claims" in item:
                    key = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
                    atomic_json(client.raw_dir / (key + ".json"), dict(url="import:" + path.name,
                        retrieved_at=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                        response={"entities": {qid: item}}))
                    entity = parse_entity(item, qid, 0, config.discovery.properties)
                    entity.raw_request = key
                    add(entity, True)
                else:
                    add(Entity(qid=qid, seed=qid))
        store.complete_task(task)
        LOG.info("Imported local entity file %s", path)
