from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ..storage.checkpoint import atomic_json
from .data import BrowserStore, summary


def export_snapshot(store: BrowserStore, site_dir: Path) -> Path:
    """Publishable source holds only the selected dataset snapshot and web assets."""
    dist = site_dir / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    for path in (Path(__file__).parent / "web").iterdir():
        if path.is_file():
            shutil.copy2(path, dist / path.name)
    catalog = store.catalog()
    catalog["mode"] = "snapshot"
    catalog["generated_at"] = datetime.now(timezone.utc).isoformat()
    for dataset in catalog["datasets"]:
        rows = []
        for record in store.iter_records(dataset["id"]):
            key = hashlib.sha256(record["id"].encode()).hexdigest()
            row = summary(record)
            row["detail_key"] = key
            # Search uses the complete source text, even when card captions are truncated.
            row["search_text"] = json.dumps(record, ensure_ascii=False).casefold()
            rows.append(row)
            atomic_json(dist / "site-data" / dataset["id"] / "records" / f"{key}.json", record)
        dataset["count"] = len(rows)
        atomic_json(dist / "site-data" / dataset["id"] / "index.json", rows)
    atomic_json(dist / "site-data/catalog.json", catalog)
    atomic_json(dist / "app-config.json", {"mode": "snapshot"})
    return dist
