from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from ..config import Config
from ..storage.checkpoint import CheckpointStore, atomic_json
from ..storage.lease import CollectorLease
from ..storage.parquet import export

LOG = logging.getLogger(__name__)


def expand(source: Path, config: Config) -> dict:
    """Fork a checkpoint with larger search bounds, retaining records and evidence."""
    config.validate()
    source = source.resolve()
    target = Path(config.output_dir).resolve()
    if target == source or target.exists():
        raise ValueError("Expansion needs a new, nonexistent output_dir")
    budget = Path(config.http.state_path) if config.http.state_path else source / "http_state.json"
    with CollectorLease(budget.with_suffix(".owner.lock")):
        with closing(sqlite3.connect((source / "checkpoint.sqlite3").as_uri() + "?mode=ro", uri=True)) as previous:
            old = json.loads(previous.execute("SELECT json FROM manifest WHERE id=1").fetchone()[0])
            new = config.manifest()
            # An expansion must preserve the acquisition/filter policy. Changed
            # filters would make inherited file rejections and item caches unsafe.
            comparable = json.loads(json.dumps(old))
            desired = json.loads(json.dumps(new))
            changes = {}
            for section, fields in {
                "discovery": ("max_entities", "max_depth", "max_links_per_entity"),
                "wikimedia": ("max_files_per_entity", "category_depth", "max_categories", "category_member_limit"),
            }.items():
                for field in fields:
                    before, after = old["config"][section][field], new["config"][section][field]
                    if after < before:
                        raise ValueError(f"Expansion cannot reduce {section}.{field}")
                    if after != before:
                        changes[f"{section}.{field}"] = {"before": before, "after": after}
                    comparable["config"][section].pop(field)
                    desired["config"][section].pop(field)
            comparable["config"].pop("output_dir")
            desired["config"].pop("output_dir")
            # Export cadence affects I/O cost, not acceptance or provenance.
            for field in ("export_every_records", "export_interval"):
                comparable["config"]["pipeline"].pop(field)
                desired["config"]["pipeline"].pop(field)
            if comparable != desired:
                raise ValueError("Expansion only supports increased search bounds with identical policy")
            if not changes:
                raise ValueError("Increase at least one search bound")
            target.mkdir(parents=True)
            with closing(sqlite3.connect(target / "checkpoint.sqlite3")) as destination:
                LOG.info("Copying checkpoint from %s to %s", source, target)
                previous.backup(destination)
                with destination:
                    destination.execute("UPDATE manifest SET json=? WHERE id=1", (json.dumps(new, sort_keys=True),))
                    destination.execute("UPDATE entities SET collected=0")
                    # Revisit cached entity claims to expand graph edges beyond
                    # the former depth/link bounds. Categories need no hydration.
                    for key, raw in destination.execute("SELECT key,json FROM entities").fetchall():
                        if json.loads(raw).get("qid"):
                            destination.execute("UPDATE entities SET discovered=0 WHERE key=?", (key,))
        copied = 0
        def copy_file(src, dst):
            nonlocal copied
            result = shutil.copy2(src, dst)
            copied += 1
            if copied % 10000 == 0:
                LOG.info("Copied %d evidence/cache files", copied)
            return result

        for name in ("raw", "items", "images"):
            if (source / name).exists():
                LOG.info("Copying %s", name)
                shutil.copytree(source / name, target / name, copy_function=copy_file)
        if not config.http.state_path and (source / "http_state.json").exists():
            shutil.copy2(source / "http_state.json", target / "http_state.json")
        with CheckpointStore(target, config) as store:
            LOG.info("Exporting inherited records")
            export(store, config.jsonl)
            report = {"source": str(source), "created_at": datetime.now(timezone.utc).isoformat(),
                      "changes": changes, "counts": store.counts()}
            atomic_json(target / "expansion.json", report)
            return report
