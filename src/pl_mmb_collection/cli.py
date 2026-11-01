from __future__ import annotations

import argparse
import json
import logging
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

from .config import Config, load_config
from .sources.base import DataSource
from .sources.http import HttpClient
from .storage.checkpoint import CheckpointStore
from .storage.lease import CollectorLease


def make_source(name: str, client: HttpClient, config: Config) -> DataSource:
    if name == "wikimedia":
        from .sources.wikimedia import WikimediaSource
        return WikimediaSource(client, config)
    raise ValueError(f"Unknown source: {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resumable Polish culture image/text collection")
    parser.add_argument("command", choices=["collect", "discover-entities", "collect-metadata",
                                          "download-images", "deduplicate", "validate", "stats", "preview"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", help="Overrides the configured source")
    parser.add_argument("--output", type=Path, help="Overrides output_dir (relative to working directory)")
    parser.add_argument("--phash", action="store_true", help="Enable optional pHash in deduplicate")
    parser.add_argument("--sample-size", type=int, default=12, help="Number of randomly sampled preview pairs")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible preview sampling seed")
    parser.add_argument("--max-seconds", type=float, help="Stop collect after this soft run budget; rerun to resume")
    parser.add_argument("--min-free-gb", type=float, default=2, help="Stop collect before disk space falls below this GiB reserve")
    parser.add_argument("--max-records", type=int, help="Persist a unique-record ceiling and stop at that total")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.output:
            config = replace(config, output_dir=str(args.output))
        if args.source:
            config = replace(config, source=args.source)
        directory = Path(config.output_dir).expanduser().resolve()
        (directory / "logs").mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                            handlers=[logging.StreamHandler(), logging.FileHandler(directory / "logs/collection.log", encoding="utf-8")])
        if args.max_seconds is not None and args.max_seconds <= 0:
            raise ValueError("max-seconds must be positive")
        if args.min_free_gb <= 0:
            raise ValueError("min-free-gb must be positive")
        if args.max_records is not None and args.max_records <= 0:
            raise ValueError("max-records must be positive")
        networking = args.command in {"collect", "discover-entities", "collect-metadata", "download-images"}
        budget_path = Path(config.http.state_path) if config.http.state_path else directory / "http_state.json"
        lease = CollectorLease(budget_path.with_suffix(".owner.lock")) if networking else nullcontext()
        transport = HttpClient(config.http, directory / "raw") if networking else nullcontext()
        with lease, CheckpointStore(directory, config) as store, transport as client:
            if args.command in {"collect", "discover-entities", "collect-metadata"}:
                source = make_source(config.source, client, config)
                if args.command == "collect" and hasattr(source, "wikidata"):
                    from .pipeline.hybrid import run
                    run(config, store, source, client, args.max_seconds, int(args.min_free_gb * 1024 ** 3),
                        max_records=args.max_records)
                elif args.command == "discover-entities":
                    source.discover(store)
                elif args.command == "collect":
                    source.discover(store)
                    from .pipeline.collect import collect
                    collect(config, store, source)
                if args.command == "collect-metadata":
                    from .pipeline.collect import collect
                    from .pipeline.limits import RecordLimitReached
                    try:
                        collect(config, store, source, max_records=args.max_records)
                    except RecordLimitReached as exc:
                        logging.getLogger(__name__).info("%s", exc)
            elif args.command == "download-images":
                from .pipeline.download import download_images
                download_images(config, store, client)
            elif args.command == "deduplicate":
                from .pipeline.dedup import deduplicate
                from .storage.parquet import export
                print(json.dumps(deduplicate(store, config.download.phash_distance,
                                             args.phash or config.download.phash), indent=2))
                export(store, config.jsonl)
            elif args.command == "preview":
                from .preview import write_preview
                path = write_preview(store, directory / "preview.html", args.sample_size, args.seed)
                print(f"Preview: {path}")
            elif args.command == "validate":
                from .pipeline.reporting import validate
                report = validate(store, config)
                print(json.dumps(report, ensure_ascii=False, indent=2))
                return 0 if report["valid"] else 1
            from .pipeline.reporting import statistics
            print(json.dumps(statistics(store), ensure_ascii=False, indent=2))
            return 2 if store.counts()["failures"] else 0
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Interrupted; rerun the same command to resume")
        return 130
    except (ValueError, ImportError, OSError) as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
