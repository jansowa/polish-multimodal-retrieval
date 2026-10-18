from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .config import Settings
from .registry import BENCHMARKS, DATASETS, resolve_target
from .runner import run_dataset


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected True or False")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate multimodal embedding benchmark text from English to Polish via an OpenAI-compatible API."
    )
    parser.add_argument("target", nargs="?", help="Dataset name or benchmark name, e.g. cirr, compositionality, mmeb_core, pl_v1")
    parser.add_argument("with_image", nargs="?", type=parse_bool, help="True/False: send related image(s) to the translation model")
    parser.add_argument("--env-file", default=".env", help="Path to .env (default: .env)")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=None, help="Translate only first N records of each dataset (useful for testing)")
    selection.add_argument("--ids-file", type=Path, help="JSON list of exact unit IDs to translate (single dataset only)")
    parser.add_argument("--list", action="store_true", help="List available datasets and benchmarks")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        print("Datasets:")
        for key in sorted(DATASETS):
            print(f"  {key:28} {DATASETS[key].description}")
        print("\nBenchmarks:")
        for key in sorted(BENCHMARKS):
            print(f"  {key:28} {len(BENCHMARKS[key])} datasets")
        return

    if args.target is None or args.with_image is None:
        parser.error("target and with_image are required, e.g. `python translate.py pl_v1 true`")

    try:
        dataset_keys = resolve_target(args.target)
    except KeyError:
        print(f"Unknown dataset/benchmark: {args.target!r}\n", file=sys.stderr)
        print("Run with --list to see available targets.", file=sys.stderr)
        raise SystemExit(2)

    settings = Settings.from_env(args.env_file)
    unit_ids = None
    if args.ids_file:
        if len(dataset_keys) != 1:
            parser.error("--ids-file requires exactly one dataset")
        try:
            ids = json.loads(args.ids_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            parser.error(f"Cannot read --ids-file: {exc}")
        if (not isinstance(ids, list) or not ids
                or any(not isinstance(item, str) or not item.strip() for item in ids)
                or len(set(ids)) != len(ids)):
            parser.error("--ids-file must contain a nonempty JSON list of unique, nonempty string IDs")
        unit_ids = tuple(ids)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.hf_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"target={args.target} -> {len(dataset_keys)} dataset(s)")
    print(f"model={settings.model}")
    print(f"base_url={settings.base_url}")
    print(f"concurrency(BATCH_SIZE)={settings.batch_size}")
    print(f"with_image={args.with_image}")
    print(f"quality_review={settings.quality_review}; save_drafts={settings.save_drafts}")

    async def run_all() -> None:
        for key in dataset_keys:
            await run_dataset(key, settings, with_image=args.with_image, limit=args.limit, unit_ids=unit_ids)

    asyncio.run(run_all())
