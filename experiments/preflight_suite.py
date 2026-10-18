"""Inspect candidate datasets and the configured model endpoint before an unattended run.

No translation requests are made. Results are saved after each dataset probe.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import Image, get_dataset_config_info, load_dataset
from openai import AsyncOpenAI
from pl_mmb_translate.adapters.hf_generic import HFGenericAdapter
from pl_mmb_translate.adapters.mmeb import MMEBAdapter, MMEB_PATH, MMEB_REVISION
from pl_mmb_translate.config import Settings
from pl_mmb_translate.registry import DATASETS


def probe(key: str, settings: Settings) -> dict:
    spec = DATASETS[key]
    adapter = spec.factory()
    if isinstance(adapter, MMEBAdapter):
        path, revision, config, split = MMEB_PATH, MMEB_REVISION, adapter.config_name, "test"
        fields, image_fields = ("qry_text", "tgt_text"), ()
    elif isinstance(adapter, HFGenericAdapter):
        path, revision, config, split = adapter.path, adapter.revision, adapter.config, adapter.split
        fields, image_fields = adapter.text_fields, adapter.image_fields
    else:
        raise ValueError("This probe supports MMEB and HFGeneric adapters")
    kwargs = {"revision": revision, "token": settings.hf_token, "cache_dir": str(settings.hf_cache_dir)}
    info = get_dataset_config_info(path, config_name=config, **kwargs)
    split_info = info.splits[split]
    expected = split_info.get("num_examples") if isinstance(split_info, dict) else split_info.num_examples
    result = {"dataset_key": key, "path": path, "revision": revision, "config": config,
              "split": split, "profile": adapter.prompt_profile,
              "expected_examples": expected,
              "features": list(info.features), "missing_text_fields": [f for f in fields if f not in info.features]}
    ds = load_dataset(path, name=config, split=split, streaming=True, **kwargs)
    # Inspect raw image fields without triggering image decoding or remote image URLs.
    for field in image_fields:
        if isinstance(ds.features.get(field), Image):
            ds = ds.cast_column(field, Image(decode=False))
    row = next(iter(ds))
    result["sample_texts"] = {
        f: row[f][:3] if isinstance(row[f], list) else row[f] for f in fields if f in row
    }
    result["image_fields_present"] = [f for f in image_fields if row.get(f) is not None]
    if isinstance(adapter, MMEBAdapter):
        target = row.get("tgt_text") or []
        result["target_count"] = len(target)
        result["unique_target_texts"] = len(set(target))
        result["unique_target_characters"] = sum(len(t) for t in set(target))
        result["query_image_present"] = bool(row.get("qry_img_path"))
        result["target_images_present"] = bool(row.get("tgt_img_path"))
        # Large candidate-caption pools need a dedicated plan, not a larger retry count.
        result["large_output_risk"] = result["unique_target_characters"] > 18000
    result["ready_for_pilot"] = not result["missing_text_fields"] and not result.get("large_output_risk", False)
    return result


async def model_ids(settings: Settings) -> list[str]:
    async with AsyncOpenAI(base_url=settings.base_url, api_key=settings.api_key,
                           timeout=20, max_retries=0) as client:
        response = await client.models.list()
        return [model.id for model in response.data]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", default=[
        "sugarcrepe", "aro_visual_relation", "aro_visual_attribution", "winoground",
        "cirr", "fashioniq", "refcoco", "mscoco_i2t",
    ])
    args = parser.parse_args()
    settings = Settings.from_env(ROOT / ".env")
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "configured_model": settings.model,
              "hf_token_present": bool(settings.hf_token), "datasets": []}
    target = ROOT / "reports/trip_preflight.json"
    target.parent.mkdir(exist_ok=True)
    def clean_error(exc):
        text = f"{type(exc).__name__}: {exc}"
        for secret in (settings.api_key, settings.hf_token):
            if secret:
                text = text.replace(secret, "[REDACTED]")
        return text[:1500]
    def save():
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(target)
    try:
        report["available_models"] = asyncio.run(model_ids(settings))
    except Exception as exc:
        report["model_list_error"] = clean_error(exc)
    print(json.dumps({k: v for k, v in report.items() if k != "datasets"}), flush=True)
    save()
    for key in args.datasets:
        started = time.monotonic()
        print(f"Probing {key}...", flush=True)
        try:
            result = probe(key, settings)
        except Exception as exc:
            result = {"dataset_key": key, "ready_for_pilot": False, "error": clean_error(exc)}
        result["seconds"] = round(time.monotonic() - started, 2)
        report["datasets"].append(result)
        save()
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
