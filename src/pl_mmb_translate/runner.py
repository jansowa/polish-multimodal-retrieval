from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Iterator

from .checkpoint import CheckpointStore
from .config import Settings
from .models import TranslationUnit
from .openai_translator import CONTRASTIVE_VALIDATION_VERSION, OpenAITranslator
from .registry import DATASETS
from .prompts import (
    PROFILE_HINTS, REVIEW_SYSTEM_PROMPT, REVIEW_TASK_PROMPT, SYSTEM_PROMPT, USER_TASK_PROMPT,
)
from .text_templates import (
    IMAGE_INSTRUCTION_EN, IMAGE_INSTRUCTION_PL, QUERY_PREFIX_EN, QUERY_PREFIX_PL,
    TEMPLATE_VERSION,
    IMAGE_TEMPLATES, QUERY_TEMPLATES,
)


PIPELINE_VERSION = "translation-v3"


def _translation_fingerprint(settings: Settings, *, with_image: bool) -> dict:
    policy = {
        "system": SYSTEM_PROMPT,
        "profiles": PROFILE_HINTS,
        "review": REVIEW_SYSTEM_PROMPT,
        "user_task": USER_TASK_PROMPT,
        "review_task": REVIEW_TASK_PROMPT,
        "templates": [TEMPLATE_VERSION, QUERY_PREFIX_EN, QUERY_PREFIX_PL,
                      IMAGE_INSTRUCTION_EN, IMAGE_INSTRUCTION_PL, QUERY_TEMPLATES, IMAGE_TEMPLATES],
        "contrastive_validation": CONTRASTIVE_VALIDATION_VERSION,
    }
    fingerprint = {
        "pipeline_version": PIPELINE_VERSION,
        "policy_sha256": hashlib.sha256(
            json.dumps(policy, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "temperature": settings.temperature,
        "max_output_tokens": settings.max_output_tokens,
        "quality_review": settings.quality_review,
        "image_settings": {
            "max_side": settings.image_max_side,
            "jpeg_quality": settings.image_jpeg_quality,
            "max_images_per_record": settings.max_images_per_record,
        } if with_image else None,
    }
    # Opt-in instrumentation does not change ordinary v2 runs or their manifests.
    if settings.save_drafts:
        fingerprint["draft_capture"] = "compact-stages-v1"
    return fingerprint


def _write_manifest(
    path: Path, *, dataset_key: str, settings: Settings, with_image: bool,
    unit_ids: tuple[str, ...] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_key": dataset_key,
        "model": settings.model,
        "base_url": settings.base_url,
        # Historical env name retained for compatibility. Semantics are concurrency.
        "batch_size": settings.batch_size,
        "request_semantics": "one_benchmark_example_per_request",
        "with_image": with_image,
        "run_name": settings.run_name,
        "translation_config": _translation_fingerprint(settings, with_image=with_image),
    }
    if unit_ids is not None:
        manifest["selected_unit_ids"] = sorted(unit_ids)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        invariant_keys = ("dataset_key", "model", "base_url", "with_image", "run_name", "translation_config", "selected_unit_ids")
        changed = {
            k: (previous.get(k), manifest.get(k))
            for k in invariant_keys
            if previous.get(k) != manifest.get(k)
        }
        if changed:
            raise RuntimeError(
                f"Existing checkpoint was created with different settings: {changed}. "
                "Use a different RUN_NAME to preserve the old results for comparison."
            )
        previous_semantics = previous.get("request_semantics")
        if previous_semantics != manifest["request_semantics"]:
            raise RuntimeError(
                "This output directory was created by the old multi-example batching pipeline. "
                "The new pipeline uses one benchmark example per request and, for MMEB, different "
                "checkpoint unit IDs. Remove this dataset/mode output directory or set a new "
                "RUN_NAME before resuming."
            )
    else:
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


async def _translate_task(
    translator: OpenAITranslator,
    unit: TranslationUnit,
    *,
    with_image: bool,
) -> tuple[TranslationUnit, dict | None, Exception | None]:
    try:
        translated = await translator.translate_one(unit, with_image=with_image)
        return unit, translated, None
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return unit, None, exc


async def _run_concurrently(
    units: Iterator[TranslationUnit],
    *,
    translator: OpenAITranslator,
    store: CheckpointStore,
    settings: Settings,
    with_image: bool,
    expected_total: int | None,
    completed_before: int,
) -> None:
    """Keep at most settings.batch_size independent one-example requests in flight."""

    concurrency = settings.batch_size
    in_flight: set[asyncio.Task] = set()
    submitted = 0
    finished = 0
    succeeded = 0
    failed_this_run = 0
    started_at = time.monotonic()
    pending_at_start = (
        max(expected_total - completed_before, 0)
        if expected_total is not None
        else None
    )
    last_line_len = 0

    def clear_progress_line() -> None:
        nonlocal last_line_len
        if sys.stdout.isatty() and last_line_len:
            print("\r" + (" " * last_line_len) + "\r", end="", flush=True)
            last_line_len = 0

    def render_progress(*, force_newline: bool = False) -> None:
        nonlocal last_line_len
        elapsed = max(time.monotonic() - started_at, 1e-9)
        rate = finished / elapsed
        completed_total = completed_before + succeeded

        if pending_at_start is not None:
            remaining = max(pending_at_start - finished, 0)
            pct = (finished / pending_at_start * 100.0) if pending_at_start else 100.0
            eta = remaining / rate if rate > 0 else None
            eta_text = _format_duration(eta) if eta is not None else "?"
            line = (
                f"  progress: {finished}/{pending_at_start} ({pct:5.1f}%) | "
                f"completed={completed_total}/{expected_total} | "
                f"failed_this_run={failed_this_run} | in_flight={len(in_flight)} | "
                f"{rate:.2f} ex/s | ETA {eta_text}"
            )
        else:
            line = (
                f"  progress: processed_this_run={finished} | "
                f"completed_total={completed_total} | failed_this_run={failed_this_run} | "
                f"in_flight={len(in_flight)} | {rate:.2f} ex/s"
            )

        if sys.stdout.isatty() and not force_newline:
            padded = line.ljust(last_line_len)
            print("\r" + padded, end="", flush=True)
            last_line_len = len(line)
        else:
            # In redirected logs avoid one line per example; print every 10 or at the end.
            if force_newline or finished == 1 or finished % 10 == 0:
                print(line, flush=True)

    def submit_next() -> bool:
        nonlocal submitted
        try:
            unit = next(units)
        except StopIteration:
            return False
        task = asyncio.create_task(
            _translate_task(translator, unit, with_image=with_image),
            name=f"translate:{unit.unit_id}",
        )
        in_flight.add(task)
        submitted += 1
        return True

    for _ in range(concurrency):
        if not submit_next():
            break

    if pending_at_start is not None:
        print(
            f"  total={expected_total}; already_completed={completed_before}; "
            f"pending_this_run={pending_at_start}"
        )
    if in_flight:
        print(
            f"  concurrency={concurrency}; one benchmark example per request; "
            f"started={len(in_flight)}"
        )
        render_progress()

    while in_flight:
        done, _ = await asyncio.wait(
            in_flight, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            in_flight.remove(task)
            unit, translated, error = await task
            finished += 1

            if error is None and translated is not None:
                store.put_completed(
                    unit,
                    translated,
                    model=settings.model,
                    with_image=with_image,
                )
                succeeded += 1
            else:
                assert error is not None
                store.put_failure(unit.unit_id, f"{type(error).__name__}: {error}")
                failed_this_run += 1
                clear_progress_line()
                print(f"  FAILED {unit.unit_id}: {error}", flush=True)

            # Refill immediately: BATCH_SIZE is max simultaneous connections.
            submit_next()
            render_progress()

    if finished:
        clear_progress_line()
        render_progress(force_newline=True)


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "?"
    seconds_i = int(round(seconds))
    hours, rem = divmod(seconds_i, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


async def run_dataset(
    dataset_key: str,
    settings: Settings,
    *,
    with_image: bool,
    limit: int | None = None,
    unit_ids: tuple[str, ...] | None = None,
) -> Path:
    if limit is not None and unit_ids is not None:
        raise ValueError("Use either limit or unit_ids, not both")
    if unit_ids is not None and (not unit_ids or len(set(unit_ids)) != len(unit_ids)):
        raise ValueError("unit_ids must be nonempty and unique")
    spec = DATASETS[dataset_key]
    mode = "with_image" if with_image else "text_only"
    out_dir = settings.output_dir / settings.run_name / dataset_key / mode
    _write_manifest(
        out_dir / "manifest.json",
        dataset_key=dataset_key,
        settings=settings,
        with_image=with_image,
        unit_ids=unit_ids,
    )

    store = CheckpointStore(out_dir)
    translator = OpenAITranslator(settings)
    adapter = spec.factory()
    print(f"\n=== {dataset_key}: {spec.description} | image={with_image} ===")
    completed_before, failed_before = store.counts()
    expected_total = spec.expected_examples
    if unit_ids is not None:
        expected_total = len(unit_ids)
    if limit is not None:
        expected_total = min(expected_total, limit) if expected_total is not None else limit
    print(f"checkpoint: completed={completed_before}, failures={failed_before}")

    def pending_units() -> Iterator[TranslationUnit]:
        seen = 0
        remaining = set(unit_ids) if unit_ids is not None else None
        for unit in adapter.iter_units(settings, with_image):
            if remaining is not None:
                if unit.unit_id not in remaining:
                    continue
                remaining.remove(unit.unit_id)
            if limit is not None and seen >= limit:
                break
            seen += 1
            if not store.is_completed(unit.unit_id):
                yield unit
            if remaining is not None and not remaining:
                break
        if remaining:
            raise RuntimeError(f"Selected IDs not found in dataset: {sorted(remaining)}")

    try:
        await _run_concurrently(
            iter(pending_units()),
            translator=translator,
            store=store,
            settings=settings,
            with_image=with_image,
            expected_total=expected_total,
            completed_before=completed_before,
        )
    except KeyboardInterrupt:
        print("\nInterrupted. Checkpoint is safe; rerun the same command to resume.")
    except asyncio.CancelledError:
        print("\nCancelled. Checkpoint is safe; rerun the same command to resume.")
        raise
    except Exception:
        print(traceback.format_exc())
        raise
    finally:
        await translator.close()
        path = store.export_jsonl()
        if settings.save_drafts:
            print(f"review comparison: {store.export_review_comparison()}")
        completed, failed = store.counts()
        print(f"export: {path} | completed={completed}, failures={failed}")
        store.close()
    return path
