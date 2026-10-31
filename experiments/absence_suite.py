"""Prepare, pilot and resume complete benchmark translations independently of chat.

WSL: .venv/bin/python -u experiments/absence_suite.py prepare|pilot|full|status
Add --detach to pilot/full. Full runs require a saved pilot assessment.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from contextlib import closing
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pl_mmb_translate.config import Settings
from pl_mmb_translate.models import ImageRef, TranslationUnit
from pl_mmb_translate.registry import DATASETS
from pl_mmb_translate.runner import _translation_fingerprint, run_dataset

PLAN = ROOT / "reports/absence_suite_plan.json"
STATE = ROOT / "reports/absence_suite_state.json"
PILOT_STATE = ROOT / "reports/absence_pilot_state.json"
ASSESSMENT = ROOT / "reports/absence_pilot_assessment.json"
RUN = "qwen3.8-27b-absence-v3-single"
COUNTS = {"sugarcrepe": 7511, "cirr": 1000, "fashioniq": 1000, "refcoco": 1000,
          "aro_visual_relation": 23937, "aro_visual_attribution": 28748}
MIN_FREE = 1536 * 1024**2


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Windows readers (including the monitor) can briefly block rename on /mnt/.
    # Keep the previous complete snapshot until the replacement succeeds.
    for attempt in range(8):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(0.05 * 2**attempt, 0.5))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def choose_ids(groups, size=40):
    """Spread samples across source strata; random sampling within each stratum."""
    rng = random.Random(20261017)
    buckets = defaultdict(list)
    for unit_id, group in groups:
        buckets[group].append(unit_id)
    keys = sorted(buckets)
    rng.shuffle(keys)
    for values in buckets.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < min(size, len(groups)):
        for key in keys:
            if buckets[key] and len(selected) < size:
                selected.append(buckets[key].pop())
    return selected


class PreparedAdapter:
    def __init__(self, job):
        self.job = job

    def iter_units(self, settings, with_image):
        path = ROOT / self.job["source"]
        if digest(path) != self.job["source_sha256"]:
            raise RuntimeError("Prepared source checksum changed")
        if with_image != self.job["with_image"]:
            raise RuntimeError("Image mode differs from the frozen source plan")
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                yield TranslationUnit(
                    unit_id=item["id"], texts=item["texts"],
                    images=[ImageRef(**ref) for ref in item["images"]],
                    metadata=item["metadata"], prompt_profile=item["profile"],
                )


def prepare():
    if PLAN.exists():
        raise RuntimeError("Frozen plan already exists; use pilot/full/status to resume it")
    from datasets import load_dataset
    from pl_mmb_translate.adapters.hf_generic import HFGenericAdapter
    from pl_mmb_translate.adapters.mmeb import MMEB_REVISION

    settings = replace(Settings.from_env(ROOT / ".env"), run_name=RUN, quality_review=False,
                       save_drafts=True, batch_size=4)
    fields = ("model", "base_url", "batch_size", "request_timeout", "max_retries",
              "max_output_tokens", "temperature", "image_max_side", "image_jpeg_quality",
              "max_images_per_record", "max_images_per_request", "quality_review", "save_drafts")
    plan = {"created_at": datetime.now(timezone.utc).isoformat(), "run_name": RUN,
            "settings": {field: getattr(settings, field) for field in fields}, "jobs": []}
    for key, expected in COUNTS.items():
        if shutil.disk_usage(ROOT).free < MIN_FREE:
            raise RuntimeError("Less than 1.5 GiB free on the output volume")
        print(f"Preparing {key} ({expected} records)...", flush=True)
        adapter = DATASETS[key].factory()
        with_image = not isinstance(adapter, HFGenericAdapter)
        destination = ROOT / ".cache/absence-v3" / f"{key}.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_suffix(".tmp")
        groups = []
        revision = getattr(adapter, "revision", MMEB_REVISION)
        with tmp.open("w", encoding="utf-8") as stream:
            if isinstance(adapter, HFGenericAdapter):
                strata = {"sugarcrepe": "Negative_type", "aro_visual_relation": "relation_name",
                          "aro_visual_attribution": "attributes"}[key]
                columns = list(adapter.text_fields) + [strata]
                # Parquet column projection avoids decoding/downloading image columns.
                ds = load_dataset(adapter.path, name=adapter.config, split=adapter.split,
                                  revision=revision, streaming=True, columns=columns,
                                  token=settings.hf_token, cache_dir=str(settings.hf_cache_dir))
                for index, row in enumerate(ds):
                    texts = {field: row[field] for field in adapter.text_fields}
                    if any(not isinstance(text, str) or not text.strip() for text in texts.values()):
                        raise RuntimeError(f"Missing source caption in {key}:{index}")
                    unit_id = f"{adapter.dataset_name}:{index}"
                    group = json.dumps(row.get(strata), ensure_ascii=False, sort_keys=True)
                    item = {"id": unit_id, "texts": texts, "images": [], "profile": adapter.prompt_profile,
                            "metadata": {"dataset": adapter.dataset_name, "hf_path": adapter.path,
                                         "hf_config": adapter.config, "split": adapter.split,
                                         "source_revision": revision, "row_index": index,
                                         "sample_stratum": group}}
                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                    groups.append((unit_id, group))
            else:
                for unit in adapter.iter_units(settings, with_image=True):
                    unit.metadata["source_revision"] = revision
                    item = {"id": unit.unit_id, "texts": unit.texts,
                            "images": [asdict(ref) for ref in unit.images], "profile": unit.prompt_profile,
                            "metadata": unit.metadata}
                    # Check all referenced query/target image paths before the pilot.
                    for ref in unit.images:
                        from pl_mmb_translate.image_utils import _safe_local_path
                        _safe_local_path(ref.value, ref.root)
                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                    groups.append((unit.unit_id, "random"))
        if len(groups) != expected or len({uid for uid, _ in groups}) != expected:
            raise RuntimeError(f"{key}: expected {expected} distinct records, got {len(groups)}")
        tmp.replace(destination)
        job = {"key": key, "expected": expected, "with_image": with_image, "revision": revision,
               "source": str(destination.relative_to(ROOT)), "source_sha256": digest(destination),
               "pilot_ids": choose_ids(groups),
               "translation_config": _translation_fingerprint(settings, with_image=with_image)}
        plan["jobs"].append(job)
        save(ROOT / "reports/absence_suite_preparation.json", plan)
        print(f"Ready: {key}, {expected} records, image={with_image}", flush=True)
    save(PLAN, plan)


def read_plan(allow_preparation=False):
    path = PLAN
    if allow_preparation and not path.exists():
        path = ROOT / "reports/absence_suite_preparation.json"
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_settings(plan, phase):
    settings = replace(Settings.from_env(ROOT / ".env"), **plan["settings"],
                       run_name=plan["run_name"] + ("-pilot" if phase == "pilot" else ""))
    for job in plan["jobs"]:
        if _translation_fingerprint(settings, with_image=job["with_image"]) != job["translation_config"]:
            raise RuntimeError("Translation policy changed since preparation; create a new plan/run")
    if phase == "pilot":
        settings = replace(settings, batch_size=8)
    return settings


def refreeze(run_name):
    """Keep source data but use a new output namespace after a pilot-driven edit."""
    import re
    if not run_name or not re.fullmatch(r"[a-z0-9._-]+", run_name):
        raise RuntimeError("Provide a new lowercase --run-name")
    plan = read_plan()
    settings = replace(Settings.from_env(ROOT / ".env"), **plan["settings"], run_name=run_name)
    if run_name == plan["run_name"] or any((settings.output_dir / name).exists()
                                         for name in (run_name, run_name + "-pilot")):
        raise RuntimeError("The new run name must not already have any outputs")
    save(ROOT / "reports" / (plan["run_name"] + "-plan.json"), plan)
    plan["previous_run_name"] = plan["run_name"]
    plan["run_name"] = run_name
    plan["refined_at"] = datetime.now(timezone.utc).isoformat()
    for job in plan["jobs"]:
        job["translation_config"] = _translation_fingerprint(settings, with_image=job["with_image"])
    save(PLAN, plan)
    print(f"Sources retained; policy frozen for new run {run_name}")


def checkpoint(job, settings):
    return settings.output_dir / settings.run_name / job["key"] / (
        "with_image" if job["with_image"] else "text_only") / "checkpoint.sqlite3"


def _progress_once(job, settings, phase):
    if phase == "pilot":
        wanted = set(job["pilot_ids"])
    else:
        with (ROOT / job["source"]).open(encoding="utf-8") as stream:
            wanted = {json.loads(line)["id"] for line in stream}
    db = checkpoint(job, settings)
    done, failures = set(), {}
    if db.exists():
        with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True, timeout=30)) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "completed" in tables:
                done = {row[0] for row in conn.execute("SELECT unit_id FROM completed")}
            if "failures" in tables:
                failures = dict(conn.execute("SELECT unit_id,error FROM failures"))
    missing, extra = wanted - done, done - wanted
    return {"expected": len(wanted), "completed": len(done), "failed": len(failures),
            "missing_count": len(missing), "missing_ids": sorted(missing), "unexpected_ids": sorted(extra),
            "failures": failures, "complete": not missing and not extra and not failures}


def progress(job, settings, phase):
    for attempt in range(4):
        try:
            return _progress_once(job, settings, phase)
        except sqlite3.OperationalError as exc:
            if attempt == 3:
                return {"expected": len(job["pilot_ids"]) if phase == "pilot" else job["expected"],
                        "completed": None, "failed": None, "missing_count": None, "complete": False,
                        "monitor_error": str(exc)}
            time.sleep(0.5 * 2**attempt)


def translate(key, phase):
    plan = read_plan(allow_preparation=phase == "pilot")
    job = next(job for job in plan["jobs"] if job["key"] == key)
    settings = frozen_settings(plan, phase)
    DATASETS[key] = replace(DATASETS[key], expected_examples=job["expected"],
                            factory=lambda: PreparedAdapter(job))
    import fcntl
    lock_path = ROOT / "reports" / f"absence-{phase}-{key}.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        asyncio.run(run_dataset(key, settings, with_image=job["with_image"],
                                unit_ids=tuple(job["pilot_ids"]) if phase == "pilot" else None))


def assessment_allows(job, plan):
    if not ASSESSMENT.exists():
        return False
    report = json.loads(ASSESSMENT.read_text(encoding="utf-8"))
    approved = report.get("approved", {}).get(job["key"], {})
    return (report.get("run_name") == plan["run_name"]
            and approved.get("source_sha256") == job["source_sha256"]
            and approved.get("translation_config") == job["translation_config"])


def work(phase, wait_for_pilots=False):
    plan = read_plan(allow_preparation=phase == "pilot")
    settings = frozen_settings(plan, phase)
    jobs = plan["jobs"]
    if phase == "full":
        pilot_settings = frozen_settings(plan, "pilot")
        while True:
            jobs = [job for job in plan["jobs"] if assessment_allows(job, plan)
                    and progress(job, pilot_settings, "pilot")["complete"]]
            declined = {}
            if ASSESSMENT.exists():
                report = json.loads(ASSESSMENT.read_text(encoding="utf-8"))
                if report.get("run_name") == plan["run_name"]:
                    declined = report.get("declined", {})
            if not wait_for_pilots or len(jobs) + len(set(declined) & {j["key"] for j in plan["jobs"]}) == len(plan["jobs"]):
                break
            save(STATE, {"phase": "full", "pid": os.getpid(), "stage": "waiting_for_pilot_assessment",
                         "run_name": settings.run_name, "jobs": {}, "approved": [j["key"] for j in jobs],
                         "updated_at": datetime.now(timezone.utc).isoformat()})
            time.sleep(15)
        if not jobs:
            raise RuntimeError("No dataset has a completed and approved pilot")
    state = {"phase": phase, "pid": os.getpid(), "run_name": settings.run_name, "jobs": {},
             "excluded": [job["key"] for job in plan["jobs"] if job not in jobs]}

    def update(stage, **kwargs):
        state.update(stage=stage, updated_at=datetime.now(timezone.utc).isoformat(), **kwargs)
        save(PILOT_STATE if phase == "pilot" else STATE, state)

    for attempt in range(1, 4):
        for job in jobs:
            key = job["key"]
            state["jobs"][key] = progress(job, settings, phase)
            if state["jobs"][key]["complete"]:
                continue
            if shutil.disk_usage(ROOT).free < MIN_FREE:
                update("disk_space_required", current=key)
                return
            update("translating", current=key, attempt=attempt)
            log_path = ROOT / "reports" / f"absence-{phase}-{key}.log"
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
            with log_path.open("a", encoding="utf-8") as log:
                child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                          "_translate", "--key", key, "--phase", phase],
                                         cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                last_count = state["jobs"][key]["completed"]
                last_change = time.monotonic()
                try:
                    while True:
                        try:
                            code = child.wait(timeout=30)
                            break
                        except subprocess.TimeoutExpired:
                            state["jobs"][key] = progress(job, settings, phase)
                            count = state["jobs"][key]["completed"]
                            if count != last_count:
                                last_count, last_change = count, time.monotonic()
                            if shutil.disk_usage(ROOT).free < MIN_FREE or time.monotonic() - last_change > 1800:
                                child.terminate()
                                try:
                                    child.wait(timeout=30)
                                except subprocess.TimeoutExpired:
                                    child.kill()
                                    child.wait()
                                code = child.returncode
                                break
                            update("translating", child_pid=child.pid)
                finally:
                    if child.poll() is None:
                        child.terminate()
                        child.wait()
            state["jobs"][key] = progress(job, settings, phase)
            state["jobs"][key]["last_exit_code"] = code
            update("checking")
        if all(state["jobs"][job["key"]]["complete"] for job in jobs):
            update("complete" if not state["excluded"] else "complete_with_exclusions", current=None)
            return
        if attempt < 3:
            update("retry_wait")
            time.sleep(60 * attempt)
    update("incomplete", current=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "pilot", "full", "status", "refreeze", "_translate"))
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--key")
    parser.add_argument("--phase", choices=("pilot", "full"))
    parser.add_argument("--run-name")
    parser.add_argument("--wait-for-pilots", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == "status":
        if not STATE.exists():
            print("No worker state yet")
            return
        state = json.loads(STATE.read_text(encoding="utf-8"))
        state["jobs"] = {key: {field: value.get(field) for field in
                              ("expected", "completed", "failed", "missing_count", "complete")}
                         for key, value in state.get("jobs", {}).items()}
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return
    if args.command == "_translate":
        translate(args.key, args.phase)
        return
    if args.detach:
        if args.command not in ("pilot", "full"):
            parser.error("--detach is only valid for pilot/full")
        log_path = ROOT / "reports/absence-suite.log"
        with log_path.open("a", encoding="utf-8") as log:
            command = [sys.executable, "-u", str(Path(__file__).resolve()), args.command]
            if args.wait_for_pilots:
                command.append("--wait-for-pilots")
            child = subprocess.Popen(command,
                                     cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"))
        print(f"Detached {args.command} worker PID={child.pid}; log={log_path}")
        return
    import fcntl
    lock_name = {"prepare": "absence-prepare.lock", "pilot": "absence-pilot-worker.lock",
                 "full": "absence-full-worker.lock", "refreeze": "absence-worker.lock"}[args.command]
    lock_path = ROOT / "reports" / lock_name
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            if args.wait_for_pilots and args.command == "full":
                save(ROOT / "reports/absence_full_pending.json", {
                    "pid": os.getpid(), "stage": "waiting_for_pilot_worker",
                    "updated_at": datetime.now(timezone.utc).isoformat()})
                fcntl.flock(lock, fcntl.LOCK_EX)
            else:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another suite worker/preparation is already running")
        if args.command == "prepare":
            prepare()
        elif args.command == "refreeze":
            refreeze(args.run_name)
        else:
            work(args.command, wait_for_pilots=args.wait_for_pilots)


if __name__ == "__main__":
    main()
