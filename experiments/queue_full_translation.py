"""Wait for the running review experiment, then translate all MSCOCO t2i rows.

Run with --detach in WSL. No scheduler service or open chat is required.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pl_mmb_translate.config import Settings
from pl_mmb_translate.runner import _write_manifest

EXPERIMENT = "qwen3.8-27b-review-ab100"
FULL_RUN = "qwen3.8-27b-full-v2-single"
STATE = ROOT / "reports/queued_full_translation.json"
LOG = ROOT / "reports/queued_full_translation.log"
LOCK = ROOT / "reports/queued_full_translation.lock"


def counts(path: Path) -> dict:
    if not path.exists():
        return {"completed": 0, "failures": 0}
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)) as conn:
        return {
            "completed": conn.execute("SELECT COUNT(*) FROM completed").fetchone()[0],
            "failures": conn.execute("SELECT COUNT(*) FROM failures").fetchone()[0],
        }


def experiment_pids() -> list[int]:
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
            if not any(arg.endswith(b"/translate.py") or arg == b"translate.py" for arg in args):
                continue
            env = (entry / "environ").read_bytes().split(b"\0")
            if f"RUN_NAME={EXPERIMENT}".encode() in env:
                result.append(int(entry.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return result


def save_state(stage: str, **details) -> None:
    row = {"state": stage, "updated_at": datetime.now(timezone.utc).isoformat(),
           "worker_pid": os.getpid(), "full_run": FULL_RUN, **details}
    tmp = STATE.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(STATE)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def command_env(run_name: str, *, review: bool, drafts: bool) -> dict[str, str]:
    return {**os.environ, "RUN_NAME": run_name, "QUALITY_REVIEW": str(review).lower(),
            "SAVE_DRAFTS": str(drafts).lower(), "TEMPERATURE": "0", "BATCH_SIZE": "4"}


def translate(env: dict[str, str], *, sample: bool = False) -> int:
    args = [sys.executable, "-u", "translate.py", "mscoco_t2i", "true"]
    if sample:
        args += ["--ids-file", "experiments/review_ab_ids.json"]
    return subprocess.run(args, cwd=ROOT, env=env, stdin=subprocess.DEVNULL).returncode


def work(settings: Settings) -> None:
    experiment_db = settings.output_dir / EXPERIMENT / "mscoco_t2i/with_image/checkpoint.sqlite3"
    full_db = settings.output_dir / FULL_RUN / "mscoco_t2i/with_image/checkpoint.sqlite3"
    expected = len(json.loads((ROOT / "experiments/review_ab_ids.json").read_text(encoding="utf-8")))
    # A separate lock is held for the entire queue, including its translation child.
    import fcntl
    with LOCK.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("A queue worker is already active; no duplicate job started.", flush=True)
            return
        deadline = time.monotonic() + 12 * 3600
        while active := experiment_pids():
            save_state("waiting_experiment", experiment_pids=active, experiment=counts(experiment_db))
            if time.monotonic() > deadline:
                raise RuntimeError("Experiment process has not finished after 12 hours")
            time.sleep(30)
        experiment_result = counts(experiment_db)
        if experiment_result["completed"] < expected:
            # If the original terminal/process went away, finish its checkpoint once.
            save_state("resuming_experiment", experiment=experiment_result)
            code = translate(command_env(EXPERIMENT, review=True, drafts=True), sample=True)
            experiment_result = {**counts(experiment_db), "exit_code": code}
        save_state("experiment_finished", experiment=experiment_result)
        # Finishing the experiment is independent of making progress on the full set.
        # A failed sample record must not prevent the requested full run from starting.
        for attempt in range(1, 4):
            result = counts(full_db)
            if result["completed"] >= 1000 and result["failures"] == 0:
                save_state("complete", full=result)
                return
            save_state("running_full", attempt=attempt, full=result, quality_review=False,
                       with_image=True, expected=1000)
            code = translate(command_env(FULL_RUN, review=False, drafts=False))
            result = counts(full_db)
            if result["completed"] >= 1000 and result["failures"] == 0:
                save_state("complete", full=result, exit_code=code)
                return
            save_state("retry_pending", attempt=attempt, full=result, exit_code=code)
            if attempt < 3:
                time.sleep(30)
        save_state("failed", full=result, reason="Three passes exhausted; checkpoint preserved")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--detach", action="store_true")
    mode.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Run this queue using the project's WSL Python interpreter")
    os.chdir(ROOT)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    settings = replace(Settings.from_env(), run_name=FULL_RUN, quality_review=False,
                       save_drafts=False, temperature=0, batch_size=4)
    # Validate/create the exact full-run manifest now, while the user is present.
    _write_manifest(settings.output_dir / FULL_RUN / "mscoco_t2i/with_image/manifest.json",
                    dataset_key="mscoco_t2i", settings=settings, with_image=True)
    if args.detach:
        import fcntl
        with LOCK.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("Queue already active. Status: " + str(STATE))
                return
            fcntl.flock(lock, fcntl.LOCK_UN)
        with LOG.open("a", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), "--worker"],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True,
            )
        print(json.dumps({"started_worker_pid": proc.pid, "waiting_for": experiment_pids(),
                          "full_output": str(settings.output_dir / FULL_RUN),
                          "status": str(STATE), "log": str(LOG)}, ensure_ascii=False))
    else:
        try:
            work(settings)
        except Exception as exc:
            save_state("failed", reason=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
