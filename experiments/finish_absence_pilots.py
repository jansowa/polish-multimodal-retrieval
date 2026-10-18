"""Finish the other pilots while a pathological SugarCrepe source is retried.

Does not stop any process. Per-dataset locks in absence_suite serialize duplicate
attempts and checkpoints skip every already completed example.
"""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    import fcntl
    with (ROOT / "reports/absence-pilot-helper.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for key in ("cirr", "fashioniq", "refcoco", "aro_visual_relation", "aro_visual_attribution"):
            with (ROOT / "reports" / f"absence-pilot-{key}.log").open("a") as log:
                subprocess.run([str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "experiments/absence_suite.py"),
                                "_translate", "--key", key, "--phase", "pilot"], cwd=ROOT,
                               env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
                               stdout=log, stderr=subprocess.STDOUT, check=False)


if __name__ == "__main__":
    main()
