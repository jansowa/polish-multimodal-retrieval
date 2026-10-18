"""Prevent automatic Windows sleep only while the full translation worker runs."""
import argparse
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--state-file", default="reports/absence_suite_state.json")
    args = parser.parse_args()
    if sys.platform != "win32":
        raise SystemExit("Run this helper with Windows Python")
    root = Path(__file__).resolve().parents[1]
    state = root / args.state_file
    set_state = ctypes.windll.kernel32.SetThreadExecutionState
    set_state.argtypes = [ctypes.c_uint]
    set_state.restype = ctypes.c_uint
    if not set_state(0x80000001):  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED; display may sleep.
        raise ctypes.WinError()
    try:
        if args.check:
            print("Windows sleep-prevention API available; temporary request released")
            return
        started = time.monotonic()
        while True:
            try:
                data = json.loads(state.read_text(encoding="utf-8"))
                if data.get("phase") == "full":
                    if data.get("stage") in {"complete", "complete_with_exclusions", "incomplete", "disk_space_required"}:
                        break
                    age = (datetime.now(timezone.utc) - datetime.fromisoformat(data["updated_at"])).total_seconds()
                    if age > 3600:
                        break
                elif time.monotonic() - started > 3600:
                    break
            except (OSError, ValueError, KeyError):
                if time.monotonic() - started > 300:
                    break
            time.sleep(30)
    finally:
        set_state(0x80000000)  # Restore ordinary automatic sleep behavior.


if __name__ == "__main__":
    main()
