"""Wait for the existing repair, then fill its one missing caption if necessary.

Never overwrites a successful translation. Any fallback is explicitly attributed
to Codex, with exact source checking and saved provenance, rather than to Qwen.
"""
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path

import absence_suite as suite
from pl_mmb_translate.checkpoint import CheckpointStore
from pl_mmb_translate.openai_translator import _validate_contrastive, _validate_same_shape

ROOT = suite.ROOT
UNIT_ID = "AROVisualAttribution:19837"
SOURCE = {"true_caption": "the green tree and the weathered fence",
          "false_caption": "the weathered tree and the green fence"}
POLISH = {"true_caption": "zielone drzewo i płot noszący ślady działania warunków atmosferycznych",
          "false_caption": "drzewo noszące ślady działania warunków atmosferycznych i zielony płot"}


def main():
    plan = suite.read_plan()
    settings = suite.frozen_settings(plan, "full")
    job = next(job for job in plan["jobs"] if job["key"] == "aro_visual_attribution")
    with (ROOT / "reports/absence-full-aro_visual_attribution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        store = CheckpointStore(suite.checkpoint(job, settings).parent)
        try:
            if not store.is_completed(UNIT_ID):
                unit = next(unit for unit in suite.PreparedAdapter(job).iter_units(settings, False)
                            if unit.unit_id == UNIT_ID)
                if unit.texts != SOURCE:
                    raise RuntimeError("Unexpected source; no fallback applied")
                _validate_same_shape(SOURCE, POLISH)
                _validate_contrastive(SOURCE, POLISH)
                prior = store.conn.execute("SELECT error,attempts FROM failures WHERE unit_id=?", (UNIT_ID,)).fetchone()
                unit.metadata["translation_method"] = "codex_manual_fallback"
                unit.metadata["translation_audit"] = {"source_compact": SOURCE, "draft_compact": POLISH,
                    "reviewed_compact": None, "review_changed": None, "stages": {},
                    "method": "codex_manual_fallback", "reason": "Repeated empty model responses at the output limit"}
                store.put_completed(unit, POLISH, model="codex-manual-fallback", with_image=False)
                suite.save(ROOT / "reports/aro_manual_fallback.json", {"id": UNIT_ID, "source": SOURCE,
                    "polish": POLISH, "previous_failure": prior, "method": "codex_manual_fallback",
                    "created_at": datetime.now(timezone.utc).isoformat()})
                print("Applied one explicitly attributed fallback; no completed record was replaced", flush=True)
            else:
                print("Existing model repair succeeded; no fallback needed", flush=True)
            store.export_jsonl()
            store.export_review_comparison()
        finally:
            store.close()
    state = json.loads(suite.STATE.read_text(encoding="utf-8"))
    pid = state.get("pid")
    if pid and (Path("/proc") / str(pid) / "cmdline").exists():
        return  # A live controller owns its status file.
    for candidate in plan["jobs"]:
        if candidate["key"] in state.get("jobs", {}):
            state["jobs"][candidate["key"]] = suite.progress(candidate, settings, "full")
    if all(job["complete"] for job in state["jobs"].values()):
        state["stage"] = "complete_with_exclusions" if state.get("excluded") else "complete"
    state["reconciled_at"] = datetime.now(timezone.utc).isoformat()
    suite.save(suite.STATE, state)


if __name__ == "__main__":
    main()
