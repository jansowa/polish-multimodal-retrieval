"""News-caption policy refinement, queued after the existing continuation.

Only this process changes its in-memory profile; running jobs retain their policy.
"""
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import absence_extension as base
from pl_mmb_translate.prompts import PROFILE_HINTS

ROOT = base.ROOT
OLD_PLAN = base.PLAN
OLD_STATE = base.STATE
PLAN = ROOT / "reports/absence_news_plan.json"
STATE = ROOT / "reports/absence_news_state.json"
RUN = "qwen3.8-27b-visualnews-v3c"

PROFILE_HINTS["retrieval"] += (
    " These are news photo captions and headlines, often with missing punctuation. "
    "Preserve personal names and names of groups, performers, organisations and brands. "
    "For example, the name 'Idiots of Ants' stays 'Idiots of Ants', not a literal Polish phrase. "
    "A positional 'left' or 'right' after a person's name describes their position in a photo: "
    "'Henry Kissinger left chats with Elie Wiesel' means 'Henry Kissinger (po lewej) rozmawia z Elie Wieselem', "
    "not that he left a conversation. Use this reading when supported by syntax; do not invent positions. "
    "Weathered means showing wear from weather ('noszący ślady działania warunków atmosferycznych'), "
    "never 'pogodzony'. Preserve corrupted/merged numbers rather than guessing a percentage or date range. "
    "You may restore missing punctuation when the syntax is clear. Check Polish grammar and agreement "
    "in full sentences, including constructions such as 'kilka osób stoi'."
)


def prepare():
    if PLAN.exists():
        raise RuntimeError("Refined news plan already exists; resume pilot/full")
    previous = json.loads(OLD_PLAN.read_text(encoding="utf-8"))
    job = dict(next(job for job in previous["jobs"] if job["key"] == "mbeir_visualnews"))
    settings = replace(base.suite.Settings.from_env(ROOT / ".env"), **previous["settings"],
                       output_dir=Path(previous["output_dir"]), run_name=RUN)
    job["translation_config"] = base.suite._translation_fingerprint(settings, with_image=False)
    plan = {"run_name": RUN, "settings": previous["settings"], "output_dir": previous["output_dir"],
            "jobs": [job], "created_at": datetime.now(timezone.utc).isoformat(),
            "source_plan": str(OLD_PLAN), "news_profile": PROFILE_HINTS["retrieval"]}
    base.suite.save(PLAN, plan)
    print("News policy frozen; all 537568 source captions retained")


if __name__ == "__main__":
    base.PLAN, base.STATE, base.RUN = PLAN, STATE, RUN
    base.PRIMARY_STATE = OLD_STATE
    base.configure()
    base.suite.__file__ = __file__
    base.suite.prepare = prepare
    base.suite.PILOT_STATE = ROOT / "reports/absence_news_pilot_state.json"
    base.suite.ASSESSMENT = ROOT / "reports/absence_news_assessment.json"
    if len(sys.argv) > 1 and sys.argv[1] == "pilot" and "--detach" not in sys.argv:
        import fcntl
        with (ROOT / "reports/absence-news-pilot-worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            base.suite.work("pilot")
    else:
        if len(sys.argv) > 1 and sys.argv[1] == "full" and "--detach" not in sys.argv:
            base.wait_for_primary()
        base.suite.main()
