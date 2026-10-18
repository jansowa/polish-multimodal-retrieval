"""Continue after the active suite with complete M-BEIR caption pools.

Reuses the tested worker in a separate process, with separate plans/checkpoints.
Never stops or edits the active suite. Full mode waits for its completion.
"""
from __future__ import annotations
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
from types import SimpleNamespace

import absence_suite as suite

ROOT = suite.ROOT
PRIMARY_PLAN = suite.PLAN
PRIMARY_STATE = suite.STATE
PLAN = ROOT / "reports/absence_extension_plan.json"
STATE = ROOT / "reports/absence_extension_state.json"
RUN = "qwen3.8-27b-absence-extension-v3b"
REVISION = "023f7cc5425e15e3c93f6146f7295c28eae767dc"
OUTPUT = Path("/mnt/c/Users/jsowa/benchmark-translations")


def prepare():
    from huggingface_hub import hf_hub_download
    from pl_mmb_translate.runner import _write_manifest
    if PLAN.exists():
        raise RuntimeError("Extension plan already exists; resume pilot/full")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(OUTPUT).free < 8 * 1024**3:
        raise RuntimeError("At least 8 GiB free required on C: for extension outputs")
    original = json.loads(PRIMARY_PLAN.read_text(encoding="utf-8"))
    plan = {"run_name": RUN, "settings": original["settings"], "output_dir": str(OUTPUT),
            "created_at": datetime.now(timezone.utc).isoformat(), "jobs": [], "original_files": {},
            "scope": "Complete M-BEIR MSCOCO test and VisualNews retrieval variants; captions translated once for both directions"}
    settings = replace(suite.Settings.from_env(ROOT / ".env"), **plan["settings"],
                       output_dir=OUTPUT, run_name=RUN)
    sugar = dict(next(job for job in original["jobs"] if job["key"] == "sugarcrepe"))
    sugar["key"] = "sugarcrepe_continuation"
    pilot_db = settings.output_dir  # actual original pilot remains on D:
    original_settings = replace(settings, output_dir=ROOT / "outputs", run_name=original["run_name"] + "-pilot")
    original_job = dict(sugar, key="sugarcrepe")
    pilot_db = suite.checkpoint(original_job, original_settings)
    with closing(sqlite3.connect(pilot_db.as_uri() + "?mode=ro", uri=True)) as conn:
        sugar["pilot_ids"] = [row[0] for row in conn.execute("SELECT unit_id FROM completed ORDER BY unit_id")]
        if len(sugar["pilot_ids"]) < 30:
            raise RuntimeError("Insufficient completed SugarCrepe pilot for reuse")
        pilot_settings = replace(settings, run_name=RUN + "-pilot")
        destination = suite.checkpoint(sugar, pilot_settings)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(destination)) as target:
            conn.backup(target)
            # The copied pilot comprises the successfully inspected rows only.
            # The full prepared source STILL contains every row, including 1821.
            target.execute("DELETE FROM failures")
            marks = ",".join("?" for _ in sugar["pilot_ids"])
            target.execute(f"DELETE FROM completed WHERE unit_id NOT IN ({marks})", sugar["pilot_ids"])
            target.commit()
    _write_manifest(destination.parent / "manifest.json", dataset_key=sugar["key"],
                    settings=pilot_settings, with_image=False, unit_ids=tuple(sugar["pilot_ids"]))
    sugar["pilot_reuse"] = {"from_run": original["run_name"] + "-pilot",
                            "source_issue": "1821: malformed English and repeated server HTTP 504; retained in full source"}
    plan["jobs"].append(sugar)
    for name in ("mscoco", "visualnews"):
        files = {}
        for role, filename in (
            ("queries_t2i", f"query/test/mbeir_{name}_task0_test.jsonl"),
            ("queries_i2t", f"query/test/mbeir_{name}_task3_test.jsonl"),
            ("images", f"cand_pool/local/mbeir_{name}_task0_{'test_' if name == 'mscoco' else ''}cand_pool.jsonl"),
            ("captions", f"cand_pool/local/mbeir_{name}_task3_{'test_' if name == 'mscoco' else ''}cand_pool.jsonl"),
        ):
            print(f"Downloading {filename}", flush=True)
            path = Path(hf_hub_download("TIGER-Lab/M-BEIR", filename, repo_type="dataset", revision=REVISION,
                                        cache_dir=str(ROOT / ".cache/huggingface")))
            files[role] = path
            plan["original_files"][filename] = {"path": str(path), "sha256": suite.digest(path)}
        queries = {}
        query_count = 0
        with files["queries_t2i"].open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                text = row["query_txt"]
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError(f"{name}: missing text in t2i query {row['qid']}")
                queries.setdefault(text, []).append(str(row["qid"]))
                query_count += 1
        source = ROOT / ".cache/absence-v3" / f"mbeir_{name}.jsonl"
        groups, ids, matched = [], set(), set()
        with source.open("w", encoding="utf-8") as out, files["captions"].open(encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                row = json.loads(line)
                text, did = row["txt"], str(row["did"])
                if not isinstance(text, str) or not text.strip() or did in ids:
                    raise RuntimeError(f"{name}: empty caption or duplicate did {did}")
                ids.add(did)
                matched.add(text)
                uid = f"MBeir{name}:{did}"
                item = {"id": uid, "texts": {"text": text}, "images": [], "profile": "retrieval",
                        "metadata": {"dataset": f"M-BEIR/{name}", "hf_path": "TIGER-Lab/M-BEIR",
                                     "source_revision": REVISION, "corpus_id": did, "row_index": index,
                                     "source_file": str(files["captions"]), "t2i_query_ids": queries.get(text, [])}}
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                groups.append((uid, "long" if len(text) > 180 else "short"))
            # Preserve the complete t2i variant even if its query text differs from its paired caption.
            missing_queries = set(queries) - matched
            for text in sorted(missing_queries):
                qids = queries[text]
                uid = f"MBeir{name}:query:{qids[0]}"
                out.write(json.dumps({"id": uid, "texts": {"text": text}, "images": [], "profile": "retrieval",
                                      "metadata": {"dataset": f"M-BEIR/{name}", "hf_path": "TIGER-Lab/M-BEIR",
                                                   "source_revision": REVISION, "t2i_query_ids": qids,
                                                   "source_file": str(files["queries_t2i"])}}, ensure_ascii=False) + "\n")
                groups.append((uid, "unmatched_query"))
        job = {"key": f"mbeir_{name}", "source": str(source.relative_to(ROOT)), "source_sha256": suite.digest(source),
               "expected": len(groups), "with_image": False, "revision": REVISION,
               "pilot_ids": suite.choose_ids(groups), "t2i_query_count": query_count,
               "i2t_caption_count": len(ids), "additional_query_texts": len(missing_queries),
               "translation_config": suite._translation_fingerprint(settings, with_image=False)}
        plan["jobs"].append(job)
        suite.save(ROOT / "reports/absence_extension_preparation.json", plan)
        print(f"Prepared {name}: {len(ids)} captions, {query_count} t2i queries, {len(missing_queries)} extra texts", flush=True)
    suite.save(PLAN, plan)
    print("Extension sources ready; no active worker was stopped", flush=True)


def configure():
    from pl_mmb_translate.registry import DatasetSpec
    suite.PLAN = PLAN
    suite.STATE = STATE
    suite.PILOT_STATE = ROOT / "reports/absence_extension_pilot_state.json"
    suite.ASSESSMENT = ROOT / "reports/absence_extension_assessment.json"
    # Children must execute this wrapper and configure their own module state.
    suite.__file__ = __file__
    suite.prepare = prepare
    if PLAN.exists():
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        os.environ["OUTPUT_DIR"] = plan["output_dir"]
        for job in plan["jobs"]:
            suite.DATASETS[job["key"]] = DatasetSpec(job["key"], job["key"],
                                                    lambda j=job: suite.PreparedAdapter(j), job["expected"])
        real_disk_usage = shutil.disk_usage
        suite.shutil = SimpleNamespace(disk_usage=lambda _: real_disk_usage(plan["output_dir"]))
        suite.MIN_FREE = 8 * 1024**3


def wait_for_primary():
    while True:
        try:
            primary = json.loads(PRIMARY_STATE.read_text(encoding="utf-8"))
            pid = primary.get("pid")
            if primary.get("stage") in {"complete", "complete_with_exclusions", "incomplete", "disk_space_required"}:
                return
            if pid and not (Path("/proc") / str(pid) / "cmdline").exists():
                return
        except (OSError, ValueError):
            pass
        suite.save(STATE, {"phase": "full", "stage": "waiting_for_primary", "pid": os.getpid(),
                           "updated_at": datetime.now(timezone.utc).isoformat(), "jobs": {}, "run_name": RUN})
        time.sleep(30)


if __name__ == "__main__":
    configure()
    if len(sys.argv) > 1 and sys.argv[1] == "pilot" and "--detach" not in sys.argv:
        import fcntl
        with (ROOT / "reports/absence-extension-pilot-worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            suite.work("pilot")
    else:
        if len(sys.argv) > 1 and sys.argv[1] == "full" and "--detach" not in sys.argv:
            wait_for_primary()
        suite.main()
