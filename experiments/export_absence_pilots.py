"""Export compact EN/PL pairs from the current suite's live pilot checkpoints."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--print-pairs", action="store_true")
    parser.add_argument("--plan", default="reports/absence_suite_plan.json")
    parser.add_argument("--report-prefix", default="absence_pilot")
    args = parser.parse_args()
    plan = json.loads((ROOT / args.plan).read_text(encoding="utf-8"))
    output_dir = Path(plan.get("output_dir", str(ROOT / "outputs")))
    rows, summary = [], {}
    for job in plan["jobs"]:
        mode = "with_image" if job["with_image"] else "text_only"
        db = output_dir / (plan["run_name"] + "-pilot") / job["key"] / mode / "checkpoint.sqlite3"
        if not db.exists():
            summary[job["key"]] = {"completed": 0, "expected": len(job["pilot_ids"])}
            continue
        with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True, timeout=30)) as conn:
            entries = list(conn.execute("SELECT unit_id,metadata_json FROM completed ORDER BY rowid"))
            failures = dict(conn.execute("SELECT unit_id,error FROM failures"))
        summary[job["key"]] = {"completed": len(entries), "expected": len(job["pilot_ids"]), "failures": failures}
        for uid, metadata in entries:
            metadata = json.loads(metadata)
            audit = metadata.get("translation_audit", {})
            row = {"dataset": job["key"], "id": uid, "en": audit.get("source_compact"),
                   "pl": audit.get("draft_compact"), "stratum": metadata.get("sample_stratum"),
                   "stages": audit.get("stages")}
            rows.append(row)
    target = ROOT / "reports" / (args.report_prefix + "_pairs.jsonl")
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (ROOT / "reports" / (args.report_prefix + "_summary.json")).write_text(
        json.dumps({"run_name": plan["run_name"], "datasets": summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    if args.print_pairs:
        for row in rows:
            if args.dataset is None or row["dataset"] == args.dataset:
                print(json.dumps({k: row[k] for k in ("id", "en", "pl")}, ensure_ascii=False))
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
