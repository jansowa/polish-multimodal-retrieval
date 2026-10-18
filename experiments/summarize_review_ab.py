"""Export a live, read-only snapshot of the paired experiment (stdlib only)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sqlite3


def summarize(db_path: Path, sample_path: Path, out_dir: Path) -> dict:
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    diagnostic = set(sample["diagnostic_ids"])
    random_ids = set(sample["random_ids"])
    pairs = []
    with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.execute("BEGIN")
        failures = conn.execute("SELECT unit_id, error FROM failures").fetchall()
        for unit_id, raw in conn.execute("SELECT unit_id, metadata_json FROM completed ORDER BY rowid"):
            metadata = json.loads(raw)
            audit = metadata.get("translation_audit")
            if not audit:
                continue
            pairs.append({
                "id": unit_id,
                "cohort": "diagnostic" if unit_id in diagnostic else "random" if unit_id in random_ids else "other",
                "source": audit["source_compact"],
                "draft": audit["draft_compact"],
                "reviewed": audit["reviewed_compact"],
                "changed": audit["review_changed"],
                "flags": metadata.get("translation_quality", {}).get("flags", []),
                "stages": audit["stages"],
            })
    expected = diagnostic | random_ids
    summary = {
        "expected": len(expected), "completed": len(pairs),
        "missing_ids": sorted(expected - {p["id"] for p in pairs}),
        "failures": failures,
        "cohorts": {}, "stages": {},
    }
    for cohort in ("diagnostic", "random"):
        group = [p for p in pairs if p["cohort"] == cohort]
        summary["cohorts"][cohort] = {
            "completed": len(group), "changed": sum(bool(p["changed"]) for p in group),
            "flagged": sum(bool(p["flags"]) for p in group),
        }
    for stage in ("draft", "review"):
        measurements = [p["stages"][stage] for p in pairs if stage in p["stages"]]
        seconds = sum(m["seconds"] for m in measurements)
        summary["stages"][stage] = {
            "requests": len(measurements),
            "mean_request_seconds": round(seconds / len(measurements), 3) if measurements else None,
            "total_tokens": sum(m["usage"]["total_tokens"] for m in measurements)
                if measurements and all(m["usage"]["total_tokens"] is not None for m in measurements) else None,
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pairs.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "pairs.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "cohort", "source", "draft", "reviewed", "flags", "semantic_outcome", "notes"])
        for p in pairs:
            writer.writerow([p["id"], p["cohort"], json.dumps(p["source"], ensure_ascii=False),
                             json.dumps(p["draft"], ensure_ascii=False), json.dumps(p["reviewed"], ensure_ascii=False),
                             json.dumps(p["flags"], ensure_ascii=False), "", ""])
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", type=Path, nargs="?", default=Path(
        "outputs/qwen3.8-27b-review-ab100/mscoco_t2i/with_image/checkpoint.sqlite3"))
    parser.add_argument("--sample", type=Path, default=Path(__file__).with_name("review_ab_sample.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/review_ab"))
    args = parser.parse_args()
    print(json.dumps(summarize(args.db, args.sample, args.out_dir), ensure_ascii=False, indent=2))
