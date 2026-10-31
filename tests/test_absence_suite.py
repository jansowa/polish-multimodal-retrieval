"""Offline checks for the full-dataset worker's source and completion guarantees."""
from contextlib import closing
import importlib.util
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_quality import settings

spec = importlib.util.spec_from_file_location(
    "absence_suite", Path(__file__).resolve().parents[1] / "experiments/absence_suite.py")
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


class SuiteTests(unittest.TestCase):
    def test_status_save_retries_temporary_windows_file_lock(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "state.json"
            target.write_text('{"stage":"old"}', encoding="utf-8")
            original_replace = Path.replace
            attempts = []

            def replace_after_unlock(source, destination):
                attempts.append(destination)
                if len(attempts) < 3:
                    self.assertEqual(json.loads(target.read_text())["stage"], "old")
                    raise PermissionError("Windows reader holds the status file")
                return original_replace(source, destination)

            with patch.object(Path, "replace", new=replace_after_unlock), patch.object(suite.time, "sleep") as sleep:
                suite.save(target, {"stage": "translating"})
            self.assertEqual(len(attempts), 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(json.loads(target.read_text())["stage"], "translating")

    def test_status_save_does_not_hide_permanent_write_errors(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "state.json"
            target.write_text('{"stage":"old"}', encoding="utf-8")
            with patch.object(Path, "replace", side_effect=PermissionError("denied")) as replace, \
                    patch.object(suite.time, "sleep"):
                with self.assertRaises(PermissionError):
                    suite.save(target, {"stage": "translating"})
            self.assertEqual(replace.call_count, 8)
            self.assertEqual(json.loads(target.read_text())["stage"], "old")

    def make_job(self, root, key="example"):
        source = root / f"{key}.jsonl"
        source.write_text("".join(json.dumps({"id": uid, "texts": {"a": "dog", "b": "cat"},
                                             "images": [], "metadata": {}, "profile": "contrastive"}) + "\n"
                                  for uid in ("row:1", "row:2")), encoding="utf-8")
        return {"key": key, "source": source.name, "source_sha256": suite.digest(source),
                "expected": 2, "with_image": False, "pilot_ids": ["row:1"], "translation_config": {}}

    def test_stratified_sample_is_repeatable_and_covers_every_type(self):
        groups = [(f"{group}:{i}", group) for group in ("add", "swap", "replace") for i in range(30)]
        selected = suite.choose_ids(groups)
        self.assertEqual(selected, suite.choose_ids(groups))
        self.assertEqual(len(set(selected)), 40)
        self.assertEqual({uid.split(":")[0] for uid in selected}, {"add", "swap", "replace"})

    def test_prepared_source_detects_corruption_and_wrong_image_mode(self):
        with TemporaryDirectory() as tmp, patch.object(suite, "ROOT", Path(tmp)):
            job = self.make_job(Path(tmp))
            adapter = suite.PreparedAdapter(job)
            self.assertEqual([u.unit_id for u in adapter.iter_units(None, False)], ["row:1", "row:2"])
            with self.assertRaises(RuntimeError):
                list(adapter.iter_units(None, True))
            (Path(tmp) / job["source"]).write_text("corrupted", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                list(adapter.iter_units(None, False))

    def test_equal_count_with_wrong_ids_is_not_complete(self):
        with TemporaryDirectory() as tmp, patch.object(suite, "ROOT", Path(tmp)):
            root = Path(tmp)
            job = self.make_job(root)
            config = settings(output_dir=root, run_name="run")
            db = suite.checkpoint(job, config)
            db.parent.mkdir(parents=True)
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.execute("CREATE TABLE completed(unit_id TEXT)")
                conn.execute("CREATE TABLE failures(unit_id TEXT,error TEXT)")
                conn.executemany("INSERT INTO completed VALUES(?)", [("row:1",), ("row:999",)])
            result = suite.progress(job, config, "full")
            self.assertFalse(result["complete"])
            self.assertEqual(result["missing_ids"], ["row:2"])
            self.assertEqual(result["unexpected_ids"], ["row:999"])

    def test_approval_is_bound_to_source_and_policy(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = self.make_job(root)
            report = root / "assessment.json"
            suite.save(report, {"run_name": "run", "approved": {job["key"]: job}})
            with patch.object(suite, "ASSESSMENT", report):
                self.assertTrue(suite.assessment_allows(job, {"run_name": "run"}))
                self.assertFalse(suite.assessment_allows(dict(job, source_sha256="changed"), {"run_name": "run"}))
                self.assertFalse(suite.assessment_allows(dict(job, translation_config={"changed": True}), {"run_name": "run"}))

    def test_failed_job_does_not_block_next_job_and_is_retried(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reports").mkdir()
            jobs = [self.make_job(root, key) for key in ("first", "second")]
            plan = {"jobs": jobs, "run_name": "run"}
            calls = []
            done = set()
            def progress(job, config, phase):
                return {"completed": 2 if job["key"] in done else 0, "complete": job["key"] in done}
            def popen(command, **kwargs):
                key = command[command.index("--key") + 1]
                calls.append(key)
                if key == "second" or calls.count(key) > 1:
                    done.add(key)
                return SimpleNamespace(pid=123, wait=lambda timeout=None: 0, poll=lambda: 0)
            with patch.object(suite, "ROOT", root), patch.object(suite, "STATE", root / "state.json"), \
                    patch.object(suite, "PILOT_STATE", root / "state.json"), \
                    patch.object(suite, "read_plan", return_value=plan), \
                    patch.object(suite, "frozen_settings", return_value=settings(run_name="run")), \
                    patch.object(suite, "progress", side_effect=progress), \
                    patch.object(suite.subprocess, "Popen", side_effect=popen), \
                    patch.object(suite.shutil, "disk_usage", return_value=SimpleNamespace(free=10**10)), \
                    patch.object(suite.time, "sleep"):
                suite.work("pilot")
            self.assertEqual(calls, ["first", "second", "first"])
            self.assertEqual(json.loads((root / "state.json").read_text())["stage"], "complete")

    def test_monitor_retries_sqlite_errors_and_does_not_claim_completion(self):
        job = {"pilot_ids": ["row:1"], "expected": 100}
        with patch.object(suite, "_progress_once", side_effect=[sqlite3.OperationalError("busy"), {"complete": True}]), \
                patch.object(suite.time, "sleep"):
            self.assertTrue(suite.progress(job, None, "pilot")["complete"])
        with patch.object(suite, "_progress_once", side_effect=sqlite3.OperationalError("unavailable")), \
                patch.object(suite.time, "sleep"):
            result = suite.progress(job, None, "full")
        self.assertFalse(result["complete"])
        self.assertIsNone(result["completed"])
        self.assertEqual(result["monitor_error"], "unavailable")


if __name__ == "__main__":
    unittest.main()
