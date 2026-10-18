"""Queue behavior without a model server, network calls or actual waiting."""
import importlib.util
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "translation_queue", Path(__file__).resolve().parents[1] / "experiments/queue_full_translation.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class QueueTests(unittest.TestCase):
    def test_read_only_counts(self):
        with TemporaryDirectory() as tmp:
            db = Path(tmp) / "checkpoint.sqlite3"
            self.assertEqual(queue.counts(db), {"completed": 0, "failures": 0})
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.execute("CREATE TABLE completed (id TEXT)")
                conn.execute("CREATE TABLE failures (id TEXT)")
                conn.execute("INSERT INTO completed VALUES ('x')")
            self.assertEqual(queue.counts(db), {"completed": 1, "failures": 0})

    @unittest.skipUnless(sys.platform == "linux", "Queue worker runs in WSL/Linux")
    def test_wait_resume_and_full_retry(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "experiments").mkdir()
            (root / "experiments/review_ab_ids.json").write_text(json.dumps(["x", "y"]))
            progress = {"experiment": 1, "full": 0}
            calls, states = [], []
            def counts(path):
                is_sample = queue.EXPERIMENT in str(path)
                return {"completed": progress["experiment" if is_sample else "full"], "failures": 0}
            def translate(env, *, sample=False):
                calls.append((env, sample))
                if sample:
                    progress["experiment"] = 2
                elif sum(not flag for _, flag in calls) == 2:
                    progress["full"] = 1000
                return 0
            with patch.object(queue, "ROOT", root), patch.object(queue, "LOCK", root / "lock"), \
                    patch.object(queue, "experiment_pids", side_effect=[[123], []]), \
                    patch.object(queue, "counts", counts), patch.object(queue, "translate", translate), \
                    patch.object(queue, "save_state", side_effect=lambda stage, **kw: states.append(stage)), \
                    patch.object(queue.time, "sleep"):
                queue.work(SimpleNamespace(output_dir=root / "outputs"))
            self.assertEqual([flag for _, flag in calls], [True, False, False])
            self.assertEqual(calls[0][0]["QUALITY_REVIEW"], "true")
            self.assertEqual(calls[-1][0]["QUALITY_REVIEW"], "false")
            self.assertEqual(calls[-1][0]["SAVE_DRAFTS"], "false")
            self.assertEqual(calls[-1][0]["RUN_NAME"], queue.FULL_RUN)
            self.assertEqual(states[0], "waiting_experiment")
            self.assertIn("retry_pending", states)
            self.assertEqual(states[-1], "complete")

    @unittest.skipUnless(sys.platform == "linux", "Queue worker runs in WSL/Linux")
    def test_completed_full_run_is_not_repeated(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "experiments").mkdir()
            (root / "experiments/review_ab_ids.json").write_text('["x"]')
            def counts(path):
                return {"completed": 1 if queue.EXPERIMENT in str(path) else 1000, "failures": 0}
            with patch.object(queue, "ROOT", root), patch.object(queue, "LOCK", root / "lock"), \
                    patch.object(queue, "experiment_pids", return_value=[]), \
                    patch.object(queue, "counts", counts), patch.object(queue, "translate") as translate, \
                    patch.object(queue, "save_state") as save:
                queue.work(SimpleNamespace(output_dir=root / "outputs"))
            translate.assert_not_called()
            self.assertEqual(save.call_args.args[0], "complete")
