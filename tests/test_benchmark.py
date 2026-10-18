import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pl_mmb_benchmark.models import check_cache
from pl_mmb_benchmark.leaderboard import build


class BenchmarkTests(unittest.TestCase):
    def test_reject_windows_cache(self):
        with patch.dict(os.environ, {'HF_HOME': '/mnt/d/models'}):
            with self.assertRaises(RuntimeError):
                check_cache()

    def test_leaderboard_empty_is_valid(self):
        with tempfile.TemporaryDirectory() as d:
            page = build(d)
            self.assertIn('Porównanie modeli', page.read_text(encoding='utf-8'))
            self.assertTrue((Path(d) / 'leaderboard.csv').exists())

    def test_historical_nan_excluded_without_modifying_original(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / 'old-run'
            run.mkdir()
            original = json.dumps({'run_id': 'old-run', 'examples': 1})
            (run / 'result.json').write_text(original)
            (run / 'predictions.jsonl').write_text('{"positive_score": NaN}\n')
            build(d)
            self.assertEqual((run / 'result.json').read_text(), original)
            audit = json.loads((Path(d) / 'validation.json').read_text())
            self.assertEqual(audit['excluded'][0]['run_id'], 'old-run')
            self.assertNotIn('old-run', (Path(d) / 'leaderboard.csv').read_text())

    def test_positive_rank_ties_are_errors(self):
        import numpy as np
        from pl_mmb_benchmark.evaluate import rank
        self.assertEqual(rank(np.array([1., .5, .1])), 1)
        self.assertEqual(rank(np.array([1., 1., .1])), 2)
        self.assertEqual(rank(np.array([.2, .5, .1])), 2)
        for values in ([float('nan'), .5], [.5, float('nan')], [.5, float('inf')]):
            with self.assertRaises(ValueError):
                rank(np.array(values))

    def test_source_export_order_is_not_row_order(self):
        # Exports are ordered by asynchronous completion, so the join must use row_index.
        from pl_mmb_benchmark.data import translation_path
        import json
        path = translation_path('mscoco_t2i')
        if not path.exists():
            self.skipTest('No local translations')
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        self.assertEqual({r['metadata']['row_index'] for r in rows}, set(range(1000)))
        self.assertEqual(len({r['id'] for r in rows}), 1000)
