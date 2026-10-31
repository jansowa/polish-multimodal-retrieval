import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pl_mmb_benchmark.models import check_cache
from pl_mmb_benchmark.leaderboard import build
from pl_mmb_benchmark.leaderboard_view import matrix
from pl_mmb_benchmark.protocol import PROTOCOL, without_instruction


class BenchmarkTests(unittest.TestCase):
    def test_instruction_removal_preserves_caption_colons_and_quotes(self):
        self.assertEqual(without_instruction('mscoco_t2i',
            'Find me an everyday image that matches the given caption: A sign: STOP.',
            'Znajdź obraz z życia codziennego pasujący do podanego opisu: Znak: STOP.'), 'Znak: STOP.')
        self.assertEqual(without_instruction('refcoco',
            '<|image_1|>\nSelect the portion of the image that follows the language expressions. "girl"',
            '<|image_1|>\nWybierz fragment obrazu odpowiadający opisowi. "dziewczynka"'), '"dziewczynka"')
        with self.assertRaises(ValueError):
            without_instruction('cirr', 'Unknown instruction: test', 'Nieznana instrukcja: test')

    def test_raw_fusion_matches_reference_and_not_unit_feature_sum(self):
        import numpy as np
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch is isolated in the benchmark environment')
        from pl_mmb_benchmark.evaluate import cosine_scores
        text, image = torch.tensor([0., 1.]), torch.tensor([100., 0.])
        targets = torch.tensor([[1., 0.], [.8, .6]])
        scores = cosine_scores(text, image, targets)
        q = (text + image).numpy(); t = targets.numpy()
        expected = t @ q / (np.linalg.norm(t, axis=1) * np.linalg.norm(q))
        np.testing.assert_allclose(scores.numpy(), expected, rtol=1e-6)
        self.assertEqual(int(scores.argmax()), 0)
        wrong = cosine_scores(torch.nn.functional.normalize(text, dim=0), torch.nn.functional.normalize(image, dim=0), targets)
        self.assertEqual(int(wrong.argmax()), 1)

    def test_matrix_latest_full_run_and_complete_mean(self):
        def result(model, dataset, score, timestamp='2026-01-01', full=True):
            return dict(model=model, dataset=dataset, metrics={'recall@1': score},
                        timestamp=timestamp, full_dataset=full)
        datasets, rows = matrix([
            result('a', 'one', .9), result('a', 'one', .3, '2026-01-02'),
            result('a', 'two', .5), result('a', 'two', 1., '2026-01-03', full=False),
            result('b', 'one', .8), result('b', 'two', .6), result('c', 'one', 1.),
        ])
        self.assertEqual(datasets, ['one', 'two'])
        self.assertEqual([r['model'] for r in rows], ['b', 'a', 'c'])
        self.assertAlmostEqual(rows[0]['mean'], .7)
        self.assertAlmostEqual(rows[1]['mean'], .4)
        self.assertIsNone(rows[2]['mean'])

    def test_ndcg_from_saved_ranks_preserves_originals(self):
        import json
        import math
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'valid'
            run.mkdir()
            result = dict(run_id='valid', model='publisher/test', dataset='example', examples=3,
                          protocol=PROTOCOL, full_dataset=True, metrics={'recall@1': 1/3})
            original = json.dumps(result)
            (run / 'result.json').write_text(original)
            predictions = ''.join(json.dumps(dict(rank=r, candidates=20, positive_score=.5)) + '\n' for r in [1, 2, 11])
            (run / 'predictions.jsonl').write_text(predictions)
            page = build(directory)
            derived = json.loads((Path(directory) / 'derived_metrics.json').read_text())
            self.assertAlmostEqual(derived['valid']['ndcg@10'], (1 + 1/math.log2(3))/3)
            self.assertEqual((run / 'result.json').read_text(), original)
            self.assertEqual((run / 'predictions.jsonl').read_text(), predictions)
            content = page.read_text(encoding='utf-8')
            self.assertNotIn('batch_size', content)
            self.assertNotIn('seconds', content)
            self.assertEqual(content.count('<table>'), 1)
            self.assertIn('>publisher/test</th>', content)

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

    def test_positive_rank_ties_follow_original_argmax(self):
        import numpy as np
        from pl_mmb_benchmark.evaluate import rank
        self.assertEqual(rank(np.array([1., .5, .1])), 1)
        self.assertEqual(rank(np.array([1., 1., .1])), 1)
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
