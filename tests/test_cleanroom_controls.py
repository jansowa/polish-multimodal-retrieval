import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module_at(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'training/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BudgetAndBlindTests(unittest.TestCase):
    def test_sampler_covers_dataset_before_repeating_even_across_batch_boundaries(self):
        cls = module_at('matched_cleanroom').RepeatedShuffle
        sampler = cls(7, 42)
        consumed = sampler.take(4) + sampler.take(6) + sampler.take(11)
        self.assertEqual(len(consumed), 21)
        for i in range(0, 21, 7):
            self.assertEqual(set(consumed[i:i+7]), set(range(7)))
        self.assertEqual(consumed, cls(7, 42).take(21))

    def test_xm3600_preserves_original_polish_captions_and_refuses_implicit_blind_eval(self):
        module = module_at('xm3600')
        row = {'pl': {'caption': [' Polski podpis ', 'Drugi podpis', 'Polski podpis']}}
        self.assertEqual(module.captions_from_row(row), ['Polski podpis', 'Drugi podpis'])
        with self.assertRaises(ValueError):
            module.captions_from_row({'pl': {'caption': 'wrong schema'}})
        with self.assertRaisesRegex(ValueError, 'FINAL BLIND'):
            module.evaluate(Path('does-not-exist'), Path('does-not-exist'), False)


if __name__ == '__main__':
    unittest.main()
