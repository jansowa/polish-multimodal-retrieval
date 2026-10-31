"""Scheduler boundaries and matched controls, without model/data downloads."""
import unittest

import torch

from pl_mmb_training.tinyclip_optimization import lr_factor, specs


class OptimizationTests(unittest.TestCase):
    def test_warmup_cosine_boundaries_and_monotonicity(self):
        factors = [lr_factor(i, 100, 10, True, .1) for i in range(100)]
        self.assertAlmostEqual(factors[0], .1)
        self.assertAlmostEqual(factors[9], 1.)
        self.assertAlmostEqual(factors[10], 1.)
        self.assertAlmostEqual(factors[-1], .1)
        self.assertTrue(all(a <= b for a, b in zip(factors[:9], factors[1:10])))
        self.assertTrue(all(a >= b for a, b in zip(factors[10:-1], factors[11:])))

    def test_lr_is_applied_to_first_update_in_all_parameter_groups(self):
        head = torch.nn.Parameter(torch.tensor(1.))
        vision = torch.nn.Parameter(torch.tensor(1.))
        optimizer = torch.optim.SGD([dict(params=[head], lr=.1), dict(params=[vision], lr=.001)])
        for step in range(20):
            multiplier = lr_factor(step, 20, 4, True)
            for group, peak in zip(optimizer.param_groups, [.1, .001]):
                group['lr'] = peak * multiplier
            head.grad = vision.grad = torch.tensor(1.)
            before = head.item(), vision.item()
            optimizer.step()
            if step == 0:
                self.assertAlmostEqual(before[0] - head.item(), .025, places=6)
                self.assertAlmostEqual(before[1] - vision.item(), .00025, places=6)
            self.assertAlmostEqual(optimizer.param_groups[0]['lr'] / optimizer.param_groups[1]['lr'], 100.)

    def test_frozen_and_adapted_controls_match_except_vision(self):
        plans = specs()
        for name in ['S2_last1_schedule', 'S3_last1_low_schedule', 'S4_last2_low_schedule']:
            control = dict(plans['S0_frozen_schedule'])
            adapted = dict(plans[name])
            for field in ['blocks', 'vision_lr']:
                control.pop(field)
                adapted.pop(field)
            self.assertEqual(control, adapted)
        for name in ['T1_last1_aligned', 'T2_last2_aligned']:
            control = dict(plans['T0_frozen_aligned'])
            adapted = dict(plans[name])
            for field in ['blocks', 'vision_lr']:
                control.pop(field)
                adapted.pop(field)
            self.assertEqual(control, adapted)

    def test_bad_schedule_is_rejected(self):
        for args in [(-1, 10, 0), (10, 10, 0), (0, 10, 10)]:
            with self.assertRaises(ValueError):
                lr_factor(*args)


if __name__ == '__main__':
    unittest.main()
