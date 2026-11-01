import unittest

import torch

from pl_mmb_training.tinyclip_generalization import augment, geometry_loss, specs


class GeneralizationTests(unittest.TestCase):
    def test_anchor_has_zero_value_and_gradient_at_identity(self):
        x = torch.randn(8, 32, requires_grad=True)
        value = geometry_loss(x, x.detach().clone())
        value.backward()
        self.assertEqual(float(value.detach()), 0.)
        self.assertEqual(float(x.grad.abs().max()), 0.)

    def test_augmentation_views_match_across_controls(self):
        # Spatially constant channels preserve semantic colors under all views.
        x = torch.tensor([.1, .4, .8])[None, :, None, None].expand(4, 3, 16, 16).clone()
        a = augment(x, torch.Generator().manual_seed(42))
        b = augment(x, torch.Generator().manual_seed(42))
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(a, x)

    def test_matching_controls_and_full_cosine_tail(self):
        values = list(specs().values())
        self.assertEqual([s['blocks'] for s in values], [0, 0, 2, 2])
        for key in ('seed', 'epochs', 'projector_lr', 'vision_lr', 'warmup_epochs'):
            self.assertEqual(len({s[key] for s in values}), 1)
        self.assertTrue(all(not s['early_stopping'] for s in values))


if __name__ == '__main__':
    unittest.main()
