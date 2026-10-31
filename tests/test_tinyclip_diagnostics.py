import unittest

import torch

from pl_mmb_training.tinyclip_diagnostics import leave_one_out_centroids, ridge_fit


class DiagnosticTests(unittest.TestCase):
    def test_self_caption_cannot_leak_into_prototype(self):
        # Each image's two captions are opposites. Including self would create ties;
        # removing it makes the other image strictly closer for every query.
        text = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        mask = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.bool)
        result = leave_one_out_centroids(text, mask)
        self.assertEqual(result['recall@1'], 0.)
        self.assertEqual(result['queries'], 4)

    def test_single_caption_images_are_not_self_oracles(self):
        result = leave_one_out_centroids(torch.eye(2), torch.eye(2, dtype=torch.bool))
        self.assertEqual(result['queries'], 0)
        self.assertEqual(result['excluded_queries'], 2)

    def test_ridge_affine_fit_does_not_penalize_intercept(self):
        x = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [2., 1.]], dtype=torch.double)
        y = x @ torch.tensor([[2.], [3.]], dtype=torch.double) + 7
        w, b, _ = ridge_fit(x, y, 0.)
        torch.testing.assert_close(x @ w + b, y)


if __name__ == '__main__':
    unittest.main()
