import copy
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from pl_mmb_training.losses import PositivePairs, PairLoss, symmetric_infonce, sigmoid_loss
from pl_mmb_training.retrieval import make_graph, evaluate_graph, best_positive_ranks
from pl_mmb_training.ablation import decide, validate_plan
from pl_mmb_training.common import ROOT, read_json


class PairLossTests(unittest.TestCase):
    def test_single_positive_infonce_matches_symmetric_cross_entropy(self):
        logits = torch.tensor([[2., .1, -.3], [.2, 1., -.2], [.3, -.1, 3.]], requires_grad=True)
        mask = torch.eye(3, dtype=torch.bool)
        expected = .5 * (F.cross_entropy(logits, torch.arange(3)) + F.cross_entropy(logits.T, torch.arange(3)))
        self.assertTrue(torch.allclose(symmetric_infonce(logits, mask), expected))
        symmetric_infonce(logits, mask).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_all_repeated_targets_and_off_batch_caption_associations_are_positive(self):
        records = [dict(target_id='a', text_pl='one'), dict(target_id='a', text_pl='two'),
                   dict(target_id='b', text_pl='one'), dict(target_id='b', text_pl='other')]
        positives = PositivePairs(records)
        # Caption 'one' also belongs to image b via a record outside this batch.
        mask = positives.mask([0, 1, 3])
        self.assertEqual(mask.tolist(), [[True, True, True], [True, True, False], [False, False, True]])
        self.assertAlmostEqual(float(symmetric_infonce(torch.ones(3, 3), torch.ones(3, 3, dtype=torch.bool))), 0.)

    def test_sigmoid_uses_pair_labels_and_batch_normalization(self):
        logits = torch.tensor([[20., -20.], [20., 20.]], requires_grad=True)
        mask = torch.tensor([[True, False], [True, True]])
        self.assertLess(float(sigmoid_loss(logits, mask).detach()), 1e-7)
        zero_loss = sigmoid_loss(torch.zeros(2, 2), mask)
        self.assertAlmostEqual(float(zero_loss), 2 * float(F.softplus(torch.tensor(0.))), places=6)
        sigmoid_loss(logits, mask).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_temperature_bias_and_projector_receive_gradients_while_features_stay_frozen(self):
        for name in ('symmetric_infonce', 'siglip_sigmoid'):
            torch.manual_seed(9)
            vision = torch.randn(4, 5)
            text = torch.randn(4, 3)
            model = torch.nn.Linear(5, 3)
            criterion = PairLoss(name)
            loss = criterion(model(vision), text, torch.eye(4, dtype=torch.bool))
            loss.backward()
            self.assertTrue(torch.isfinite(criterion.logit_scale.grad))
            self.assertGreater(float(model.weight.grad.norm()), 0)
            if criterion.logit_bias is not None:
                self.assertTrue(torch.isfinite(criterion.logit_bias.grad))
            self.assertIsNone(vision.grad)
            self.assertIsNone(text.grad)


class BidirectionalRetrievalTests(unittest.TestCase):
    def test_alias_images_multiple_captions_and_negative_only_gallery_images(self):
        pairs = [dict(image='a', image_id='A', text='one'), dict(image='alias_a', image_id='A', text='two'),
                 dict(image='b', image_id='B', text='three')]
        graph = make_graph(pairs, ['a', 'alias_a', 'b', 'negative'], ['one', 'two', 'three'],
                           [dict(image='negative', image_id='C')])
        images = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [-1., -1.]])
        texts = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        metrics = evaluate_graph(graph, images, texts)
        self.assertEqual(metrics['positive_pairs'], 3)
        self.assertEqual(metrics['t2i']['candidates'], 3)
        self.assertEqual(metrics['i2t']['queries'], 2)
        self.assertEqual(metrics['i2t']['candidates'], 3)
        self.assertEqual(metrics['mean_r1'], 1.)

    def test_best_of_multiple_positives_and_ties_follow_gallery_order(self):
        ranks = best_positive_ranks(torch.tensor([[1., 1., 1.], [3., 2., 1.]]),
                                    torch.tensor([[False, True, True], [False, True, True]]))
        self.assertEqual(ranks.tolist(), [2, 2])
        with self.assertRaises(ValueError):
            best_positive_ranks(torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.bool))


class AblationPolicyTests(unittest.TestCase):
    def test_replication_uses_validation_and_does_not_read_test(self):
        p0 = dict(runs=[dict(metadata=dict(seed=42), validation={'t2i': {'recall@1': .4}})])
        scouts = {'E2a': {'selected_validation': {'t2i': {'recall@1': .41}}, 'benchmark': {'recall@1': 0.}},
                  'E2b': {'selected_validation': {'t2i': {'recall@1': .39}}, 'benchmark': {'recall@1': 1.}}}
        result = decide(scouts, p0, dict(scout_seed=42, repeat_gain=.01))
        self.assertEqual(result['scout_repeat'], ['E2a'])

    def test_loss_ablation_rejects_changed_budget(self):
        config = read_json(ROOT / 'training/configs/p0-e2.json')
        base = read_json(ROOT / 'training/configs/e1.json')
        validate_plan(config, base)
        changed = copy.deepcopy(config)
        changed['optimization']['batch_size'] *= 2
        with self.assertRaises(ValueError):
            validate_plan(changed, base)


if __name__ == '__main__':
    unittest.main()
