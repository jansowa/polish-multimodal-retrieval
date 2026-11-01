import copy
import tempfile
import unittest
from pathlib import Path

import torch

from pl_mmb_training.common import ROOT, read_json, write_json
from pl_mmb_training.vision_study import validate_study, dino_seeds, VisionContext


class FrozenVisionStudyTests(unittest.TestCase):
    def test_streamed_snapshot_preserves_candidate_order_and_shared_values(self):
        from pl_mmb_training.evaluation_snapshot import compact_snapshot
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, target = root / 'old.json', root / 'compact.json'
            image_values = ['a.jpg', 'b.jpg']
            text_values = ['opis z {nawiasami} i\\nznakiem', 'inny']
            original = {'one': [{'id': 'q', 'texts': text_values[:1],
                                'images': ['b.jpg', 'a.jpg', 'b.jpg'], 'query_image': 'a.jpg'}],
                        'two': [{'id': 'r', 'texts': text_values[1:], 'images': ['a.jpg'], 'query_image': None}]}
            write_json(source, {'datasets': original, 'provenance': {}})
            for _ in range(2):
                result = compact_snapshot(source, target, image_values, text_values, 's', 'm', {'one': 1, 'two': 1})
                self.assertEqual(vars(result['one'][0]), original['one'][0])
                self.assertEqual(vars(result['two'][0]), original['two'][0])
                self.assertIs(result['one'][0].images[0], image_values[1])

    def test_study_stops_after_four_and_replicates_dino_at_requested_test_threshold(self):
        config = read_json(ROOT / 'training/configs/vision-four.json')
        validate_study(config)
        self.assertEqual(dino_seeds(.379, config), [42])
        self.assertEqual(dino_seeds(.38, config), [42, 43, 44])
        self.assertEqual(dino_seeds(.40, config), [42, 43, 44])
        bad = copy.deepcopy(config)
        bad['backbones'].append('convnext-small')
        with self.assertRaises(ValueError):
            validate_study(bad)

    def test_changing_backbone_rebuilds_training_and_validation_features(self):
        with tempfile.TemporaryDirectory() as folder:
            ctx = VisionContext.__new__(VisionContext)
            ctx.root = Path(folder)
            ctx.reference = ctx.root / 'reference'
            write_json(ctx.reference / 'lock.json', {'test': True})
            ctx.original_base = {'vision': {'model': 'old'}}
            ctx.original_manifest = {'images': ['a', 'b']}
            ctx.original_identity = {'source': {}}
            ctx.study_config = {'test': True}
            ctx.ii, ctx.ti = {'a': 0, 'b': 1}, {'one': 0, 'two': 1}
            ctx.rows = {'train': [{'image': 'a', 'text_pl': 'one'}],
                        'validation': [{'image': 'b', 'text_pl': 'two'}]}
            ctx.images = torch.zeros(2, 5)
            ctx.texts = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
            replacement = torch.arange(14.).reshape(2, 7)
            metadata = {'model_stats': {'model': 'new', 'revision': 'r', 'feature_dim': 7}}
            ctx.activate('dinov2-small', replacement, metadata)
            self.assertTrue(torch.equal(ctx.splits['train'][0], replacement[:1]))
            self.assertTrue(torch.equal(ctx.splits['validation'][0], replacement[1:]))
            self.assertEqual(ctx.base['vision']['model'], 'new')
            self.assertEqual(ctx.projector().in_features, 7)
            self.assertEqual(ctx.projector().out_features, 3)
            self.assertTrue(torch.equal(ctx.splits['train'][1], ctx.texts[:1]))


if __name__ == '__main__':
    unittest.main()
