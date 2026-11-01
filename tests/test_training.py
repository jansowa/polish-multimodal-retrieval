import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pl_mmb_training.common import read_config, write_json
from pl_mmb_training.leakage import image_identity, parent_ids, make_blacklist, LeakageGuard, grouped_split
from pl_mmb_training.evaluation import evaluate_features, scores_table, delta


class LeakageTests(unittest.TestCase):
    def test_blacklist_includes_negatives_and_query_image_and_renamed_pixels(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for i, color in enumerate(('red', 'green', 'blue')):
                p = root / f'{i}.png'
                Image.new('RGB', (8, 9), color).save(p)
                paths.append(str(p))
            e = SimpleNamespace(id='eval:0', texts=['opis ewaluacyjny'], images=paths[:2], query_image=paths[2])
            blacklist = make_blacklist({'one': [e]}, {'eval:0': 'evaluation caption'})
            self.assertEqual(set(blacklist['images']), set(paths))
            guard = LeakageGuard(blacklist, extra_holdout=None)
            for p in paths:
                record = dict(id='train:7', text_en='different', text_pl='inny opis',
                              parent_ids=['vg:7'], lineage_verified=True, **image_identity(p))
                self.assertIn('eval_image_content', guard.reasons(record))
            # Re-encoded files retain the pixel identity even when byte hashes differ.
            png = root / 'renamed.bmp'
            Image.open(paths[1]).save(png)
            self.assertNotEqual(image_identity(png)['sha256'], image_identity(paths[1])['sha256'])
            self.assertEqual(image_identity(png)['pixels_sha256'], image_identity(paths[1])['pixels_sha256'])

    def test_vg_crop_cannot_leak_from_coco_parent(self):
        blacklist = dict(ids=['eval:2'], texts=['eval caption'], images={
            'eval': dict(sha256='a', pixels_sha256='b', parent_ids=['coco:391895'])})
        guard = LeakageGuard(blacklist, extra_holdout=None)
        r = dict(id='aro:4', text_en='door near car', text_pl='drzwi obok auta', sha256='crop',
                 pixels_sha256='cropped-pixels', parent_ids=['vg:99', 'coco:391895'], lineage_verified=True)
        self.assertEqual(guard.reasons(r), ['eval_parent_image'])
        r['parent_ids'] = []
        r['lineage_verified'] = False
        self.assertIn('unknown_parent_lineage', guard.reasons(r))
        r.update(id='eval:2', text_en='  EVAL caption  ')
        self.assertIn('eval_id', guard.reasons(r))
        self.assertIn('eval_text', guard.reasons(r))

    def test_coco_parent_is_preserved_for_crops(self):
        self.assertEqual(parent_ids('/RefCOCO/COCO_train2014_000000000123_5.jpg'), ['coco:123'])
        self.assertEqual(parent_ids('/RefCOCO/000000000123_5.jpg'), ['coco:123'])
        self.assertEqual(parent_ids('/MSCOCO_t2i/COCO_val2014_000000000123.jpg'), ['coco:123'])
        self.assertEqual(parent_ids('/RefCOCO/image_1537681_crop.jpg', {'1537681': 123}), ['coco-ann:1537681', 'coco:123'])
        with self.assertRaises(ValueError):
            parent_ids('/RefCOCO/image_1537681_crop.jpg')

    def test_transitive_parent_and_pixel_aliases_stay_in_one_split(self):
        records = [dict(id=str(i), parent_ids=[f'vg:{i}', f'coco:{i}'], pixels_sha256=f'pixels{i}') for i in range(100)]
        records += [dict(id='alias', parent_ids=['vg:0', 'coco:1000'], pixels_sha256='pixels1000'),
                    dict(id='transitive', parent_ids=['coco:1000'], pixels_sha256='pixels1001'),
                    dict(id='pixel-alias', parent_ids=['vg:999'], pixels_sha256='pixels0')]
        train, val = grouped_split(records, .2, 42)
        train_ids, val_ids = {r['id'] for r in train}, {r['id'] for r in val}
        group = {'0', 'alias', 'transitive', 'pixel-alias'}
        self.assertTrue(group <= train_ids or group <= val_ids)
        self.assertFalse({p for r in train for p in r['parent_ids']} & {p for r in val for p in r['parent_ids']})
        self.assertEqual((train, val), grouped_split(records, .2, 42))


class TrainingNumericsTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch is isolated in the training/benchmark environment')
        self.torch = torch

    def test_evaluator_raw_fusion_ties_and_metrics(self):
        t = self.torch
        examples = [SimpleNamespace(id='q', texts=['caption'], images=['positive', 'negative'], query_image='query')]
        metric, predictions = evaluate_features(examples, ['positive', 'negative', 'query'],
            t.tensor([[1., 0.], [.8, .6], [100., 0.]]), ['caption'], t.tensor([[0., 1.]]))
        self.assertEqual(predictions[0]['rank'], 1)
        self.assertEqual(metric['recall@1'], 1.)
        self.assertEqual(metric['ndcg@10'], 1.)
        _, ties = evaluate_features([SimpleNamespace(id='tie', texts=['caption'], images=['a', 'a'], query_image=None)],
                                    ['a'], t.tensor([[1., 1.]]), ['caption'], t.tensor([[1., 1.]]))
        self.assertEqual(ties[0]['rank'], 1)

    def test_multi_positive_infonce_duplicate_targets(self):
        from pl_mmb_training.runner import multi_positive_infonce
        t = self.torch
        q = t.tensor([[1., 0.], [1., 0.], [0., 1.]], requires_grad=True)
        images = q.detach().clone()
        loss = multi_positive_infonce(q, images, ['same-image', 'same-image', 'other'], temperature=1.)
        logits = q @ images.T
        expected = (t.logsumexp(logits, 1) - t.stack([
            t.logsumexp(logits[0, :2], 0), t.logsumexp(logits[1, :2], 0), logits[2, 2]])).mean()
        self.assertTrue(t.allclose(loss, expected))
        loss.backward()
        self.assertTrue(t.isfinite(q.grad).all())
        all_positive = multi_positive_infonce(q, images, ['one'] * 3)
        self.assertAlmostEqual(float(all_positive.detach()), 0., places=6)
        with self.assertRaises(ValueError):
            multi_positive_infonce(q, images, ['one'] * 3, temperature=0.)

    def test_regression_trains_only_projector_from_detached_cache(self):
        from pl_mmb_training.runner import regression_loss
        t = self.torch
        t.manual_seed(0)
        x = t.randn(32, 4)
        target = x @ t.randn(4, 3)
        projector = t.nn.Linear(4, 3)
        optimizer = t.optim.AdamW(projector.parameters(), lr=.05)
        before = float(regression_loss(projector(x), target).detach())
        for _ in range(60):
            optimizer.zero_grad()
            loss = regression_loss(projector(x), target)
            loss.backward()
            optimizer.step()
        self.assertLess(float(regression_loss(projector(x), target).detach()), before * .1)
        self.assertIsNone(x.grad)
        self.assertIsNone(target.grad)

    def test_interrupted_feature_chunk_is_recomputed(self):
        from pl_mmb_training.features import load_chunk
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'chunk.pt'
            expected = self.torch.tensor([[1., 2.]])
            self.torch.save(expected, path)
            self.assertTrue(self.torch.equal(load_chunk(['x'], path), expected))
            path.write_bytes(b'')
            self.assertIsNone(load_chunk(['x'], path))
            self.torch.save(self.torch.zeros(1, 2), path)
            self.assertIsNone(load_chunk(['x'], path))

    def test_bad_features_are_rejected(self):
        from pl_mmb_training.features import validate_features
        t = self.torch
        for feature in (t.tensor([[float('nan'), 1.]]), t.tensor([[0., 0.]]), t.tensor([1., 2.])):
            with self.assertRaises(ValueError):
                validate_features(['x'], feature)
        e = SimpleNamespace(id='cancel', texts=['text'], images=['image'], query_image='image')
        with self.assertRaises(ValueError):
            evaluate_features([e], ['image'], t.tensor([[1., 0.]]), ['text'], t.tensor([[-1., 0.]]))


class ConfigurationTests(unittest.TestCase):
    def test_average_requires_four_benchmarks(self):
        self.assertIsNone(scores_table({'cirr': {'recall@1': .5}})['AVG'])
        scores = scores_table({k: {'recall@1': .5} for k in ('cirr', 'fashioniq', 'mscoco_t2i', 'refcoco')})
        self.assertEqual(scores['AVG'], .5)
        self.assertEqual(delta(scores, {k: .25 for k in scores})['AVG'], .25)

    def test_no_silent_stage_or_frozen_policy_changes(self):
        from pl_mmb_training.common import ROOT
        config = read_config(ROOT / 'training/configs/e1.json')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            for field, value in (('stage', 'E2'), ('loss', 'infonce'), ('projector', 'mlp')):
                invalid = copy.deepcopy(config)
                invalid[field] = value
                write_json(path, invalid)
                with self.assertRaises(ValueError):
                    read_config(path)
            invalid = copy.deepcopy(config)
            invalid['vision']['frozen'] = False
            write_json(path, invalid)
            with self.assertRaises(ValueError):
                read_config(path)


if __name__ == '__main__':
    unittest.main()
