import copy
import tempfile
import unittest
from pathlib import Path

from pl_mmb_training.cleanroom import select_variant, counts, phash
from pl_mmb_training.cleanroom_training import checkpoint_key


class CleanroomTests(unittest.TestCase):
    def test_non_coco_follows_transitive_parent_and_pixel_aliases(self):
        rows = [dict(id='1', dataset='mbeir_mscoco', target_id='A', text_pl='one', parent_ids=['coco:1']),
                dict(id='2', dataset='aro_visual_relation', target_id='A', text_pl='two', parent_ids=['vg:2']),
                dict(id='3', dataset='aro_visual_attribution', target_id='B', text_pl='three', parent_ids=['vg:2', 'flickr:3']),
                dict(id='4', dataset='aro_visual_relation', target_id='C', text_pl='four', parent_ids=['flickr:3']),
                dict(id='5', dataset='aro_visual_relation', target_id='D', text_pl='five', parent_ids=['vg:5'])]
        before = copy.deepcopy(rows)
        self.assertEqual([r['id'] for r in select_variant(rows, 'NON_COCO_CLEANROOM')], ['5'])
        self.assertEqual(len(select_variant(rows, 'FULL_CLEAN')), 5)
        self.assertEqual(len(select_variant(rows, 'NO_MBEIR_MSCOCO')), 4)
        self.assertEqual(rows, before)

    def test_nested_variants_and_duplicate_counts(self):
        rows = [dict(dataset='sugarcrepe', target_id='s', text_pl='x', parent_ids=['coco:2']),
                dict(dataset='aro_visual_relation', target_id='a', text_pl='y', parent_ids=['vg:1']),
                dict(dataset='aro_visual_relation', target_id='a', text_pl='y', parent_ids=['vg:1'])]
        self.assertEqual(counts(rows)['unique_image_text_pairs'], 2)
        self.assertEqual(len(select_variant(rows, 'NON_COCO_CLEANROOM')), 2)
        with self.assertRaises(ValueError):
            select_variant(rows, 'typo')

    def test_checkpoint_uses_t2i_then_bidirectional_then_earliest_not_loss(self):
        def row(epoch, t2i, mean, loss):
            return dict(epoch=epoch, validation_loss=loss,
                        validation={'t2i': {'recall@1': t2i}, 'mean_r1': mean})
        rows = [row(1, .5, .8, 0.), row(2, .6, .7, 9.), row(3, .6, .75, 10.), row(4, .6, .75, 1.)]
        self.assertEqual(max(rows, key=checkpoint_key)['epoch'], 3)

    def test_phash_detects_reencoding_and_resize_of_structured_image(self):
        from PIL import Image, ImageDraw
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / 'original.png', Path(temp) / 'resized.jpg'
            image = Image.new('RGB', (400, 300), 'white')
            draw = ImageDraw.Draw(image)
            draw.rectangle((15, 20, 140, 170), fill='navy')
            draw.ellipse((170, 60, 350, 270), fill='red')
            image.save(a)
            image.resize((200, 150)).save(b, quality=92)
            self.assertLessEqual((phash(a) ^ phash(b)).bit_count(), 6)


if __name__ == '__main__':
    unittest.main()
