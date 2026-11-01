import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pl_mmb_training.common import file_sha, write_json
from pl_mmb_training.translations import snapshot
from pl_mmb_training.full_data import build_mbeir_links, prepare_full


class FullTrainingDataTests(unittest.TestCase):
    def test_live_checkpoint_snapshot_does_not_grow_after_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path = root / 'translations.sqlite3'
            with sqlite3.connect(db_path) as db:
                db.execute('CREATE TABLE completed(unit_id,source_json,translated_json,metadata_json)')
                db.execute('INSERT INTO completed VALUES(?,?,?,?)', ('one', '{"text":"dog"}', '{"text":"pies"}', '{}'))
            source = dict(dataset='news', checkpoint=str(db_path))
            path, metadata = snapshot(source, root / 'snapshots')
            with sqlite3.connect(db_path) as db:
                db.execute('INSERT INTO completed VALUES(?,?,?,?)', ('two', '{"text":"cat"}', '{"text":"kot"}', '{}'))
            _, again = snapshot(source, root / 'snapshots')
            self.assertEqual(metadata, again)
            self.assertEqual(metadata['completed'], 1)
            self.assertEqual(len(path.read_text().splitlines()), 1)

    def test_mbeir_pairs_use_qrels_instead_of_adjacent_document_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            payloads = {
                'cand_pool/local/mbeir_mscoco_task3_test_cand_pool.jsonl': [dict(did='caption:99', txt='A dog')],
                'query/test/mbeir_mscoco_task3_test.jsonl': [dict(query_img_path='images/COCO_val2014_000000000007.jpg', pos_cand_list=['caption:99'])],
            }
            for index, (key, rows) in enumerate(payloads.items()):
                path = root / f'{index}.jsonl'
                path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                files[key] = dict(path=str(path), sha256=file_sha(path))
            captions, links = build_mbeir_links('mscoco', {'original_files': files})
            self.assertEqual(captions['caption:99'], 'A dog')
            self.assertEqual(links['caption:99'], {'images/COCO_val2014_000000000007.jpg'})

    def test_full_sugarcrepe_uses_filename_for_lineage_and_all_positive_rows(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            from PIL import Image
        except ImportError:
            self.skipTest('Full data tests require the training environment')
        import io
        from pl_mmb_translate.registry import DATASETS
        revision = DATASETS['sugarcrepe'].factory().revision
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, patches = [], []
            for index in range(40):
                payload = io.BytesIO()
                Image.new('RGB', (10, 10), (index * 5, index * 3, index * 2)).save(payload, format='PNG')
                en = dict(caption=f'positive {index}', negative_caption=f'negative {index}')
                rows.append(dict(filename=f'{10000 + index:012d}.jpg', images=dict(bytes=payload.getvalue(), path=None), **en))
                patches.append(dict(id=f'SugarCrepe:{index}', source=en,
                    translated=dict(caption=f'pozytyw {index}', negative_caption=f'negatyw {index}'),
                    metadata=dict(row_index=index, source_revision=revision)))
            translations = root / 'translations.jsonl'
            translations.write_text(''.join(json.dumps(p) + '\n' for p in patches))
            parquet = root / 'source.parquet'
            pq.write_table(pa.Table.from_pylist(rows), parquet)
            cache = root / 'cache'
            cache.mkdir()
            write_json(cache / 'visual_genome_image_data.json', [])
            config = dict(data=dict(sources=[dict(dataset='sugarcrepe', translations=str(translations),
                        shards=['data/train-00000-of-00001.parquet'])], include_ids=None, exclude_ids=[],
                        max_pairs=None, selection_seed=1, validation_fraction=.2, split_seed=1701))
            blacklist = dict(ids=[], texts=[], images={}, sha256='empty')
            with patch('pl_mmb_training.full_data.published_datasets', return_value=set()), \
                    patch('huggingface_hub.hf_hub_download', return_value=str(parquet)):
                records, audit = prepare_full(config, blacklist, cache, root / 'run')
            self.assertEqual(len(records), 40)
            self.assertEqual(audit['safe_before_limit'], 40)
            self.assertIsNone(audit['selection_limit'])
            self.assertEqual(audit['rejected'], 0)
            self.assertTrue(all(r['text_en'].startswith('positive') for r in records))
            self.assertTrue(all(r['parent_ids'][0].startswith('coco:') for r in records))
            train = {p for r in records if r['split'] == 'train' for p in r['parent_ids']}
            val = {p for r in records if r['split'] == 'validation' for p in r['parent_ids']}
            self.assertFalse(train & val)


if __name__ == '__main__':
    unittest.main()
