import io
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from experiments.visualnews_images_production import collect_completed, ImageLoader
from experiments.visualnews_image_assets import json_array_items


class VisualNewsContinuationTests(unittest.TestCase):
    def test_completed_union_covers_methods_pilots_and_mmeb_aliases(self):
        canonical = {'MBeirvisualnews:0:1': 'Caption one.', 'MBeirvisualnews:0:3': 'Caption two.',
                     'MBeirvisualnews:0:5': 'Caption three.'}
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name, rows in [('text_only', [('MBeirvisualnews:0:1', {'text': 'Caption one.'})]),
                               ('with_image', [('MBeirvisualnews:0:3', {'text': 'Caption two.'})]),
                               ('pilot', [('MBeirvisualnews:0:1', {'text': 'Caption one.'})]),
                               ('mmeb', [('VisualNews_t2i:7', {'qry_text': 'Retrieve an image of this news caption. Caption three.'})])]:
                path = Path(directory) / (name + '.sqlite3')
                with closing(sqlite3.connect(path)) as db:
                    db.execute('CREATE TABLE completed (unit_id TEXT, source_json TEXT)')
                    db.executemany('INSERT INTO completed VALUES (?,?)', [(uid, json.dumps(source)) for uid, source in rows])
                    db.commit()
                paths.append(path)
            done, _ = collect_completed(paths, canonical)
            self.assertEqual(done, set(canonical))

    def test_same_id_with_different_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.sqlite3'
            with closing(sqlite3.connect(path)) as db:
                db.execute('CREATE TABLE completed (unit_id TEXT, source_json TEXT)')
                db.execute('INSERT INTO completed VALUES (?,?)', ('MBeirvisualnews:0:1', json.dumps({'text': 'Wrong source.'})))
                db.commit()
            with self.assertRaisesRegex(ValueError, 'source mismatch'):
                collect_completed([path], {'MBeirvisualnews:0:1': 'Original caption.'})

    def test_image_loader_checks_identity_across_row_group_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            buffer = io.BytesIO()
            Image.new('RGB', (8, 8), 'red').save(buffer, format='JPEG')
            payload = buffer.getvalue()
            path = Path(directory) / 'images.parquet'
            rows = [dict(did=str(i), img_path=f'images/{i}.jpg', img=dict(bytes=payload, path=f'{i}.jpg')) for i in range(5)]
            pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
            loader = ImageLoader()
            self.assertEqual(loader.get(str(path), 3, '3', 'images/3.jpg'), payload)
            with self.assertRaisesRegex(ValueError, 'verified caption/image join'):
                loader.get(str(path), 3, '4', 'images/4.jpg')
            loader.close()

    def test_original_metadata_reader_handles_large_unicode_object(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metadata.json'
            expected = [{'caption': 'żółć' * 90000, 'image_path': './image.jpg'}, {'caption': 'Second'}]
            path.write_text(json.dumps(expected, ensure_ascii=False), encoding='utf-8')
            self.assertEqual(list(json_array_items(path)), expected)


if __name__ == '__main__':
    unittest.main()
