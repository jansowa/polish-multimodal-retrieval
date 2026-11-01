import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from retrieval_benchmark import requested_queue as queue


class RequestedQueueTests(unittest.TestCase):
    def test_completed_any_profile_is_not_scheduled_again(self):
        rows = [dict(model_id='a', dataset_id='x', prompt_mode='v2_no_instructions', profile='old'),
                dict(model_id='a', dataset_id='x', prompt_mode='v2_no_instructions', profile='new'),
                dict(model_id='b', dataset_id='y', prompt_mode='author_recommended')]
        self.assertEqual(queue.completed_cells(dict(results=rows), 'a'), {('x','v2_no_instructions')})
        self.assertEqual(queue.completed_cells(dict(results=rows), 'c'), set())

    def test_only_one_weight_format_is_downloaded(self):
        files = [dict(name=n, bytes=1, lfs=dict(sha256='a'*64)) for n in
                 ['model.safetensors', 'open_clip_model.safetensors', 'pytorch_model.bin', 'config.json']]
        spec = queue.component(dict(repository='laion/example', revision='a'*40, files=files))
        self.assertEqual([f['name'] for f in spec['weights']], ['open_clip_model.safetensors'])
        self.assertNotIn('model.safetensors', spec['download_files'])

    def test_cleanup_only_registered_private_weight_not_results(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            cache = home/'.cache/pl_mmb_requested_20261101/hub'
            payload = b'test weight'
            digest = hashlib.sha256(payload).hexdigest()
            blob = cache/'models--a--b/blobs'/digest
            blob.parent.mkdir(parents=True)
            blob.write_bytes(payload)
            keep = blob.with_name('unregistered')
            keep.write_text('retain')
            job = dict(repository='a/b', alias='test', components=[dict(repository='a/b',
                weights=[dict(name='model.safetensors', bytes=len(payload), lfs=dict(sha256=digest))])])
            with patch.object(queue, 'CACHE', cache), patch.object(queue, 'RUN', home/'runs'), patch.object(Path, 'home', return_value=home):
                queue.clean(job)
            self.assertFalse(blob.exists())
            self.assertTrue(keep.exists())

    def test_cleanup_refuses_modified_weight(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            cache = home/'.cache/pl_mmb_requested_20261101/hub'
            digest = 'a'*64
            blob = cache/'models--a--b/blobs'/digest
            blob.parent.mkdir(parents=True)
            blob.write_bytes(b'changed')
            job = dict(repository='a/b', alias='test', components=[dict(repository='a/b',
                weights=[dict(name='model.safetensors', bytes=7, lfs=dict(sha256=digest))])])
            with patch.object(queue, 'CACHE', cache), patch.object(Path, 'home', return_value=home):
                with self.assertRaisesRegex(ValueError, 'SHA/size'):
                    queue.clean(job)
            self.assertTrue(blob.exists())


if __name__ == '__main__':
    unittest.main()
