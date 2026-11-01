"""Shared-cache cleanup must preserve weights referenced by other repos."""
import importlib
import os
from pathlib import Path
import sys
import tempfile
import unittest


def snapshot(repo, target):
    link = repo / 'snapshots' / 'revision' / 'model.safetensors'
    link.parent.mkdir(parents=True)
    link.symlink_to(target)
    return link


@unittest.skipUnless(os.name == 'posix', 'The worker uses Linux HF cache paths')
class SharedWeightOwnership(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
        self.cleanup_module = importlib.import_module('recover_series_reference_execution')
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.hub = Path(scratch.name) / 'hub'
        self.target = self.hub / 'blobs' / 'weight'
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b'weight')
        self.owner = self.hub / 'models--google--embeddinggemma-2'
        self.link = snapshot(self.owner, self.target)

    def test_exclusive_weight_is_accepted(self):
        self.assertEqual(self.cleanup_module.exclusive_references(self.hub, self.target, {self.owner}), [str(self.link)])
        self.assertEqual(self.target.read_bytes(), b'weight')

    def reject_foreign(self, name):
        snapshot(self.hub / name, self.target)
        with self.assertRaisesRegex(ValueError, 'another repository'):
            self.cleanup_module.exclusive_references(self.hub, self.target, {self.owner})
        self.assertEqual(self.target.read_bytes(), b'weight')

    def test_poldense_reference_stops_cleanup(self):
        self.reject_foreign('models--OPI-PIB--PolDense-32M')

    def test_tinyclip_reference_stops_cleanup(self):
        self.reject_foreign('models--wkcn--TinyCLIP-ViT-39M-16-Text-19M-YFCC15M')

    def test_other_repository_reference_stops_cleanup(self):
        self.reject_foreign('models--another--reference')
