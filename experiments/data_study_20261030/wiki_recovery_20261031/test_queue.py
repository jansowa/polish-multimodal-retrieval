"""CPU-only checks of queue completeness and generation isolation."""
import unittest
from config import *
import queue as std_queue
import importlib.util


class QueueTests(unittest.TestCase):
    def test_profile_smoke_bit_exact(self):
        receipts=[read(p) for p in (STORE/'compact_receipts').glob('*.json') if read(p).get('status')=='ready']
        self.assertGreaterEqual(len(receipts),3)
        for r in receipts:
            self.assertTrue(r['preprocessing_bit_exact'])
            self.assertEqual(r['native_dimensions'],[224,224])
            self.assertLessEqual(max(r['master_dimensions']),1536)
            self.assertEqual(sha256(r['native_path']),r['native_sha256'])
            self.assertEqual(sha256(r['master_path']),r['master_sha256'])

    def test_catalog_covers_exact_export(self):
        catalog=list(rows(local_path(read(REPORT/'active_catalog.json')['path'])))
        self.assertEqual(len(catalog),24125)
        self.assertEqual({r['sha1'] for r in catalog},{source_sha1(r['source_record']['source_sha1']) for r in rows(EXPORT)})
        groups={};last=None
        for r in catalog:
            group=r['metadata_component']
            if group!=last:
                self.assertNotIn(group,groups);groups[group]=True;last=group

    def test_original_generation_preserved(self):
        registration=read(REPORT/'registration.json')
        for name,h in registration['generation_1_markers'].items():self.assertEqual(sha256(COORD.parent/(name+'.json')),h)
        self.assertNotEqual(STORE,OLD_STORE);self.assertEqual(read(REPORT/'compact_smoke.json')['status'],'smoke_complete')

    def test_training_recipe_bindings(self):
        import train_generation as training
        training.configure()
        recipe=training.prior.read(REPORT/'registration.json')
        self.assertEqual(recipe['checkpoint_selection'],read(OLD_REPORT/'registration.json')['checkpoint_selection'])
        self.assertEqual(training.prior.STORAGE,STORE)
        self.assertEqual(training.prior.COORD,COORD)


if __name__=='__main__':unittest.main()
