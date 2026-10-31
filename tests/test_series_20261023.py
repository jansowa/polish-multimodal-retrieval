import importlib.util
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
from pl_mmb_training.series_20261023 import freeze, native_original, deterministic_images, require_space
from pl_mmb_training.series_holdout import require_wiki_training_release,ExternalHoldoutGuard
from pl_mmb_training.leakage import LeakageGuard
from pl_mmb_benchmark.full_gallery import evaluate_features, per_query, paired_bootstrap, evaluate_encoder

spec = importlib.util.spec_from_file_location('series_annotation', Path(__file__).resolve().parents[1] / 'experiments/series_annotation.py')
annotation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(annotation)


class PreparationTests(unittest.TestCase):
    def test_provenance_cannot_accept_recycled_or_translated_or_mismatched(self):
        row = dict(route='ready_native_original', source_language='pl', text_pl='oryginał', accepted_text='oryginał',
                   source_caption='oryginał', selected_candidate=dict(language='pl', text='oryginał'),
                   stages=dict(judge=dict(visual_grounding=3, image_caption_mismatch=False)))
        self.assertTrue(native_original(row))
        for change in (dict(text_pl='model changed text'), dict(route='ready_native_recycled'),
                       dict(source_language='en'), dict(stages=dict(judge=dict(visual_grounding=4, image_caption_mismatch=True)))):
            self.assertFalse(native_original(dict(row, **change)))
        row['stages']['translate'] = {}
        self.assertFalse(native_original(row))

    def test_frozen_manifest_refuses_changed_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            first = freeze(path, dict(a=1))
            self.assertEqual(freeze(path, dict(a=1)), first)
            with self.assertRaises(ValueError):
                freeze(path, dict(a=2))

    def test_sampling_is_independent_of_row_order_and_pixel_aliases(self):
        records = [dict(id=str(i), target_id=str(i//2)) for i in range(20)]
        a = deterministic_images(records, 8, 'fixture')
        b = deterministic_images(list(reversed(records)), 8, 'fixture')
        self.assertEqual(a, b)
        self.assertEqual(len({r['target_id'] for r in a}), 8)

    def test_wiki_training_blocks_until_new_data_and_release(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'blocked'):
                require_wiki_training_release(directory)


class RetrievalTests(unittest.TestCase):
    def result(self):
        return evaluate_features(np.eye(3), np.eye(3), ['i0','i1','i2'], ['q0','q1','q2'],
                                 [(0,0),(1,1),(2,2)], ['a','b','c'], ['a','b','c'], 'fixture', 'manifest')

    def test_multi_positive_ndcg_uses_all_known_positives_and_stable_ties(self):
        result = per_query([[1,1,1,0]], [[False,True,True,False]])
        self.assertEqual(result['recall@1'][0], 0)
        self.assertEqual(result['mrr'][0], .5)
        expected = (1/np.log2(3)+1/np.log2(4))/(1+1/np.log2(3))
        self.assertAlmostEqual(result['ndcg@10'][0], expected)

    def test_paired_bootstrap_identity_and_component_protection(self):
        a = self.result()
        paired = paired_bootstrap(a, a, 't2i', replicates=50)
        self.assertEqual(paired['metrics']['recall@1'], dict(delta_pp=0, ci95_pp=[0,0]))
        b = json.loads(json.dumps(a)); b['directions']['t2i']['query_ids'].reverse()
        with self.assertRaisesRegex(ValueError, 'Query IDs'):
            paired_bootstrap(a, b, 't2i', 10)
        b = dict(a, gallery_sha256='different')
        with self.assertRaisesRegex(ValueError, 'gallery'):
            paired_bootstrap(a, b, 't2i', 10)

    def test_execution_gate_before_encoder_access(self):
        with self.assertRaises(ValueError):
            evaluate_encoder(None, {}, {'status':'model_list_frozen_execution_blocked'})

    def test_xm3600_uses_the_same_native_scorer(self):
        result=evaluate_features(np.eye(3),np.eye(3),['i0','i1','i2'],['q0','q1','q2'],[(0,0),(1,1),(2,2)],
            ['a','b','c'],['a','b','c'],'xm3600_pl','manifest')
        self.assertEqual(result['directions']['t2i']['metrics']['recall@1'],1)
        self.assertEqual(result['directions']['i2t']['metrics']['recall@1'],1)

    def test_asymmetric_candidate_prompts_affect_i2t_only(self):
        a=self.result()
        b=evaluate_features(np.eye(3),np.eye(3),['i0','i1','i2'],['q0','q1','q2'],[(0,0),(1,1),(2,2)],
            ['a','b','c'],['a','b','c'],'fixture','manifest',text_candidate_features=np.eye(3)[[1,2,0]])
        self.assertEqual(a['directions']['t2i'],b['directions']['t2i'])
        self.assertEqual(b['directions']['i2t']['metrics']['recall@1'],0)


class HoldoutGuardTests(unittest.TestCase):
    def guard(self):
        return dict(ids=['dev'],texts=['opis'],parent_ids=['commons:dev'],source_sha1_hex=['a'*40],
            images={'dev':dict(sha256='file-dev',pixels_sha256='pixel-dev',parent_ids=['commons:dev'],phash='0000000000000000')})

    def test_captionless_training_cannot_use_dev_pixels(self):
        guard=LeakageGuard(dict(ids=[],texts=[],images={}),extra_holdout=self.guard())
        record=dict(id='alias',sha256='reencoded',pixels_sha256='pixel-dev',parent_ids=['other'],lineage_verified=True,phash='000000000000ffff')
        self.assertIn('wiki_dev_file_or_pixels',guard.reasons(record))

    def test_phash_six_is_excluded_seven_is_allowed(self):
        guard=ExternalHoldoutGuard(self.guard())
        self.assertIn('wiki_dev_phash_le6',guard.reasons(dict(phash='000000000000003f')))
        self.assertNotIn('wiki_dev_phash_le6',guard.reasons(dict(phash='000000000000007f')))

    def test_unknown_image_hash_is_not_silently_accepted(self):
        self.assertIn('wiki_dev_unverified_phash',ExternalHoldoutGuard(self.guard()).reasons(dict(image='missing-fixture://image')))

    def test_source_sha1_survives_rendering_and_text_is_optional(self):
        guard=ExternalHoldoutGuard(self.guard())
        result=guard.reasons(dict(source_sha1_hex='a'*40,sha256='different',pixels_sha256='different',phash='ffffffffffffffff'))
        self.assertIn('wiki_dev_source_sha1',result)


class AnnotationHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        folder = Path(self.temp.name)
        base = dict(id='fixture:1', target_id='pixels', component='component', source='fixture', split='dev',
                    cache_index=0, dataset='VN', asset='vn_native_rgb')
        query = freeze(folder/'query_tasks.jsonl', [base], jsonl=True)
        audit = freeze(folder/'audit_tasks.jsonl', [dict(base,id='fixture:2',text_pl='PRIVATE CAPTION')], jsonl=True)
        freeze(folder/'sampling.json', dict(query=dict(manifest_sha256=query),audit=dict(manifest_sha256=audit)))
        self.collection = annotation.Collection(folder)
        self.server = ThreadingHTTPServer(('127.0.0.1',0), annotation.handler(self.collection))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(); self.temp.cleanup()

    def test_query_api_hides_caption_and_save_is_validated_append_only(self):
        with urllib.request.urlopen(self.url+'/api/tasks?mode=query') as response:
            public = response.read().decode()
        self.assertNotIn('caption', public)
        self.assertNotIn('PRIVATE', public)
        self.assertNotIn('cache_index', public)
        data = json.dumps(dict(mode='query',index=0,queries=['fixture query'])).encode()
        request = urllib.request.Request(self.url+'/api/save', data=data, headers={'Content-Type':'application/json'})
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        self.assertEqual(error.exception.code, 403)
        request.add_header('X-Annotation-Token', self.collection.token)
        for _ in range(2):
            with urllib.request.urlopen(request) as response:
                self.assertEqual(response.status, 200)
        saved = (Path(self.temp.name)/'human_query.jsonl').read_text().splitlines()
        self.assertEqual(len(saved), 2)
        self.assertEqual(json.loads(saved[0])['image_id'], 'pixels')
        self.assertIn('timestamp', json.loads(saved[0]))

    def test_foreign_host_and_origin_cannot_save(self):
        for headers in ({'Host':'attacker.test'}, {'Origin':'https://attacker.test'}):
            request = urllib.request.Request(self.url+'/api/state', headers=headers)
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            self.assertEqual(error.exception.code,403)

    def test_audit_requires_explicit_boolean_and_integer(self):
        with self.assertRaises(ValueError):
            annotation.validate_annotation('audit', dict(grounding=True,mismatch=False))
        with self.assertRaises(ValueError):
            annotation.validate_annotation('audit', dict(grounding=3,mismatch='no'))


if __name__ == '__main__':
    unittest.main()
