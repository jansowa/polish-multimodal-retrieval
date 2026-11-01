"""Contract-critical CPU checks; no model loading or CUDA import."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare
from protocol import graph, choose_components


def row(identifier, pixel, phash, en, pl):
    return dict(id=identifier, pixels_sha256=f'{pixel:064x}', phash=f'{phash:016x}',
                image=f'/original/{pixel}.jpg', text_en=en, text_pl=pl)


class ProtocolTests(unittest.TestCase):
    def test_near_images_do_not_create_positive_edges(self):
        result = graph([row('a', 1, 0, 'dog', 'pies'), row('b', 2, 1, 'cat', 'kot')], 'docci_pl_local', 'x')
        self.assertEqual(result['edges'], [(0, 0), (1, 1)])
        self.assertEqual(len(set(result['image_components'])), 1)
        self.assertEqual(len(result['image_ids']), 2)
        self.assertEqual(len(choose_components(result, target=1)), 2)

    def test_identical_en_unions_original_positives(self):
        result = graph([row('a', 1, 0, ' A dog ', 'pies'), row('b', 2, 2**64-1, 'a DOG', 'psina')], 'flickr_pl_local', 'x')
        self.assertEqual(len(result['texts']), 1)
        self.assertEqual(result['edges'], [(0, 0), (0, 1)])
        self.assertEqual(len(set(result['image_components'])), 1)

    def test_exact_pixels_canonicalized_keep_all_captions(self):
        result = graph([row('a', 1, 0, 'dog', 'pies'), row('b', 1, 0, 'animal', 'zwierze')], 'local', 'x')
        self.assertEqual(len(result['images']), 1)
        self.assertEqual(result['edges'], [(0, 0), (1, 0)])

    def test_input_order_does_not_affect_graph_or_selection(self):
        rows = [row('a', 1, 0, 'dog', 'pies'), row('b', 2, 2**64-1, 'cat', 'kot')]
        a, b = graph(rows, 'local', 'x'), graph(rows[::-1], 'local', 'x')
        self.assertEqual(a, b)
        self.assertEqual(choose_components(a, 1), choose_components(b, 1))

    def test_no_empty_translations_or_blind_gallery(self):
        for name, rows in [('local', [row('a', 1, 0, 'dog', '')]), ('XM3600', [row('a', 1, 0, 'dog', 'pies')])]:
            with self.assertRaises(ValueError):
                graph(rows, name, 'x')


class HandoffTests(unittest.TestCase):
    def test_tampered_dependency_and_wrong_phase_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / 'COORDINATION.md').write_text('contract')
            payload = folder / 'artifact'
            payload.write_text('good')
            artifact = dict(role='common_helper', path='C:/artifact', sha256=prepare.sha(payload), bytes=4)
            marker = dict(schema_version=1, study_id='data-ood-20261030', generation=1, owner='A',
                          phase='COMMON_READY', status='complete', contract_sha256=prepare.sha(folder / 'COORDINATION.md'),
                          artifacts=[dict(artifact, role=role) for role in prepare.ROLES['COMMON_READY']])
            target = folder / 'COMMON_READY.json'
            target.write_text(json.dumps(marker))
            with patch.object(prepare, 'COORD', folder), patch.object(prepare, 'local_path', lambda value: payload):
                prepare.verify_marker('COMMON_READY')
                payload.write_text('evil')
                with self.assertRaises(ValueError):
                    prepare.verify_marker('COMMON_READY')
                payload.write_text('good')
                marker['phase'] = 'A_GPU_DONE'
                target.write_text(json.dumps(marker))
                with self.assertRaises(ValueError):
                    prepare.verify_marker('COMMON_READY')

    def test_missing_marker_is_waiting_not_permission(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(prepare, 'COORD', Path(temporary)):
            self.assertTrue(all(r['status'] == 'missing' for r in prepare.marker_status().values()))


class LeakageTests(unittest.TestCase):
    def test_cross_source_near_originals_do_not_enter_both_galleries(self):
        from leakage import ProtectedIndex, filter_components
        rows=[dict(id='a',dataset='flickr',phash='0000000000000000'),
              dict(id='b',dataset='vizwiz',phash='0000000000000001')]
        self.assertTrue(all(row['excluded'] for row in filter_components(rows,ProtectedIndex())))

    def test_phash_threshold_and_transitive_candidate_rejection(self):
        from leakage import ProtectedIndex, filter_components
        index = ProtectedIndex()
        index.add(dict(phash='ffffffffffffffff'))
        rows = [dict(id='a', phash='ffffffffffffffc0'), dict(id='b', phash='fffffffffffff000'),
                dict(id='c', phash='0000000000000000')]
        result = filter_components(rows, index)
        self.assertTrue(result[0]['excluded'])
        self.assertTrue(result[1]['excluded'])
        self.assertFalse(result[2]['excluded'])
        self.assertEqual(result[1]['exclusion_reasons'], ['transitive_B_component'])

    def test_normalized_texts_and_flickr_url_identity(self):
        from leakage import ProtectedIndex
        index = ProtectedIndex()
        index.add(dict(normalized_texts=[' A DOG '], source_url='https://www.flickr.com/photos/author/123456789/'))
        self.assertTrue(index.reasons(dict(text_pl='a dog', phash='0000000000000000')))
        self.assertTrue(index.reasons(dict(parent_ids=['flickr30k:123456789'], phash='0000000000000000')))

    def test_sha_and_pixel_hash_aliases_match(self):
        from leakage import ProtectedIndex
        index = ProtectedIndex()
        index.add(dict(sha256='a'*64))
        self.assertTrue(index.reasons(dict(pixels_sha256='a'*64, phash='0000000000000000')))

    def test_gpu_gates_fail_before_importing_torch(self):
        import sys
        import evaluate
        before='torch' in sys.modules
        with patch.object(evaluate,'load_common',return_value=object()),patch.object(evaluate,'verify_A',side_effect=ValueError('missing handoff')):
            with self.assertRaises(ValueError):
                evaluate.gates()
        self.assertEqual('torch' in sys.modules,before)


class SyntaxTests(unittest.TestCase):
    def test_all_B_scripts_parse_without_execution(self):
        import ast
        for path in Path(__file__).parent.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'),filename=str(path))


if __name__ == '__main__':
    unittest.main()
