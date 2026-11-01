import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from retrieval_benchmark.core import PROTOCOL, parameters, public_check, sha, validate, write
from retrieval_benchmark.runner import Originals, plan, selected_datasets
from retrieval_benchmark.space.view import cell_styles, csv_bytes, default_columns, group, overview


def document():
    return dict(schema_version=1, datasets={'test': dict(status='ready')}, models={
        'a': dict(label='A', revision='a'*40, representative=True, role='reference',
                  parameters=parameters(dict(vision=30_000_000, text=20_000_000, head=3_000_000))),
        'b': dict(label='B', revision='b'*40, representative=True, role='reference',
                  parameters=parameters(dict(vision=70_000_000, text=30_000_000)))}, results=[dict(
        model_id='a', dataset_id='test', prompt_mode='v2_no_instructions', profile='historical-native-v2',
        protocol=PROTOCOL, manifest_sha256='1'*64, gallery_sha256='2'*64, text_gallery_sha256='3'*64,
        qrels_sha256='4'*64, source_sha256='5'*64, exposure='unknown', precision='test-fp32',
        metrics={d: dict(zip(('recall@1', 'recall@5', 'recall@10', 'mrr', 'ndcg@10'), values))
                 for d, values in [('t2i', (.2, .4, .6, .3, .35)), ('i2t', (.8, .9, 1., .85, .86))]})])


class PublicDataTests(unittest.TestCase):
    def test_finite_complete_and_parameter_policy(self):
        d = public_check(document())
        self.assertEqual(d['models']['a']['parameters']['total'], 50_000_000)
        self.assertEqual(d['models']['a']['parameters']['head'], 3_000_000)

    def test_reject_nan_and_percent_in_fraction_field(self):
        for invalid in (math.nan, math.inf, 85, -.1):
            d = document(); d['results'][0]['metrics']['t2i']['recall@1'] = invalid
            with self.assertRaises(ValueError): validate(d)

    def test_reject_incomplete_metrics(self):
        d = document(); del d['results'][0]['metrics']['i2t']['mrr']
        with self.assertRaises(ValueError): validate(d)

    def test_reject_mixed_gallery_versions(self):
        d = document(); row = copy.deepcopy(d['results'][0]); row['model_id'] = 'b'; row['qrels_sha256'] = '9'*64
        d['results'].append(row)
        with self.assertRaisesRegex(ValueError, 'Mixed dataset'): validate(d)

    def test_reject_duplicate_and_blind_results(self):
        d = document(); d['results'].append(copy.deepcopy(d['results'][0]))
        with self.assertRaises(ValueError): validate(d)
        d = document(); d['datasets']['test']['status'] = 'final_blind'
        with self.assertRaises(ValueError): validate(d)

    def test_reject_host_paths_in_public_json(self):
        for path in ('C:/Users/alice/private', '/home/alice/data', '/mnt/d/data'):
            d = document(); d['notes'] = [path]
            with self.assertRaises(ValueError): public_check(d)

    def test_public_https_links_are_allowed(self):
        d = document(); d['notes'] = ['https://huggingface.co/datasets/example']
        public_check(d)

    def test_immutable_write_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'result.json'; write(path, {'score': .1}, immutable=True)
            previous = sha(path)
            with self.assertRaises(FileExistsError): write(path, {'score': .2}, immutable=True)
            self.assertEqual(previous, sha(path))


class ViewTests(unittest.TestCase):
    def test_direction_and_display_units(self):
        view = overview(document(), visible_columns=['average', 't2i', 'i2t'])
        self.assertEqual(view['rows'][0], ['A', 50., 20., 80.])

    def test_missing_results_are_absent_not_zero(self):
        self.assertEqual([r[0] for r in overview(document())['rows']], ['A'])
        d = document(); d['datasets']['missing'] = dict(status='ready')
        v = overview(d, visible_columns=['average', 'coverage', 'task:missing'])
        self.assertEqual(v['rows'][0], ['A', None, '1/2', None])
        self.assertEqual(v['complete'], 0)

    def test_parameter_bounds_are_inclusive(self):
        self.assertEqual(overview(document(), max_parameters=49)['rows'], [])
        self.assertEqual(len(overview(document(), max_parameters=50)['rows']), 1)
        self.assertEqual(len(overview(document(), min_parameters=50)['rows']), 1)
        self.assertEqual(overview(document(), min_parameters=51)['rows'], [])
        self.assertEqual(len(overview(document(), max_parameters=100)['rows']), 1)

    def test_seed_and_english_control_filters(self):
        d = document(); d['models']['a']['representative'] = False
        self.assertEqual(overview(d)['rows'], [])
        self.assertEqual(len(overview(d, show_seeds=True)['rows']), 1)
        d['models']['a'].update(representative=True, role='english_control')
        self.assertEqual(overview(d)['rows'], [])
        self.assertEqual(len(overview(d, show_controls=True)['rows']), 1)

    def test_csv_neutralizes_formula_strings(self):
        view = overview(document()); view['rows'][0][0] = '=HYPERLINK("test")'
        self.assertIn("'=HYPERLINK", csv_bytes(view).decode('utf-8-sig').replace('""', '"'))

    def test_machine_translation_and_synthetic_are_separate_groups(self):
        d = document(); d['results'] = []; d['datasets'] = {}
        for name, origin, value in [('human', 'native_human', .2), ('generated', 'generated_recycled_translated', .4), ('translated', 'machine_translation', .9)]:
            d['datasets'][name] = dict(status='ready', caption_origin=origin)
            r = copy.deepcopy(document()['results'][0]); r['dataset_id'] = name
            r['metrics']['t2i']['recall@1'] = r['metrics']['i2t']['recall@1'] = value
            d['results'].append(r)
        view = overview(d, visible_columns=['organic', 'synthetic', 'mt', 'average'])
        self.assertEqual(view['rows'][0][:4], ['A', 20., 40., 90.])
        self.assertAlmostEqual(view['rows'][0][4], 50.)
        self.assertEqual(group('human_translation'), 'organic')

    def test_partial_results_do_not_beat_complete_model(self):
        d = document(); d['datasets']['missing'] = dict(status='ready')
        complete = copy.deepcopy(d['results'][0]); complete['model_id'] = 'b'
        second = copy.deepcopy(complete); second['dataset_id'] = 'missing'
        for row in (complete, second):
            for dr in ('t2i', 'i2t'):row['metrics'][dr]['recall@1'] = .1
        d['results'].extend([complete, second])
        view = overview(d, visible_columns=['average'])
        self.assertEqual(view['rows'], [['B', 10.], ['A', None]])

    def test_hiding_columns_does_not_change_average(self):
        self.assertEqual(overview(document(), visible_columns=['average'])['rows'][0][1], 50.)
        self.assertEqual(overview(document(), average_tasks=[], visible_columns=['average'])['rows'][0][1], None)

    def test_score_gradient_excludes_all_parameter_columns(self):
        d = document(); row = copy.deepcopy(d['results'][0]); row['model_id'] = 'b'
        for dr in ('t2i', 'i2t'):row['metrics'][dr]['recall@1'] = .9
        d['results'].append(row)
        view = overview(d, visible_columns=['average', 'total', 'vision', 'text', 'head'])
        styles = cell_styles(view)
        self.assertIn('rgb(165, 214, 167)', styles[0][1])
        self.assertIn('rgb(255, 205, 210)', styles[1][1])
        self.assertTrue(all(s == '' for row in styles for s in row[2:]))

    def test_default_columns_hide_group_and_direction_scores(self):
        visible = default_columns(document())
        self.assertIn('task:test', visible)
        self.assertFalse(set(visible) & {'t2i', 'i2t', 'organic', 'synthetic', 'mt'})


class GateTests(unittest.TestCase):
    def test_original_file_tamper_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'image.jpg'; path.write_bytes(b'original pixels')
            manifest = Path(tmp)/'manifest.json'
            write(manifest, [dict(image=str(path), sha256=sha(path))])
            originals = Originals(manifest, {})
            with originals.paths([str(path)]) as paths:
                self.assertEqual(paths, [str(path)])
            path.write_bytes(b'changed pixels')
            with self.assertRaisesRegex(ValueError, 'image SHA changed'):
                with originals.paths([str(path)]): pass

    def test_planned_refused_before_model_load(self):
        with self.assertRaisesRegex(ValueError, 'not released'):
            selected_datasets({'datasets': {}}, ['docci_pl'])

    def test_protocol_tamper_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'protocol.json'; p.write_text('{}')
            config = {'datasets': {'flickr30k_pl_local_test': dict(protocol=str(p), protocol_sha256='0'*64)}}
            with self.assertRaisesRegex(ValueError, 'changed'): selected_datasets(config, ['flickr30k_pl_local_test'])

    def test_revision_and_download_budget_required(self):
        config = {'models': {'test': {'repository': 'example/test', 'revision': 'main'}}}
        with self.assertRaisesRegex(ValueError, 'Pinned'): plan(config, 'test', ['core'], ['v2_no_instructions'], 64)
        config['models']['test']['revision'] = 'a'*40
        with self.assertRaisesRegex(ValueError, 'budget'): plan(config, 'test', ['core'], ['v2_no_instructions'], 64)


if __name__ == '__main__':
    unittest.main()
