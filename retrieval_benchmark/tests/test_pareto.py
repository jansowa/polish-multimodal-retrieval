import copy
import unittest
import xml.etree.ElementTree as ET

from retrieval_benchmark.space.view import commercial_use, frontier, label_rows, license_category, model_choices, organizations, overview, pareto_data, pareto_html, place_point_labels
from retrieval_benchmark.tests.test_suite import document


class ParetoTests(unittest.TestCase):
    def test_frontier_ties_dominance_and_missing_scores(self):
        points = [dict(id=name, size=size, score=score) for name,size,score in
                  [('small',10,20), ('same-size-lower',10,10), ('tie-larger',20,20),
                   ('dominated',30,15), ('best',40,50), ('pending',5,None)]]
        self.assertEqual([p['id'] for p in frontier(points)], ['small','best'])
        self.assertEqual(frontier([]), [])

    def test_license_declarations_and_owner_override(self):
        for info in ({'id':'apache-2.0'}, {'spdx':'MIT'}, 'BSD-3-Clause'):
            self.assertTrue(commercial_use('hf:example',dict(license=info)))
        for info in ({'id':'cc-by-nc-4.0'}, {'id':'apple-amlr'}, {}, {'display':'Brak deklaracji'},
                     {'id':'cc-by-nc-4.0', 'display':'MIT'}):
            self.assertFalse(commercial_use('hf:example',dict(license=info)))
        self.assertTrue(commercial_use('checkpoint:own',dict(license={'id':'apple-amlr'})))
        self.assertTrue(commercial_use('own',dict(role='project')))

    def test_chart_and_table_share_seed_control_and_profile_visibility(self):
        doc = document()
        doc['models']['a'].update(representative=False, role='english_control')
        self.assertEqual(overview(doc)['rows'], [])
        data = pareto_data(doc)
        self.assertEqual(data['points'], [])
        data = pareto_data(doc,show_seeds=True,show_controls=True)
        self.assertEqual({p['id'] for p in data['points']}, {'a'})
        self.assertEqual(next(p['score'] for p in data['points'] if p['id']=='a'),50)
        self.assertEqual([p['id'] for p in data['all_frontier']],['a'])
        self.assertEqual(pareto_data(doc,show_seeds=True,show_controls=True,profile='missing')['points'],[])

    def test_organization_options_and_owner_group(self):
        doc = document()
        self.assertNotIn('jansowa',organizations(doc))
        doc['models']['a']['repository'] = 'alphaedge-ai/a'
        doc['models']['b']['role'] = 'project'
        doc['models']['checkpoint:own'] = dict(doc['models']['b'],role='diagnostic')
        self.assertEqual(organizations(doc),['alphaedge-ai','jansowa'])
        self.assertEqual(pareto_data(doc,selected_organizations=['jansowa'])['points'],[])
        self.assertEqual([p['id'] for p in pareto_data(doc,selected_organizations=['alphaedge-ai'])['points']],['a'])
        self.assertEqual(pareto_data(doc,selected_organizations=[])['points'],[])

    def test_combined_parameter_organization_filters_and_frontiers(self):
        doc = document()
        doc['models']['a']['repository'] = 'one/a'
        doc['models']['b']['repository'] = 'two/b'
        row = copy.deepcopy(doc['results'][0]); row['model_id']='b'
        doc['results'].append(row)
        options = dict(min_parameters=50,max_parameters=100,selected_organizations=['two'])
        table = overview(doc,**options)
        data = pareto_data(doc,**options)
        self.assertEqual([p['id'] for p in data['points']],[p['id'] for p in table['model_data']])
        self.assertEqual([p['id'] for p in data['all_frontier']],['b'])
        self.assertEqual(pareto_data(doc,min_parameters=101,max_parameters=100)['points'],[])

    def test_commercial_frontier_computed_independently(self):
        doc = document()
        doc['models']['a']['license'] = {'id':'cc-by-nc-4.0'}
        doc['models']['b']['license'] = {'id':'mit'}
        row = copy.deepcopy(doc['results'][0])
        row['model_id'] = 'b'
        doc['results'].append(row)
        data = pareto_data(doc)
        self.assertEqual([p['id'] for p in data['all_frontier']], ['a'])
        self.assertEqual([p['id'] for p in data['commercial_frontier']], ['b'])

    def test_same_metric_task_and_size_selection_as_table(self):
        doc = document()
        doc['datasets']['pending'] = dict(status='ready')
        self.assertTrue(all(p['score'] is None for p in pareto_data(doc)['points']))
        data = pareto_data(doc, metric='mrr', max_parameters=60, average_tasks=['test'])
        self.assertEqual(len(data['points']),1)
        table = overview(doc, metric='mrr', average_tasks=['test'], visible_columns=['average'])
        self.assertEqual(data['points'][0]['score'], table['rows'][0][1])

    def test_svg_escapes_labels_and_handles_log_missing_and_empty(self):
        doc = document()
        doc['models']['a']['label'] = '<script>alert("x")</script>'
        doc['datasets']['pending'] = dict(status='ready')
        for logarithmic in (False, True):
            markup = pareto_html(doc, log_scale=logarithmic)
            self.assertNotIn('<script>',markup)
            start = markup.index('<svg ')
            svg = markup[start:markup.index('</svg>')+6]
            root = ET.fromstring(svg)
            ns = {'s':'http://www.w3.org/2000/svg'}
            self.assertEqual(len(root.findall('s:a', ns)),1)
            self.assertEqual(len(root.findall('s:polyline',ns)),2)
            self.assertIn('Brak pełnej średniej',markup)
        self.assertIn('Brak modeli',pareto_html(doc,max_parameters=1))

    def test_dense_identical_targets_have_nonoverlapping_full_names(self):
        points = [dict(id=str(i),size=50,name='organization/'+('long-model-name-'*5)+str(i)) for i in range(16)]
        rows = label_rows(points)
        positions = {p['id']:(500,400) for p in points}
        placed = place_point_labels(rows,positions,(85,35,1235,1100))
        self.assertIsNotNone(placed)
        self.assertEqual(len(placed),len(points))
        boxes = list(placed.values())
        for i,a in enumerate(boxes):
            for b in boxes[i+1:]:
                self.assertFalse(a['x']<b['x']+b['width'] and a['x']+a['width']>b['x'] and
                                 a['y']<b['y']+b['height'] and a['y']+a['height']>b['y'])
        for point in points:
            self.assertEqual(''.join(placed[point['id']]['lines']),point['name'])

    def test_sparse_labels_stay_next_to_their_dots(self):
        points = [dict(id='a',size=50,name='R1_seed44'),dict(id='b',size=100,name='other/model')]
        positions = dict(a=(200,200),b=(700,400))
        placed = place_point_labels(label_rows(points),positions,(85,35,1235,800))
        for mid,row in placed.items():
            px,py = positions[mid]
            dx=max(row['x']-px,0,px-row['x']-row['width'])
            dy=max(row['y']-py,0,py-row['y']-row['height'])
            self.assertLessEqual((dx*dx+dy*dy)**.5,15)

    def test_specific_model_selection_intersects_other_filters(self):
        doc = document()
        row = copy.deepcopy(doc['results'][0]); row['model_id']='b'; doc['results'].append(row)
        selected = overview(doc,selected_models=['b'])
        self.assertEqual([p['id'] for p in selected['model_data']],['b'])
        self.assertEqual([p['id'] for p in pareto_data(doc,selected_models=['b'])['points']],['b'])
        self.assertEqual(overview(doc,selected_models=['b'],max_parameters=60)['rows'],[])
        self.assertEqual(overview(doc,selected_models=[])['rows'],[])
        self.assertEqual({mid for name,mid in model_choices(doc)},{'a','b'})

    def test_license_filter_separates_commercial_restricted_unknown_and_own(self):
        doc=document()
        doc['models']['a']['license']={'id':'cc-by-nc-4.0'}
        doc['models']['b']['license']={'id':'apache-2.0'}
        doc['models']['checkpoint:own']=dict(doc['models']['a'],role='project',label='Own')
        doc['models']['unknown']=dict(doc['models']['a'],license={},label='Unknown')
        for mid in ('b','checkpoint:own','unknown'):
            row=copy.deepcopy(doc['results'][0]); row['model_id']=mid; doc['results'].append(row)
        expected={'commercial':{'b','checkpoint:own'},'noncommercial':{'a'},'unknown':{'unknown'}}
        for category,ids in expected.items():
            table=overview(doc,license_filter=category)
            graph=pareto_data(doc,license_filter=category)
            self.assertEqual({p['id'] for p in table['model_data']},ids)
            self.assertEqual({p['id'] for p in graph['points']},ids)
        self.assertEqual(license_category('checkpoint:own',doc['models']['checkpoint:own']),'commercial')

    def test_labels_are_on_by_default_and_hover_mode_omits_them(self):
        doc = document()
        markup = pareto_html(doc)
        self.assertEqual(markup.count('class="model-label"'),len(overview(doc)['rows']))
        self.assertIn('class="label-leader"',markup)
        hover = pareto_html(doc,show_labels=False)
        self.assertNotIn('class="model-label"',hover)
        self.assertNotIn('class="label-leader"',hover)
        self.assertIn('<title>',hover)


if __name__ == '__main__':
    unittest.main()
