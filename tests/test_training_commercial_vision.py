import copy
import unittest

from pl_mmb_training.common import ROOT, read_json
from pl_mmb_training.commercial_study import validate_study, replication_decisions


def candidate(params, score, license='mit'):
    return dict(encoder=dict(vision_parameters=params, license=dict(declared_weights_license=license)),
                runs=[dict(benchmark=dict(scores=dict(mscoco_t2i=score)))])


class CommercialVisionTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json(ROOT / 'training/configs/vision-commercial-five.json')

    def test_only_five_in_order_and_fixed_recipe(self):
        validate_study(self.config)
        for mutate in (lambda c: c['backbones'].append('rn101-yfcc15m'),
                       lambda c: c['backbones'].reverse(),
                       lambda c: c.update(loss='symmetric_infonce'),
                       lambda c: c.update(projector='mlp'),
                       lambda c: c.update(checkpoint_selection='test_r1')):
            bad = copy.deepcopy(self.config)
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_study(bad)

    def test_quality_threshold_and_only_best_permissive_model_per_class(self):
        models = {'a': candidate(39000000, .46), 'b': candidate(40000000, .47),
                  'c': candidate(41000000, .43), 'unlicensed': candidate(35000000, .465, 'unknown')}
        decisions = replication_decisions(models, self.config)
        self.assertEqual(decisions['a']['seeds'], [42])
        self.assertEqual(decisions['b']['seeds'], [42, 43, 44])
        self.assertEqual(len(decisions['b']['reasons']), 2)
        self.assertEqual(decisions['c']['seeds'], [42, 43, 44])
        self.assertEqual(decisions['unlicensed']['seeds'], [42])

    def test_existing_permissive_baselines_can_prevent_weak_class_winner(self):
        decisions = replication_decisions({'weak': candidate(39000000, .20)}, self.config)
        self.assertEqual(decisions['weak']['seeds'], [42])

    def test_mobileclip_size_rule_boundary_and_tied_class_winners(self):
        count = int(self.config['mobileclip_reference_parameters'] * .8)
        decisions = replication_decisions({'small': candidate(count, .471),
                                          'bigger': candidate(count + 1, .471)}, self.config)
        self.assertIn('within_2pp_of_MobileCLIP_with_at_least_20_percent_fewer_parameters', decisions['small']['reasons'])
        self.assertNotIn('within_2pp_of_MobileCLIP_with_at_least_20_percent_fewer_parameters', decisions['bigger']['reasons'])
        self.assertEqual(decisions['small']['seeds'], [42, 43, 44])
        self.assertEqual(decisions['bigger']['seeds'], [42, 43, 44])


if __name__ == '__main__':
    unittest.main()
