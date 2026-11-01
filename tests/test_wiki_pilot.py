import copy
import random
import unittest

import torch

from pl_mmb_training import wiki_pilot as pilot
from pl_mmb_training import wiki_pilot_data as data
from pl_mmb_training import source_ablation as old


class WikiPilotTests(unittest.TestCase):
    def rows(self):
        return [dict(target_id=str(i//2),text_pl=str(i)) for i in range(21)]

    def test_paired_slots_budget_and_resume(self):
        news=self.rows();wiki=[dict(target_id='wiki'+str(i)) for i in range(7)]
        samplers=[pilot.Sampler(news,wiki,p) for p in (0,3,10)]
        counts=[0,0,0];total=0
        for n in (256,145,13,256,145):
            batches=[s.draw(n) for s in samplers];total+=n
            masks=[b>=len(news) for b in batches]
            self.assertTrue((~masks[1]|masks[2]).all())
            for j in (1,2):
                self.assertTrue(torch.equal(batches[j][~masks[j]],batches[0][~masks[j]]))
            for j,m in enumerate(masks):counts[j]+=int(m.sum())
        self.assertEqual(counts,[total*p//100 for p in (0,3,10)])
        clone=pilot.Sampler(news,wiki,10);clone.restore(copy.deepcopy(samplers[2].state()))
        for n in (145,256,3):self.assertTrue(torch.equal(clone.draw(n),samplers[2].draw(n)))

    def test_baseline_matches_completed_source_study(self):
        news=self.rows();wiki=[dict(target_id='wiki')]
        current=pilot.Sampler(news,wiki,0)
        previous=old.Sampler(dict(mode='half_add',base_pool=[]),8,news,8)
        for n in (256,256,145,7,256):
            self.assertTrue(torch.equal(current.draw(n),previous.draw(n)-8))

    def test_entity_ids_are_associated_subjects(self):
        source=dict(wikidata_qid='Q12',associations=[dict(wikidata_qid='Q13')],
                    depicts=['Q5'],seed_entity=dict(wikidata_qid='Q99'))
        self.assertEqual(data.entities(source),['Q12','Q13'])

    def test_split_preserves_linked_components_and_propagates_rejection(self):
        rng=random.Random(915);rows=[]
        for i in range(60):
            rows.append(dict(id=f'r{i}',target_id=f'pixel{i}',source_sha1_hex=f'{i:040x}',
                entity_ids=[f'Q{i//3}'],text_pl=f'unique caption {i}',source_caption=f'original {i}',
                phash=f'{rng.getrandbits(64):016x}'))
        # Connect otherwise separate entities using a shared caption and a near image.
        rows[4]['source_caption']=rows[1]['text_pl']
        rows[7]['phash']=rows[10]['phash']
        train,dev,summary=data.split_groups(rows,{0:['test rejection']},target=6,maximum=9,minimum=3,min_groups=2)
        accepted={r['id'] for r in train+dev}
        self.assertTrue(all(f'r{i}' not in accepted for i in range(6)))
        where={r['id']:r['split'] for r in train+dev}
        self.assertEqual(where['r7'],where['r10'])
        self.assertEqual(summary['propagated_rejected_rows'],6)
        self.assertGreaterEqual(summary['dev_components'],2)


if __name__=='__main__':unittest.main()
