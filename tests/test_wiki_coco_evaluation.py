import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
import numpy as np
from evaluate_wiki_coco import query_groups


class QueryGroupingTests(unittest.TestCase):
    def test_query_permutation_preserves_shared_source_components(self):
        metadata=dict(images={p:dict(parent_ids=[source]) for p,source in
            [('a','shared'),('b','shared'),('c','other'),('d','distractor')]})
        graph=dict(images=['a','b','c','d'],image_ids=['ia','ib','ic','id'],
                   texts=['alpha','beta','gamma'],edges=[(0,0),(1,1),(2,2)])
        original=query_groups(graph,metadata)
        perm=[2,0,1];inverse={old:new for new,old in enumerate(perm)}
        changed=dict(graph,texts=[graph['texts'][i] for i in perm],
                     edges=[(inverse[t],i) for t,i in graph['edges']])
        reordered=query_groups(changed,metadata)
        actual=reordered['t2i'][np.argsort(perm)]
        np.testing.assert_array_equal(actual[:,None]==actual,original['t2i'][:,None]==original['t2i'])
        self.assertEqual(original['t2i'][0],original['t2i'][1])
        self.assertNotEqual(original['t2i'][0],original['t2i'][2])
        self.assertEqual(len(original['i2t']),3)  # A distractor is not an i2t query.


if __name__=='__main__':unittest.main()
