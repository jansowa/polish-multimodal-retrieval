import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from retrieval_benchmark.completion import parameter_guard,parity_check
from retrieval_benchmark.runner import tokenization
from types import SimpleNamespace


class CompletionGuards(unittest.TestCase):
    def test_openclip_hf_tokenizer_unwrapped_for_length_report(self):
        class Native:
            def __call__(self,texts,**kwargs):
                self.assert_untruncated=kwargs['truncation'] is False
                return {'input_ids':[list(range(len(t))) for t in texts]}
        class Wrapper:
            tokenizer=Native()
            def __call__(self,texts):raise AssertionError('Wrapper would truncate before counting')
        encoder=SimpleNamespace(tokenizer=Wrapper(),text_max_length=4)
        result=tokenization(encoder,['abc','123456'],'siglip-i18n-256','v2_no_instructions')
        self.assertEqual(result['query']['truncation_fraction'],.5)
        self.assertEqual(result['query']['maximum'],6)

    def test_500m_boundary_counts_only_used_encoders(self):
        self.assertEqual(parameter_guard(dict(vision=300_000_000,text=200_000_000,shared=2))['total'],500_000_000)
        with self.assertRaisesRegex(ValueError,'exceed'):
            parameter_guard(dict(vision=300_000_001,text=200_000_000))

    def test_parity_rejects_different_query_metrics_or_gallery(self):
        value={k:'same' for k in ('protocol','dataset_id','manifest_sha256','gallery_sha256','text_gallery_sha256','qrels_sha256','prompt_mode')}
        value['directions']={d:dict(metrics={'recall@1':.5},per_query={'recall@1':[0.,1.]}) for d in ('t2i','i2t')}
        parity_check(value,copy.deepcopy(value))
        changed=copy.deepcopy(value);changed['directions']['t2i']['per_query']['recall@1']=[1.,0.]
        with self.assertRaisesRegex(ValueError,'per-query'):parity_check(changed,value)
        changed=copy.deepcopy(value);changed['qrels_sha256']='different'
        with self.assertRaisesRegex(ValueError,'identity'):parity_check(changed,value)


if __name__=='__main__':unittest.main()
