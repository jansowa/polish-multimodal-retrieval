import unittest

from retrieval_benchmark.requested_models import last_token_pool,require_last_token_pooling


class PoolingTests(unittest.TestCase):
    def test_published_new_and_legacy_last_token_formats(self):
        require_last_token_pooling(dict(pooling_mode='lasttoken',include_prompt=True))
        require_last_token_pooling(dict(pooling_mode_lasttoken=True,pooling_mode_mean_tokens=False))
        for invalid in ({},dict(pooling_mode='mean'),dict(pooling_mode_lasttoken=False),
                        dict(pooling_mode_lasttoken=True,pooling_mode_mean_tokens=True)):
            with self.assertRaises(ValueError): require_last_token_pooling(invalid)

    def test_last_real_token_with_left_right_and_mixed_padding(self):
        import torch
        hidden=torch.arange(24).reshape(2,4,3)
        mask=torch.tensor([[0,1,1,0],[1,1,0,0]])
        result=last_token_pool(hidden,mask)
        self.assertTrue(torch.equal(result,torch.stack([hidden[0,2],hidden[1,1]])))
        with self.assertRaises(ValueError): last_token_pool(hidden,torch.zeros_like(mask))


if __name__=='__main__': unittest.main()
