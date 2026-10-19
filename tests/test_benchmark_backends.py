import unittest


class MClipCheckpointTests(unittest.TestCase):
    def test_legacy_position_buffer_and_reference_pooling(self):
        try:
            import torch
            from transformers import XLMRobertaConfig, XLMRobertaModel
            from pl_mmb_benchmark.backends import MultilingualText
        except ImportError:
            self.skipTest('Model libraries live in the benchmark environment')
        config = XLMRobertaConfig(vocab_size=24, hidden_size=8, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=16, max_position_embeddings=20,
            hidden_dropout_prob=0., attention_probs_dropout_prob=0.)
        reference = XLMRobertaModel(config).eval()
        projection = torch.nn.Linear(8, 640)
        state = {'transformer.' + key: value for key, value in reference.state_dict().items()}
        state.update({'LinearTransformation.' + key: value for key, value in projection.state_dict().items()})
        state['transformer.embeddings.position_ids'] = torch.arange(20).expand(1, -1)
        model = MultilingualText(config, state).eval()
        self.assertFalse(any(b.is_meta for b in model.buffers()))
        tokens = {'input_ids': torch.tensor([[0, 4, 2, 1], [0, 7, 8, 2]]),
                  'attention_mask': torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])}
        with torch.inference_mode():
            hidden = reference(**tokens).last_hidden_state
            mask = tokens['attention_mask'].unsqueeze(-1)
            expected = projection((hidden * mask).sum(1) / mask.sum(1))
            torch.testing.assert_close(model(tokens), expected)
        state['transformer.embeddings.position_ids'] = torch.zeros(1, 20, dtype=torch.long)
        with self.assertRaises(ValueError):
            MultilingualText(config, state)
