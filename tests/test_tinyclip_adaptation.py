"""Structural checks for partial adaptation; no data or model downloads."""
import unittest

import torch
from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

from pl_mmb_training.tinyclip_adaptation import Suffix, projector
from pl_mmb_training.tinyclip_followups import LoRALinear, similarity_kd


class AdaptationTests(unittest.TestCase):
    def test_suffix_matches_native_representation_and_only_selected_blocks_train(self):
        torch.manual_seed(42)
        config = CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                                 num_attention_heads=2, image_size=16, patch_size=8, projection_dim=12)
        vision = CLIPVisionModelWithProjection(config).eval().requires_grad_(False)
        captured = []
        handle = vision.vision_model.encoder.layers[-2].register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach()))
        pixels = torch.randn(2, 3, 16, 16)
        with torch.no_grad():
            native = vision.vision_model(pixel_values=pixels).pooler_output
            expected = vision.visual_projection(native)
        handle.remove()
        for feature in ['pooled', 'projected']:
            for blocks in [1, 2]:
                suffix = Suffix(vision, feature, blocks)
                output = suffix(captured[0])
                torch.testing.assert_close(output, native if feature == 'pooled' else expected)
                output.square().mean().backward()
                for i, layer in enumerate(suffix.layers):
                    self.assertTrue(all(p.requires_grad == (i >= 2 - blocks) for p in layer.parameters()))
                    self.assertEqual(any(p.grad is not None for p in layer.parameters()), i >= 2 - blocks)
                self.assertFalse(any(p.requires_grad for p in suffix.post_layernorm.parameters()))
                self.assertFalse(any(p.requires_grad for p in suffix.projection.parameters()))
                self.assertFalse(any(p.grad is not None for p in vision.parameters()))

    def test_projector_has_exact_requested_architecture(self):
        self.assertIsInstance(projector(512, 0), torch.nn.Linear)
        mlp = projector(512, 768)
        self.assertEqual([type(m) for m in mlp], [torch.nn.Linear, torch.nn.GELU, torch.nn.Linear])
        self.assertEqual(mlp(torch.randn(2, 512)).shape, (2, 384))

    def test_lora_is_initially_identical_and_freezes_base_weights(self):
        linear = torch.nn.Linear(16, 12)
        layer = LoRALinear(linear, rank=4, alpha=8)
        x = torch.randn(3, 16)
        torch.testing.assert_close(layer(x), linear(x))
        layer(x).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in layer.base.parameters()))
        self.assertIsNotNone(layer.b.grad)
        self.assertGreater(layer.b.grad.norm().item(), 0)
        self.assertEqual(sum(p.numel() for p in layer.parameters() if p.requires_grad), 4 * (16 + 12))

    def test_kd_compares_similarities_in_different_embedding_dimensions(self):
        student_i = torch.randn(5, 7, requires_grad=True)
        student_t = torch.randn(5, 7)
        teacher_i = torch.randn(5, 11, requires_grad=True)
        teacher_t = torch.randn(5, 11, requires_grad=True)
        loss = similarity_kd(student_i, student_t, teacher_i, teacher_t)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(student_i.grad.norm().item(), 0)
        self.assertIsNone(teacher_i.grad)
        self.assertIsNone(teacher_t.grad)
        identical = similarity_kd(student_i.detach(), student_t, student_i.detach(), student_t)
        self.assertLess(abs(identical.item()), 1e-5)


if __name__ == '__main__':
    unittest.main()
