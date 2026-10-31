import random
import unittest
from pl_mmb_training.visualnews_data import HammingIndex, linux_path
from pl_mmb_training.tinyclip_visualnews import Bag, specs, copy_suffix
import torch
from types import SimpleNamespace

class VisualNewsTrainingTests(unittest.TestCase):
    def test_hamming_index_does_not_miss_distributed_six_bit_changes(self):
        gen=random.Random(42)
        index=HammingIndex()
        for i in range(60):
            index.add(i,gen.getrandbits(64))
        for i,h in index.hashes.items():
            changed=h
            for bit in gen.sample(range(64),6):
                changed ^= 1<<bit
            self.assertIn((i,6),index.near(changed))
            actual={k for k,d in index.near(changed)}
            expected={k for k,v in index.hashes.items() if (v^changed).bit_count()<=6}
            self.assertEqual(actual,expected)

    def test_windows_asset_path_maps_to_read_only_wsl_mount(self):
        self.assertEqual(str(linux_path('C:\\Users\\jsowa\\cache.parquet')),'/mnt/c/Users/jsowa/cache.parquet')

    def test_sampler_visits_all_images_before_repeating_and_resumes_exactly(self):
        a=Bag(100,42)
        draws=torch.cat([a.take(17),a.take(83)])
        self.assertEqual(set(draws.tolist()),set(range(100)))
        a.take(37)
        saved=a.state()
        b=Bag(100,999)
        b.restore(saved)
        torch.testing.assert_close(a.take(287),b.take(287))

    def test_compute_and_source_fraction_match_across_volume_and_unfreeze(self):
        values=list(specs().values())
        self.assertEqual(len(values),4)
        for field in ('steps','window_steps','warmup_steps','projector_lr','vision_lr','batch_size','full_clean_fraction','init'):
            self.assertEqual(len({v[field] for v in values}),1)

    def test_reference_hook_is_not_copied_into_trainable_suffix(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear=torch.nn.Linear(8,8)
            def forward(self,x,**kwargs):
                return (self.linear(x),)
        layers=torch.nn.ModuleList([Block(),Block()])
        vision=SimpleNamespace(vision_model=SimpleNamespace(encoder=SimpleNamespace(layers=layers),
            post_layernorm=torch.nn.LayerNorm(8)),visual_projection=torch.nn.Linear(8,8))
        captured=[]
        hook=layers[0].register_forward_pre_hook(lambda module,args:captured.append(args[0]))
        student=copy_suffix(vision,2,hook)
        x=torch.randn(3,4,8)
        layers[0](x)
        self.assertEqual(len(captured),1)
        captured.clear()
        student(x).sum().backward()
        self.assertEqual(captured,[])
        hook.remove()

if __name__=='__main__':
    unittest.main()
