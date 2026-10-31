"""Estimate seed variance on validation; no benchmark/test access."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from pl_mmb_training import tinyclip_adaptation as base
from pl_mmb_training import tinyclip_optimization as opt
from pl_mmb_training.common import ROOT, file_sha, read_json, write_json


def main():
    output = ROOT / 'training/runs/tinyclip-frozen-replication'
    output.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    data = base.Inputs(output)
    meta = read_json(opt.CACHE / 'complete.json')
    for name, sha in meta['files'].items():
        assert file_sha(opt.CACHE / name) == sha
    results = [read_json(base.OUT / 'A1/result.json')]
    for seed in (43, 44):
        spec = dict(opt.specs()['S0_frozen_schedule'], seed=seed, warmup_epochs=0, cosine=False,
                    patience=21, minimum_epochs=20)
        results.append(opt.train(data, f'frozen_seed_{seed}', spec, meta))
    scores = torch.tensor([r['selected_validation']['t2i']['recall@1'] for r in results], dtype=torch.double)
    value = dict(results=results, t2i_r1_mean=float(scores.mean()), t2i_r1_sample_std=float(scores.std()),
                 interpretation='Three projector-training seeds; checkpoint per seed by existing validation retrieval; not a minimum detectable effect estimate.',
                 inputs=data.hashes, test_used=False, xm3600_used=False)
    data.verify()
    write_json(ROOT / 'reports/tinyclip_frozen_replication.json', value)
    print(json.dumps(dict(mean=value['t2i_r1_mean'], sample_std=value['t2i_r1_sample_std'])), flush=True)


if __name__ == '__main__':
    main()
