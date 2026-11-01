"""Conditional aligned-head LR midpoint, after the eight planned comparisons."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from pl_mmb_training import tinyclip_adaptation as base
from pl_mmb_training import tinyclip_optimization as study
from pl_mmb_training.common import ROOT, file_sha, read_json, write_json


def main():
    import fcntl
    with (study.OUT / 'worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4)
        for n in study.specs():
            assert read_json(study.OUT / n / 'result.json')['status'] == 'complete'
        data = base.Inputs(study.OUT)
        meta = read_json(study.CACHE / 'complete.json')
        for filename, sha in meta['files'].items():
            assert file_sha(study.CACHE / filename) == sha
        control = read_json(study.OUT / 'T0_frozen_aligned/result.json')
        initial = [read_json(study.OUT / n / 'result.json') for n in ('T1_last1_aligned', 'T2_last2_aligned')]
        gain = max(base.key(r)[0] for r in initial) - base.key(control)[0]
        trigger = gain < .005
        decision = dict(run=trigger, maximum_gain_pp=100*gain,
                        rule='If aligned-head vision LR=1e-6 gains less than 0.5pp over its matched frozen control, test only the 3e-6 midpoint.',
                        source_sha256=file_sha(__file__), selection_uses_test=False, xm3600_used=False)
        write_json(study.OUT / 'midpoint_decision.json', decision)
        checks = {}
        if trigger:
            for name, original in [('T3_last1_aligned_mid', 'T1_last1_aligned'), ('T4_last2_aligned_mid', 'T2_last2_aligned')]:
                spec = dict(study.specs()[original], vision_lr=3e-6, launcher_sha256=file_sha(__file__))
                r = study.train(data, name, spec, meta)
                best = max(r['history'], key=base.checkpoint_key)
                assert best['validation'] == r['selected_validation'] and best['epoch'] == r['best_epoch']
                assert file_sha(study.OUT / name / 'best.pt') == r['checkpoint_sha256']
                assert not r['selection_uses_test'] and not r['xm3600_used']
                assert r['starting_validation'] == control['starting_validation']
                for row in r['history'][1:]:
                    for group, diag in row['diagnostics'].items():
                        peak = spec['vision_lr'] if group == 'vision' else spec['projector_lr']
                        assert 0 < diag['lr_min'] <= diag['lr_max'] <= peak * (1+1e-12)
                checks[name] = dict(checkpoint_verified=True, validation_selection_verified=True, lr_verified=True)
                study.export(study.OUT)
        initial_selection = study.OUT / 'initial_selection.json'
        if not initial_selection.exists():
            write_json(initial_selection, read_json(study.OUT / 'selection.json'))
        results = [read_json(p) for p in sorted(study.OUT.glob('*/result.json'))]
        winner = max(results, key=base.key)
        write_json(study.OUT / 'selection.json', dict(winner=winner['experiment'], validation=winner['selected_validation'],
                   selection_uses_test=False, xm3600_used=False, conditional_midpoint_ran=trigger))
        data.verify()
        write_json(ROOT / 'reports/tinyclip_optimization_extended_verification.json',
                   dict(status='passed', decision=decision, midpoint_runs=checks,
                        original_eight_verification_sha256=file_sha(ROOT / 'reports/tinyclip_optimization_verification.json'),
                        unchanged_inputs=data.hashes, selection_uses_test=False, xm3600_used=False))
        write_json(study.OUT / 'status.json', dict(status='complete', stage='midpoint', winner=winner['experiment'], pid=os.getpid()))
        print(dict(midpoint_decision=decision, winner=winner['experiment']), flush=True)


if __name__ == '__main__':
    main()
