"""Repeat the fixed full-VisualNews recipe, with matched frozen/FT sampling seeds."""
import gc
import json
import os
import sys
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from pl_mmb_training import tinyclip_visualnews as study
from pl_mmb_training.common import ROOT,file_sha,read_json,write_json


def main():
    import fcntl
    with (study.OUT/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        existing=read_json(ROOT/'reports/tinyclip_visualnews.json')['results']
        results={r['experiment']:r for r in existing}
        assert all(name in results for name in ['VN_all_frozen','VN_all_last2_anchor','VN_0_frozen'])
        gain=results['VN_all_frozen']['selected_validation']['t2i']['recall@1']-results['VN_0_frozen']['selected_validation']['t2i']['recall@1']
        decision=study.OUT/'replication_decision.json'
        if not decision.exists():
            write_json(decision,dict(created_at=datetime.now(timezone.utc).isoformat(),
                reason='Primary COCO gain from adding the full dataset exceeds 1pp; prioritize replication of the fixed data recipe over another warm-start hyperparameter scout.',
                primary_data_gain_percentage_points=100*gain,seeds=[42,43,44],
                conditions='Same A1 epoch-9 warm start; vary only adaptation sampling/augmentation seed. Not variance of original linear-probe pretraining.',
                changes_from_scout='No model/loss/LR/source-mixture changes; two extra seeds for each full-data model.',
                vn_dev_selects_checkpoints=False,test_used=False,xm3600_scored=False))
        assert gain>=.01
        ctx=study.Context();engine=study.Engine(ctx)
        for seed in (43,44):
            for original in ('VN_all_frozen','VN_all_last2_anchor'):
                spec=dict(study.specs()[original],seed=seed,replication_of=original,launcher_sha256=file_sha(__file__))
                name=f'{original}_seed{seed}'
                result=study.train(ctx,engine,name,spec)
                results[name]=result
                study.export(ctx,list(results.values()))
                gc.collect();torch.cuda.empty_cache()
        ctx.verify()
        write_json(study.OUT/'status.json',dict(status='complete',stage='three_adaptation_seeds',
            experiments=list(results),snapshot_unchanged=True,pid=os.getpid()))

if __name__=='__main__':main()
