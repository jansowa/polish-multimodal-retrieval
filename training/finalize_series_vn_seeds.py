"""P1.A post-hoc COCO only after immutable VN-only finalist decision."""
import fcntl
import json
import statistics
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from run_series_vn_seeds import OUTPUT,HISTORICAL,load_selected,reuse_historical_text_cache
from pl_mmb_training import source_ablation as study
from pl_mmb_training.common import ROOT,file_sha,read_json,write_json
from pl_mmb_training.series_20261023 import OUT as REPORT,freeze,require_space
from verify_source_ablation import benchmark


def main():
    decision=read_json(REPORT/'vn_full_finalist_decision.json')
    if decision['status']!='all_training_closed_before_new_COCO_evaluation':raise ValueError('Training not closed')
    selected=decision['seed']
    for seed,item in decision['seed_checkpoints'].items():
        if file_sha(item['path'])!=item['sha256']:raise ValueError('Registered checkpoint changed')
    require_space(['/mnt/c','/mnt/d','/'],expected_bytes=16*2**20)
    freeze(REPORT/'vn_full_posthoc_execution_decision.json',dict(status='registered_before_posthoc',
        script_sha256=file_sha(__file__),finalist_decision_sha256=file_sha(REPORT/'vn_full_finalist_decision.json'),
        legacy_test_seeds=[selected],COCO_validation_seeds=[42,43,44],selection_closed=True,
        protocol='reuse existing legacy and fixed-gallery verifier; no configuration feedback; XM3600 prohibited'))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with (study.vn.OUT/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        study.OUT=OUTPUT/'posthoc';study.OUT.mkdir(exist_ok=True)
        study.vn.encode_cached=reuse_historical_text_cache
        data=study.Data()
        results={42:read_json(HISTORICAL/'VN_FULL/result.json')}
        results.update({seed:read_json(OUTPUT/f'seed-{seed}'/'VN_FULL/result.json') for seed in (43,44)})
        validation={}
        for seed,result in results.items():
            model,_=load_selected(data,result)
            validation[str(seed)]=data.retrieve(model,data.ctx.base.graph)
            print('COCO_VAL_POSTHOC '+str(seed)+' '+str(validation[str(seed)]['t2i']['recall@1']),flush=True)
        (study.OUT/'VN_FULL').mkdir(exist_ok=True)
        tests,protocol=benchmark(data,[results[selected]],dict(finalists=['VN_FULL']))
        selected_test=tests['VN_FULL']
        summaries={}
        for direction in ('t2i','i2t'):
            summaries[direction]={}
            for metric in ('recall@1','recall@5','recall@10'):
                values=[validation[str(seed)][direction][metric] for seed in (42,43,44)]
                summaries[direction][metric]=dict(values=values,mean=statistics.mean(values),sample_sd=statistics.stdev(values))
        summary=dict(status='P1.A_complete',selected_seed=selected,finalist_decision_sha256=file_sha(REPORT/'vn_full_finalist_decision.json'),
            checkpoint_sha256=results[selected]['checkpoint_sha256'],COCO_val_posthoc=summaries,
            selected_legacy_test=selected_test['local'],selected_fixed_gallery=selected_test['fixed_gallery'],
            protocol=protocol,COCO_used_for_selection=False,XM3600_scored=False,
            training_summary_sha256=file_sha(REPORT/'vn_full_seeds_training_summary.json'))
        freeze(REPORT/'vn_full_seeds_posthoc.json',summary)
        freeze(REPORT/'vn_full_COCO_val_per_seed.json',validation)
        data.verify()
        print('P1_A_COMPLETE '+json.dumps({k:v for k,v in summary.items() if k!='selected_fixed_gallery'}),flush=True)


if __name__=='__main__':main()
