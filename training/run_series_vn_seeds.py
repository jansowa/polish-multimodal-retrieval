"""P1.A: reuse exact source-ablation functions in isolated, preregistered outputs."""
import argparse
import copy
import fcntl
import json
import os
import shutil
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import torch
from pl_mmb_training import source_ablation as study
from pl_mmb_training import tinyclip_adaptation as base
from pl_mmb_training.common import ROOT, file_sha, read_json, write_json, digest
from pl_mmb_training.losses import PairLoss
from pl_mmb_training.series_20261023 import OUT as REPORT, freeze, require_space

OUTPUT = Path('/mnt/c/Users/jsowa/benchmark-translations/series-2026-10-23/runs/vn_full_seeds')
HISTORICAL = ROOT/'training/runs/tinyclip-source-ablation-v1'
OriginalSampler = study.Sampler
EXECUTION_DECISION='vn_full_seeds_execution_decision_v2.json'


def reuse_historical_text_cache(values,kind,config,lock,cache_root):
    if kind!='texts':raise ValueError('This runner never encodes images')
    metadata=read_json(study.vn.OUT/'native_features.json')['identity']['data']['text_features']
    path=Path(metadata['path']);identity=read_json(path.with_suffix('.json'))
    if digest(identity)!=metadata['identity_sha256'] or identity['values']!=list(values):
        raise ValueError('Historical text-cache input identity differs')
    if identity['encoder']!=config['text'] or identity['revision']!=lock['text']['revision']:
        raise ValueError('Historical text encoder identity differs')
    if file_sha(path)!=metadata['sha256']:raise ValueError('Historical text-cache SHA differs')
    tensor=torch.load(path,weights_only=True,map_location='cpu')
    if tensor.dtype!=torch.float32 or tensor.shape!=(len(values),384):raise ValueError('Unexpected frozen text tensor')
    print(f'EXACT_HISTORICAL_TEXT_CACHE {len(values)} SHA256={metadata["sha256"]}',flush=True)
    return tensor,metadata


def sampler_for_seed(seed):
    def factory(*args, **kwargs):
        kwargs.setdefault('seed', seed)
        return OriginalSampler(*args, **kwargs)
    return factory


def test_equivalence():
    torch.manual_seed(42)
    head=base.projector(512,0); loss=PairLoss('siglip_sigmoid',.07,-10.)
    saved=torch.load(HISTORICAL/'random_initialization.pt',weights_only=True,map_location='cpu')
    for group, current in [('projector',head.state_dict()),('loss',loss.state_dict())]:
        if not all(torch.equal(value,saved[group][name]) for name,value in current.items()):
            raise ValueError('Seed42 initialization no longer matches historical recipe')
    fixture=[dict(target_id=str(i),id=str(i)) for i in range(16)]
    plan=dict(mode='replace',base_pool=list(range(16)))
    old=OriginalSampler(plan,16,fixture,16,seed=42)
    new=sampler_for_seed(42)(plan,16,fixture,16)
    other=sampler_for_seed(43)(plan,16,fixture,16)
    same_seed=True; different_seed=False
    for _ in range(5):
        a,b,c=old.draw(32),new.draw(32),other.draw(32)
        same_seed &= torch.equal(a,b); different_seed |= not torch.equal(b,c)
    if not same_seed or not different_seed:raise ValueError('Seeded sampler wrapper failed parity')
    return dict(initialization_seed42_exact=True,sampler_seed42_exact=True,sampler_seed43_distinct=True)


def initialize_seed(seed):
    torch.manual_seed(seed)
    model=base.projector(512,0);loss=PairLoss('siglip_sigmoid',.07,-10.)
    expected=dict(projector=model.state_dict(),loss=loss.state_dict(),seed=seed,
        initialized_from=f'random_torch_seed_{seed}_no_data_no_pretrained_projector')
    path=study.OUT/'random_initialization.pt'
    if path.exists():
        saved=torch.load(path,weights_only=True,map_location='cpu')
        if saved['seed']!=seed:raise ValueError('Wrong initialization seed')
        for group in ('projector','loss'):
            for name,value in expected[group].items():
                if not torch.equal(saved[group][name],value):raise ValueError('Initialization changed')
    else:
        study.opt.atomic_save(expected,path)
    return path,expected


def load_selected(data,result):
    path=Path(result['checkpoint_path'])
    if file_sha(path)!=result['checkpoint_sha256']:raise ValueError('Checkpoint SHA changed')
    saved=torch.load(path,weights_only=True,map_location='cpu')
    chosen=max(result['history'],key=study.key)
    if chosen['updates']!=result['best_updates'] or saved['updates']!=chosen['updates']:
        raise ValueError('Wrong VN-selected checkpoint')
    model=torch.nn.Linear(512,384).to('cuda').eval();model.load_state_dict(saved['projector'])
    actual=data.retrieve(model,data.ctx.dev_graph)
    expected=result['selected_VN_dev']
    score=dict(actual);score.pop('ranks')
    if score!=expected:raise ValueError('Reloaded VN metrics differ')
    criterion=PairLoss('siglip_sigmoid',.07,-10.).to('cuda');criterion.load_state_dict(saved['loss'])
    with torch.inference_mode():
        val_loss=sum(float(criterion(model(data.dev_x[p]),data.dev_t[p],mask))*len(p) for p,mask in data.dev_batches)/len(data.ctx.dev)
    if abs(val_loss-result['best_validation_loss'])>2e-5:raise ValueError('Reloaded validation loss differs')
    return model,actual


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    require_space(['/mnt/c','/mnt/d','/'],expected_bytes=64*2**20)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    checks=test_equivalence()
    if args.check_only:
        print(json.dumps(checks),flush=True);return
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    expected=read_json(study.vn.OUT/'native_features.json')['identity']['data']
    current=read_json(study.vn.OUT/'context.json')
    incident=dict(status='rejected_before_training',cause='library-dependent cache identity triggered an unused text-cache recomputation',
        expected_text_cache=expected['text_features'],unused_recomputed_text_cache=current['text_features'],
        mitigation='Read exact historical tensor and its pinned input identity; never invoke encoder in P1.A',
        no_training_results_from_failed_attempt=True)
    if not (REPORT/'vn_full_cache_mismatch_incident.json').exists():freeze(REPORT/'vn_full_cache_mismatch_incident.json',incident)
    study.vn.encode_cached=reuse_historical_text_cache
    execution=dict(status='registered_before_training',prior_decision_sha256=file_sha(REPORT/'vn_full_seeds_decision.json'),
        supersedes_execution_sha256=file_sha(REPORT/'vn_full_seeds_execution_decision.json'),
        runner_sha256=file_sha(__file__),output=str(OUTPUT),seeds=[42,43,44],reuse_seed=42,
        cache_only=True,encoder_weights_loaded=False,precision='FP32 cached features and FP32 head/loss; historical frozen cache convention retained',
        sampler='Historical Sampler with explicit per-run seed (its Python default is bound to 42)',
        parity_checks=checks,reserve_bytes=4*2**30,
        historical_result_sha256=file_sha(HISTORICAL/'VN_FULL/result.json'),
        historical_checkpoint_sha256=file_sha(HISTORICAL/'VN_FULL/best.pt'))
    freeze(REPORT/EXECUTION_DECISION,execution)
    with (study.vn.OUT/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        study.OUT=OUTPUT
        data=study.Data()
        if data.native.dtype!=torch.float32 or data.ctx.texts.dtype!=torch.float32:raise ValueError('Unexpected cached feature dtype')
        original_identity=copy.deepcopy(data.identity)
        results={42:read_json(HISTORICAL/'VN_FULL/result.json')}
        load_selected(data,results[42])
        for seed in (43,44):
            require_space(['/mnt/c','/mnt/d','/'],expected_bytes=32*2**20)
            study.SEED=seed;study.OUT=OUTPUT/f'seed-{seed}';study.OUT.mkdir(exist_ok=True)
            study.Sampler=sampler_for_seed(seed)
            data.identity=dict(original_identity,seed=seed,series_execution_sha256=file_sha(REPORT/EXECUTION_DECISION))
            initial_path,initial=initialize_seed(seed)
            results[seed]=study.train(data,'VN_FULL',data.plans['VN_FULL'],initial_path,initial)
            _,actual=load_selected(data,results[seed])
            write_json(study.OUT/'verification.json',dict(status='passed',checkpoint_sha256=results[seed]['checkpoint_sha256'],
                selected_updates=results[seed]['best_updates'],reload_VN_metrics_and_loss_reproduced=True))
            torch.cuda.empty_cache()
        ranked=sorted(results,key=lambda seed:(results[seed]['selected_VN_dev']['t2i']['recall@1'],seed))
        median=ranked[1]
        registration=dict(status='all_training_closed_before_new_COCO_evaluation',seed=median,
            selection_rule='median selected VN dev t2i R@1; ascending seed breaks ties',
            checkpoint_path=results[median]['checkpoint_path'],checkpoint_sha256=results[median]['checkpoint_sha256'],
            finalists=[dict(seed=median,checkpoint_path=results[median]['checkpoint_path'],sha256=results[median]['checkpoint_sha256'])],
            seed_checkpoints={str(seed):dict(path=r['checkpoint_path'],sha256=r['checkpoint_sha256'],
                selected_update=r['best_updates'],VN_dev=r['selected_VN_dev']) for seed,r in results.items()},
            COCO_used_for_selection=False,XM3600_scored=False)
        freeze(REPORT/'vn_full_finalist_decision.json',registration)
        metrics={}
        for direction in ('t2i','i2t'):
            metrics[direction]={}
            for metric in ('recall@1','recall@5','recall@10'):
                values=[results[s]['selected_VN_dev'][direction][metric] for s in (42,43,44)]
                metrics[direction][metric]=dict(values=values,mean=statistics.mean(values),sample_sd=statistics.stdev(values))
        summary=dict(status='training_and_VN_verification_complete',seeds=[42,43,44],selected_seed=median,VN_dev=metrics,
            checkpoint_selection='VN dev only',COCO_posthoc_pending=True,legacy_test_pending=True,XM3600_scored=False,
            GPU_seconds_new_runs={str(s):results[s]['history'][-1]['elapsed_seconds'] for s in (43,44)})
        freeze(REPORT/'vn_full_seeds_training_summary.json',summary)
        data.verify()
        print('P1_A_TRAINING_COMPLETE '+json.dumps(summary),flush=True)


if __name__=='__main__':main()
