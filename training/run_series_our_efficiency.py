"""Measure our registered model once; projector seed does not change architecture."""
import fcntl
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
os.environ['HF_HUB_OFFLINE']='1'
from pl_mmb_training.series_20261023 import ROOT,OUT,read,freeze,file_sha,require_space


def main():
    from pl_mmb_benchmark.projector_adapter import ProjectorEncoder
    from pl_mmb_benchmark.efficiency import measure_efficiency
    registration=read(OUT/'reference_eval_registration_v2_20261029.json')
    checkpoint=registration['our_checkpoints'][registration['selected_finalist']]
    require_space(['/mnt/c','/mnt/d','/'])
    proof=OUT/'our_efficiency_decision_20261029.json'
    freeze(proof,dict(status='registered_before_efficiency_measurement',checkpoint_sha256=checkpoint['sha256'],
        procedure_sha256=file_sha(ROOT/'src/pl_mmb_benchmark/efficiency.py'),
        model=registration['selected_finalist'],cpu_fp32_threads=2,gpu_batch_size=64,
        warmup=10,repetitions=100,no_embedding_cache=True,no_retrieval_scores=True))
    with (ROOT/'training/runs/tinyclip-visualnews-v1/worker.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        encoder=ProjectorEncoder(checkpoint['path'],batch_size=64)
        dataset=read(OUT/'vn_dev_protocol/retrieval_protocol.json')
        result=measure_efficiency(encoder,dataset['images'],dataset['texts'])
        result.update(model=registration['selected_finalist'],decision_sha256=file_sha(proof),
            registration_sha256=file_sha(OUT/'reference_eval_registration_v2_20261029.json'))
        freeze(OUT/'our_efficiency_20261029.json',result)
        print(result,flush=True)


if __name__=='__main__':main()
