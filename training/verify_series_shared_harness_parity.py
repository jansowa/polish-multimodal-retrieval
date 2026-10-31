"""Verify shared FP32 cosine/ranking against preselected historical cache ranks."""
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_DATASETS_OFFLINE']='1'
import numpy as np
import torch
from pl_mmb_benchmark.data import load
from pl_mmb_benchmark.full_gallery import cosine_matrix,per_query
from pl_mmb_training.series_20261023 import ROOT,OUT,read,freeze,file_sha,require_space


def main():
    require_space(['/mnt/c','/mnt/d','/'])
    require_space(['/mnt/c'],expected_bytes=128*2**20)
    torch.set_num_threads(2)
    prior=read(OUT/'reference_cache_parity_decision.json')
    examples,translation_sha=load('mscoco_t2i')
    if translation_sha!=prior['translations_sha256']:raise ValueError('Historical translations changed')
    images=list(dict.fromkeys(p for e in examples for p in e.images))
    texts=list(dict.fromkeys(t for e in examples for t in e.texts))
    ii={p:i for i,p in enumerate(images)};ti={t:i for i,t in enumerate(texts)}
    decision=dict(status='registered_before_shared_harness_rank_verification',models=prior['models'],
        inputs=prior['inputs'],prior_decision_sha256=file_sha(OUT/'reference_cache_parity_decision.json'),
        script_sha256=file_sha(__file__),harness_sha256=file_sha(ROOT/'src/pl_mmb_benchmark/full_gallery.py'),
        model_list_unchanged=True,encoder_runs=0,on_mismatch='stop before new-dataset retrieval')
    decision_path=OUT/'reference_shared_harness_parity_decision_20261029.json'
    freeze(decision_path,decision)
    checks={}
    for alias in prior['models']:
        item=prior['inputs'][alias]
        for field in ('image_cache','text_cache'):
            if file_sha(Path(item[field]))!=item[field+'_sha256']:raise ValueError('Frozen feature cache differs')
        im=torch.load(item['image_cache'],weights_only=True,map_location='cpu').float().numpy()
        text=torch.load(item['text_cache'],weights_only=True,map_location='cpu').float().numpy()
        expected=[json.loads(line) for line in Path(item['predictions_path']).read_text().splitlines()]
        mismatches=[]
        for e,old in zip(examples,expected):
            score=cosine_matrix(text[[ti[e.texts[0]]]],im[[ii[p] for p in e.images]])
            positives=np.zeros(score.shape,dtype=bool);positives[0,0]=True
            rank=int(round(1/per_query(score,positives)['mrr'][0]))
            if e.id!=old['id'] or rank!=old['rank']:
                mismatches.append(dict(id=e.id,actual_rank=rank,expected_rank=old['rank']))
        passed=len(examples)==len(expected)==1000 and not mismatches
        checks[alias]=dict(status='passed' if passed else 'failed',queries=len(expected),mismatches=mismatches[:10])
        if not passed:
            freeze(OUT/'reference_shared_harness_parity_failure_20261029.json',checks)
            raise ValueError('Shared harness historical rank mismatch')
        print('SHARED_HARNESS_PARITY '+alias+' 1000/1000',flush=True)
    freeze(OUT/'reference_shared_harness_parity_20261029.json',dict(status='passed',checks=checks,
        decision_sha256=file_sha(decision_path),harness_sha256=decision['harness_sha256'],encoder_runs=0))


if __name__=='__main__':main()
