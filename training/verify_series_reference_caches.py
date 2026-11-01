"""Reproduce three prior MSCOCO rank vectors from their existing raw caches."""
import hashlib
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from pl_mmb_benchmark.data import load
from pl_mmb_benchmark.evaluate import cosine_scores,rank
from pl_mmb_benchmark.models import MODELS,CLIP_IMAGE_REPO,CLIP_IMAGE_REVISION
from pl_mmb_training.series_20261023 import OUT,ROOT,read,freeze,file_sha,require_space


def cache_path(result,values,modality):
    identity=dict(policy='native-features-v2',repo=result['model'],revision=result['revision'],
        components=result.get('components',{}),precision=result['precision'],text_max_length=result['text_max_length'],
        modality=modality,values=values,torch=result['torch_version'],transformers=result['libraries']['transformers'])
    if modality=='images' and result['model'] in (MODELS['clip-b32-pl'],MODELS['clip-b32-pl-32768']):
        identity.update(repo=CLIP_IMAGE_REPO,revision=CLIP_IMAGE_REVISION,components={},text_max_length=None)
    if modality=='images' and result['model'] in (MODELS['siglip2-pl'],MODELS['siglip2-pl-32768'],MODELS['siglip2-base']):
        identity.update(repo='siglip2-512-shared-vision',revision='57b0579795b818d558016142ad391bb62323f6f653a987150f0f8e57ab7f830f',components={},text_max_length=None)
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    return Path.home()/'.cache/huggingface/pl_mmb_benchmark/embeddings'/f'{key}.pt'


def main():
    require_space(['/mnt/c','/mnt/d','/'],expected_bytes=64*2**20)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1'
    torch.set_num_threads(4)
    examples,sha=load('mscoco_t2i')
    images=list(dict.fromkeys(p for e in examples for p in e.images));texts=list(dict.fromkeys(t for e in examples for t in e.texts))
    registration=read(OUT/'reference_eval_registration.json')
    available={}
    for item in registration['reference_models']:
        for previous in item['historical_results']:
            path=ROOT/previous['path'].replace('\\','/')
            result=read(path)
            if result.get('dataset')!='mscoco_t2i' or result['translations_sha256']!=sha:continue
            image_cache=cache_path(result,images,'images');text_cache=cache_path(result,texts,'texts')
            if image_cache.is_file() and text_cache.is_file():
                available[item['alias']]=dict(result_path=str(path),result_sha256=file_sha(path),
                    predictions_path=str(path.parent/'predictions.jsonl'),
                    image_cache=str(image_cache),text_cache=str(text_cache),
                    image_cache_sha256=file_sha(image_cache),text_cache_sha256=file_sha(text_cache))
    priority=['siglip2-base-256','tipsv2-b14','bge-vl-base']
    if not all(alias in available for alias in priority):
        freeze(OUT/'reference_cache_availability.json',available)
        raise ValueError('A preselected cache-parity model is unavailable; do not substitute after observing ranks')
    decision=dict(status='registered_before_cache_rank_reproduction',models=priority,
        supersedes_initial_parity_models=['siglip2-base-256','metaclip-s16','clip-b32-pl'],
        reason='Initial historical MetaCLIP/CLIP runs predate raw embedding cache persistence. Select three available caches before any new-dataset model scores.',
        model_list_unchanged=True,script_sha256=file_sha(__file__),translations_sha256=sha,
        inputs={alias:available[alias] for alias in priority},on_rank_mismatch='stop series')
    freeze(OUT/'reference_cache_parity_decision.json',decision)
    checks={}
    for alias in priority:
        item=available[alias]
        image_features=torch.load(item['image_cache'],weights_only=True,map_location='cpu')
        text_features=torch.load(item['text_cache'],weights_only=True,map_location='cpu')
        ii={p:i for i,p in enumerate(images)};ti={t:i for i,t in enumerate(texts)}
        expected=[json.loads(line) for line in Path(item['predictions_path']).read_text().splitlines()]
        actual=[]
        for e in examples:
            scores=cosine_scores(text_features[ti[e.texts[0]]],None,image_features[[ii[p] for p in e.images]])
            actual.append(dict(id=e.id,rank=rank(scores)))
        mismatches=[dict(id=a['id'],actual=a['rank'],expected=b['rank']) for a,b in zip(actual,expected)
                    if a['id']!=b['id'] or a['rank']!=b['rank']]
        checks[alias]=dict(queries=len(actual),all_ranks_identical=len(actual)==len(expected)==1000 and not mismatches,
                          mismatch_examples=mismatches[:10],prior_result_sha256=item['result_sha256'])
        if not checks[alias]['all_ranks_identical']:
            freeze(OUT/'reference_cache_parity_failure.json',checks);raise ValueError('Historical cache rank mismatch; series stopped')
        print('CACHE_PARITY_PASSED '+alias+' 1000/1000 exact ranks',flush=True)
    freeze(OUT/'reference_cache_parity.json',dict(status='passed',checks=checks,
        decision_sha256=file_sha(OUT/'reference_cache_parity_decision.json'),encoder_runs=0,weights_loaded=False))


if __name__=='__main__':main()
