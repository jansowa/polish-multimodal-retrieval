"""Evaluate preregistered frozen projectors with the shared full-gallery harness."""
import fcntl
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
os.environ['HF_HUB_OFFLINE']='1'
import torch
from pl_mmb_benchmark.full_gallery import evaluate_encoder
from pl_mmb_benchmark.projector_adapter import ProjectorEncoder
from pl_mmb_training.series_20261023 import ROOT,OUT,read,freeze,file_sha,require_space

CACHE=Path('/mnt/c/Users/jsowa/benchmark-translations/series-2026-10-23/evaluation')


class CachedEncoder:
    def __init__(self,encoder):self.encoder=encoder;self.entries=[]

    def encode(self,values,images=False):
        modality='images' if images else 'texts'
        identity=dict(modality=modality,values=values,precision=self.encoder.precision,
            torch=torch.__version__,max_text_length=128,
            components=dict(self.encoder.components))
        if not images:
            identity['components']={k:v for k,v in identity['components'].items() if k in ('text','text_revision','text_prefix')}
        fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        path=CACHE/'features'/f'{fingerprint}.pt'
        if path.exists():
            result=torch.load(path,weights_only=True,map_location='cpu')
        else:
            require_space(['/mnt/c','/mnt/d','/'])
            require_space(['/mnt/c'],expected_bytes=len(values)*384*4+1024**2)
            result=self.encoder.encode(values,images=images)
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream:torch.save(result,stream)
            freeze(path.with_suffix('.json'),dict(identity=identity,sha256=file_sha(path)))
        if result.shape!=(len(values),384) or not torch.isfinite(result).all() or (result.norm(dim=1)==0).any():
            raise ValueError('Invalid cached native features')
        meta=read(path.with_suffix('.json'))
        if meta['identity']!=identity or file_sha(path)!=meta['sha256']:raise ValueError('Embedding cache identity/SHA differs')
        self.entries.append(dict(path=str(path),sha256=meta['sha256'],modality=modality))
        return result

    def encode_text_candidates(self,values):return self.encode(values)


def main():
    require_space(['/mnt/c','/mnt/d','/'])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    reg_path=OUT/'reference_eval_registration_v2_20261029.json';registration=read(reg_path)
    if registration['status']!='closed_ready_for_execution':raise ValueError('Execution registration not closed')
    decision_path=OUT/'our_full_gallery_execution_decision_20261029.json'
    freeze(decision_path,dict(status='registered_before_first_new_gallery_score',registration_sha256=file_sha(reg_path),
        runner_sha256=file_sha(__file__),models=list(registration['our_checkpoints']),
        datasets=[d['name'] for d in registration['datasets']],GPU_workers=1,
        modes=['v2_no_instructions','author_recommended'],
        author_mode_note='Our [query]: prefix is intrinsic and identical in both modes; reuse identical features',
        no_training=True,no_XM3600=True,output=str(CACHE),automatic_deletion=False))
    CACHE.mkdir(parents=True,exist_ok=True)
    names=[registration['selected_finalist']]+[k for k in registration['our_checkpoints'] if k!=registration['selected_finalist']]
    summaries={}
    with (ROOT/'training/runs/tinyclip-visualnews-v1/worker.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in names:
            checkpoint=registration['our_checkpoints'][name]
            if file_sha(Path(checkpoint['path']))!=checkpoint['sha256']:raise ValueError('Registered checkpoint changed')
            completed=[CACHE/name/d['name']/'v2_no_instructions.json' for d in registration['datasets']]
            if all(p.exists() for p in completed):
                print('OUR_ALREADY_COMPLETE '+name,flush=True)
                continue
            encoder=ProjectorEncoder(checkpoint['path'],batch_size=64)
            cached=CachedEncoder(encoder)
            summaries[name]={}
            for item in registration['datasets']:
                target=CACHE/name/item['name']/'v2_no_instructions.json'
                if target.exists():continue
                require_space(['/mnt/c','/mnt/d','/'])
                path=ROOT/item['protocol_path']
                if file_sha(path)!=item['protocol_sha256']:raise ValueError('Frozen retrieval protocol changed')
                dataset=dict(read(path),prompt_mode='v2_no_instructions')
                started=time.monotonic()
                result=evaluate_encoder(cached,dataset,registration)
                result.update(model=name,checkpoint_sha256=checkpoint['sha256'],
                    registration_sha256=file_sha(reg_path),runner_sha256=file_sha(__file__),
                    protocol_file_sha256=file_sha(path),precision=encoder.precision,
                    torch=torch.__version__,feature_caches=list(cached.entries),elapsed_seconds=time.monotonic()-started)
                freeze(target,result)
                sensitivity=dict(result,prompt_mode='author_recommended',
                    sensitivity_note='Same registered intrinsic [query]: prefix; identical features, no second encoder run')
                freeze(target.with_name('author_recommended.json'),sensitivity)
                summaries[name][item['name']]={direction:value['metrics'] for direction,value in result['directions'].items()}
                if name==registration['selected_finalist'] and item['name']=='wiki_native_dev':
                    baseline=OUT/'wiki_dev_zero_shot_pre_wiki_training.json'
                    freeze(baseline,dict(status='immutable_pre_Wiki_baseline',model=name,
                        checkpoint_sha256=checkpoint['sha256'],never_trained_on_Wiki=True,
                        registration_sha256=file_sha(reg_path),result_path=str(target),result_sha256=file_sha(target),
                        result=result,note='Historical Wiki-trained pilot checkpoints are separate and are not used here; no new Wiki training released'))
                print('OUR_RESULT '+name+' '+item['name']+' '+json.dumps(summaries[name][item['name']]),flush=True)
            del cached,encoder;gc.collect();torch.cuda.empty_cache()
    freeze(OUT/'our_full_gallery_summary_20261029.json',dict(status='complete',results=summaries,
        registration_sha256=file_sha(reg_path),output=str(CACHE),XM3600_scored=False,new_Wiki_training_started=False))


if __name__=='__main__':main()
