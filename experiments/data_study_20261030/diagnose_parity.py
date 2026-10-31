"""Read-only encoder/cache diagnosis, no fitting or benchmark scoring."""
import gc
import json
import time
import torch
from run import Data, ROOT, REPORT, SERIES, VN, read, reload, gpu_lock, status, atomic_json, rows
from pl_mmb_benchmark.projector_adapter import ProjectorEncoder


def compare(a,b):
    cos=torch.nn.functional.cosine_similarity(a,b)
    return dict(min_cosine=float(cos.min()),per_row_cosine=cos.tolist(),
                max_abs=float((a-b).abs().max()),norm_a=a.norm(dim=1).tolist(),norm_b=b.norm(dim=1).tolist())


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    status('diagnosing_encoder_cache_parity',new_training_runs_started=False)
    with gpu_lock(False):
        data=Data();reg=read(SERIES/'vn_full_finalist_decision.json')
        r0=read(__import__('pathlib').Path(reg['seed_checkpoints']['44']['path']).with_name('result.json'))
        model=reload(data,r0);adapter=ProjectorEncoder(r0['checkpoint_path'],batch_size=32)
        images=data.protocols['wiki_native_dev']['images'][:8];texts=data.text_values[:16]
        with torch.inference_mode():
            cached=model(data.native[[data.ii[s] for s in images]].to('cuda')).cpu()
        actual=adapter.encode(images,images=True);expected_t=data.texts[[data.ti[s] for s in texts]]
        actual_t=adapter.encode(texts)
        report=dict(status='diagnosis_only_no_training',images=compare(cached,actual),texts=compare(expected_t,actual_t),
            components=adapter.components,torch=torch.__version__,text_config=data.base_config['text'],
            original_cache_identity=read(read(VN/'native_features.json')['identity']['data']['text_features']['path'].replace('.pt','.json')))
        report['original_cache_identity'].pop('values',None)
        # Reconstruct exact extraction batches. BF16 kernels may differ with
        # shape; this tests equal inputs and batching without weakening a gate.
        val=list(rows(ROOT/'training/runs/clean-v1/validation.jsonl'))
        candidate=list(rows(VN/'data/candidates.jsonl'))
        known=set(sorted({r['image'] for r in data.full+val})+[r['image'] for r in candidate])
        missing={r['image']:r for r in data.new+data.wiki if r['image'] not in known}
        by_target={r['target_id'] for r in data.full+data.old+data.dev}
        for protocol in data.protocols.values():
            for p,uid in zip(protocol['images'],protocol['image_ids']):
                if p not in known and uid not in by_target:missing.setdefault(p,dict(image=p))
        plain=[p for p,r in missing.items() if not r.get('shard')]
        matched=[]
        for start in range(0,len(plain),32):
            batch=plain[start:start+32]
            if not set(batch)&set(images):continue
            actual=adapter.encode(batch,images=True)
            with torch.inference_mode():
                expected=model(data.native[[data.ii[s] for s in batch]].to('cuda')).cpu()
            matched.append(dict(batch_start=start,batch_size=len(batch),comparison=compare(expected,actual)))
        report['matched_extraction_batches']=matched
        atomic_json(REPORT/'parity_diagnosis.json',report)
        print(json.dumps(report),flush=True)
        del adapter,data;gc.collect();torch.cuda.empty_cache()
    status('encoder_cache_parity_diagnosed',new_training_runs_started=False)


if __name__=='__main__':main()
