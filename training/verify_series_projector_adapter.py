"""Numerical encoder/checkpoint parity before any new-dataset retrieval score."""
import fcntl
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
os.environ['HF_HUB_OFFLINE']='1'
import torch
from pl_mmb_benchmark.projector_adapter import ProjectorEncoder
from pl_mmb_training.series_20261023 import ROOT,OUT,VN,read,rows,freeze,file_sha,require_space


def compare(actual,expected):
    cosine=torch.nn.functional.cosine_similarity(actual,expected)
    relative=(actual-expected).norm(dim=1)/expected.norm(dim=1)
    return dict(min_cosine=float(cosine.min()),max_relative_l2=float(relative.max()),
                max_absolute=float((actual-expected).abs().max()))


def main():
    require_space(['/mnt/c','/mnt/d','/'])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    checkpoint=Path('/mnt/c/Users/jsowa/benchmark-translations/series-2026-10-23/runs/vn_full_seeds/seed-44/VN_FULL/best.pt')
    original=rows(VN/'vn_dev.jsonl');prepared={r['id']:r for r in rows(OUT/'vn_dev_protocol/manifest.jsonl')}
    selected=sorted(original,key=lambda r:r['id'])[:16]
    decision=dict(status='registered_before_numerical_adapter_verification',checkpoint_sha256=file_sha(checkpoint),
        selected_ids=[r['id'] for r in selected],adapter_sha256=file_sha(ROOT/'src/pl_mmb_benchmark/projector_adapter.py'),
        image_tolerance=dict(min_cosine=.9999,max_relative_l2=.02),
        text_tolerance=dict(min_cosine=.9999,max_relative_l2=.02),
        no_retrieval_scores=True,models_loaded='our two frozen encoders only; local files only')
    decision_path=OUT/'projector_adapter_verification_decision_20261029.json'
    freeze(decision_path,decision)
    base=rows(ROOT/'training/runs/clean-v1/FULL_CLEAN.jsonl')+rows(ROOT/'training/runs/clean-v1/validation.jsonl')
    all_images=sorted({r['image'] for r in base})+[r['image'] for r in rows(VN/'candidates.jsonl')]
    ii={path:i for i,path in enumerate(all_images)}
    native_meta=read(VN.parent/'native_features.json')
    native_path=VN.parent/'native_features.pt'
    if file_sha(native_path)!=native_meta['sha256']:raise ValueError('Historical native cache changed')
    native=torch.load(native_path,weights_only=True,map_location='cpu')
    initial=read(ROOT/'training/runs/e1-full/cache.json')
    old_text_meta=initial['text_features']
    extra_meta=native_meta['identity']['data']['text_features']
    tensors={}
    for name,meta in [('old',old_text_meta),('extra',extra_meta)]:
        path=Path(meta['path'])
        if file_sha(path)!=meta['sha256']:raise ValueError('Historical text cache changed')
        tensors[name]=torch.load(path,weights_only=True,map_location='cpu')
    old_index={t:i for i,t in enumerate(initial['texts'])}
    extra_index={t:i for i,t in enumerate(read(Path(extra_meta['path']).with_suffix('.json'))['values'])}
    expected_text=torch.stack([tensors['old'][old_index[r['text_pl']]] if r['text_pl'] in old_index
                               else tensors['extra'][extra_index[r['text_pl']]] for r in selected])
    with (VN.parent/'worker.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        encoder=ProjectorEncoder(checkpoint,batch_size=16)
        with torch.inference_mode():
            expected_image=encoder.projector(native[[ii[r['image']] for r in selected]].to('cuda')).cpu()
        actual_image=encoder.encode([prepared[r['id']]['image'] for r in selected],images=True)
        actual_text=encoder.encode([r['text_pl'] for r in selected])
        comparisons=dict(images=compare(actual_image,expected_image),texts=compare(actual_text,expected_text))
        passed=all(v['min_cosine']>=.9999 and v['max_relative_l2']<=.02 for v in comparisons.values())
        report=dict(status='passed' if passed else 'failed',comparisons=comparisons,examples=len(selected),
            decision_sha256=file_sha(decision_path),torch=torch.__version__,no_retrieval_scores=True,
            parameter_count=encoder.parameter_count,precision=encoder.precision,
            notes='Original VN bytes are decoded by the pinned Linux Pillow version; historical native RGB quantization may introduce tiny rounding differences')
        freeze(OUT/'projector_adapter_verification_20261029.json',report)
        print(json.dumps(report,indent=2),flush=True)
        if not passed:raise ValueError('Projector adapter does not reproduce frozen embedding space')


if __name__=='__main__':main()
