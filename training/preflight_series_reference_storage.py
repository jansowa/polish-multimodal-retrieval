"""Estimate retained reference weights from pinned HF metadata; never download weights."""
import concurrent.futures
import json
import os
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pl_mmb_training.series_20261023 import ROOT,OUT,read,freeze,file_sha,require_space
os.environ['REQUESTS_CA_BUNDLE']=str(ROOT/'combined-ca.pem')
os.environ['SSL_CERT_FILE']=str(ROOT/'combined-ca.pem')
from huggingface_hub import HfApi


def main():
    require_space(['/mnt/c','/mnt/d','/'])
    registration=read(OUT/'reference_eval_registration.json')
    jobs=[dict(repo=r['model'],revision=r['revision'],alias=r['alias']) for r in registration['reference_models']]
    jobs.extend([
        dict(repo='openai/clip-vit-base-patch32',revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',alias='shared-clip-image'),
        dict(repo='timm/vit_base_patch16_plus_clip_240.laion400m_e31',revision='b83f7d1477f5cb56091b6da84b5a9d47674bc93b',alias='mclip-image-component')])
    # Read the actual image-component revision from the implementation.
    from pl_mmb_benchmark.backends import MCLIP_IMAGE_REVISION
    jobs[-1]['revision']=MCLIP_IMAGE_REVISION
    def one(item):
        api=HfApi()
        info=api.model_info(item['repo'],revision=item['revision'],files_metadata=True,timeout=40)
        if info.sha!=item['revision']:raise ValueError('Pinned revision differs')
        siblings=info.siblings
        alias=item['alias']
        if alias=='mobileclip2-s2':names={'mobileclip2_s2.pt'}
        elif alias=='pe-core-b16':names={'PE-Core-B16-224.pt'}
        elif alias=='mclip-b16plus':names={'pytorch_model.bin'}
        elif alias in ('siglip-i18n-256','siglip2-b32-256','fashion-siglip','mclip-image-component'):
            names={'open_clip_model.safetensors'}
        else:
            safes=[s.rfilename for s in siblings if s.rfilename.endswith('.safetensors') and 'open_clip' not in s.rfilename]
            names=set(safes or [s.rfilename for s in siblings if s.rfilename.endswith('.bin')])
            if alias in ('clip-b32-pl','clip-b32-pl-32768'):
                names.update(s.rfilename for s in siblings if s.rfilename.endswith('.bin') and '/' in s.rfilename)
        files=[]
        snapshot=Path.home()/'.cache/huggingface/hub'/('models--'+item['repo'].replace('/','--'))/'snapshots'/item['revision']
        for sibling in siblings:
            if sibling.rfilename not in names:continue
            target=snapshot/sibling.rfilename
            cached=target.is_file() and target.stat().st_size==sibling.size
            files.append(dict(name=sibling.rfilename,bytes=sibling.size,cached=cached))
        if not files:raise ValueError('No loader-compatible weights in metadata: '+alias)
        result=dict(**item,files=files,weight_bytes=sum(f['bytes'] for f in files),
                    missing_weight_bytes=sum(f['bytes'] for f in files if not f['cached']))
        print('WEIGHT_METADATA '+alias+' missing_MiB='+str(round(result['missing_weight_bytes']/2**20,1)),flush=True)
        return result
    results=[];errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(one,item):item for item in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:results.append(future.result())
            except Exception as exc:errors.append(dict(**futures[future],error=type(exc).__name__+': '+str(exc)))
    missing=sum(r['missing_weight_bytes'] for r in results)
    free=shutil.disk_usage('/mnt/c').free
    overhead=2*2**30
    result=dict(status='metadata_complete' if not errors else 'metadata_incomplete',
        registration_sha256=file_sha(OUT/'reference_eval_registration.json'),
        policy='retain all downloaded weights; user performs every deletion manually',
        models=sorted(results,key=lambda r:r['alias']),errors=errors,
        missing_weight_bytes=missing,additional_cache_and_metadata_reserve_bytes=overhead,
        physical_C_free_bytes=free,minimum_free_reserve_bytes=4*2**30,
        entire_series_fits_without_deletion=not errors and free-missing-overhead>=4*2**30,
        no_weights_downloaded=True)
    freeze(OUT/'reference_storage_preflight_20261029.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='models'},indent=2),flush=True)


if __name__=='__main__':main()
