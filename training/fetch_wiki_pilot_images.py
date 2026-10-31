"""Windows: freeze published wiki JSONL and use the producer's shared HTTP limiter.

Original images are checked against pinned SHA1; official renders require API
confirmation of that original SHA1. Raw downloads are transient; retain native
224 RGB, small previews and downloaded-image identities.
"""
import base64
import hashlib
import io
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime,timezone
from contextlib import contextmanager,ExitStack
from pathlib import Path
from urllib.parse import urlparse
from urllib.error import HTTPError

import numpy as np
from PIL import Image,ImageOps
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pl_mmb_collection.config import HttpConfig,FilterConfig
from pl_mmb_collection.pipeline.filters import license_is_allowed
from pl_mmb_collection.sources.http import HttpClient
from pl_mmb_collection.sources.rate_limit import RetryLater,RateLimiter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'training/runs/tinyclip-wiki-pilot-v1'


class WikiRateLimiter(RateLimiter):
    @contextmanager
    def shared_state(self):
        # Windows byte locks are mandatory: even the existing initialization
        # read can fail while another producer owns byte zero. Wait and retry
        # acquisition, retaining the original lock and shared budget protocol.
        while True:
            with ExitStack() as stack:
                try:stack.enter_context(super().shared_state())
                except PermissionError:
                    time.sleep(.2)
                    continue
                yield
                return


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(path)


def freeze():
    folder=OUT/'data';folder.mkdir(parents=True,exist_ok=True)
    state=folder/'snapshot.json'
    if state.exists():
        value=json.loads(state.read_text(encoding='utf-8'))
        assert sha(folder/'wiki_pairs_snapshot.jsonl')==value['pairs_sha256']
        assert sha(folder/'pipeline_manifest.json')==value['manifest_sha256']
        return value
    source=(ROOT/'reports/wikimedia_pipeline/v1').resolve()
    shutil.copyfile(source/'final_pairs.jsonl',folder/'wiki_pairs_snapshot.jsonl')
    shutil.copyfile(source/'manifest.json',folder/'pipeline_manifest.json')
    rows=[json.loads(l) for l in (folder/'wiki_pairs_snapshot.jsonl').read_text(encoding='utf-8').splitlines()]
    manifest=json.loads((folder/'pipeline_manifest.json').read_text(encoding='utf-8'))
    assert len({r['id'] for r in rows})==len(rows)
    assert all(r['pipeline_fingerprint']==manifest['fingerprint'] for r in rows)
    value=dict(created_at=datetime.now(timezone.utc).isoformat(),source=str(source),rows=len(rows),
        pairs_sha256=sha(folder/'wiki_pairs_snapshot.jsonl'),manifest_sha256=sha(folder/'pipeline_manifest.json'),
        fingerprint=manifest['fingerprint'],live_SQLite_accessed=False)
    write(state,value);print('FROZEN_WIKI',json.dumps(value),flush=True)
    return value


def native_rgb(image):
    """Slow CLIP resize(shortest=224, bicubic), then center crop; parity checked in WSL."""
    width,height=image.size
    if width<=height:size=(224,int(224*height/width))
    else:size=(int(224*width/height),224)
    resized=image.resize(size,Image.Resampling.BICUBIC)
    left=(size[0]-224)//2;top=(size[1]-224)//2
    return np.array(resized.crop((left,top,left+224,top+224)),dtype=np.uint8).transpose(2,0,1)


def sha1number(value):
    value=value.lower()
    return int(value,16) if len(value)==40 and all(c in '0123456789abcdef' for c in value) else int(value,36)


def decode(raw,record,api_evidence=None):
    expected=record['source_sha1'].lower()
    actual=hashlib.sha1(raw).hexdigest()
    original_matches=sha1number(actual)==sha1number(expected)
    if api_evidence is None:
        assert original_matches,'Original image no longer matches published SHA1'
    else:
        assert sha1number(api_evidence['sha1'])==sha1number(expected),'API source SHA1 differs from frozen record'
    with Image.open(io.BytesIO(raw)) as original:
        canonical=ImageOps.exif_transpose(original).convert('RGB')
        pixel=hashlib.sha256(str(canonical.size).encode()+canonical.tobytes()).hexdigest()
        gray=canonical.convert('L').resize((32,32),Image.Resampling.LANCZOS)
        rgb=original.convert('RGB')
        native=native_rgb(rgb)
        identity=dict(sha256=hashlib.sha256(raw).hexdigest(),pixels_sha256=pixel,
            source_sha1_verified=expected,source_original_bytes_verified=original_matches,
            acquisition_mode='original_SHA1_verified' if api_evidence is None else 'API_SHA1_verified_official_render_1280',
            api_evidence=api_evidence,decoded_size=list(original.size),
            gray32_base64=base64.b64encode(gray.tobytes()).decode(),
            native_row_sha256=hashlib.sha256(native.tobytes()).hexdigest())
    return native,identity


def render_plan(rows,client,folder):
    target=folder/'render_plan.json'
    if target.exists():return json.loads(target.read_text(encoding='utf-8'))
    pending=[(i,p) for i,p in enumerate(rows) if not (folder/'decoded'/f'{i:05d}.json').exists()
        or json.loads((folder/'decoded'/f'{i:05d}.json').read_text(encoding='utf-8')).get('status')!='ready']
    entries={}
    for start in range(0,len(pending),40):
        batch=pending[start:start+40]
        titles=['File:'+p['source_record']['image_filename'].removeprefix('File:') for _,p in batch]
        while True:
            try:
                response=client.get('https://commons.wikimedia.org/w/api.php',action='query',prop='imageinfo',
                    iiprop='url|sha1|size|timestamp',iiurlwidth=1280,titles='|'.join(titles))
                break
            except RetryLater as exc:
                print(f'HTTP_API_WAIT seconds={max(0,exc.retry_at-time.time()):.1f}',flush=True)
                time.sleep(min(10,max(.1,exc.retry_at-time.time())))
        images={sha1number(info['sha1']):dict(info,api_request_key=response['_request_key'])
            for page in response.get('query',{}).get('pages',{}).values() for info in page.get('imageinfo',[])}
        for index,pair in batch:
            expected=sha1number(pair['source_record']['source_sha1']);info=images.get(expected)
            if info:
                entries[str(index)]=dict(url=info.get('thumburl') or info['url'],sha1=info['sha1'],
                    source_timestamp=info.get('timestamp'),api_request_key=info['api_request_key'],
                    original_width=info['width'],original_height=info['height'],original_bytes=info['size'])
            else:entries[str(index)]=dict(error='Original SHA1 missing or changed in current Commons API response')
        print(f'WIKI_RENDER_METADATA {min(start+40,len(pending))}/{len(pending)}',flush=True)
    write(target,entries);return entries


def main():
    os.chdir(ROOT)
    snapshot=freeze();folder=OUT/'data'
    rows=[json.loads(l) for l in (folder/'wiki_pairs_snapshot.jsonl').read_text(encoding='utf-8').splitlines()]
    (folder/'decoded').mkdir(exist_ok=True);(folder/'previews').mkdir(exist_ok=True);(folder/'parity_raw').mkdir(exist_ok=True)
    complete=folder/'download_complete.json'
    if complete.exists():
        value=json.loads(complete.read_text(encoding='utf-8'));assert value['snapshot']==snapshot
        assert sha(folder/'wiki_native_rgb.npy')==value['native_sha256']
        print('WIKI_DOWNLOAD_CACHE_COMPLETE',flush=True);return
    path=folder/'wiki_native_rgb.npy'
    pixels=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype='uint8',shape=(len(rows),3,224,224))
    assert pixels.shape==(len(rows),3,224,224)
    source_manifest=json.loads((ROOT/'outputs/polish_culture_fast/manifest.json').read_text(encoding='utf-8'))
    config=dict(source_manifest['config']['http'])
    state=Path(config['state_path']);config['state_path']=str(state if state.is_absolute() else ROOT/state)
    os.environ['SSL_CERT_FILE']=str(ROOT/'combined-ca.pem');os.environ['REQUESTS_CA_BUNDLE']=str(ROOT/'combined-ca.pem')
    successes=failures=0;started=time.monotonic();parity_count=len(list((folder/'parity_raw').glob('*.bin')))
    with HttpClient(HttpConfig(**config),OUT/'network/raw') as client:
        client.limiter=WikiRateLimiter(client.config,client.limiter.path)
        renders=render_plan(rows,client,folder)
        for index,pair in enumerate(rows):
            meta=folder/'decoded'/f'{index:05d}.json'
            if meta.exists():
                old=json.loads(meta.read_text(encoding='utf-8'))
                if old['status']=='ready':
                    assert old['native_row_sha256']==hashlib.sha256(pixels[index].tobytes()).hexdigest();successes+=1
                    continue
            record=pair['source_record'];evidence=renders.get(str(index))
            if not evidence or evidence.get('error'):
                write(meta,dict(status='failed',id=pair['id'],cache_index=index,error=(evidence or {}).get('error','No verified render')))
                failures+=1;continue
            url=evidence['url'];parsed=urlparse(url)
            assert parsed.scheme=='https' and parsed.hostname in {'upload.wikimedia.org','thumb.wikimedia.org'}
            assert license_is_allowed(record.get('license'),record.get('license_url'),list(FilterConfig().allowed_licenses))
            temporary=folder/'current_original.bin';attempt=0
            while True:
                try:
                    client.download(url,temporary,max_bytes=50_000_000)
                    # download() does not call success(), whereas get() does.
                    # Use the same gradual recovery after successful media responses.
                    client.limiter.success()
                    raw=temporary.read_bytes();native,identity=decode(raw,record,evidence)
                    pixels[index]=native;pixels.flush()
                    preview=folder/'previews'/f'{index:05d}.jpg'
                    Image.fromarray(native.transpose(1,2,0)).save(preview,'JPEG',quality=90)
                    if parity_count<12:
                        raw_path=folder/'parity_raw'/f'{index:05d}.bin';raw_path.write_bytes(raw);parity_count+=1
                    write(meta,dict(status='ready',id=pair['id'],cache_index=index,original_url=url,
                        native_preview=str(preview),preview_sha256=sha(preview),**identity))
                    temporary.unlink(missing_ok=True);successes+=1
                    break
                except RetryLater as exc:
                    print(f'HTTP_BUDGET_WAIT {exc.scope} seconds={max(0,exc.retry_at-time.time()):.1f}',flush=True)
                    time.sleep(min(10,max(.1,exc.retry_at-time.time())))
                except Exception as exc:
                    if isinstance(exc,PermissionError) and attempt==0:
                        print('LOCAL_DOWNLOAD_PERMISSION_TRACE '+traceback.format_exc(limit=8),flush=True)
                    attempt+=1;temporary.unlink(missing_ok=True)
                    permanent=isinstance(exc,(AssertionError,ValueError)) or (isinstance(exc,HTTPError) and exc.code in {400,401,403,404,410})
                    if attempt<3 and not permanent:
                        time.sleep(min(10,2**attempt));continue
                    write(meta,dict(status='failed',id=pair['id'],cache_index=index,error=type(exc).__name__+': '+str(exc)[:500]))
                    print(f'IMAGE_FAILED {pair["id"]}: {type(exc).__name__}: {str(exc)[:160]}',flush=True)
                    failures+=1;break
            if (index+1)%25==0:
                print(f'WIKI_IMAGES {index+1}/{len(rows)} ready={successes} failed={failures} seconds={time.monotonic()-started:.0f}',flush=True)
                write(OUT/'status.json',dict(status='downloading',done=index+1,total=len(rows),ready=successes,failed=failures))
    pixels.flush()
    write(complete,dict(snapshot=snapshot,ready=successes,failed=failures,native_sha256=sha(path),
        native_size_bytes=path.stat().st_size,preprocessing='PIL bicubic shortest224 then center224; verify against pinned HF processor before training',
        shared_HTTP_state=config['state_path'],raw_bulk_retained=False,pillow=Image.__version__,numpy=np.__version__))
    print(f'WIKI_DOWNLOAD_COMPLETE ready={successes} failed={failures}',flush=True)


if __name__=='__main__':main()
