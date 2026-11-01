"""Acquire native Wiki candidate originals on C through the scraper's shared budget."""
import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pl_mmb_training.series_20261023 import native_original, freeze, rows, read, file_sha, require_space
from pl_mmb_collection.config import HttpConfig
from pl_mmb_collection.sources.http import HttpClient
from pl_mmb_collection.sources.rate_limit import RetryLater
from fetch_wiki_pilot_images import WikiRateLimiter, sha1number

OUT=ROOT/'reports/series_2026-10-23/wiki_v2'
ASSETS=Path('C:/Users/jsowa/benchmark-translations/series-2026-10-23/wiki-native-dev/source-assets')


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(path)


def snapshot():
    target=OUT/'snapshot.json'
    if target.exists():
        value=read(target)
        if file_sha(OUT/'candidates.jsonl')!=value['candidate_manifest_sha256']:raise ValueError('Candidate snapshot changed')
        return value
    payload=(ROOT/'reports/wikimedia_pipeline/v1/final_pairs.jsonl').read_bytes()
    records=[json.loads(line) for line in payload.splitlines() if line.strip()]
    candidates=[r for r in records if native_original(r)]
    sha=freeze(OUT/'candidates.jsonl',candidates,jsonl=True)
    value=dict(source_export_sha256=hashlib.sha256(payload).hexdigest(),export_rows=len(records),
        native_candidates=len(candidates),candidate_manifest_sha256=sha,
        source='read-only atomically published final_pairs.jsonl; no SQLite/WAL',
        previous_candidate_snapshot='wiki_native_candidates.jsonl remains immutable',
        no_translated_or_recycled_candidates=True)
    freeze(target,value);return value


def image_metadata(path,record):
    expected=f"{sha1number(record['source_sha1']):040x}"
    raw=path.read_bytes()
    if hashlib.sha1(raw).hexdigest()!=expected:raise ValueError('Original SHA1 differs from frozen source record')
    with Image.open(io.BytesIO(raw)) as image:
        canonical=ImageOps.exif_transpose(image).convert('RGB')
        pixel=hashlib.sha256(str(canonical.size).encode()+canonical.tobytes()).hexdigest()
        gray=canonical.convert('L').resize((32,32),Image.Resampling.LANCZOS)
        coefficients=dctn(np.asarray(gray,dtype=float),type=2)[:8,:8]
        phash=int.from_bytes(np.packbits((coefficients>np.median(coefficients)).reshape(-1)).tobytes(),'big')
        dimensions=list(canonical.size)
    return dict(sha256=hashlib.sha256(raw).hexdigest(),pixels_sha256=pixel,phash=f'{phash:016x}',
        source_sha1_hex=expected,source_original_bytes_verified=True,dimensions=dimensions,
        bytes=len(raw),image=str(path),license=record.get('license'),license_url=record.get('license_url'),
        author=record.get('creator'),source_attribution=record.get('attribution'),source_page_url=record['source_page_url'],
        attribution=' | '.join(str(v) for v in (record.get('creator'),record['source_page_url'],record.get('license'),record.get('license_url')) if v))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);args=parser.parse_args()
    require_space(['C:/','D:/'],expected_bytes=100*2**20)
    meta=snapshot()
    if meta['native_candidates']<300:
        save(OUT/'download_state.json',dict(status='stopped_below_300',snapshot=meta));return
    candidates=rows(OUT/'candidates.jsonl')
    source=read(ROOT/'outputs/polish_culture_fast/manifest.json')['config']['http']
    source['state_path']=str((ROOT/source['state_path']).resolve())
    config=HttpConfig(**source)
    decision=dict(status='registered_before_acquisition',snapshot_sha256=file_sha(OUT/'snapshot.json'),
        script_sha256=file_sha(__file__),asset_root=str(ASSETS),max_assets_bytes=8*2**30,
        max_file_bytes=100_000_000,local_minimum_request_interval_seconds=3,
        shared_HTTP_state=config.state_path,shared_budget_unchanged=True,pipeline_slots_unchanged=True,
        source_identity='original bytes must equal frozen Wikimedia SHA1; canonical RGB hash and DCT64 pHash',
        captions='native provenance accepted directly; no model generation',
        no_training_or_retrieval=True)
    freeze(OUT/'acquisition_decision.json',decision)
    ASSETS.mkdir(parents=True,exist_ok=True);(OUT/'decoded').mkdir(exist_ok=True)
    os.environ['SSL_CERT_FILE']=str(ROOT/'combined-ca.pem');os.environ['REQUESTS_CA_BUNDLE']=str(ROOT/'combined-ca.pem')
    successes=failures=0;last_request=0.;consecutive_forbidden=0
    native_limit=args.limit or len(candidates)
    with HttpClient(config,ASSETS.parent/'network/raw') as client:
        client.limiter=WikiRateLimiter(client.config,client.limiter.path)
        for index,pair in enumerate(candidates[:native_limit]):
            receipt=OUT/'decoded'/f'{index:05d}.json'
            if receipt.exists() and read(receipt).get('status')=='ready':
                old=read(receipt)
                if file_sha(old['image'])!=old['sha256']:raise ValueError('Previously acquired asset changed')
                successes+=1;continue
            record=pair['source_record'];expected=f"{sha1number(record['source_sha1']):040x}"
            target=ASSETS/f'{expected}.bin';temporary=ASSETS/f'{expected}.part'
            try:
                if not record.get('license') or not record.get('creator'):raise ValueError('Missing license or author provenance')
                if record.get('mime_type') not in ('image/jpeg','image/png','image/webp','image/tiff','image/gif'):
                    raise ValueError('Original format requires separately verified rendering; excluded from this acquisition')
                url=record['image_original_url'];parsed=urlparse(url)
                if parsed.scheme!='https' or parsed.hostname not in ('upload.wikimedia.org','thumb.wikimedia.org'):
                    raise ValueError('Unexpected asset host')
                if not target.exists():
                    require_space(['C:/','D:/'],expected_bytes=100*2**20)
                    if sum(p.stat().st_size for p in ASSETS.glob('*.bin'))>8*2**30:
                        raise RuntimeError('Registered disk budget exhausted')
                    while True:
                        try:
                            time.sleep(max(0,3-(time.monotonic()-last_request)))
                            last_request=time.monotonic()
                            client.download(url,temporary,max_bytes=100_000_000)
                            client.limiter.success();break
                        except RetryLater as error:
                            print('SHARED_HTTP_WAIT '+str(error),flush=True)
                            time.sleep(min(10,max(.2,error.retry_at-time.time())))
                    temporary.replace(target)
                identity=image_metadata(target,record)
                save(receipt,dict(id=pair['id'],status='ready',source_class=pair['selected_candidate'].get('quality'),**identity))
                successes+=1;consecutive_forbidden=0
            except RuntimeError:
                raise
            except Exception as error:
                temporary.unlink(missing_ok=True)
                save(receipt,dict(id=pair['id'],status='failed',error=type(error).__name__+': '+str(error)[:400]))
                failures+=1
                print(f'WIKI_ASSET_FAILED {index} {pair["id"]} {type(error).__name__}: {str(error)[:150]}',flush=True)
                if isinstance(error,HTTPError) and error.code==403:
                    consecutive_forbidden+=1
                    if consecutive_forbidden>=5:raise RuntimeError('Five consecutive HTTP403 responses; stop rather than burden the shared scraper')
            if (index+1)%10==0 or index+1==native_limit:
                state=dict(status='acquiring',snapshot=meta,processed=index+1,ready=successes,failed=failures,
                    asset_bytes=sum(p.stat().st_size for p in ASSETS.glob('*.bin')),
                    shared_HTTP_state=config.state_path,minimum_local_interval=3)
                save(OUT/'download_state.json',state)
                print('WIKI_ASSETS '+json.dumps(state),flush=True)
    save(OUT/'download_state.json',dict(status='complete' if native_limit==len(candidates) else 'pilot_complete',
        snapshot=meta,processed=native_limit,ready=successes,failed=failures,
        asset_bytes=sum(p.stat().st_size for p in ASSETS.glob('*.bin'))))


if __name__=='__main__':main()
