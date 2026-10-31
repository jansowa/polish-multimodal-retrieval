"""One sequential worker. Shared limiter; restartable receipts; no raw bulk archive."""
import argparse
import hashlib
import json
import ssl
import time
from collections import Counter
from dataclasses import replace
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from config import *
from image_assets import create
from pl_mmb_collection.config import HttpConfig
from pl_mmb_collection.sources.http import HttpClient
from pl_mmb_collection.sources.rate_limit import RetryLater


def main(limit=None):
    catalog_path=local_path(read(REPORT/'active_catalog.json')['path'])
    catalog=list(rows(catalog_path))
    cfg=HttpConfig(**read(ROOT/'outputs/polish_culture_fast/manifest.json')['config']['http'])
    cfg=replace(cfg,state_path=str(ROOT/'outputs/wikimedia_http_state.json'))
    client=HttpClient(cfg,STORE/'compact_network/raw')
    # The legacy writer uses one shared .json.tmp filename, which can collide
    # across Windows/WSL clients. This worker uses a unique atomic temp name;
    # the shared path, pacing, cooldowns and other pipelines remain unchanged.
    client.limiter.save=lambda:atomic_json(client.limiter.path,client.limiter.state)
    context=ssl.create_default_context(cafile=str(ROOT/'combined-ca.pem'))
    agent=cfg.user_agent+(f' ({cfg.contact})' if cfg.contact else '')
    # Preserve the 20 original smoke downloads. No dataset/cache deletion.
    original_receipts={p.stem:read(p) for p in (STORE/'receipts').glob('*.json')}
    receipt_dir=STORE/'compact_receipts';receipt_dir.mkdir(exist_ok=True)
    done=0;last_status=0.;last_reserve=0.;started=time.monotonic()
    try:
        for round_number in range(4):
            retry_deadlines=[]
            for item in catalog:
                h=item['sha1'];receipt=receipt_dir/(h+'.json');previous=read(receipt) if receipt.exists() else {}
                if previous.get('status') in ('ready','failed'):continue
                if previous.get('next_retry_at',0)>time.time():retry_deadlines.append(previous['next_retry_at']);continue
                try:
                    if time.monotonic()-last_reserve>120:
                        reserve();last_reserve=time.monotonic()
                    free=disk_snapshot()
                    if free['C']['free']<4*GIB+100*1024**2 or free['D']['free']<4*GIB+2*1024**2:
                        raise RuntimeError('Physical disk reserve reached')
                    old=original_receipts.get(h,{})
                    path=item.get('existing_path') or (old.get('path') if old.get('status')=='ready' else None)
                    if path:
                        raw=Path(path).read_bytes()
                    else:
                        def fetch():
                            client.limiter.reserve(item['url'])
                            request=Request(item['url'],headers={'User-Agent':agent,'Accept':'*/*','Accept-Encoding':'identity'})
                            with urlopen(request,timeout=cfg.timeout,context=context) as response:
                                raw=response.read(50_000_001)
                                if len(raw)>50_000_000:raise ValueError('Original exceeds existing pipeline 50 MB limit')
                                expected=int(response.headers.get('Content-Length','0'))
                                if expected and len(raw)!=expected:raise ValueError('Incomplete original response')
                                return raw
                        raw=client._retry(fetch,item['url']);client.limiter.success()
                    assets=create(raw,h)
                    atomic_json(receipt,dict(status='ready',sha1=h,id=item['id'],url=item['url'],**assets,
                        source_page_url=item['source_page_url'],retained_original_path=path,updated_at=utc()),immutable=not receipt.exists())
                    del raw;done+=1
                except RetryLater as exc:
                    when=max(time.time()+5,exc.retry_at)
                    atomic_json(receipt,dict(status='retry',sha1=h,reason=str(exc),next_retry_at=when,attempt=round_number+1))
                    retry_deadlines.append(when);time.sleep(min(60,max(1,when-time.time())))
                except Exception as exc:
                    if isinstance(exc,RuntimeError) and ('reserve' in str(exc).lower() or 'cap' in str(exc).lower()):raise
                    permanent=(isinstance(exc,(ValueError,AssertionError)) and not isinstance(exc,json.JSONDecodeError)) or isinstance(exc,HTTPError) and exc.code in (400,403,404,410)
                    failed=permanent or round_number==3
                    when=time.time()+60*(round_number+1)
                    atomic_json(receipt,dict(status='failed' if failed else 'retry',sha1=h,
                        reason=type(exc).__name__+': '+str(exc),next_retry_at=when,attempt=round_number+1,updated_at=utc()))
                    if not failed:retry_deadlines.append(when)
                if time.monotonic()-last_status>15:
                    counts=Counter();stored=0
                    for p in receipt_dir.glob('*.json'):
                        v=read(p);counts[v['status']]+=1;stored+=v.get('stored_bytes',0)
                    status('downloading_compact_archive',counts=dict(counts),stored_asset_bytes=stored,
                        target_unique_originals=len(catalog),elapsed_seconds=time.monotonic()-started,
                        profile='master1536 JPEG92 + exact TinyCLIP224 PNG + original fingerprints',GPU_planned_after_audit=True)
                    print('COMPACT_DOWNLOAD',dict(counts),stored,flush=True);last_status=time.monotonic()
                if limit is not None and done>=limit:break
            if limit is not None and done>=limit:break
            if not retry_deadlines:break
            while min(retry_deadlines)>time.time():time.sleep(min(60,min(retry_deadlines)-time.time()))
        counts=Counter();source_bytes=0;stored_bytes=0
        for p in receipt_dir.glob('*.json'):
            r=read(p)
            if r['status']=='retry' and limit is None:
                r.update(status='failed',reason='Retry budget exhausted: '+r.get('reason',''));atomic_json(p,r)
            counts[r['status']]+=1;source_bytes+=r.get('original_bytes',0);stored_bytes+=r.get('stored_bytes',0)
        summary=dict(status='acquisition_finished' if limit is None else 'smoke_complete',counts=dict(counts),
            expected_unique_originals=len(catalog),source_bytes_downloaded_or_reused=source_bytes,stored_asset_bytes=stored_bytes,
            original_fingerprints_computed_before_resize=True,full_frame_master_edge=1536,native224_lossless=True,elapsed_seconds=time.monotonic()-started)
        atomic_json(REPORT/('download_summary.json' if limit is None else 'compact_smoke.json'),summary)
        reserve();print('COMPACT_ACQUISITION_DONE',json.dumps(summary),flush=True)
        return summary
    finally:client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);args=parser.parse_args();main(args.limit)
