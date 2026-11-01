"""Durable original-only acquisition, one worker, existing shared limiter."""
import argparse
import hashlib
import json
import os
import ssl
import time
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request,urlopen
from dataclasses import replace
from config import *
from pl_mmb_collection.config import HttpConfig
from pl_mmb_collection.sources.http import HttpClient
from pl_mmb_collection.sources.rate_limit import RetryLater

MAX_FILE=50_000_000


def original_hash(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha1').hexdigest()


def summarize(catalog):
    counts={};total=0;success_sizes=[]
    for item in catalog:
        receipt=STORE/'receipts'/(item['sha1']+'.json')
        if not receipt.exists():state='pending'
        else:
            value=read(receipt);state=value['status']
            if state=='ready':
                total+=value['bytes'];success_sizes.append(value['bytes'])
        counts[state]=counts.get(state,0)+1
    sample=sum(success_sizes)/max(1,len(success_sizes))
    value=dict(status='acquisition_finished' if not counts.get('pending') and not counts.get('retry') else 'acquiring',
        counts=counts,verified_original_bytes_including_reuse=total,mean_ready_original_bytes=sample,
        projected_all_bytes_from_observed_mean=int(sample*len(catalog)),updated_at=utc())
    atomic_json(REPORT/'download_summary.json',value)
    return value


def main(limit=None):
    catalog_path=Path(read(REPORT/'active_catalog.json')['path']) if (REPORT/'active_catalog.json').exists() else STORE/'download_catalog.jsonl'
    catalog_path=local_path(catalog_path)
    catalog=list(rows(catalog_path))
    cfg=HttpConfig(**read(ROOT/'outputs/polish_culture_fast/manifest.json')['config']['http'])
    cfg=replace(cfg,state_path=str(ROOT/'outputs/wikimedia_http_state.json'))
    client=HttpClient(cfg,STORE/'network/raw')
    context=ssl.create_default_context(cafile=str(ROOT/'combined-ca.pem'))
    agent=cfg.user_agent+(f' ({cfg.contact})' if cfg.contact else '')
    raw_total=sum(p.stat().st_size for p in (STORE/'originals').glob('*') if p.is_file())
    reserve(used_hint=raw_total)
    worked=0;last_status=0.;budget_exhausted=False
    try:
        for round_number in range(4):
            retry_at=[];unresolved=False
            for item in catalog:
                h=item['sha1'];receipt=STORE/'receipts'/(h+'.json');dest=STORE/'originals'/(h+'.bin')
                prior=read(receipt) if receipt.exists() else {}
                if prior.get('status')=='ready':continue
                if prior.get('status') in ('failed','capacity_excluded'):continue
                if prior.get('next_retry_at',0)>time.time():
                    retry_at.append(prior['next_retry_at']);unresolved=True;continue
                path=Path(item['existing_path']) if item.get('existing_path') else dest
                try:
                    if path.exists():
                        assert original_hash(path)==h,'Retained asset source SHA1 mismatch'
                        size=path.stat().st_size
                    else:
                        # Each new original must fit the raw archive and both
                        # physical reserve floors. No thumbnail fallback.
                        if raw_total+MAX_FILE>RAW_CAP:
                            budget_exhausted=True
                            atomic_json(receipt,dict(status='capacity_excluded',sha1=h,reason='32 GiB original archive cap',updated_at=utc()))
                            continue
                        disk=disk_snapshot()
                        if disk['C']['free']<4*GIB+MAX_FILE or disk['D']['free']<4*GIB+2*1024**2:
                            raise RuntimeError('Physical capacity floor reached')
                        if urlparse(item['url']).hostname!='upload.wikimedia.org':raise ValueError('Unexpected original host')
                        partial=dest.with_suffix('.part');old_partial=partial.stat().st_size if partial.exists() else 0
                        def fetch():
                            nonlocal raw_total
                            client.limiter.reserve(item['url'])
                            request=Request(item['url'],headers={'User-Agent':agent,'Accept':'*/*','Accept-Encoding':'identity'})
                            before=partial.stat().st_size if partial.exists() else 0
                            digest=hashlib.sha1();size=0
                            try:
                                with urlopen(request,timeout=cfg.timeout,context=context) as response,partial.open('wb') as out:
                                    length=int(response.headers.get('Content-Length','0'))
                                    if length>MAX_FILE:raise ValueError('Original exceeds pipeline 50 MB limit')
                                    for chunk in iter(lambda:response.read(1024*1024),b''):
                                        size+=len(chunk)
                                        if size>MAX_FILE:raise ValueError('Original exceeds pipeline 50 MB limit')
                                        if size+raw_total-before>RAW_CAP:raise RuntimeError('Archive cap reached mid-stream')
                                        if disk_snapshot()['C']['free']<4*GIB+len(chunk):raise RuntimeError('C free-space floor reached')
                                        out.write(chunk);digest.update(chunk)
                                    out.flush();os.fsync(out.fileno())
                                if length and size!=length:raise ValueError('Incomplete HTTP body')
                                if digest.hexdigest()!=h:raise ValueError('Original source SHA1 changed; no replacement accepted')
                                partial.replace(dest)
                                return size
                            finally:
                                after=dest.stat().st_size if dest.exists() else (partial.stat().st_size if partial.exists() else 0)
                                raw_total+=after-before
                        size=client._retry(fetch,item['url'])
                        client.limiter.success();path=dest
                    atomic_json(receipt,dict(status='ready',sha1=h,path=str(path),bytes=size,url=item['url'],
                        reused=path!=dest,verified_source_sha1=True,updated_at=utc()))
                    worked+=1
                except RetryLater as exc:
                    when=max(time.time()+5,exc.retry_at)
                    atomic_json(receipt,dict(status='retry',sha1=h,next_retry_at=when,reason=str(exc),attempt=round_number+1))
                    # Global pacing/cooldown is shared: wait rather than issuing
                    # a storm of deferred reservations for other records.
                    time.sleep(min(60,max(1,when-time.time())))
                    unresolved=True;retry_at.append(when)
                except Exception as exc:
                    permanent=isinstance(exc,(ValueError,AssertionError)) or isinstance(exc,HTTPError) and exc.code in (400,403,404,410)
                    terminal=permanent or round_number==3
                    when=time.time()+60*(round_number+1)
                    atomic_json(receipt,dict(status='failed' if terminal else 'retry',sha1=h,reason=type(exc).__name__+': '+str(exc),
                        next_retry_at=when,attempt=round_number+1,updated_at=utc()))
                    if not terminal:unresolved=True;retry_at.append(when)
                if time.monotonic()-last_status>15:
                    reserve(used_hint=raw_total)
                    status('downloading_originals',successful_this_invocation=worked,raw_archive_bytes=raw_total,
                        total_unique_originals=len(catalog),last_original=item['id'],GPU_planned_after_audit=True)
                    print('DOWNLOAD_PROGRESS',worked,raw_total,item['id'],flush=True);last_status=time.monotonic()
                if limit is not None and worked>=limit:return summarize(catalog)
            if not unresolved:break
            while retry_at and min(retry_at)>time.time():time.sleep(min(60,min(retry_at)-time.time()))
        # End-of-acquisition retry deadlines do not masquerade as completed images.
        for item in catalog:
            p=STORE/'receipts'/(item['sha1']+'.json')
            if p.exists() and read(p)['status']=='retry':
                v=read(p);v.update(status='failed',reason='Retry budget exhausted: '+v.get('reason',''))
                atomic_json(p,v)
        value=summarize(catalog);reserve()
        print('DOWNLOAD_COMPLETE',json.dumps(value),flush=True)
        return value
    finally:client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);args=parser.parse_args()
    main(args.limit)
