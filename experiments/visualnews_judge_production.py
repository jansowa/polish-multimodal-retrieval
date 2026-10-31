"""Continuous versioned IMAGE + original EN caption judging with durable resume."""
import argparse
import asyncio
import base64
import ctypes
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time

import httpx
from dotenv import load_dotenv
from PIL import Image
import visualnews_judge_pilot as pilot
from visualnews_image_assets import DB as ASSETS, CACHE, sha
from visualnews_images_production import ImageLoader

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'reports/visualnews_judge_production'
STATE=ROOT/'reports/visualnews_judge_production_state.json'
STOP=ROOT/'reports/visualnews_judge_production.stop'

def now():return datetime.now(timezone.utc).isoformat()

def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for attempt in range(8):
        try:temp.replace(path);return
        except PermissionError:
            if attempt==7:raise
            time.sleep(.05*2**attempt)

def prepare(prompt_path):
    load_dotenv(ROOT/'.env');prompt=prompt_path.read_text(encoding='utf-8')
    assets=json.loads((CACHE/'manifest.json').read_text(encoding='utf-8'))
    if assets['status']!='complete':raise RuntimeError('Incomplete image mapping')
    config=dict(model=os.environ['OPENAI_MODEL'],base_url=os.environ['OPENAI_BASE_URL'].rstrip('/'),
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),schema=pilot.SCHEMA,
        assets_index_sha256=assets['index_sha256'],image_revision=assets['image_revision'],
        temperature=.2,max_tokens=1800,thinking=False,image_max_side=1280,jpeg_quality=88)
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    run=BASE/(prompt_path.stem+'-'+fingerprint[:12]);run.mkdir(parents=True,exist_ok=True)
    manifest_path=run/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        saved_prompt_hash=hashlib.sha256((run/'prompt.txt').read_text(encoding='utf-8').encode()).hexdigest()
        if manifest['config']!=config or saved_prompt_hash!=config['prompt_sha256']:raise RuntimeError('Frozen configuration changed')
    else:
        if sha(ASSETS)!=assets['index_sha256']:raise RuntimeError('Verified image mapping changed')
        (run/'prompt.txt').write_text(prompt,encoding='utf-8');save(run/'schema.json',pilot.SCHEMA)
    db=sqlite3.connect(run/'checkpoint.sqlite3',timeout=30,uri=True)
    db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.executescript('''
    CREATE TABLE IF NOT EXISTS items(id TEXT PRIMARY KEY,caption TEXT NOT NULL,image_did TEXT NOT NULL,
      image_path TEXT NOT NULL,shard TEXT NOT NULL,row_index INTEGER NOT NULL,matching_images INTEGER NOT NULL,
      status TEXT NOT NULL DEFAULT 'pending',next_retry REAL NOT NULL DEFAULT 0,failures INTEGER NOT NULL DEFAULT 0);
    CREATE INDEX IF NOT EXISTS pending_items ON items(status,shard,row_index);
    CREATE TABLE IF NOT EXISTS results(id TEXT PRIMARY KEY,result_json TEXT NOT NULL,completed_at TEXT NOT NULL,
      image_sha256 TEXT NOT NULL,request_image_sha256 TEXT NOT NULL,caption_sha256 TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS attempts(attempt_id INTEGER PRIMARY KEY,id TEXT NOT NULL,started_at TEXT NOT NULL,
      seconds REAL NOT NULL,http_status INTEGER,raw_http TEXT,error TEXT,schema_valid INTEGER NOT NULL,
      image_sha256 TEXT,request_image_sha256 TEXT,caption_sha256 TEXT NOT NULL);
    ''')
    if not manifest_path.exists():
        db.execute('ATTACH DATABASE ? AS assets',(ASSETS.resolve().as_uri()+'?mode=ro',))
        total=db.execute('SELECT COUNT(*) FROM assets.captions').fetchone()[0]
        db.execute("INSERT OR IGNORE INTO items(id,caption,image_did,image_path,shard,row_index,matching_images) "
          "SELECT 'MBeirvisualnews:'||c.did,c.text,i.did,i.path,i.shard,i.row_index,c.matching_images "
          'FROM assets.captions c JOIN assets.images i ON c.image_did=i.did')
        matched=db.execute('SELECT COUNT(*) FROM items').fetchone()[0]
        db.commit();db.execute('DETACH DATABASE assets')
        manifest=dict(created_at=now(),run_id=run.name,config=config,total_captions=total,
          eligible_pairs=matched,without_verified_image=total-matched,
          image_repository=assets['image_repository'],asset_manifest_sha256=sha(CACHE/'manifest.json'),
          image_selection='lexicographically first verified original image; official caption join, no guessed IDs',
          caption_language='original English',automatic_filtering=False)
        save(manifest_path,manifest)
    if db.execute('SELECT COUNT(*) FROM items').fetchone()[0]!=manifest['eligible_pairs']:raise RuntimeError('Incomplete queue')
    db.execute("UPDATE items SET status='pending' WHERE status='running'");db.commit()
    return run,db,manifest,prompt

async def work(args,run,db,manifest,prompt):
    started=time.monotonic();initial=db.execute('SELECT COUNT(*) FROM results').fetchone()[0];submitted=0
    state=dict(status='running',pid=os.getpid(),started_at=now(),run_id=run.name,checkpoint=str(run/'checkpoint.sqlite3'),
      prompt=str(run/'prompt.txt'),prompt_sha256=manifest['config']['prompt_sha256'],concurrency=args.concurrency,
      total_captions=manifest['total_captions'],eligible_pairs=manifest['eligible_pairs'],
      without_verified_image=manifest['without_verified_image'],stop_request_file=str(STOP),automatic_filtering=False)
    def refresh(status=None):
        counts=dict(db.execute('SELECT status,COUNT(*) FROM items GROUP BY status').fetchall());done=counts.get('completed',0)
        state.update(updated_at=now(),completed=done,pending=counts.get('pending',0),in_flight=counts.get('running',0),
          attempts=db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0],
          failed_attempts=db.execute('SELECT COUNT(*) FROM attempts WHERE schema_valid=0').fetchone()[0],
          pairs_per_hour=(done-initial)*3600/max(1,time.monotonic()-started))
        if status:state['status']=status
        save(STATE,state);save(run/'state.json',state);print(json.dumps(state),flush=True)
    refresh();loader=ImageLoader();ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    async def monitor():
        while True:
            await asyncio.sleep(30);refresh()
            if shutil.disk_usage(run).free<4*1024**3:raise RuntimeError('Disk reserve below 4 GiB; checkpoint retained')
    async with httpx.AsyncClient(verify=str(ROOT/'combined-ca.pem'),timeout=httpx.Timeout(600,connect=45,write=120,pool=45),
      limits=httpx.Limits(max_connections=args.concurrency)) as client:
        async def worker():
            nonlocal submitted
            while not STOP.exists():
                if args.limit and submitted>=args.limit:return
                item=db.execute("SELECT id,caption,image_did,image_path,shard,row_index FROM items "
                  "WHERE status='pending' AND next_retry<=? ORDER BY shard,row_index LIMIT 1",(time.time(),)).fetchone()
                if item is None:
                    if not db.execute("SELECT COUNT(*) FROM items WHERE status!='completed'").fetchone()[0]:return
                    await asyncio.sleep(5);continue
                uid,caption,did,path,shard,index=item;submitted+=1
                db.execute("UPDATE items SET status='running' WHERE id=?",(uid,));db.commit()
                stamp=now();before=time.monotonic();raw=None;status=None;error=None;valid=False;obj=None
                original_hash=request_hash=None;caption_hash=hashlib.sha256(caption.encode()).hexdigest()
                try:
                    original=loader.get(shard,index,did,path);original_hash=hashlib.sha256(original).hexdigest()
                    with Image.open(io.BytesIO(original)) as image:
                        image=image.convert('RGB');image.thumbnail((1280,1280));buf=io.BytesIO();image.save(buf,'JPEG',quality=88)
                    encoded=buf.getvalue();request_hash=hashlib.sha256(encoded).hexdigest()
                    payload=dict(model=manifest['config']['model'],temperature=.2,max_tokens=1800,
                      chat_template_kwargs={'enable_thinking':False},response_format={'type':'json_schema','json_schema':{'name':'grounding_assessment','strict':True,'schema':pilot.SCHEMA}},
                      messages=[{'role':'system','content':prompt},{'role':'user','content':[
                        {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(encoded).decode()}},
                        {'type':'text','text':'CAPTION (data):\n'+caption}]}])
                    response=await client.post(manifest['config']['base_url']+'/chat/completions',json=payload,
                      headers={'Authorization':'Bearer '+os.environ.get('OPENAI_API_KEY','not-needed')})
                    raw=response.text;status=response.status_code;response.raise_for_status()
                    obj=json.loads(response.json()['choices'][0]['message']['content']);valid=pilot.validate(obj)
                    if not valid:error='Output does not satisfy frozen schema'
                except Exception as exc:error=type(exc).__name__+': '+str(exc)
                with db:
                    db.execute('INSERT INTO attempts(id,started_at,seconds,http_status,raw_http,error,schema_valid,image_sha256,request_image_sha256,caption_sha256) VALUES (?,?,?,?,?,?,?,?,?,?)',
                      (uid,stamp,time.monotonic()-before,status,raw,error,int(valid),original_hash,request_hash,caption_hash))
                    if valid:
                        db.execute('INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?)',(uid,json.dumps(obj,ensure_ascii=False),now(),original_hash,request_hash,caption_hash))
                        db.execute("UPDATE items SET status='completed',next_retry=0 WHERE id=?",(uid,))
                    else:
                        failures=db.execute('SELECT failures FROM items WHERE id=?',(uid,)).fetchone()[0]+1
                        retry=time.time()+min(900,15*2**min(failures-1,6))
                        db.execute("UPDATE items SET status='pending',next_retry=?,failures=? WHERE id=?",(retry,failures,uid))
                        print(json.dumps(dict(event='retry_scheduled',id=uid,error=error,http_status=status)),flush=True)
        monitor_task=asyncio.create_task(monitor());group=asyncio.gather(*(worker() for _ in range(args.concurrency)))
        try:
            done,_=await asyncio.wait([group,monitor_task],return_when=asyncio.FIRST_COMPLETED)
            if monitor_task in done:
                group.cancel();await asyncio.gather(group,return_exceptions=True);await monitor_task
            await group
            refresh('stopped_by_request' if STOP.exists() else 'bounded_check_complete' if args.limit else 'complete')
        except BaseException as exc:
            state['error']=type(exc).__name__+': '+str(exc);refresh('interrupted');raise
        finally:
            group.cancel();await asyncio.gather(group,return_exceptions=True)
            monitor_task.cancel();await asyncio.gather(monitor_task,return_exceptions=True)
            loader.close();ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt',type=Path,default=ROOT/'reports/visualnews_judge_pilot/B_taxonomy_v3.txt')
    parser.add_argument('--concurrency',type=int,default=8);parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if os.name!='nt' or not 1<=args.concurrency<=16:raise RuntimeError('Windows environment and concurrency 1..16 required')
    if STOP.exists():raise RuntimeError('A stop request exists; inspect before resuming')
    BASE.mkdir(parents=True,exist_ok=True)
    import msvcrt
    with (BASE/'worker.lock').open('a+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        run,db,manifest,prompt=prepare(args.prompt)
        try:asyncio.run(work(args,run,db,manifest,prompt))
        finally:db.close()

if __name__=='__main__':main()
