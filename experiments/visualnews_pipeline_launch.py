"""Adopt complete tested examples, then continue all verified pairs without duplicates."""
import asyncio
import argparse
import json
from pathlib import Path
import sqlite3

import visualnews_pipeline as p
from visualnews_pipeline_benchmark import source_path,BASE as BENCH

def adopt(target,run):
    if not (run/'checkpoint.sqlite3').exists():return 0
    db=target.db
    db.execute('CREATE TABLE IF NOT EXISTS imports(source_run TEXT PRIMARY KEY,items INTEGER NOT NULL,imported_at TEXT NOT NULL)')
    if db.execute('SELECT 1 FROM imports WHERE source_run=?',(str(run.resolve()),)).fetchone():return 0
    source=sqlite3.connect((run/'checkpoint.sqlite3').resolve().as_uri()+'?mode=ro',uri=True);source.row_factory=sqlite3.Row
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    for key,value in manifest['config'].items():
        if key not in ['fresh_judge','selected_ids'] and target.manifest['config'].get(key)!=value:
            raise RuntimeError('Cannot adopt outputs from a different pipeline configuration: '+key)
    count=0
    with db:
        for row in source.execute("SELECT * FROM items WHERE status='completed'"):
            item=dict(row);uid=item['id']
            original=db.execute('SELECT caption,image_did,image_path FROM items WHERE id=?',(uid,)).fetchone()
            if original is None:continue
            if tuple(original)!=(item['caption'],item['image_did'],item['image_path']):raise RuntimeError('Adopted source changed')
            if item['final_pl'] and p.translation_issue(item['final_en'],{'text':item['final_pl']}):continue
            for output in source.execute('SELECT * FROM outputs WHERE id=?',(uid,)):
                origin=dict(imported_run=str(run.resolve()),source_pipeline_fingerprint=manifest['fingerprint'])
                if output['origin_json']:origin['previous_origin']=json.loads(output['origin_json'])
                db.execute('INSERT OR REPLACE INTO outputs VALUES (?,?,?,?,?)',
                    (uid,output['stage'],output['output_json'],output['created_at'],json.dumps(origin,ensure_ascii=False)))
            db.execute('UPDATE items SET stage=?,status=?,priority=?,route=?,final_en=?,final_pl=?,next_retry=0,failures=0 WHERE id=?',
                (item['stage'],item['status'],item['priority'],item['route'],item['final_en'],item['final_pl'],uid))
            for request in source.execute('SELECT * FROM requests WHERE id=? ORDER BY request_id',(uid,)):
                keys=[k for k in request.keys() if k!='request_id']
                db.execute('INSERT INTO requests('+','.join(keys)+') VALUES ('+','.join('?' for _ in keys)+')',tuple(request[k] for k in keys))
            count+=1
        db.execute('INSERT INTO imports VALUES (?,?,?)',(str(run.resolve()),count,p.now()))
    source.close();return count

async def main(concurrency_override=None):
    comparison=json.loads((BENCH/'comparison.json').read_text(encoding='utf-8'))
    if comparison['status']!='complete':raise RuntimeError('Benchmark incomplete')
    runtime_path=p.BASE/'runtime_config.json'
    runtime=json.loads(runtime_path.read_text(encoding='utf-8')) if runtime_path.exists() else {}
    concurrency=concurrency_override or runtime.get('concurrency',comparison['selected_concurrency'])
    if not 1<=concurrency<=32:raise ValueError('Concurrency must be 1..32')
    target=p.Pipeline(p.BASE/'v1',source_path(require_idle=False),publish=True)
    try:
        imported=adopt(target,p.BASE/'smoke_checked')
        # Use one measured round, retain the other three as independent raw benchmark artifacts.
        candidate=next(r for r in comparison['rounds'] if r['concurrency']==comparison['selected_concurrency'])
        imported+=adopt(target,Path(candidate['run_dir']))
        p.save(p.BASE/'launch.json',dict(started_at=p.now(),concurrency=concurrency,adopted_completed_examples=imported,
            benchmark=str(BENCH/'comparison.json'),run_dir=str(target.run)))
        print('ADOPTED '+str(imported),flush=True)
        target.export()
        await target.work(concurrency)
    finally:await target.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--concurrency',type=int)
    args=parser.parse_args()
    import msvcrt
    p.BASE.mkdir(parents=True,exist_ok=True)
    with (p.BASE/'worker.lock').open('a+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        asyncio.run(main(args.concurrency))
