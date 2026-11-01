"""Run VisualNews and expanded Wiki with configured shares and a hard shared request cap."""
import argparse
import asyncio
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sqlite3

import visualnews_pipeline as v
import wikimedia_grounding_pipeline as w
from visualnews_pipeline_benchmark import source_path
from pl_mmb_collection.storage.lease import CollectorLease
from inference_slots import SharedSlots
from migrate_wikimedia_source import migrate

ROOT=v.ROOT
SOURCE=Path('C:/Users/jsowa/datasets/polish_culture')
STOP=ROOT/'reports/balanced_inference.stop'


def has_work(pipeline,stop):
    if stop.exists():return False
    return bool(pipeline.db.execute("SELECT 1 FROM items WHERE status IN ('pending','running') LIMIT 1").fetchone())


async def main(total=24,shares=None,work_conserving=False):
    wiki=visual=None
    tasks=[]
    try:
        v.save(ROOT/'reports/balanced_inference_state.json',dict(status='preparing',pid=os.getpid(),updated_at=v.now(),stage='verify_and_expand_wiki_source'))
        print(json.dumps(migrate(w.BASE/'v1',SOURCE)),flush=True)
        wiki=w.WikiPipeline(source=SOURCE)
        visual=v.Pipeline(v.BASE/'v1',source_path(require_idle=False),publish=True)
        pipelines={'visualnews':visual,'wikimedia':wiki}
        slots=SharedSlots(total,{'visualnews':lambda:has_work(visual,v.STOP),'wikimedia':lambda:has_work(wiki,w.STOP)},shares=shares,work_conserving=work_conserving)
        for name,pipeline in pipelines.items():
            original= pipeline.process
            async def process(client,item,_original=original,_name=name,_pipeline=pipeline):
                if work_conserving and _name=='wikimedia':
                    await _pipeline.ensure_image(item)
                    # Pin the prepared image while waiting for a model slot; the LRU may evict it.
                    item['_prepared_image']=_pipeline.image(item)
                async with slots.slot(_name):return await _original(client,item)
            pipeline.process=process;pipeline.shared_slots=slots
        # Neither queue receives more than its fair share while both have work.
        tasks=[asyncio.create_task(visual.work(total)),asyncio.create_task(wiki.work(total))]
        workers=tuple(tasks)
        async def monitor():
            while True:
                if STOP.exists():
                    v.STOP.write_text('Shared worker stop requested',encoding='utf-8')
                    w.STOP.write_text('Shared worker stop requested',encoding='utf-8')
                state=dict(status='stopping' if STOP.exists() else 'running',pid=os.getpid(),updated_at=v.now(),
                    wiki_source=str(SOURCE),**slots.snapshot(),
                    queues={name:dict(pending=p.db.execute("SELECT COUNT(*) FROM items WHERE status='pending'").fetchone()[0],
                        ready=p.db.execute("SELECT COUNT(*) FROM items WHERE status='completed' AND route LIKE 'ready%' AND final_pl IS NOT NULL").fetchone()[0]) for name,p in pipelines.items()})
                v.save(ROOT/'reports/inference_allocation.json',state)
                v.save(ROOT/'reports/balanced_inference_state.json',state)
                if all(task.done() for task in workers):return
                await asyncio.sleep(10)
        watcher=asyncio.create_task(monitor())
        tasks.append(watcher)
        # Monitor only the two workers, rather than including itself in its completion check.
        await asyncio.gather(*tasks)
    except BaseException as exc:
        v.save(ROOT/'reports/balanced_inference_state.json',dict(status='interrupted',pid=os.getpid(),updated_at=v.now(),error=str(exc)))
        raise
    finally:
        for task in tasks:task.cancel()
        if tasks:await asyncio.gather(*tasks,return_exceptions=True)
        if visual:await visual.close()
        if wiki:await wiki.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--total',type=int,choices=range(2,33,2))
    args=parser.parse_args()
    config_path=ROOT/'reports/balanced_inference_config.json'
    config=json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
    if config.get('visualnews_enabled') is False:
        v.STOP.write_text('VisualNews disabled in shared coordinator configuration',encoding='utf-8')
    total=args.total if args.total is not None else config.get('total',24)
    if total not in range(2,33,2):parser.error('total must be even, between 2 and 32')
    with ExitStack() as stack:
        for lock in [ROOT/'reports/balanced_inference.lock',v.BASE/'worker.lock',w.BASE/'v1/worker.lock']:
            stack.enter_context(CollectorLease(lock))
        asyncio.run(main(total,config.get('shares'),config.get('work_conserving',False)))
