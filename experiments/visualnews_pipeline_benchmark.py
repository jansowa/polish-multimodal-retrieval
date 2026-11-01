"""Matched end-to-end ABBA throughput comparison: 16,32,32,16 global requests."""
import asyncio
from collections import defaultdict
import json
from pathlib import Path
import random
import sqlite3
import statistics

import visualnews_pipeline as p

BASE=p.BASE/'benchmark'

def source_path(require_idle=True):
    state=json.loads((p.ROOT/'reports/visualnews_judge_production_state.json').read_text(encoding='utf-8'))
    if require_idle and state['status'] not in ['stopped_by_request','complete']:raise RuntimeError('Old judge must drain before benchmarking')
    translation=json.loads((p.ROOT/'reports/visualnews_production_state.json').read_text(encoding='utf-8'))
    if require_idle and translation['status'] not in ['stopped_by_request','complete','incomplete']:raise RuntimeError('Old translation must drain before benchmarking')
    return Path(state['checkpoint'])

def select(source):
    if (BASE/'ids.json').exists():return json.loads((BASE/'ids.json').read_text(encoding='utf-8'))
    db=sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True)
    candidates=[(uid,json.loads(raw)) for uid,raw in db.execute('SELECT id,result_json FROM results ORDER BY id')]
    db.close();rng=random.Random(20261020);rng.shuffle(candidates)
    groups=defaultdict(list)
    for uid,obj in candidates:
        if not obj['image_caption_mismatch']:groups[obj['visual_grounding']].append(uid)
    ids=(groups[3]+groups[4])[:8]+groups[1][:28]+groups[2][:28]
    if len(ids)!=64:raise RuntimeError('Insufficient representative 64-record benchmark')
    rng.shuffle(ids);p.save(BASE/'ids.json',ids)
    p.save(BASE/'protocol.json',dict(order=[16,32,32,16],same_ids_each_round=True,fresh_initial_judge_each_round=True,
      sample=dict(direct=8,grounding_1=28,grounding_2=28),seed=20261020,
      metric='Valid prompt outputs per second, finalized source pairs per hour, final bilingual ready pairs per hour; include all pipeline stages, cache reuse and errors.',
      note='ABBA counterbalances warm-up and drift. A small matched local experiment is not a universal server benchmark.'))
    return ids

async def main():
    BASE.mkdir(parents=True,exist_ok=True);source=source_path()
    if not p.CACHE_DB.exists():print('Cached English translations:',p.refresh_translation_cache(),flush=True)
    ids=select(source);summaries=[]
    for index,concurrency in enumerate([16,32,32,16],1):
        run=BASE/f'round{index}-c{concurrency}'
        summary_path=run/'summary.json'
        if summary_path.exists():
            summary=json.loads(summary_path.read_text(encoding='utf-8'))
            if summary['status']!='complete':raise RuntimeError('Incomplete saved round; inspect before resume')
        else:
            pipeline=p.Pipeline(run,source,ids=ids,fresh_judge=True)
            try:summary=await pipeline.work(concurrency)
            finally:await pipeline.close()
        requests=sum(r['n'] for r in summary['request_stats']);valid=sum(r['valid'] for r in summary['request_stats'])
        seconds=summary['elapsed_seconds']
        summaries.append(dict(round=index,concurrency=concurrency,seconds=seconds,requests=requests,valid=valid,
          valid_requests_per_second=valid/seconds,finalized_per_hour=summary['completed']*3600/seconds,
          ready_per_hour=summary['ready']*3600/seconds,ready=summary['ready'],routes=summary['routes'],
          request_stats=summary['request_stats'],run_dir=str(run)))
        p.save(BASE/'progress.json',dict(status='benchmarking',rounds=summaries))
        print('BENCHMARK '+json.dumps(summaries[-1]),flush=True)
    aggregates={}
    for concurrency in [16,32]:
        rows=[r for r in summaries if r['concurrency']==concurrency];seconds=sum(r['seconds'] for r in rows)
        aggregates[str(concurrency)]=dict(seconds=seconds,valid_requests_per_second=sum(r['valid'] for r in rows)/seconds,
          finalized_per_hour=sum(64 for r in rows)*3600/seconds,ready_per_hour=sum(r['ready'] for r in rows)*3600/seconds,
          valid=sum(r['valid'] for r in rows),requests=sum(r['requests'] for r in rows))
    # Prefer higher useful output throughput; require 5% to justify the larger request queue.
    gain=aggregates['32']['ready_per_hour']/aggregates['16']['ready_per_hour']-1
    selected=32 if gain>.05 and aggregates['32']['valid']/aggregates['32']['requests']>=.98 else 16
    comparison=dict(status='complete',selected_concurrency=selected,gain_32_ready_throughput=gain,
      selection_rule='32 only if >5% faster in final ready-pair throughput and >=98% successful prompt outputs; otherwise 16.',
      aggregates=aggregates,rounds=summaries)
    p.save(BASE/'comparison.json',comparison)
    print('SELECTED '+json.dumps(comparison),flush=True)

if __name__=='__main__':asyncio.run(main())
