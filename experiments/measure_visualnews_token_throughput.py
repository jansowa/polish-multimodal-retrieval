"""Read server-reported token usage and extrapolate active-work windows."""
import json
import shutil
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(run, lower=None):
    db = sqlite3.connect((run/'checkpoint.sqlite3').resolve().as_uri()+'?mode=ro', uri=True)
    rows = db.execute('SELECT stage,started_at,seconds,response_json,error FROM requests WHERE started_at>=? ORDER BY started_at', (lower or '',))
    out=[]
    for stage, stamp, seconds, raw, error in rows:
        body=json.loads(raw) if raw else {}
        usage=body.get('usage') or {}
        details=usage.get('completion_tokens_details') or {}
        prompt_details=usage.get('prompt_tokens_details') or {}
        start=datetime.fromisoformat(stamp).timestamp()
        out.append(dict(stage=stage,start=start,end=start+seconds,seconds=seconds,error=error,
          has_usage=bool(usage),input=usage.get('prompt_tokens',usage.get('input_tokens',0)) or 0,
          output=usage.get('completion_tokens',usage.get('output_tokens',0)) or 0,
          reasoning=details.get('reasoning_tokens') or 0,cached=prompt_details.get('cached_tokens') or 0,
          usage_keys=sorted(usage),detail_keys=sorted(details)))
    db.close()
    return out


def aggregate(rows,start,end):
    selected=[r for r in rows if start <= r['end'] < end]
    seconds=end-start
    totals={key:sum(r[key] for r in selected) for key in ['input','output','reasoning','cached']}
    return dict(seconds=seconds,requests=len(selected),with_usage=sum(r['has_usage'] for r in selected),
      errors=sum(bool(r['error']) for r in selected),**totals,
      per_second={key:round(value/seconds,2) for key,value in totals.items()},
      per_day={key:round(value*86400/seconds) for key,value in totals.items()})


def main():
    state=json.loads((ROOT/'reports/visualnews_pipeline_state.json').read_text())
    run=Path(state['run_dir'])
    rows=read(run,state['started_at'])
    begin=datetime.fromisoformat(state['started_at']).timestamp()
    stop=datetime.fromisoformat(state['updated_at']).timestamp()
    # Exclude startup and final minute, where stop/cancellation can affect throughput.
    end=stop-60
    result=dict(status=state['status'],error=state.get('error'),started_at=state['started_at'],stopped_at=state['updated_at'],
      concurrency=state['concurrency'],ready=state['ready'],disk_free_bytes=shutil.disk_usage(ROOT).free,
      whole_run=aggregate(rows,begin,stop),stable=aggregate(rows,begin+60,end),
      last_5min=aggregate(rows,end-300,end),last_10min=aggregate(rows,end-600,end))
    candidates=[aggregate(rows,begin+60+i*60,begin+60+i*60+300) for i in range(max(0,int((end-begin-60-300)//60)+1))]
    result['best_5min_by_output']=max(candidates,key=lambda r:r['output']) if candidates else None
    result['best_5min_by_input']=max(candidates,key=lambda r:r['input']) if candidates else None
    result['by_stage']={stage:aggregate([r for r in rows if r['stage']==stage],begin+60,end) for stage in sorted({r['stage'] for r in rows})}
    result['usage_shapes']=[dict(stage=stage,usage_keys=next((r['usage_keys'] for r in rows if r['stage']==stage and r['has_usage']),[]),
      detail_keys=next((r['detail_keys'] for r in rows if r['stage']==stage and r['has_usage']),[])) for stage in result['by_stage']]
    comparison=json.loads((ROOT/'reports/visualnews_pipeline/benchmark/comparison.json').read_text())
    benchmarks=defaultdict(lambda:dict(seconds=0,input=0,output=0,reasoning=0,requests=0,with_usage=0))
    for rnd in comparison['rounds']:
        records=read(Path(rnd['run_dir']))
        target=benchmarks[str(rnd['concurrency'])]
        target['seconds']+=rnd['seconds']
        target['requests']+=len(records)
        target['with_usage']+=sum(r['has_usage'] for r in records)
        for key in ['input','output','reasoning']:
            target[key]+=sum(r[key] for r in records)
    for record in benchmarks.values():
        record['per_second']={key:round(record[key]/record['seconds'],2) for key in ['input','output','reasoning']}
        record['per_day']={key:round(record[key]*86400/record['seconds']) for key in ['input','output','reasoning']}
    result['benchmarks']=dict(benchmarks)
    target=ROOT/'reports/visualnews_pipeline/token_throughput.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
