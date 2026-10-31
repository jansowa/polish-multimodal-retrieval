"""Snapshot comparable first-caption outcomes and all Wiki candidate outcomes by provenance grade."""
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def statistics(trials):
    count=Counter()
    for stages in trials:
        judge=stages.get('judge')
        if not judge: continue
        count['judged']+=1
        if judge['image_caption_mismatch']:
            count['mismatch']+=1
        elif judge['visual_grounding']==0:
            count['grounding_zero']+=1
        elif judge['visual_grounding']>=3:
            count['direct']+=1
        else:
            count['recycle_routed']+=1
            recycled=stages.get('recycle')
            if recycled:
                count['recycle_completed']+=1
                if not recycled['usable']:count['recycle_unusable']+=1
                else:count['recycle_usable']+=1
            check=stages.get('recheck')
            if check:
                count['rechecked']+=1
                if check['visual_grounding']>=3 and not check['image_caption_mismatch']:
                    count['recheck_pass']+=1
                else:count['recheck_fail']+=1
    keys=['judged','direct','recycle_routed','grounding_zero','mismatch','recycle_completed','recycle_usable',
          'recycle_unusable','rechecked','recheck_pass','recheck_fail']
    result={key:count[key] for key in keys}
    done=count['rechecked']+count['recycle_unusable']
    result['recycle_resolved']=done
    result['initial_percent']={key:round(100*count[key]/count['judged'],4) if count['judged'] else None
        for key in ['direct','recycle_routed','grounding_zero','mismatch']}
    result['recheck_pass_percent']=round(100*count['recheck_pass']/count['rechecked'],4) if count['rechecked'] else None
    result['recycle_rescue_percent']=round(100*count['recheck_pass']/done,4) if done else None
    return result


def snapshot(name):
    db=sqlite3.connect((ROOT/f'reports/{name}_pipeline/v1/checkpoint.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row;db.execute('BEGIN')
    items={r['id']:dict(r) for r in db.execute('SELECT * FROM items')}
    trials=defaultdict(dict)
    if name=='visualnews':
        for row in db.execute("SELECT id,stage,output_json FROM outputs WHERE stage IN ('judge','recycle','recheck')"):
            trials[row['id'],0][row['stage']]=json.loads(row['output_json'])
    else:
        for row in db.execute("SELECT id,stage,input_json,response_json FROM requests WHERE valid=1 AND stage IN ('judge','recycle','recheck') ORDER BY request_id"):
            inputs=json.loads(row['input_json']);body=json.loads(row['response_json'])
            obj=json.loads(body['choices'][0]['message']['content'])
            trials[row['id'],inputs['candidate_index']][row['stage']]=obj
    exclusions=dict(db.execute('SELECT reason,COUNT(*) FROM exclusions GROUP BY reason')) if name=='wikimedia' else {}
    db.rollback();db.close()
    ready=[r for r in items.values() if r['status']=='completed' and (r['route'] or '').startswith('ready') and r['final_pl']]
    result=dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),queued_images=len(items),
        judged_images=len({uid for (uid,index),stages in trials.items() if 'judge' in stages}),
        first_caption=statistics([stages for (uid,index),stages in trials.items() if index==0]),
        all_candidates=statistics(list(trials.values())),ready_polish=len(ready),
        ready_routes=dict(Counter(r['route'] for r in ready)),exclusions=exclusions)
    if name=='wikimedia':
        groups=defaultdict(list);queue_classes=Counter();ready_classes=Counter()
        for item in items.values():
            texts=json.loads(item['candidates_json'])
            queue_classes[texts[0].get('quality') or 'unknown']+=1
        for (uid,index),stages in trials.items():
            candidate=json.loads(items[uid]['candidates_json'])[index]
            groups[candidate.get('quality') or 'unknown'].append(stages)
        for item in ready:
            selected=json.loads(item['candidates_json'])[item['candidate_index']]
            ready_classes[selected.get('quality') or 'unknown']+=1
        result['by_grade']={grade:dict(statistics(groups[grade]),queued_first_caption=queue_classes[grade],ready_polish=ready_classes[grade])
            for grade in sorted({'A','B','C','D'}|groups.keys())}
        result['fallback_images_judged']=len({uid for (uid,index),stages in trials.items() if index>0 and 'judge' in stages})
        result['ready_after_fallback']=sum(r['candidate_index']>0 for r in ready)
    return result


def main():
    result={name:snapshot(name) for name in ['visualnews','wikimedia']}
    path=ROOT/'reports/grounding_dataset_comparison.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
