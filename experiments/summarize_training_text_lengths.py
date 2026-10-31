"""Read-only current pipeline inventory and Polish caption length distributions."""
import json
import math
from collections import Counter, defaultdict
from datetime import datetime,timezone
from pathlib import Path
import re
import sqlite3
import statistics

ROOT=Path(__file__).resolve().parents[1]
WORD=re.compile(r"\b\w+(?:[-’']\w+)*\b",re.UNICODE)
BINS=[('1–5',1,5),('6–10',6,10),('11–20',11,20),('21–40',21,40),('41–80',41,80),('81–160',81,160),('161+',161,math.inf)]


def distribution(values):
    if not values:return None
    values=sorted(values)
    def percentile(p):
        position=(len(values)-1)*p
        low=math.floor(position);high=math.ceil(position)
        return round(values[low]+(values[high]-values[low])*(position-low),2)
    return dict(n=len(values),min=values[0],mean=round(statistics.mean(values),2),
        median=percentile(.5),p10=percentile(.1),p90=percentile(.9),p95=percentile(.95),
        p99=percentile(.99),max=values[-1],total=sum(values))


def summary(rows):
    words=[len(WORD.findall(row['final_pl'])) for row in rows]
    chars=[len(row['final_pl']) for row in rows]
    histogram={label:dict(n=sum(low<=n<=high for n in words)) for label,low,high in BINS}
    for value in histogram.values():value['percent']=round(100*value['n']/len(rows),2) if rows else None
    return dict(words=distribution(words),characters_including_spaces=distribution(chars),word_histogram=histogram,
        exact_unique_polish_texts=len({row['final_pl'] for row in rows}),
        longest=[dict(id=row['id'],words=len(WORD.findall(row['final_pl'])),characters=len(row['final_pl']),
            preview=row['final_pl'][:200]) for row in sorted(rows,key=lambda r:len(WORD.findall(r['final_pl'])),reverse=True)[:3]])


def snapshot(name):
    run=ROOT/'reports'/f'{name}_pipeline'/'v1'
    db=sqlite3.connect((run/'checkpoint.sqlite3').resolve().as_uri()+'?mode=ro',uri=True,timeout=30)
    db.row_factory=sqlite3.Row;db.execute('BEGIN')
    columns='id,route,final_pl,final_en'
    if name=='wikimedia':columns+=',language,candidate_index,candidates_json,queue_class'
    rows=[dict(row) for row in db.execute("SELECT "+columns+" FROM items WHERE status='completed' AND route LIKE 'ready%' AND final_pl IS NOT NULL AND length(trim(final_pl))>0")]
    result=dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),total=db.execute('SELECT COUNT(*) FROM items').fetchone()[0],
        statuses=dict(db.execute('SELECT status,COUNT(*) FROM items GROUP BY status')),
        routes=dict(db.execute("SELECT route,COUNT(*) FROM items WHERE status='completed' GROUP BY route")),
        ready_polish=len(rows),ready_after_recycling=sum('recycled' in r['route'] for r in rows),
        ready_without_recycling=sum('recycled' not in r['route'] for r in rows),
        ready_with_final_english=sum(bool(r['final_en'] and r['final_en'].strip()) for r in rows),
        latest_request=db.execute('SELECT MAX(started_at),COUNT(*) FROM requests').fetchone()[:])
    db.rollback();db.close()
    result['lengths']=summary(rows)
    result['lengths_by_route']={label:summary([r for r in rows if ('recycled' in r['route'])==recycled])
        for label,recycled in [('original',False),('recycled',True)]}
    if name=='wikimedia':
        grades=defaultdict(list)
        for row in rows:
            grade=json.loads(row['candidates_json'])[row['candidate_index']].get('quality') or 'unknown'
            grades[grade].append(row)
        result['ready_by_selected_caption_grade']={grade:len(items) for grade,items in sorted(grades.items())}
        result['lengths_by_grade']={grade:summary(items) for grade,items in sorted(grades.items())}
    state_path=ROOT/'reports'/f'{name}_pipeline_state.json'
    state=json.loads(state_path.read_text(encoding='utf-8'))
    result['worker_state']={key:state.get(key) for key in ['status','pid','updated_at','concurrency','error']}
    return result,rows


def main():
    result={};combined=[]
    for name in ['visualnews','wikimedia']:
        result[name],rows=snapshot(name);combined.extend(rows)
    result['combined']=dict(ready_polish=len(combined),lengths=summary(combined))
    result['definitions']=dict(scope='Completed ready pairs only; historical translations and rejected/incomplete items excluded from lengths.',
        words='Unicode word/number sequences, keeping internal hyphens and apostrophes together.',characters='Unicode characters including spaces and punctuation.')
    output=ROOT/'reports/training_text_lengths_current.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
