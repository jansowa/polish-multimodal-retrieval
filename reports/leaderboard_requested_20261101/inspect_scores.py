"""Read-only score comparison using the actual leaderboard aggregation."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from retrieval_benchmark.space.view import overview, default_tasks
from retrieval_benchmark.core import write

if __name__ == '__main__':
    raw = (ROOT/'retrieval_benchmark/space/results.json').read_bytes()
    doc = json.loads(raw)
    registration = json.loads((Path(__file__).with_name('registration.json')).read_text())
    requested = {j['repository'] for j in registration['jobs']}
    columns = ['total', 'license', 'average', 'coverage']+['task:'+n for n in default_tasks(doc)]
    table = overview(doc, visible_columns=columns, show_controls=True)
    rows = [dict(model=row[0], **dict(zip(columns,row[1:]))) for row in table['rows']]
    result = dict(source='retrieval_benchmark/space/results.json', source_sha256=hashlib.sha256(raw).hexdigest(),
                  metric='recall@1', directions='mean t2i/i2t', mode='v2_no_instructions',
                  tasks=table['average_tasks'], requested=[r for r in rows if r['model'] in requested], all_models=rows)
    write(Path(__file__).with_name('score-snapshot.json'), result)
    print('SOURCE_SHA256',result['source_sha256'])
    print('TASKS',result['tasks'])
    for r in rows[:16]:
        print('TOP', r['model'], round(r['total'],2), r['license'],
              None if r['average'] is None else round(r['average'],2), r['coverage'])
    for r in result['requested']:
        print('REQUESTED',json.dumps(r,ensure_ascii=False))
