"""Count downloaded MAUPQA CSV rows without downloading or changing datasets."""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
root = Path.home() / '.cache/huggingface/hub/datasets--ipipan--maupqa/snapshots/cdaceb38daba793c1dd5bb13fb3d91450242d464'
csv.field_size_limit(10_000_000)
sources = []
for path in sorted(root.rglob('*.csv')):
    count, questions, labels = 0, set(), Counter()
    with path.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            count += 1
            questions.add(row['question_id'])
            labels[row['relevant']] += 1
    value = dict(subset=path.parent.name, rows=count, unique_question_ids=len(questions),
                 relevance_labels=dict(labels), bytes=path.stat().st_size)
    sources.append(value)
    print(json.dumps(value), flush=True)
path = ROOT / 'reports/current_data_inventory.json'
inventory = json.loads(path.read_text(encoding='utf-8'))
inventory['additional_text_sources'] = dict(maupqa=dict(revision=root.name, subsets=sources, rows=sum(s['rows'] for s in sources)),
                                           mfaq_pl=dict(train=63157, validation=2025, rows=65182))
path.write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('MAUPQA TOTAL',sum(s['rows'] for s in sources),flush=True)
