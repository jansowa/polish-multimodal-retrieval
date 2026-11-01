"""Read-only inventory of the available VisualNews training assets."""
import json
import sqlite3
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path('/mnt/c/Users/jsowa/benchmark-translations/visualnews-image-cache')


def main():
    rows = [json.loads(s) for s in (ROOT / 'reports/visualnews_pipeline/v1/final_pairs.jsonl').read_text(encoding='utf-8').splitlines()]
    db = sqlite3.connect((ASSETS / 'assets.sqlite3').as_uri() + '?mode=ro')
    print(json.dumps(dict(rows=len(rows), unique_paths=len({r['source_image'] for r in rows}),
        sources=dict(Counter(r['source_image'].split('/')[2] for r in rows)),
        schema=db.execute("SELECT name,sql FROM sqlite_master WHERE type='table'").fetchall(),
        example_mapping=db.execute('SELECT * FROM images WHERE did=?', (rows[0]['image_did'],)).fetchone()), ensure_ascii=False), flush=True)
    print('METADATA PREFIX', (ASSETS / 'original-data.json').open(encoding='utf-8').read(1700), flush=True)
    for filename in ['training/runs/xm3600-pl/blacklist.json', 'training/runs/clean-v1/perceptual_hashes.json']:
        value = json.loads((ROOT / filename).read_text())
        print(filename, 'keys', list(value)[:8], 'first', str(next(iter(value.values())))[:350], flush=True)


if __name__ == '__main__':
    main()
