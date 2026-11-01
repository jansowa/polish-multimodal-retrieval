"""Measure export payloads from one read-only snapshot, without materializing packages."""
import json
import sqlite3
import zlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'reports/visualnews_pipeline/v1'


class Meter:
    def __init__(self):
        self.raw = self.compressed = self.rows = 0
        self.encoder = zlib.compressobj(6, zlib.DEFLATED, 31)

    def add(self, obj):
        payload = (json.dumps(obj, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')
        self.raw += len(payload)
        self.compressed += len(self.encoder.compress(payload))
        self.rows += 1

    def finish(self):
        self.compressed += len(self.encoder.flush())
        return dict(rows=self.rows, bytes=self.raw, gzip_bytes=self.compressed)


def main():
    db = sqlite3.connect((RUN / 'checkpoint.sqlite3').as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    db.execute('BEGIN')
    stamp = datetime.now(timezone.utc).isoformat()
    ready = [dict(r) for r in db.execute("SELECT * FROM items WHERE status='completed' AND route LIKE 'ready%' AND final_pl IS NOT NULL ORDER BY id")]
    ready_ids = {r['id'] for r in ready}
    meters = {k: Meter() for k in ['pl', 'pl_en', 'requests_all', 'requests_ready', 'outputs_all', 'outputs_ready']}
    for r in ready:
        base = dict(id=r['id'], image_did=r['image_did'], image_path=r['image_path'], text_pl=r['final_pl'])
        meters['pl'].add(base)
        meters['pl_en'].add(dict(base, text_en=r['final_en'], original_en=r['caption']))
    processed = [dict(r) for r in db.execute("SELECT id,image_did,image_path,shard,row_index FROM items WHERE EXISTS (SELECT 1 FROM outputs o WHERE o.id=items.id) OR EXISTS (SELECT 1 FROM requests r WHERE r.id=items.id)")]
    for row in db.execute('SELECT * FROM requests ORDER BY request_id'):
        record = dict(row)
        for key in ['input_json', 'response_json']:
            raw = record.pop(key)
            record[key.removesuffix('_json')] = json.loads(raw) if raw else None
        meters['requests_all'].add(record)
        if record['id'] in ready_ids:
            meters['requests_ready'].add(record)
    for row in db.execute('SELECT * FROM outputs ORDER BY id,stage'):
        record = dict(row)
        for key in ['output_json', 'origin_json']:
            raw = record.pop(key)
            record[key.removesuffix('_json')] = json.loads(raw) if raw else None
        meters['outputs_all'].add(record)
        if record['id'] in ready_ids:
            meters['outputs_ready'].add(record)
    db.rollback()
    db.close()
    result = dict(snapshot_utc=stamp, ready_pairs=len(ready), processed_pairs=len(processed), payloads={k: v.finish() for k,v in meters.items()})
    print(json.dumps(dict(event='texts_measured', **result)), flush=True)
    # Read binary lengths from only the necessary row groups; do not recompress or resize images.
    ready_keys = {(r['shard'], r['row_index']) for r in ready}
    all_keys = {(r['shard'], r['row_index']) for r in processed} | ready_keys
    shards = defaultdict(set)
    for shard, index in all_keys:
        shards[shard].add(index)
    ready_bytes = all_bytes = 0
    for shard, indices in sorted(shards.items()):
        with pq.ParquetFile(shard) as file:
            offset = 0
            for group in range(file.num_row_groups):
                count = file.metadata.row_group(group).num_rows
                selected = sorted(i for i in indices if offset <= i < offset + count)
                if selected:
                    table = file.read_row_group(group, columns=['img'])
                    lengths = pc.binary_length(pc.struct_field(table['img'], 'bytes'))
                    sizes = pc.take(lengths, pa.array([i-offset for i in selected])).to_pylist()
                    for index, size in zip(selected, sizes):
                        if not size:
                            raise ValueError('Empty image')
                        all_bytes += size
                        if (shard, index) in ready_keys:
                            ready_bytes += size
                offset += count
        print(json.dumps(dict(event='shard_measured', shard=Path(shard).name)), flush=True)
    result['images'] = dict(ready_unique=len(ready_keys), ready_bytes=ready_bytes, processed_unique=len(all_keys), processed_bytes=all_bytes)
    result['local_files'] = {p.name: dict(bytes=p.stat().st_size, modified_utc=datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()) for p in RUN.iterdir() if p.is_file()}
    result['notes'] = ['JSONL sizes measured exactly; gzip level 6 measured exactly without writing archives.', 'Images are original encoded bytes, deduplicated by verified shard/row; archive overhead and recompression not included.', 'Full flow is requests plus outputs plus manifest and prompt files; includes rejected examples and failures, without reasoning.']
    target = ROOT / 'reports/visualnews_pipeline/package_sizes.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
