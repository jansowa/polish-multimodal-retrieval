"""Match project Arrow caches to retained source Parquet files without mutation."""
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/series_2026-10-23'
HF = Path('/home/jsowa/.cache/huggingface')
inspection = json.loads((OUT / 'hf_storage_inspection_2026-10-24.json').read_text())
sources = []
for repo in ('datasets--TIGER-Lab--MMEB-eval', 'datasets--mteb--ARO-Visual-Relation', 'datasets--mteb--ARO-Visual-Attribution'):
    for path in sorted((HF / 'hub' / repo / 'snapshots').glob('**/*.parquet')):
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        columns = [name for name in ('qry_text', 'qry_img_path', 'true_caption', 'false_caption', 'relation_name') if name in names]
        first = next(parquet.iter_batches(batch_size=3, columns=columns), None) if columns else None
        sources.append(dict(path=str(path), exists=path.is_file(), rows=parquet.metadata.num_rows,
                            columns=names, probe=first.to_pylist() if first is not None else []))
matches = []
for config in inspection['datasets']:
    if config['kind'] != 'parquet' or config['sources']:
        continue
    path = Path(config['path'])
    arrow_files = sorted(path.glob('parquet-train*.arrow'))
    rows = 0
    probe = []
    columns = [name for name in ('qry_text', 'qry_img_path', 'true_caption', 'false_caption', 'relation_name') if name in config['features']]
    for arrow in arrow_files:
        with pa.memory_map(str(arrow)) as mapped:
            reader = pa.ipc.open_stream(mapped)
            for batch in reader:
                rows += batch.num_rows
                if not probe:
                    probe = batch.select(columns).slice(0, 3).to_pylist()
    matched = [s['path'] for s in sources if s['rows'] == rows and s['probe'] == probe]
    matches.append(dict(cache=str(path), bytes=config['bytes'], rows=rows, columns=config['features'],
                        text_probe=probe, matching_local_sources=matched,
                        verification='same row count, first three text/path records, and expected columns; not a full-table hash comparison'))
result = dict(mode='read only; no deletions', project_caches=matches, retained_source_parquets=sources)
with (OUT / 'parquet_cache_source_matches_2026-10-24.json').open('x', encoding='utf-8') as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
print(json.dumps(dict(matches=matches, total_project_bytes=sum(m['bytes'] for m in matches)), ensure_ascii=False, indent=2))
