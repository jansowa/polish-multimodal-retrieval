"""Local-only inventory; read checkpoints without changing training snapshots."""
import json
import sqlite3
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_RUNS = {
    'cirr': 'qwen3.8-27b-absence-v3b-single',
    'fashioniq': 'qwen3.8-27b-absence-v3b-single',
    'refcoco': 'qwen3.8-27b-absence-v3b-single',
    'mscoco_t2i': 'qwen3.8-27b-full-v2-single',
    'aro_visual_relation': 'qwen3.8-27b-absence-v3b-single',
    'aro_visual_attribution': 'qwen3.8-27b-absence-v3b-single',
    'sugarcrepe_continuation': 'qwen3.8-27b-absence-extension-v3b',
    'mbeir_mscoco': 'qwen3.8-27b-absence-extension-v3b',
    'mbeir_visualnews': 'qwen3.8-27b-visualnews-v3c',
}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    result = dict(observed_at=datetime.now(timezone.utc).isoformat(), checkpoints=[], parquet_sources=[], image_files={})
    checkpoint_report = ROOT / 'reports/current_translation_inventory.json'
    if '--checkpoints-only' in sys.argv:
        extra = Path('C:/Users/jsowa/benchmark-translations') if os.name == 'nt' else Path('/mnt/c/Users/jsowa/benchmark-translations')
        for parent in (ROOT / 'outputs', extra):
            for path in sorted(parent.glob('**/checkpoint.sqlite3')):
                if ACTIVE_RUNS.get(path.parent.parent.name) != path.parents[2].name:
                    continue
                with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
                    completed = db.execute('SELECT COUNT(*) FROM completed').fetchone()[0]
                    failed = db.execute('SELECT COUNT(*) FROM failures').fetchone()[0]
                    first = db.execute('SELECT source_json, metadata_json FROM completed LIMIT 1').fetchone()
                    columns = [r[1] for r in db.execute('PRAGMA table_info(completed)')]
                entry = dict(path=str(path), run=path.parents[2].name, dataset=path.parent.parent.name,
                             variant=path.parent.name, completed=completed, failures=failed,
                             pilot='pilot' in str(path), schema=columns)
                if first:
                    entry.update(source_keys=list(json.loads(first[0])), sample_metadata=json.loads(first[1]))
                result['checkpoints'].append(entry)
        checkpoint_report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps([{k:v for k,v in e.items() if k not in ('sample_metadata','schema','source_keys')} for e in result['checkpoints']],indent=2))
        return
    translations = read(checkpoint_report)
    result['checkpoints'] = [e for e in translations['checkpoints'] if ACTIVE_RUNS.get(e['dataset']) == e['run']]
    result['translations_observed_at'] = translations['observed_at']
    import pyarrow.parquet as pq
    roots = [ROOT / '.cache/huggingface', ROOT / '.cache/huggingface-windows', Path.home() / '.cache/huggingface/hub']
    for cache in roots:
        for dataset in sorted(cache.glob('datasets--*')):
            for snapshot in sorted((dataset / 'snapshots').glob('*')):
                files = []
                for path in sorted(snapshot.rglob('*.parquet')):
                    info = pq.ParquetFile(path)
                    files.append(dict(name=str(path.relative_to(snapshot)), rows=info.metadata.num_rows,
                                      bytes=path.stat().st_size, columns=info.schema_arrow.names))
                result['parquet_sources'].append(dict(dataset=dataset.name, revision=snapshot.name, files=files))
    for cache in roots:
        images = cache / 'pl_mmb_images'
        if images.exists():
            for group in sorted(images.iterdir()):
                if group.is_dir():
                    counts = Counter()
                    total_bytes = 0
                    for path in group.rglob('*'):
                        if path.is_file() and path.suffix.lower() in ('.jpg','.jpeg','.png','.webp','.bmp'):
                            counts[path.relative_to(group).parts[0]] += 1
                            total_bytes += path.stat().st_size
                    result['image_files'][str(group)] = dict(files=sum(counts.values()), bytes=total_bytes, subgroups=dict(counts))
    manifest = read(ROOT / 'training/runs/e1-full/cache.json')
    audit = read(ROOT / 'training/runs/e1-full/data_audit.json')
    result['training_snapshot'] = dict(image_cache_entries=len(manifest['images']), text_cache_entries=len(manifest['texts']),
                                      sources=audit['sources'], audit_summary={k:v for k,v in audit.items() if k != 'sources'})
    result['prepared_sources'] = []
    for path in sorted((ROOT / '.cache/absence-v3').glob('*.jsonl')):
        with path.open('rb') as stream:
            count = sum(1 for _ in stream)
        result['prepared_sources'].append(dict(path=str(path), records=count, bytes=path.stat().st_size))
    output = ROOT / 'reports/current_data_inventory.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(observed_at=result['observed_at'], checkpoints=[{k:v for k,v in e.items() if k not in ('sample_metadata','source_keys','schema')} for e in result['checkpoints']],
                         parquet_sources=[dict(dataset=e['dataset'],revision=e['revision'],shards=len(e['files']),rows=sum(f['rows'] for f in e['files'])) for e in result['parquet_sources']],
                         image_files=result['image_files'],prepared_sources=result['prepared_sources'],
                         training_cache=dict(images=len(manifest['images']),texts=len(manifest['texts']))),indent=2), flush=True)

if __name__ == '__main__':
    main()
