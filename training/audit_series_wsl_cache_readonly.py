"""Inspect derived Hugging Face cache provenance without changing the cache."""
import json
import sys
from pathlib import Path

base = Path('/home/jsowa/.cache/huggingface/datasets/json')
records = []
for info in base.glob('**/dataset_info.json'):
    value = json.loads(info.read_text())
    files = list(info.parent.glob('*.arrow'))
    sources = []
    for name, details in value.get('download_checksums', {}).items():
        path = Path(name)
        sources.append(dict(path=name, exists=path.is_file(),
                            current_bytes=path.stat().st_size if path.is_file() else None,
                            recorded=details))
    records.append(dict(path=str(info.parent), arrow_bytes=sum(p.stat().st_size for p in files),
                        source_files=sources, splits=value.get('splits')))
records.sort(key=lambda r: -r['arrow_bytes'])
if len(sys.argv) > 1:
    with Path(sys.argv[1]).open('x', encoding='utf-8') as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    print(json.dumps(dict(report=sys.argv[1], records=len(records), arrow_bytes=sum(r['arrow_bytes'] for r in records))))
else:
    print(json.dumps(records[:12], ensure_ascii=False, indent=2))
