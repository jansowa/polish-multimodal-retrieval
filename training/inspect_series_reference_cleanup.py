"""Read-only inspection of the exact reference weight targets."""
import json
from run_series_reference_evaluation import model_weight_items, file_sha
from pathlib import Path

hub = Path.home() / '.cache/huggingface/hub'
for item in model_weight_items('embeddinggemma-2'):
    repo = hub / ('models--' + item['repo'].replace('/', '--'))
    for entry in item['files']:
        link = repo / 'snapshots' / item['revision'] / entry['name']
        print(json.dumps(dict(repo=str(repo), resolved_repo=str(repo.resolve()),
            snapshot_path=str(link), snapshot_symlink=link.is_symlink(),
            snapshot_link_target=str(link.readlink()) if link.is_symlink() else None,
            resolved_target=str(link.resolve()), exists=link.is_file(),
            bytes=link.stat().st_size if link.is_file() else None,
            expected_bytes=entry['bytes'], metadata=entry,
            sha256=file_sha(link) if link.is_file() else None)))
        target = link.resolve()
        for other_repo in hub.glob('models--*'):
            for other in (other_repo / 'snapshots').rglob('*'):
                if other.is_symlink() and other.resolve() == target:
                    print(json.dumps({'weight_reference': str(other)}))
