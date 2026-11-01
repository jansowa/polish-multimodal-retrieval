"""Read-only structural inspection for F01-a; run in WSL."""
import json
from pl_mmb_training.current_holdouts import Registry, digest, local_path

registry = Registry()
rows = list(registry.rows())
viz = [r for r, label in rows if label == 'holdout:vizwiz_pl_local_validation']
print(json.dumps({'registry_status': 'registry_verified', 'registry_sha256': registry.sha256,
                  'protected_rows': len(rows), 'viz_rows': len(viz),
                  'viz_keys': sorted(set().union(*(r.keys() for r in viz))),
                  'viz_identity_sample': {k: viz[0].get(k) for k in ('id', 'image', 'sha256', 'pixels_sha256', 'phash', 'component_id')}}, indent=2))
config = json.loads(local_path('retrieval_benchmark/local/config.json').read_text())
entry = config['datasets']['vizwiz_pl_local_validation']
protocol = json.loads(local_path(entry['protocol']).read_text())
print(json.dumps({'protocol_path': entry['protocol'], 'protocol_sha256': digest(local_path(entry['protocol'])),
                  'protocol_structure': {k: {'type': type(v).__name__, 'size': len(v) if hasattr(v, '__len__') else None,
                                             'sample': v[:1] if isinstance(v, list) else list(v.items())[:1] if isinstance(v, dict) else v}
                                         for k, v in protocol.items()}}, indent=2, ensure_ascii=False))
