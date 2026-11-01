"""Verify small deletion proposals; this program never deletes anything."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/series_2026-10-23'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    inventory = json.loads((REPORT / 'storage_readonly_inventory.json').read_text())
    verified, skipped = [], []
    for entry in inventory['resume_files']:
        path = Path(entry['path'])
        result_path, best = path.with_name('result.json'), path.with_name('best.pt')
        if not result_path.is_file() or not best.is_file():
            skipped.append(dict(path=str(path), reason='no result.json or best.pt'))
            continue
        result = json.loads(result_path.read_text())
        expected = result.get('checkpoint_sha256')
        actual = sha(best)
        if not expected or expected != actual:
            skipped.append(dict(path=str(path), reason='selected checkpoint hash not verified', keys=list(result)))
            continue
        verified.append(dict(path=str(path), bytes=path.stat().st_size,
                             links=path.stat().st_nlink, sha256=sha(path),
                             retained_best=str(best), retained_best_sha256=actual,
                             lost_capability='resume optimizer/RNG state for this completed run'))
    incident = json.loads((REPORT / 'vn_full_cache_mismatch_incident.json').read_text())
    key = incident['unused_recomputed_text_cache']['identity_sha256']
    folder = ROOT / 'training/runs/tinyclip-visualnews-v1/text_cache/embeddings'
    unused = []
    for path in folder.iterdir():
        if path.name == key or path.name.startswith(key + '.'):
            if path.is_dir():
                unused.extend(dict(path=str(p), bytes=p.stat().st_size, links=p.stat().st_nlink)
                              for p in path.rglob('*') if p.is_file())
            else:
                unused.append(dict(path=str(path), bytes=path.stat().st_size,
                                   links=path.stat().st_nlink, sha256=sha(path)))
    tensor = folder / (key + '.pt')
    assert sha(tensor) == incident['unused_recomputed_text_cache']['sha256']
    historical = folder / (incident['expected_text_cache']['identity_sha256'] + '.pt')
    assert sha(historical) == incident['expected_text_cache']['sha256']
    duplicate = [ROOT / 'training/runs/p0-e2/evaluation_examples.json',
                 ROOT / 'training/runs/vision-four/reference/artifacts/evaluation_examples.json']
    identities = [dict(path=str(p), bytes=p.stat().st_size, inode=p.stat().st_ino,
                       device=p.stat().st_dev, links=p.stat().st_nlink) for p in duplicate]
    result = dict(policy='proposal only; no deletions', verified_resume_candidates=verified,
                  resume_total_bytes=sum(v['bytes'] for v in verified), skipped_resume_files=skipped,
                  unused_text_cache_candidates=unused, unused_text_cache_bytes=sum(v['bytes'] for v in unused),
                  historical_text_cache_sha256=sha(historical),
                  evaluation_snapshot_paths=identities,
                  evaluation_snapshots_already_same_hardlink=len({(v['device'], v['inode']) for v in identities}) == 1)
    output = REPORT / 'storage_deletion_proposal_evidence.json'
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(dict(report=str(output), resume_count=len(verified),
                          resume_total_bytes=result['resume_total_bytes'],
                          unused_text_cache_bytes=result['unused_text_cache_bytes'],
                          skipped_resume_files=skipped,
                          evaluation_snapshots_already_same_hardlink=result['evaluation_snapshots_already_same_hardlink']), indent=2))


if __name__ == '__main__':
    main()
