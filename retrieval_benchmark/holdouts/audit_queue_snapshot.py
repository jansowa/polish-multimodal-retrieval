"""Read-only observation of the actively downloaded Wiki candidates, not a release."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pl_mmb_training.holdouts import Components, Registry, digest, local_path, read_jsonl


def main():
    base = local_path('C:/Users/jsowa/benchmark-translations')
    store = base / 'data-ood-20261030/A/wiki-recovery-20261031'
    export = base / 'series-2026-10-23/expanded-inputs-20261029/wikimedia_final_pairs.jsonl'
    registry = Registry()
    graph = Components()
    registry.seed(graph)
    records = read_jsonl(export.read_bytes())
    nodes, ready = [], set()
    receipt_digests = []
    for i, row in enumerate(records):
        value = row['source_record']['source_sha1']
        sha1 = f'{int(value, 16 if len(value) == 40 else 36):040x}'
        receipt = store / 'compact_receipts' / (sha1 + '.json')
        if receipt.is_file():
            payload = receipt.read_bytes()  # atomic committed JSON receipts only; no SQLite/WAL
            r = json.loads(payload)
            receipt_digests.append(sha1 + ':' + hashlib.sha256(payload).hexdigest())
            if r.get('status') == 'ready':
                if r['original_sha1'] != sha1 or not r['preprocessing_bit_exact']:
                    raise RuntimeError('Unverified ready receipt: ' + sha1)
                row = dict(row, **{k: r[k] for k in ('sha256', 'pixels_sha256', 'phash')}, source_sha1_hex=sha1)
                ready.add(i)
        nodes.append(graph.add(row))
    all_labels, ready_labels = Counter(), Counter()
    for i, node in enumerate(nodes):
        labels = graph.reasons(node)
        all_labels.update(labels)
        if i in ready:
            ready_labels.update(labels)
    proof = dict(status='partial_observation_not_a_training_release', observed_at=datetime.now(timezone.utc).isoformat(),
                 registry_sha256=registry.sha256, export_sha256=digest(export), exported_candidate_rows=len(records),
                 ready_candidate_rows=len(ready), candidates_without_ready_original=len(records) - len(ready),
                 metadata_or_verified_fingerprint_matches=dict(all_labels), ready_candidate_matches=dict(ready_labels),
                 receipt_snapshot_sha256=hashlib.sha256('\n'.join(sorted(receipt_digests)).encode()).hexdigest(),
                 interpretation='Raw candidates are expected to include protected rows; the final audit must reject them. Missing originals are not certified. Receipts observed without rereading image payloads; no admission approval is issued.')
    report = ROOT / 'reports/benchmark_decontamination_20261031/queue_snapshot.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(proof, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
