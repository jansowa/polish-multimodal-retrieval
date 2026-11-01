"""Versioned, CPU-only holdout protection shared by training and evaluation.

No model, scoring, image download or destructive operation is performed here.
The registry pins existing audited identity manifests, not embedding caches.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / 'retrieval_benchmark/holdouts/registry-v2.json'
GUARD = ROOT / 'experiments/data_study_20261030/guard.py'
_spec = importlib.util.spec_from_file_location('_holdout_components', GUARD)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_path(value):
    value = str(value).replace('\\', '/')
    if os.name == 'nt':
        match = re.match(r'^/mnt/([a-z])/(.*)$', value)
        if match:
            return Path(match[1].upper() + ':/' + match[2])
        if value.startswith('/home/'):
            return Path('//wsl.localhost/Ubuntu' + value)
    elif re.match(r'^[A-Za-z]:/', value):
        return Path('/mnt/' + value[0].lower() + value[2:])
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def canonical_path(value):
    value = str(value).replace('\\', '/')
    if re.match(r'^[A-Za-z]:/', value):
        value = '/mnt/' + value[0].lower() + value[2:]
    value = re.sub(r'^//wsl(?:\.localhost|\$)/[^/]+/', '/', value, flags=re.I)
    return value


def extra_tokens(row):
    """Unify cache paths, Flickr URL aliases and EN/PL normalized text pools."""
    result = {'text:' + _base.normalized(t) for t in row.get('normalized_texts', []) if t}
    if row.get('image'):
        result.add('image_path:' + canonical_path(row['image']))

    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                # License/author/project URLs are NOT image identities. Sharing
                # CC-BY or an uploader must never merge unrelated photographs.
                if key in ('source_record', 'associations', 'parent_ids', 'article_ids',
                           'id', 'original_image_id', 'source_image', 'image_did',
                           'source_page_url', 'source_image_url', 'image_url',
                           'original_url', 'original_image_url', 'image_original_url', 'flickr_url', 'article_url'):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            match = re.search(r'flickr(?:30k|8k)?[:/-](\d{5,})', value, re.I)
            if match:
                result.add('parent:flickr:' + str(int(match[1])))
            if value.startswith(('http://', 'https://')):
                url = urlsplit(value)
                query = '&'.join(p for p in url.query.split('&') if p and not p.lower().startswith('utm_'))
                result.add('url:' + urlunsplit(('', url.netloc.lower(), unquote(url.path), query, '')))
                match = re.search(r'flickr.com/photos/[^/]+/(\d+)', value)
                match = match or re.search(r'(?:staticflickr|static.flickr)\.com/[^?]*/(\d+)_', value)
                if match:
                    result.add('parent:flickr:' + str(int(match[1])))
    visit(row)
    return result


class Components(_base.Components):
    """Existing transitive pHash<=6 guard plus cross-pipeline aliases."""
    def add(self, row, labels=()):
        node = super().add(row, labels)
        for token in extra_tokens(row):
            key = hashlib.sha256(token.encode()).digest()
            if key in self.owner:
                self.join(node, self.owner[key])
            else:
                self.owner[key] = node
        return node


def read_jsonl(payload):
    return [json.loads(line) for line in payload.decode('utf-8-sig').splitlines() if line.strip()]


class Registry:
    def __init__(self, path=DEFAULT_REGISTRY):
        self.path = Path(path)
        self.sha256 = digest(self.path)
        self.value = json.loads(self.path.read_text(encoding='utf-8'))
        if self.value.get('schema') != 1 or self.value.get('guard_sha256') != digest(GUARD):
            raise RuntimeError('Unknown registry schema or changed component algorithm; register a new version')
        self.inputs = []
        for item in self.value['inputs']:
            payload = local_path(item['path']).read_bytes()
            if hashlib.sha256(payload).hexdigest() != item['sha256']:
                raise RuntimeError('Changed holdout input: ' + item['key'])
            value = None if item['format'] == 'evidence' else read_jsonl(payload) if item['format'] in ('jsonl', 'fingerprint_cache') else json.loads(payload)
            self.inputs.append((item, value))
        self._rows = None

    def rows(self):
        if self._rows is not None:
            yield from self._rows
            return
        by_hash, by_path = {}, {}
        for item, value in self.inputs:
            if item['format'] == 'fingerprint_cache':
                for row in value:
                    if row.get('phash'):
                        for field in ('sha256', 'pixels_sha256'):
                            if row.get(field):
                                by_hash[row[field]] = row['phash']
            elif item['format'] == 'phash_cache':
                by_path.update({canonical_path(k): v for k, v in value['hashes'].items()})
        output = []
        for item, value in self.inputs:
            if item['format'] in ('fingerprint_cache', 'phash_cache', 'catalog', 'evidence'):
                continue  # caches supply fingerprints ONLY; their train rows are never holdouts
            label = 'holdout:' + item['key']
            if item['format'] in ('jsonl', 'json_array'):
                rows = value
            elif item['format'] == 'guard':
                rows = [dict(row, **({'image': key} if not key.isdigit() else {})) for key, row in value['images'].items()]
                for field, target in (('ids', 'id'), ('texts', 'text_pl'), ('source_sha1_hex', 'source_sha1_hex')):
                    rows += [{target: v} for v in value.get(field, [])]
                rows += [dict(parent_ids=[v]) for v in value.get('parent_ids', [])]
            else:
                raise RuntimeError('Unsupported registry input: ' + item['format'])
            for original in rows:
                row = dict(original)
                if row.get('sha256') or row.get('pixels_sha256'):
                    phash = row.get('phash') or by_hash.get(row.get('sha256')) or by_hash.get(row.get('pixels_sha256')) or by_path.get(canonical_path(row.get('image', '')))
                    if not phash:
                        raise RuntimeError('Missing holdout original-image pHash: ' + item['key'] + ':' + str(row.get('image', row.get('id'))))
                    row['phash'] = phash
                output.append((row, label))
        self._rows = output
        self.inputs = []  # release the large historical fingerprint cache
        yield from output

    def seed(self, graph):
        counts = Counter()
        for row, label in self.rows():
            graph.add(row, [label])
            counts[label] += 1
        return dict(counts)

    def audit(self, records, require_verified=True):
        graph = Components()
        counts = self.seed(graph)
        indices = [graph.add(row) for row in records]
        rejected, unverified = [], []
        reasons = Counter()
        for row, node in zip(records, indices):
            labels = graph.reasons(node)
            if labels:
                rejected.append(row.get('id', row.get('image')))
                reasons.update(labels)
            if require_verified and not all(row.get(k) for k in ('sha256', 'pixels_sha256', 'phash', 'lineage_verified')):
                unverified.append(row.get('id', row.get('image')))
        return dict(schema=1, status='passed' if not rejected and not unverified else 'blocked',
                    registry_sha256=self.sha256, guard_sha256=digest(__file__), candidate_rows=len(records),
                    protected_rows=counts, rejected_rows=len(rejected), rejected_ids=rejected,
                    rejection_labels=dict(reasons), unverified_rows=len(unverified), unverified_ids=unverified,
                    scope='New admissions only; historical base exposure is not retroactively repaired')


def require_training_release(store, report, registry_path=DEFAULT_REGISTRY):
    """Fail before GPU acquisition if either additions or the frozen registry changed."""
    registry = Registry(registry_path)
    binding = json.loads((Path(report) / 'holdout_registry_binding.json').read_text(encoding='utf-8'))
    if binding['registry_sha256'] != registry.sha256 or binding['guard_sha256'] != digest(__file__):
        raise RuntimeError('Holdout registry/guard changed after data freeze; a new audited generation is required')
    records = []
    manifests = {}
    for name in ('Wiki_train.jsonl', 'new_VN_train.jsonl'):
        path = Path(store) / name
        manifests[name] = digest(path)
        records += read_jsonl(path.read_bytes())
    if binding.get('manifests') != manifests:
        raise RuntimeError('Training admissions changed after their audited freeze')
    proof = registry.audit(records)
    proof['manifests'] = manifests
    if proof['status'] != 'passed':
        raise RuntimeError('New training admissions overlap holdouts or lack verified identities: ' + json.dumps(proof, ensure_ascii=False)[:3000])
    return proof


def verify_release_proof(proof, store, registry_path=DEFAULT_REGISTRY):
    """Recheck immutable inputs after waiting for the GPU, without redoing DSU."""
    if digest(registry_path) != proof['registry_sha256'] or digest(__file__) != proof['guard_sha256']:
        raise RuntimeError('Holdout registry or guard changed while waiting for GPU')
    registry = json.loads(Path(registry_path).read_text(encoding='utf-8'))
    if digest(GUARD) != registry['guard_sha256']:
        raise RuntimeError('Base component algorithm changed while waiting for GPU')
    for item in registry['inputs']:
        if digest(local_path(item['path'])) != item['sha256']:
            raise RuntimeError('Holdout input changed while waiting for GPU: ' + item['key'])
    for name, expected in proof['manifests'].items():
        if digest(Path(store) / name) != expected:
            raise RuntimeError('Training admissions changed while waiting for GPU: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument('--manifest', type=Path, action='append', default=[])
    args = parser.parse_args()
    registry = Registry(args.registry)
    if not args.manifest:
        counts = Counter(label for _, label in registry.rows())
        print(json.dumps(dict(status='registry_verified', registry_sha256=registry.sha256, protected_rows=counts), indent=2))
        return
    records = [row for path in args.manifest for row in read_jsonl(path.read_bytes())]
    proof = registry.audit(records)
    print(json.dumps(proof, ensure_ascii=False, indent=2))
    if proof['status'] != 'passed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
