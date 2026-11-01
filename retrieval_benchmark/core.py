"""Small, dependency-free storage layer. Scores are fractions, never percentages."""
import hashlib
import json
import math
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CATALOG = HERE / 'catalog-current.json'
METRICS = ('recall@1', 'recall@5', 'recall@10', 'mrr', 'ndcg@10')
MODES = ('v2_no_instructions', 'author_recommended')
PROTOCOL = 'v2-no-instructions-native-cosine-full-gallery-multi-positive'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def write(path, value, immutable=False):
    path = Path(path)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                          allow_nan=False) + '\n').encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == payload:
        return
    if immutable:
        with path.open('xb') as stream:
            stream.write(payload)
        return
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('xb') as stream:
        stream.write(payload)
    temporary.replace(path)


def local_path(value):
    value = str(value)
    if os.name == 'nt':
        if re.match(r'^/mnt/[a-z]/', value):
            return Path(value[5].upper() + ':' + value[6:])
        if value.startswith('/home/'):
            return Path('//wsl.localhost/Ubuntu' + value)
    elif re.match(r'^[A-Za-z]:[\\/]', value):
        return Path('/mnt/' + value[0].lower() + value[2:].replace('\\', '/'))
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def parameters(value):
    result = {key: value.get(key, 0) for key in ('vision', 'text', 'head', 'shared')}
    if any(type(v) is not int or v < 0 for v in result.values()):
        raise ValueError('Parameter counts must be nonnegative integers')
    result['total'] = result['vision'] + result['text']
    return result


def validate(document):
    if document.get('schema_version') != 1:
        raise ValueError('Unsupported results schema')
    datasets, models = document['datasets'], document['models']
    seen = set()
    for model in models.values():
        p = model['parameters']
        if parameters(p) != p:
            raise ValueError('V+T must exclude head/shared/other modalities')
    for row in document['results']:
        if row['model_id'] not in models or row['dataset_id'] not in datasets:
            raise ValueError('Unknown model/dataset')
        if datasets[row['dataset_id']]['status'] != 'ready':
            raise ValueError('A pending/blind dataset cannot contain scores')
        if row['protocol'] != PROTOCOL or row['prompt_mode'] not in MODES:
            raise ValueError('Incompatible protocol or prompt mode')
        key = (row['model_id'], row['dataset_id'], row['prompt_mode'], row['profile'])
        if key in seen:
            raise ValueError('Duplicate model/dataset/mode/profile result')
        seen.add(key)
        for field in ('manifest_sha256', 'gallery_sha256', 'text_gallery_sha256', 'qrels_sha256', 'source_sha256'):
            if not re.fullmatch('[0-9a-f]{64}', row[field]):
                raise ValueError('Missing or invalid provenance: ' + field)
        for direction in ('t2i', 'i2t'):
            metrics = row['metrics'][direction]
            if set(metrics) != set(METRICS):
                raise ValueError('Incomplete metric set')
            if any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1
                   for v in metrics.values()):
                raise ValueError('Metrics must be finite fractions in [0,1]')
        # A different qrels/gallery must be a separate dataset version, not another row.
        expected = datasets[row['dataset_id']].setdefault('fingerprint', {
            k: row[k] for k in ('manifest_sha256', 'gallery_sha256', 'text_gallery_sha256', 'qrels_sha256')})
        if any(row[k] != v for k, v in expected.items()):
            raise ValueError('Mixed dataset versions: ' + row['dataset_id'])
    return document


def public_check(document):
    """Public files contain identifiers and scores, never local paths or captions."""
    text = json.dumps(document, ensure_ascii=False)
    if re.search(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/]|/mnt/[a-z]/|/home/|\\\\wsl|hf_[A-Za-z0-9]{20,}', text):
        raise ValueError('Local path or credential-like value in public export')
    return validate(document)
