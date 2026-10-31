import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = ('cirr', 'fashioniq', 'mscoco_t2i', 'refcoco')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_config(path):
    config = read_json(path)
    if config['stage'] != 'E1' or config['loss'] != 'cosine_positive_pair_regression':
        raise ValueError('This runner implements E0/E1 only; E2-E11 are planned separately')
    if config['projector'] != 'linear' or not config['text']['frozen'] or not config['vision']['frozen']:
        raise ValueError('E1 requires a linear projector and two frozen encoders')
    if not config['data']['sources'] or not config['seeds'] or len(set(config['seeds'])) != len(config['seeds']):
        raise ValueError('Choose sources and unique seeds explicitly')
    for field in ('epochs', 'batch_size', 'learning_rate'):
        if config['optimization'][field] <= 0:
            raise ValueError(f'Invalid {field}')
    if not 0 < config['data']['validation_fraction'] < 1:
        raise ValueError('A separate, image-grouped train validation split is required')
    return config
