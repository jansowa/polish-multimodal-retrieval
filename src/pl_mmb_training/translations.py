"""Freeze a consistent, read-only view of completed translations, including live runs."""
import json
import sqlite3
from pathlib import Path

from .common import ROOT, file_sha, write_json, read_json


def snapshot(source, directory):
    key = source['dataset']
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{key}.jsonl'
    metadata_path = directory / f'{key}.json'
    origin = Path(source.get('checkpoint') or source['translations'])
    if not origin.is_absolute():
        origin = ROOT / origin
    if path.exists() and metadata_path.exists():
        metadata = read_json(metadata_path)
        if metadata['origin'] != str(origin) or metadata['sha256'] != file_sha(path):
            raise ValueError('Translation snapshot changed; choose a new run directory')
        return path, metadata
    if 'checkpoint' in source:
        # One SQLite transaction fixes the cutoff even while translation continues.
        with sqlite3.connect(origin.as_uri() + '?mode=ro', uri=True) as db, path.open('w', encoding='utf-8') as out:
            db.execute('BEGIN')
            count = 0
            for uid, en, pl, metadata in db.execute(
                    'SELECT unit_id,source_json,translated_json,metadata_json FROM completed ORDER BY unit_id'):
                out.write(json.dumps(dict(id=uid, source=json.loads(en), translated=json.loads(pl),
                                          metadata=json.loads(metadata)), ensure_ascii=False) + '\n')
                count += 1
    else:
        import shutil
        shutil.copyfile(origin, path)
        with path.open() as stream:
            count = sum(bool(line.strip()) for line in stream)
    metadata = dict(origin=str(origin), sha256=file_sha(path), completed=count)
    write_json(metadata_path, metadata)
    return path, metadata


def patches_at(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]
