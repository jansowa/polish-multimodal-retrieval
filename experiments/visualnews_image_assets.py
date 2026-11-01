"""Pin and verify VisualNews image assets and original caption/image joins."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time

import httpx
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path('C:/Users/jsowa/benchmark-translations/visualnews-image-cache')
IMAGE_REPO = 'MrZilinXiao/mbeir_test_visualnews_task0'
IMAGE_REV = '6864d46a2e94d456ac6262c7aa12b4abcfc1cd4b'
META_REPO = 'twelcone/VisualNews'
META_REV = '8a68a1f31d3ed8ac30a86e8e78478f45dd80bc24'
DB = CACHE / 'assets.sqlite3'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def download(client, url, path, expected_size=None):
    if path.exists() and (expected_size is None or path.stat().st_size == expected_size):
        return path
    if shutil.disk_usage(CACHE).free < 8 * 1024 ** 3 + (expected_size or 400 * 1024 ** 2):
        raise RuntimeError('Insufficient disk space; keep an 8 GiB reserve')
    for attempt in range(4):
        try:
            with client.stream('GET', url) as response:
                response.raise_for_status()
                with path.with_suffix('.part').open('wb') as stream:
                    for block in response.iter_bytes(1024 ** 2):
                        stream.write(block)
            if expected_size is not None and path.with_suffix('.part').stat().st_size != expected_size:
                raise RuntimeError('Incomplete asset download')
            path.with_suffix('.part').replace(path)
            print(json.dumps(dict(event='downloaded', file=path.name, bytes=path.stat().st_size)), flush=True)
            return path
        except (httpx.HTTPError, RuntimeError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def json_array_items(path):
    """Incremental JSON-array reader: never materialize all original metadata."""
    decoder = json.JSONDecoder()
    with path.open(encoding='utf-8') as stream:
        buf = ''
        ended = False
        first = True
        while True:
            buf = buf.lstrip()
            if first and buf:
                if buf[0] != '[':
                    raise ValueError('Expected an original metadata array')
                buf = buf[1:]
                first = False
            buf = buf.lstrip(' \r\n\t,')
            if buf.startswith(']'):
                return
            if not first and buf:
                try:
                    value, end = decoder.raw_decode(buf)
                except json.JSONDecodeError:
                    if ended:
                        raise
                else:
                    yield value
                    buf = buf[end:]
                    continue
            block = stream.read(256 * 1024)
            if not block:
                if ended:
                    raise ValueError('Truncated original metadata array')
                ended = True
            buf += block


def format_caption(value):
    # The pinned UniIR author's format_string implementation.
    value = (value or '').replace('\r', '').strip().strip('"')
    if value:
        value = value[0].upper() + value[1:]
        if value[-1] not in '.?!':
            value += '.'
    return value


def main():
    CACHE.mkdir(exist_ok=True)
    with httpx.Client(verify=str(ROOT / 'combined-ca.pem'), follow_redirects=True, timeout=180) as client:
        response = client.get(f'https://huggingface.co/api/datasets/{IMAGE_REPO}/tree/{IMAGE_REV}/corpus?recursive=true&limit=1000')
        response.raise_for_status()
        shards = sorted([x for x in response.json() if x['path'].endswith('.parquet')], key=lambda x: x['path'])
        if len(shards) != 49:
            raise ValueError('Unexpected image repository layout')
        def acquire(item):
            index = int(item['path'].split('test-')[1].split('-')[0])
            return download(client, f'https://huggingface.co/datasets/{IMAGE_REPO}/resolve/{IMAGE_REV}/{item["path"]}',
                            CACHE / f'corpus-test-{index:05d}.parquet', item['size'])
        with ThreadPoolExecutor(max_workers=2) as pool:
            paths = list(pool.map(acquire, shards))
        metadata = download(client, f'https://huggingface.co/datasets/{META_REPO}/resolve/{META_REV}/data.json',
                            CACHE / 'original-data.json', 356272348)
    if DB.exists():
        with sqlite3.connect(DB) as connection:
            if connection.execute("SELECT value FROM provenance WHERE key='status'").fetchone() == ('complete',):
                print('Existing complete verified asset index retained', flush=True)
                return
        raise RuntimeError('Incomplete asset index exists; inspect it before rebuilding')
    originals = ROOT / '.cache/visualnews-image-sources'
    official_images = {}
    with (originals / 'mbeir_visualnews_task0_cand_pool.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            official_images[row['did']] = row['img_path']
    captions = {}
    with (originals / 'mbeir_visualnews_task3_cand_pool.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['txt'] in captions:
                raise ValueError('Duplicate source caption')
            captions[row['txt']] = row['did']
    with sqlite3.connect(DB) as connection:
        connection.execute('CREATE TABLE images (did TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL, shard TEXT NOT NULL, row_index INTEGER NOT NULL)')
        connection.execute('CREATE TABLE captions (did TEXT PRIMARY KEY, text TEXT UNIQUE NOT NULL, image_did TEXT, image_path TEXT, matching_images INTEGER NOT NULL DEFAULT 0)')
        connection.execute('CREATE TABLE provenance (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        connection.executemany('INSERT INTO captions(did,text) VALUES (?,?)', [(did, text) for text, did in captions.items()])
        connection.commit()
        fingerprints = []
        for path in paths:
            pf = pq.ParquetFile(path)
            offset = 0
            for batch in pf.iter_batches(batch_size=4096, columns=['did', 'img_path', 'corpus-id']):
                records = []
                for row in batch.to_pylist():
                    if row['did'] != row['corpus-id'] or official_images.get(row['did']) != row['img_path']:
                        raise ValueError('Image mirror differs from the pinned official M-BEIR pool')
                    records.append((row['did'], row['img_path'], str(path), offset))
                    offset += 1
                connection.executemany('INSERT INTO images VALUES (?,?,?,?)', records)
            connection.commit()
            fingerprints.append(dict(path=str(path), sha256=sha(path), rows=offset))
            print(json.dumps(dict(event='indexed_images', shard=path.name, rows=offset)), flush=True)
        if connection.execute('SELECT COUNT(*) FROM images').fetchone()[0] != len(official_images):
            raise ValueError('Image mirror does not cover the full official pool')
        # Exact normalized caption + exact original image path; never derive image IDs arithmetically.
        path_to_did = {path: did for did, path in official_images.items()}
        selected, matches = {}, {}
        original_rows = 0
        for item in json_array_items(metadata):
            original_rows += 1
            text = format_caption(item.get('caption'))
            did = captions.get(text)
            relative = item.get('image_path', '').replace('\\', '/').removeprefix('./')
            full = 'mbeir_images/visualnews_images/' + relative
            image_did = path_to_did.get(full)
            if did is not None and image_did is not None:
                matches.setdefault(did, set()).add(image_did)
                if did not in selected or full < selected[did][1]:
                    selected[did] = (image_did, full)
        connection.executemany('UPDATE captions SET image_did=?,image_path=?,matching_images=? WHERE did=?',
                               [(image, path, len(matches[did]), did) for did, (image, path) in selected.items()])
        # Independently corroborate every available official test qrel.
        for filename in ['mbeir_visualnews_task3_test.jsonl']:
            with (originals / filename).open(encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line)
                    image = path_to_did[row['query_img_path']]
                    for did in row['pos_cand_list']:
                        if image not in matches.get(did, set()):
                            raise ValueError('Original caption/image join contradicts official test qrels')
        manifest = dict(status='complete', images=len(official_images), captions=len(captions),
                        matched_captions=len(selected), unmatched_captions=len(captions) - len(selected),
                        original_metadata_rows=original_rows, image_repository=IMAGE_REPO, image_revision=IMAGE_REV,
                        metadata_repository=META_REPO, metadata_revision=META_REV, metadata_sha256=sha(metadata),
                        image_shards=fingerprints, mapping='exact author-normalized caption and original image path; checked against official test qrels')
        connection.executemany('INSERT INTO provenance VALUES (?,?)', [(k, str(v)) for k, v in manifest.items() if not isinstance(v, (dict, list))])
        connection.commit()
        manifest['index_sha256'] = sha(DB)
        (CACHE / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({k: v for k, v in manifest.items() if k != 'image_shards'}), flush=True)


if __name__ == '__main__':
    main()
