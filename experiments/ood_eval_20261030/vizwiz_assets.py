"""Acquire original VizWiz validation ZIP members with HTTP ranges, never full ZIP."""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '1'
from concurrent.futures import ThreadPoolExecutor
import hashlib
import argparse
import io
import json
from pathlib import Path
import ssl
import struct
import threading
import time
import urllib.request
import zipfile
import zlib

from prepare import ROOT, OUT, read, write, load_common, committed_bytes, sha, status
from acquire import ASSETS


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size, etag):
        self.url, self.size, self.etag, self.position = url, size, etag, 0
        self.context = ssl.create_default_context(cafile=str(ROOT / 'combined-ca.pem'))
        self.sessions = threading.local()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if self.position < 0:
            raise ValueError('Negative ZIP seek')
        return self.position

    def read(self, size=-1):
        size = min(size if size >= 0 else self.size, self.size - self.position)
        result = self.range(self.position, size)
        self.position += len(result)
        return result

    def range(self, start, length):
        if length == 0:
            return b''
        if not (0 <= start < self.size) or length > 20 * 1024**2:
            raise ValueError('Range outside bounded archive access')
        end = min(self.size - 1, start + length - 1)
        headers = {
            'Range': f'bytes={start}-{end}', 'If-Match': self.etag,
            'User-Agent': 'OOD-B-original-member-audit/1.0', 'Accept-Encoding': 'identity'}
        import requests
        if not hasattr(self.sessions, 'session'):
            self.sessions.session = requests.Session()
        for attempt in range(3):
            try:
                with self.sessions.session.get(self.url, headers=headers, stream=True, timeout=(15, 45),
                        verify=str(ROOT / 'combined-ca.pem')) as response:
                    if response.status_code in (429, 500, 502, 503, 504):
                        response.raise_for_status()
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                        raise ValueError('Server did not honor exact byte range; no full archive read')
                    if response.headers.get('ETag') != self.etag:
                        raise ValueError('Archive identity changed')
                    payload = response.raw.read(length + 1)
                    if len(payload) != end - start + 1:
                        raise ValueError('Incomplete/oversized range')
                    return payload
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    def member(self, info):
        # ZIP local header, then exact compressed member. No adjacent originals.
        # Usually a single persistent HTTP request per member. Read only a small
        # header allowance beyond the compressed member; discard adjacent bytes.
        prefix = self.range(info.header_offset, min(30 + 4096 + info.compress_size, self.size-info.header_offset))
        fields = struct.unpack('<IHHHHHIIIHH', prefix[:30])
        if fields[0] != 0x04034b50 or fields[2] & 1:
            raise ValueError('Unsupported ZIP local header')
        local_offset = 30 + fields[-2] + fields[-1]
        payload = prefix[local_offset:local_offset + info.compress_size]
        if len(payload) < info.compress_size:
            payload += self.range(info.header_offset + local_offset + len(payload), info.compress_size-len(payload))
        if info.file_size > 16 * 1024**2:
            raise ValueError('Oversized image member')
        if info.compress_type == zipfile.ZIP_DEFLATED:
            decoder = zlib.decompressobj(-15)
            raw = decoder.decompress(payload, info.file_size + 1)
            if not decoder.eof or decoder.unconsumed_tail:
                raise ValueError('Invalid bounded compressed image')
        elif info.compress_type == zipfile.ZIP_STORED:
            raw = payload
        else:
            raise ValueError('Unsupported ZIP compression')
        if len(raw) != info.file_size or zlib.crc32(raw) & 0xffffffff != info.CRC:
            raise ValueError('Original member size/CRC mismatch')
        return raw


def fingerprints(raw):
    # Same convention as the historical audited A universe, one image decode.
    import numpy as np
    from PIL import Image, ImageOps
    from scipy.fft import dctn
    with Image.open(io.BytesIO(raw)) as image:
        oriented = ImageOps.exif_transpose(image)
        rgb = oriented.convert('RGB')
        pixel = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        gray = oriented.convert('L').resize((32, 32), Image.Resampling.LANCZOS)
        coeff = dctn(np.asarray(gray, dtype=float), type=2)[:8, :8]
        value = int.from_bytes(np.packbits((coeff > np.median(coeff)).reshape(-1)).tobytes(), 'big')
        width, height = rgb.size
    return dict(sha256=hashlib.sha256(raw).hexdigest(), pixels_sha256=pixel, phash=f'{value:016x}',
                width=width, height=height)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    common = load_common()
    checks = read(OUT / 'source_capacity_checks.json')['sources']
    source = next(r for r in checks if r['url'].endswith('/images/val.zip'))
    headers = {k.lower(): v for k, v in source['headers'].items()}
    remote = RemoteZip(source['url'], int(headers['content-length']), headers['etag'])
    with zipfile.ZipFile(remote) as archive:
        members = archive.infolist()
    with zipfile.ZipFile(ASSETS / 'vizwiz/annotations.zip') as annotations:
        validation = json.loads(annotations.read('annotations/val.json'))
    accepted = {}
    for caption in validation['annotations']:
        if not caption['is_rejected'] and not caption['is_precanned']:
            accepted.setdefault(caption['image_id'], []).append(caption)
    images = [image for image in validation['images'] if image['id'] in accepted]
    # Candidate pool; final component-aware sampling occurs after A-universe audit.
    images.sort(key=lambda image: hashlib.sha256(('20261023:vizwiz:' + image['file_name']).encode()).hexdigest())
    selected = images[:1250]
    by_name = {Path(item.filename).name: item for item in members if not item.is_dir()}
    infos = [by_name[image['file_name']] for image in selected]
    if len({i.filename for i in infos}) != len(selected):
        raise ValueError('Ambiguous ZIP image mapping')
    expected = sum(info.file_size for info in infos)
    if expected > 1024**3:
        raise ValueError('Candidate image pool exceeds registered 1 GiB bound')
    decision = dict(status='candidate_acquisition_not_frozen', source='VizWiz-Captions', split='validation',
        original_archive=dict(url=remote.url, bytes=remote.size, etag=remote.etag),
        archive_downloaded=False, byte_ranges_only=True, candidates=len(selected),
        expected_original_bytes=expected, pool_seed=20261023, pool_selection='sha256(seed:vizwiz:original_filename)',
        final_selection='After verified A universe, whole exact/near/text components; target 1000, minimum 500',
        captions='All original captions except official is_rejected or is_precanned flags',
        license='CC-BY-4.0, original VizWiz-Captions website', workers=args.workers,
        original_members=[dict(name=i.filename, size=i.file_size, compressed_bytes=i.compress_size, crc32=i.CRC,
                               header_offset=i.header_offset) for i in infos])
    write(OUT / 'vizwiz_acquisition_decision.json', decision)
    folder = ASSETS / 'vizwiz/images'
    folder.mkdir(parents=True, exist_ok=True)
    tasks = [(image, info) for image, info in zip(selected, infos) if not (folder / image['file_name']).exists()]
    remaining = sum(info.file_size for _, info in tasks) + 8 * 1024**2
    common.reserve('B', 'vizwiz-originals', {'C': remaining}, committed_bytes=committed_bytes())

    def acquire(task):
        image, info = task
        raw = remote.member(info)
        return image, info, raw, fingerprints(raw)

    # map eagerly queues jobs but each returns <=16 MiB; bounded submission below
    # prevents completed image payloads accumulating behind a slow early request.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, len(tasks), args.workers):
            for image, info, raw, identity in pool.map(acquire, tasks[start:start + args.workers]):
                target = folder / image['file_name']
                with target.open('xb') as stream:
                    stream.write(raw)
                common.atomic_json(target.with_suffix('.json'), dict(identity, bytes=len(raw),
                    image_id=image['id'], filename=image['file_name'], original_zip_member=info.filename,
                    image=common.windows_path(target), parent_ids=['vizwiz:' + image['file_name']],
                    captions=accepted[image['id']], source_archive_etag=remote.etag))
                remaining -= len(raw)
                common.reserve('B', 'vizwiz-originals', {'C': max(0, remaining)}, committed_bytes=committed_bytes())
            if start % 50 == 0:
                print(f'VIZWIZ_ORIGINALS {start + min(args.workers, len(tasks)-start)}/{len(tasks)}', flush=True)
    inventory = []
    for image in selected:
        target = folder / image['file_name']
        metadata = read(target.with_suffix('.json'))
        if target.stat().st_size != metadata['bytes'] or sha(target) != metadata['sha256']:
            raise ValueError('Saved original identity mismatch')
        inventory.append(metadata)
    inventory_path = ASSETS / 'vizwiz/candidate_images.json'
    common.atomic_json(inventory_path, inventory)
    common.reserve('B', 'vizwiz-originals', {'C': 0}, committed_bytes=committed_bytes())
    write(OUT / 'vizwiz_original_inventory.json', dict(status='originals_prepared_pending_A_audit_and_MT',
        images=len(inventory), captions=sum(len(r['captions']) for r in inventory),
        bytes=sum(r['bytes'] for r in inventory),
        artifact=dict(path=common.windows_path(inventory_path), sha256=sha(inventory_path), bytes=inventory_path.stat().st_size),
        algorithm='phash-dct32-low8-median-including-dc-exif-v1', no_thumbnails=True))
    status()
    print('VIZWIZ_ORIGINALS_COMPLETE', len(inventory), flush=True)


if __name__ == '__main__':
    main()
