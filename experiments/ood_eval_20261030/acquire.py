"""Reserve-accounted acquisition of explicitly pinned B-owned CPU inputs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import ssl
import urllib.request

from prepare import ROOT, OUT, COORD, load_common, local_path, committed_bytes, read, sha, write

ASSETS = local_path('C:/Users/jsowa/benchmark-translations/data-ood-20261030/B/')


def download(url, relative, expected_bytes, expected_sha256=None):
    common = load_common()
    target = ASSETS / relative
    if not target.resolve().is_relative_to(ASSETS.resolve()):
        raise ValueError('Download destination escapes B')
    if target.exists():
        digest = sha(target)
        if target.stat().st_size > expected_bytes or (expected_sha256 and digest != expected_sha256):
            raise ValueError('Existing asset does not match pinned source')
        return dict(path=common.windows_path(target), sha256=digest, bytes=target.stat().st_size, url=url)
    key = 'acquire-' + hashlib.sha256(relative.encode()).hexdigest()[:16]
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.part')
    initial_size = partial.stat().st_size if partial.exists() else 0
    if initial_size > expected_bytes or (initial_size and not expected_sha256):
        raise ValueError('Cannot safely resume unpinned or oversized partial')
    baseline = committed_bytes()
    def account():
        # Do not recursively stat thousands of images after every network block.
        # Keep another B writer's last committed accounting if it is newer.
        ledger = read(COORD / 'DISK_RESERVATIONS.json') if (COORD / 'DISK_RESERVATIONS.json').exists() else {}
        return max(baseline + size - initial_size, ledger.get('committed', {}).get('B', 0))
    size = initial_size
    common.reserve('B', key, {'C': expected_bytes-size}, committed_bytes=baseline)
    context = ssl.create_default_context(cafile=str(ROOT / 'combined-ca.pem'))
    headers = {'User-Agent': 'OOD-B-source-audit/1.0', 'Accept-Encoding': 'identity'}
    if initial_size:
        headers['Range'] = f'bytes={initial_size}-'
    request = urllib.request.Request(url, headers=headers)
    digest = hashlib.sha256()
    if initial_size:
        with partial.open('rb') as stream:
            for data in iter(lambda: stream.read(4*1024**2),b''):
                digest.update(data)
    if initial_size == expected_bytes and expected_sha256:
        if digest.hexdigest() != expected_sha256:
            raise ValueError('Complete partial has wrong SHA; preserve for inspection')
        partial.rename(target)
        common.reserve('B', key, {'C': 0}, committed_bytes=account())
        return dict(url=url,path=common.windows_path(target),sha256=digest.hexdigest(),bytes=size)
    with urllib.request.urlopen(request, timeout=45, context=context) as response, partial.open('ab' if initial_size else 'xb') as stream:
        if initial_size and (response.status != 206 or response.headers.get('Content-Range') != f'bytes {initial_size}-{expected_bytes-1}/{expected_bytes}'):
            raise ValueError('Pinned range resume not honored; preserve partial')
        length = response.headers.get('Content-Length')
        if length and int(length) > expected_bytes-initial_size:
            raise ValueError('Download exceeds preregistered byte bound')
        while True:
            data = response.read(min(1024**2, expected_bytes - size + 1))
            if not data:
                break
            size += len(data)
            if size > expected_bytes:
                raise ValueError('Download exceeds preregistered byte bound')
            # Preserve the reservation for unwritten bytes; no duplicate debit for already-written bytes.
            stream.write(data)
            stream.flush()
            digest.update(data)
            if (size-initial_size) % (8*1024**2) == 0:
                common.reserve('B', key, {'C': expected_bytes - size}, committed_bytes=account())
    if expected_sha256 and digest.hexdigest() != expected_sha256:
        raise ValueError('Source SHA mismatch; keep partial for inspection')
    partial.rename(target)
    common.reserve('B', key, {'C': 0}, committed_bytes=account())
    return dict(url=url, path=common.windows_path(target), sha256=digest.hexdigest(), bytes=size)


def main():
    from prepare import fetch, status
    sources = read(OUT / 'source_metadata.json')['sources']
    docci_api = json.loads(sources['docci'][0]['text'])
    revision = docci_api['sha']
    urls = [
        f'https://huggingface.co/datasets/google/docci/raw/{revision}/docci.py',
        'https://api.github.com/repos/Delphboy/karpathy-splits/contents',
        'https://api.github.com/repos/karpathy/neuraltalk2/contents/data',
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        metadata = list(pool.map(fetch, urls))
    write(OUT / 'additional_metadata.json', dict(sources=metadata, workers=2))
    flickr = sources['flickr30k']
    rev = json.loads(flickr[0]['text'])['sha']
    files = json.loads(next(item['text'] for item in flickr if '/tree/' in item['url'] and item['url'].endswith('/data')))
    test = next(item for item in files if item['path'] == 'data/test-00000-of-00001.parquet')
    decision = dict(status='candidate_acquisition_only_not_approved_benchmark', source='aplominski/flickr30k-multilingual',
        revision=rev, split='test', expected_rows=1000, file=test,
        original_license_evidence='https://shannon.cs.illinois.edu/DenotationGraph/',
        photos='Flickr copyright remains with owners; original distributor permits non-commercial research/education subject to Flickr terms; do not label all photographs CC-BY.',
        repack_license='CC-BY-4.0 claim applies to repack; not independent proof of photo rights',
        validation_not_downloaded=True, train_not_downloaded=True,
        limitations=['Verify embedded original image filenames and Karpathy split before dataset decision',
                     'No frozen gallery until complete A universe audit; not official Flickr30k1K'],
        prerequisites=dict(COMMON_READY_sha256=sha(COORD / 'COMMON_READY.json')))
    write(OUT / 'flickr30k_acquisition_decision.json', decision)
    result = download(f'https://huggingface.co/datasets/aplominski/flickr30k-multilingual/resolve/{rev}/{test["path"]}',
                      'flickr30k/test.parquet', test['size'], test['lfs']['oid'])
    write(OUT / 'flickr30k_download.json', result)
    print(json.dumps(dict(flickr30k=result, metadata_status=[r['status'] for r in metadata])))
    status()


if __name__ == '__main__':
    main()
