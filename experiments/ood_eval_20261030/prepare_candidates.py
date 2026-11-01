"""CPU source inspection and bounded original-caption acquisition, no models."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
from pathlib import Path
import ssl
import urllib.error
import urllib.request
import zipfile

from prepare import ROOT, OUT, load_common, read, write, fetch, status
from acquire import ASSETS, download


def head(url):
    try:
        context = ssl.create_default_context(cafile=str(ROOT / 'combined-ca.pem'))
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30, context=context) as response:
            return dict(url=url, status=response.status, headers={k: v for k, v in response.headers.items()
                if k.lower() in ('content-length', 'content-type', 'accept-ranges', 'etag', 'last-modified')})
    except (OSError, urllib.error.URLError) as error:
        return dict(url=url, error=str(error))


def main():
    load_common()
    urls = [
        'https://storage.googleapis.com/docci/data/docci_images.tar.gz',
        'https://storage.googleapis.com/docci/images/test_00000.jpg',
        'https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip',
        'https://vizwiz.cs.colorado.edu/VizWiz_final/caption/annotations.zip',
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        sizing = list(pool.map(head, urls))
    write(OUT / 'source_capacity_checks.json', dict(sources=sizing, workers=2))
    # The original descriptions are 10.5 MB; never download the 7.1 GB image archive.
    docci = download('https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines',
                     'docci/descriptions.jsonl', 16 * 1024**2)
    annotation_size = next((int(h.get('headers', {}).get('Content-Length', 0)) for h in sizing if h['url'].endswith('annotations.zip')), 0)
    if annotation_size > 32 * 1024**2:
        raise ValueError('VizWiz annotations exceed preregistered 32 MiB bound')
    vizwiz = download(urls[-1], 'vizwiz/annotations.zip', 32 * 1024**2)
    captions = [json.loads(line) for line in (ASSETS / 'docci/descriptions.jsonl').open()]
    counts = Counter(row['split'] for row in captions)
    with zipfile.ZipFile(ASSETS / 'vizwiz/annotations.zip') as archive:
        members = archive.namelist()
        validations = [name for name in members if Path(name).name in ('val.json', 'validation.json')]
        if len(validations) != 1:
            raise ValueError('Ambiguous VizWiz validation annotations: ' + repr(members))
        info = archive.getinfo(validations[0])
        if info.file_size > 32 * 1024**2:
            raise ValueError('Oversized uncompressed annotations')
        validation = json.loads(archive.read(validations[0]))
    annotations = validation['annotations']
    if not all(type(row.get('is_rejected')) is bool and type(row.get('is_precanned')) is bool for row in annotations):
        raise ValueError('Missing official boolean rejection flags')
    accepted = [row for row in annotations if not row['is_rejected'] and not row['is_precanned']]
    result = dict(status='candidate_metadata_prepared_not_audited', docci_input=docci, vizwiz_input=vizwiz,
        docci_split_counts=dict(counts),
        docci_test_word_counts=dict(min=min(len(r['description'].split()) for r in captions if r['split']=='test'),
            max=max(len(r['description'].split()) for r in captions if r['split']=='test')),
        vizwiz_archive_members=members, vizwiz_validation_images=len(validation['images']),
        vizwiz_validation_captions=len(annotations), vizwiz_accepted_captions=len(accepted),
        vizwiz_images_with_accepted_captions=len({r['image_id'] for r in accepted}),
        vizwiz_rejected=sum(r['is_rejected'] for r in annotations),
        vizwiz_precanned=sum(r['is_precanned'] for r in annotations),
        translations=dict(status='not_started', new_manual_annotations=0, CPU_only_until_A_done=True),
        no_full_image_archive_downloaded=True)
    write(OUT / 'candidate_caption_inventory.json', result)
    # The existing repo virtualenv provides pyarrow without installing/modifying environments.
    try:
        import pyarrow.parquet as pq
        path = ASSETS / 'flickr30k/test.parquet'
        parquet = pq.ParquetFile(path)
        table = parquet.read()
        rows = table.to_pylist()
        image_paths = [row['image'].get('path') for row in rows]
        result = dict(status='candidate_only', rows=len(rows), columns=table.column_names,
            schema=str(parquet.schema_arrow), original_image_paths_present=sum(bool(p) for p in image_paths),
            image_paths_sample=image_paths[:5], captions_en=sum(len(row['captions_en']) for row in rows),
            captions_pl=sum(len(row['captions_pl']) for row in rows),
            aligned_caption_lengths=all(len(row['captions_en']) == len(row['captions_pl']) for row in rows),
            embedded_image_bytes=sum(len(row['image']['bytes']) for row in rows),
            repack_split='test', official_flickr30k1k_claim=False)
        write(OUT / 'flickr30k_parquet_inventory.json', result)
        print('FLICKR_INVENTORY', json.dumps(result))
    except ImportError as error:
        write(OUT / 'flickr30k_parquet_inventory.json', dict(status='requires_existing_pyarrow_environment', reason=str(error)))
    print('CAPTION_INVENTORY', json.dumps(result))
    status()


if __name__ == '__main__':
    main()
