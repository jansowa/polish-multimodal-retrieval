"""Immutable VisualNews addendum, lineage/duplicate audit, and independent dev split."""
import hashlib
import io
import json
import shutil
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.fft import dctn

from . import tinyclip_adaptation as base
from .common import ROOT, digest, file_sha, read_json, write_json
from .leakage import image_identity, normalized_text

OUT = ROOT / 'training/runs/tinyclip-visualnews-v1'
ASSETS = Path('/mnt/c/Users/jsowa/benchmark-translations/visualnews-image-cache')


def linux_path(value):
    value = str(value).replace('\\', '/')
    return Path('/mnt/' + value[0].lower() + value[2:]) if len(value) > 1 and value[1] == ':' else Path(value)


def write_rows(path, rows):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


class HammingIndex:
    """Exact radius<=6 search: seven disjoint bit bands guarantee a matching band."""
    def __init__(self):
        self.buckets = defaultdict(list)
        self.hashes = {}

    @staticmethod
    def bands(value):
        shift = 0
        for i, width in enumerate((10, 9, 9, 9, 9, 9, 9)):
            yield i, (value >> shift) & ((1 << width) - 1)
            shift += width

    def add(self, key, value):
        self.hashes[key] = value
        for band in self.bands(value):
            self.buckets[band].append(key)

    def near(self, value):
        candidates = {k for band in self.bands(value) for k in self.buckets[band]}
        return [(k, (self.hashes[k] ^ value).bit_count()) for k in candidates if (self.hashes[k] ^ value).bit_count() <= 6]


def freeze():
    OUT.mkdir(exist_ok=True)
    folder = OUT / 'data'
    folder.mkdir(exist_ok=True)
    target = folder / 'pipeline_pairs_snapshot.jsonl'
    metadata = folder / 'snapshot.json'
    if metadata.exists():
        meta = read_json(metadata)
        assert file_sha(target) == meta['pairs_sha256']
        assert file_sha(folder / 'pipeline_manifest.json') == meta['manifest_sha256']
        return meta
    source = ROOT / 'reports/visualnews_pipeline/v1'
    # The publisher atomically replaces final_pairs; never access the live SQLite/WAL.
    shutil.copyfile(source / 'final_pairs.jsonl', target)
    shutil.copyfile(source / 'manifest.json', folder / 'pipeline_manifest.json')
    rows = base.records(target)
    manifest = read_json(folder / 'pipeline_manifest.json')
    assert len({r['id'] for r in rows}) == len(rows)
    assert all(r['pipeline_fingerprint'] == manifest['fingerprint'] for r in rows)
    value = dict(created_at=datetime.now(timezone.utc).isoformat(), source=str(source / 'final_pairs.jsonl'),
                 rows=len(rows), pairs_sha256=file_sha(target), manifest_sha256=file_sha(folder / 'pipeline_manifest.json'),
                 pipeline_fingerprint=manifest['fingerprint'], live_database_modified=False)
    write_json(metadata, value)
    print('FROZEN', json.dumps(value), flush=True)
    return value


def phash_image(im):
    small = ImageOps.exif_transpose(im).convert('L').resize((32, 32), Image.Resampling.LANCZOS)
    coefficients = dctn(np.asarray(small, dtype=float), type=2)[:8, :8]
    bits = coefficients > np.median(coefficients)
    return int.from_bytes(np.packbits(bits.reshape(-1)).tobytes(), 'big')


def join_metadata(records, metadata_path):
    # Reuse the original streaming parser; importing this module performs no downloads.
    import sys
    sys.path.insert(0, str(ROOT / 'experiments'))
    from visualnews_image_assets import json_array_items
    by_path = {r['source_image']: r for r in records}
    joined = set()
    for row in json_array_items(metadata_path):
        path = 'mbeir_images/visualnews_images/' + row['image_path'].replace('\\', '/').removeprefix('./')
        if path in by_path:
            record = by_path[path]
            article = row.get('article_path')
            if article:
                record['article_ids'].add(str(article).removeprefix('./'))
            record['topics'].add(row.get('topic', 'unknown'))
            joined.add(path)
    assert joined == set(by_path), 'Selected image missing from original VisualNews metadata'
    for r in records:
        r['article_ids'] = sorted(r['article_ids'])
        r['topics'] = sorted(r['topics'])


def extract(data, snapshot):
    folder = OUT / 'data'
    complete = folder / 'extraction.json'
    if complete.exists():
        meta = read_json(complete)
        for path, sha in meta['files'].items():
            assert file_sha(folder / path) == sha
        return base.records(folder / 'candidates.jsonl'), meta
    from transformers import AutoImageProcessor
    import pyarrow.parquet as pq
    asset_meta = read_json(ASSETS / 'manifest.json')
    assert asset_meta['status'] == 'complete'
    assert file_sha(ASSETS / 'assets.sqlite3') == asset_meta['index_sha256']
    assert file_sha(ASSETS / 'original-data.json') == asset_meta['metadata_sha256']
    db = sqlite3.connect((ASSETS / 'assets.sqlite3').as_uri() + '?mode=ro')
    patches = base.records(folder / 'pipeline_pairs_snapshot.jsonl')
    records = []
    for i, patch in enumerate(patches):
        did = patch['id'].removeprefix('MBeirvisualnews:')
        caption = db.execute('SELECT text,image_did,image_path FROM captions WHERE did=?', (did,)).fetchone()
        assert caption == (patch['original_en'], patch['image_did'], patch['source_image'])
        asset = db.execute('SELECT path,shard,row_index FROM images WHERE did=?', (patch['image_did'],)).fetchone()
        assert asset[0] == patch['source_image']
        assert patch['text_en'].strip() and patch['text_pl'].strip() and patch['route'].startswith('ready')
        grounding = patch['stages'].get('recheck', patch['stages']['judge'])['output']
        assert grounding['visual_grounding'] >= 3 and not grounding['image_caption_mismatch']
        records.append(dict(id=patch['id'], translation_id=patch['id'], dataset='visualnews', source=patch['source_image'].split('/')[2],
             text_en=patch['text_en'], text_pl=patch['text_pl'], original_en=patch['original_en'],
             source_image=patch['source_image'], image_did=patch['image_did'],
             image='visualnews-cache://' + patch['image_did'], cache_index=i,
             shard=str(linux_path(asset[1])), shard_row=asset[2], route=patch['route'],
             article_ids=set(), topics=set(), lineage_verified=True,
             parent_ids=['visualnews-image:' + patch['image_did'], 'visualnews-path:' + patch['source_image']]))
    db.close()
    join_metadata(records, ASSETS / 'original-data.json')
    print('Original caption/image/article joins verified', len(records), flush=True)
    processor = AutoImageProcessor.from_pretrained(data.lock['model'], revision=data.lock['revision'], local_files_only=True, use_fast=False)
    mean = torch.tensor(processor.image_mean)[None, :, None, None]
    std = torch.tensor(processor.image_std)[None, :, None, None]
    required = len(records)*3*224*224
    if shutil.disk_usage(OUT).free < required + 2_000_000_000:
        raise RuntimeError('Insufficient disk space for native RGB cache and checkpoint reserve')
    pixels = np.lib.format.open_memmap(folder / 'native_rgb.npy', mode='w+', dtype='uint8', shape=(len(records), 3, 224, 224))
    selected = defaultdict(dict)
    for r in records:
        selected[r['shard']][r['shard_row']] = r
    shard_meta = {str(linux_path(s['path'])): s for s in asset_meta['image_shards']}
    verified_shards = {}

    def decode(item):
        record, raw = item
        record.update(image_identity(payload=raw))
        record['target_id'] = record['pixels_sha256']
        with Image.open(io.BytesIO(raw)) as im:
            record['phash'] = f'{phash_image(im):016x}'
            # Match original CLIP RGB conversion (the exact duplicate guard handles EXIF independently).
            rgb = im.convert('RGB')
        return record, rgb

    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for shard, mapping in sorted(selected.items()):
            expected = shard_meta[shard]
            assert file_sha(shard) == expected['sha256']
            verified_shards[shard] = expected['sha256']
            parquet = pq.ParquetFile(shard)
            offset = 0
            for group in range(parquet.num_row_groups):
                length = parquet.metadata.row_group(group).num_rows
                wanted = sorted(i for i in mapping if offset <= i < offset+length)
                if wanted:
                    table = parquet.read_row_group(group, columns=['did', 'img_path', 'img'])
                    for start in range(0, len(wanted), 32):
                        items = []
                        for index in wanted[start:start+32]:
                            record = mapping[index]
                            row = table.slice(index-offset, 1).to_pylist()[0]
                            assert row['did'] == record['image_did'] and row['img_path'] == record['source_image']
                            payload = row['img']['bytes']
                            assert payload is not None
                            items.append((record, payload))
                        decoded = list(pool.map(decode, items))
                        try:
                            native = processor(images=[im for _, im in decoded], return_tensors='pt')['pixel_values']
                            rgb = ((native*std+mean)*255).round().clamp(0,255).byte()
                            assert float((((rgb.float()/255-mean)/std)-native).abs().max()) < 2e-6
                            for j, (record, _) in enumerate(decoded):
                                pixels[record['cache_index']] = rgb[j].numpy()
                        finally:
                            for _, im in decoded:
                                im.close()
                        done += len(items)
                    del table
                offset += length
            pixels.flush()
            print(f'VN RGB {done}/{len(records)}', flush=True)
    assert done == len(records)
    write_rows(folder / 'candidates.jsonl', records)
    value = dict(snapshot=snapshot, assets_index_sha256=asset_meta['index_sha256'], metadata_sha256=asset_meta['metadata_sha256'],
                 verified_shards=verified_shards, files={p: file_sha(folder / p) for p in ('native_rgb.npy','candidates.jsonl')})
    write_json(complete, value)
    return records, value


def audit_split(data, records, extraction):
    folder = OUT / 'data'
    complete = folder / 'audit.json'
    if complete.exists():
        value = read_json(complete)
        for name, sha in value['files'].items():
            assert file_sha(folder / name) == sha
        return value
    blacklists = [ROOT / 'training/runs/e1-full/blacklist.json', ROOT / 'training/runs/xm3600-pl/blacklist.json']
    xm = read_json(ROOT / 'reports/xm3600_pl_preparation.json')
    assert file_sha(blacklists[1]) == xm['blacklist_sha256']
    protected_images, protected_texts = {}, set()
    for path in blacklists:
        value = read_json(path)
        protected_images.update(value['images'])
        protected_texts.update(value['texts'])
    for r in data.train + data.val:
        protected_images[r['image']] = dict(sha256=r['sha256'], pixels_sha256=r['pixels_sha256'])
    protected_texts.update(normalized_text(r[f]) for r in data.val for f in ('text_en','text_pl'))
    train_texts = {normalized_text(r[f]) for r in data.train for f in ('text_en','text_pl')}
    exact = {h for v in protected_images.values() for h in (v['sha256'], v['pixels_sha256'])}
    old_phash = read_json(ROOT / 'training/runs/clean-v1/perceptual_hashes.json')['hashes']
    reference = HammingIndex()
    for path, identity in protected_images.items():
        if path in old_phash:
            h = int(old_phash[path],16)
        else:
            assert file_sha(path) == identity['sha256']
            with Image.open(path) as im:
                h = phash_image(im)
        reference.add(path, h)
    print('Protected image pHash index', len(reference.hashes), flush=True)
    parents = list(range(len(records)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(i,j):
        parents[find(i)] = find(j)

    seen, vn_near, rejected, near_examples = {}, HammingIndex(), {}, []
    from .series_holdout import workspace_guard
    wiki_guard=workspace_guard()
    for i, r in enumerate(records):
        reasons = wiki_guard.reasons(r) if wiki_guard else []
        if r['sha256'] in exact or r['pixels_sha256'] in exact:
            reasons.append('protected_or_base_image_content')
        if any(normalized_text(r[f]) in protected_texts for f in ('text_en','text_pl')):
            reasons.append('protected_text')
        h = int(r['phash'],16)
        near = reference.near(h)
        if near:
            reasons.append('conservative_phash_le6_to_protected_or_base')
            near_examples.append(dict(id=r['id'], matches=sorted(near,key=lambda p:p[1])[:3]))
        if reasons:
            rejected[i] = reasons
        keys = ['pixels:'+r['pixels_sha256'], 'original:'+normalized_text(r['original_en'])]
        keys += ['article:'+s for s in r['article_ids']]
        keys += ['en:'+normalized_text(r['text_en']), 'pl:'+normalized_text(r['text_pl'])]
        for key in keys:
            if key in seen:
                union(i, seen[key])
            seen[key] = i
        for j, _ in vn_near.near(h):
            union(i,j)
        vn_near.add(i,h)
    groups = defaultdict(list)
    for i in range(len(records)):
        groups[find(i)].append(i)
    bad_groups = {find(i) for i in rejected}
    kept_groups = [v for k,v in groups.items() if k not in bad_groups]
    train_groups, dev_groups = [], []
    for group in sorted(kept_groups, key=lambda g:digest([1701, sorted(records[i]['id'] for i in g)])):
        targets = {records[i]['target_id'] for i in group}
        forced = len(targets)>100 or any(normalized_text(records[i][f]) in train_texts for i in group for f in ('text_en','text_pl'))
        if not forced and sum(len({records[j]['target_id'] for j in g}) for g in dev_groups) < 2000:
            dev_groups.append(group)
        else:
            train_groups.append(group)
    train = [dict(records[i], split='train', component=digest(sorted(records[j]['id'] for j in group))) for group in train_groups for i in group]
    dev = [dict(records[i], split='vn_dev', component=digest(sorted(records[j]['id'] for j in group))) for group in dev_groups for i in group]
    assert {r['target_id'] for r in train}.isdisjoint(r['target_id'] for r in dev)
    assert {p for r in train for p in r['article_ids']}.isdisjoint(p for r in dev for p in r['article_ids'])
    assert {normalized_text(r[f]) for r in train+data.train for f in ('text_en','text_pl')}.isdisjoint(normalized_text(r[f]) for r in dev for f in ('text_en','text_pl'))
    # Image-balanced nested subset, proportional to the available source mixture.
    image_rows = defaultdict(list)
    for r in train:
        image_rows[r['target_id']].append(r)
    by_source = defaultdict(list)
    for key, rr in image_rows.items():
        by_source[rr[0]['source']].append(key)
    selected = []
    nsmall = min(10000, len(image_rows))
    for source, keys in sorted(by_source.items()):
        number = round(nsmall*len(keys)/len(image_rows))
        selected.extend(sorted(keys,key=lambda key:digest([42,key]))[:number])
    selected = sorted(set(selected),key=lambda key:digest([42,key]))
    if len(selected)<nsmall:
        selected += sorted(set(image_rows)-set(selected),key=lambda key:digest([42,key]))[:nsmall-len(selected)]
    selected = set(selected[:nsmall])
    small = [r for r in train if r['target_id'] in selected]
    filenames = [('vn_train.jsonl',train),('vn_10k.jsonl',small),('vn_dev.jsonl',dev)]
    for name, rows in filenames:
        write_rows(folder/name,sorted(rows,key=lambda r:r['id']))
    write_json(folder/'rejections.json',dict(direct={records[i]['id']:why for i,why in rejected.items()},
        propagated_component_rows=sum(len(groups[g]) for g in bad_groups), near_examples=near_examples))
    def counts(rows):
        return dict(rows=len(rows), images=len({r['target_id'] for r in rows}), sources=dict(Counter(r['source'] for r in rows)),
                    routes=dict(Counter(r['route'] for r in rows)), components=len({r['component'] for r in rows}))
    value = dict(status='passed', snapshot=extraction['snapshot'], base_inputs=data.hashes,
         blacklists={str(p):file_sha(p) for p in blacklists}, extraction_sha256=file_sha(folder/'extraction.json'),
         candidates=len(records), candidate_unique_pixels=len({r['target_id'] for r in records}),
         rejected_direct=len(rejected), rejected_component_rows=sum(len(groups[g]) for g in bad_groups),
         largest_component_images=max(len({records[i]['target_id'] for i in g}) for g in groups.values()),
         train=counts(train), small=counts(small), dev=counts(dev),
         grouping='canonical pixels, all original article paths, original and final captions, exhaustive pHash Hamming<=6 via guaranteed 7-band index',
         filtering='entire connected component excluded on exact image, normalized text, or pHash<=6 match to protected/base images; conservative false exclusions possible',
         limitation='Known article paths and pHash do not guarantee different news events or detect every crop/transformation; VN captions are model-generated/translated, not human gold.',
         xm3600_blacklist_used_only_for_overlap=True, xm3600_scores_computed=False, test_used=False,
         files={name:file_sha(folder/name) for name,_ in filenames})
    data.verify()
    write_json(complete,value)
    write_json(ROOT/'reports/tinyclip_visualnews_data_audit.json',value)
    print('AUDIT',json.dumps(value,ensure_ascii=False),flush=True)
    return value


def prepare():
    OUT.mkdir(exist_ok=True)
    data = base.Inputs(OUT)
    snapshot = freeze()
    records, extraction = extract(data,snapshot)
    return audit_split(data,records,extraction)

