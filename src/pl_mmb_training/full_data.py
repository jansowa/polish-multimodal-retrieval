"""Full production-source ingestion; every exclusion is accounted for explicitly."""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .common import ROOT, TASKS, file_sha, read_json
from .data import published_datasets
from .leakage import LeakageGuard, grouped_split, image_identity, parent_ids
from .translations import snapshot, patches_at


def build_mbeir_links(name, plan):
    """Join caption document IDs through official positive qrels, never ID arithmetic."""
    files = plan['original_files']
    def rows(key):
        info = files[key]
        if file_sha(info['path']) != info['sha256']:
            raise ValueError(f'Pinned M-BEIR source changed: {key}')
        with Path(info['path']).open() as stream:
            return [json.loads(line) for line in stream]
    captions_name = f'cand_pool/local/mbeir_{name}_task3_{"test_" if name == "mscoco" else ""}cand_pool.jsonl'
    captions = {r['did']: r['txt'] for r in rows(captions_name)}
    links = defaultdict(set)
    for row in rows(f'query/test/mbeir_{name}_task3_test.jsonl'):
        for did in row['pos_cand_list']:
            if did not in captions:
                raise ValueError('Positive document missing from caption corpus')
            links[did].add(row['query_img_path'])
    return captions, links


def prepare_full(config, blacklist, cache_root, dest):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from pl_mmb_translate.registry import DATASETS
    guard = LeakageGuard(blacklist)
    protected = published_datasets() | set(TASKS)
    vg_file = cache_root / 'visual_genome_image_data.json'
    vg_rows = read_json(vg_file)
    vg = {str(r['image_id']): r for r in vg_rows}
    for row in vg_rows:
        vg.setdefault(Path(row['url']).stem, row)
    images_dir = cache_root / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    include = config['data']['include_ids']
    include = set(include) if include is not None else None
    exclude = set(config['data']['exclude_ids'])
    records, rejected, sources, seen, requested_seen = [], [], [], set(), set()
    fingerprints = {}
    def reject(uid, reasons, **extra):
        rejected.append(dict(id=uid, reasons=reasons, **extra))
    def requested(p):
        requested_seen.add(p['id'])
        return (include is None or p['id'] in include) and p['id'] not in exclude
    def accept(record, payload=None):
        if record['id'] in seen:
            raise ValueError('Duplicate full-training pair ID')
        seen.add(record['id'])
        if not record['text_en'].strip() or not record['text_pl'].strip():
            raise ValueError('Empty positive caption')
        if payload is not None:
            identity = image_identity(payload=payload)
            path = images_dir / (identity['sha256'] + '.jpg')
            if not path.exists():
                path.write_bytes(payload)
        else:
            path = Path(record['image'])
            if str(path) not in fingerprints:
                fingerprints[str(path)] = image_identity(path)
            identity = fingerprints[str(path)]
        record.update(identity, image=str(path), target_id=identity['pixels_sha256'])
        reasons = guard.reasons(record)
        if reasons:
            reject(record['id'], reasons, parent_ids=record['parent_ids'])
        else:
            records.append(record)
    for source in config['data']['sources']:
        key = source['dataset']
        if key in protected:
            raise ValueError(f'{key} is protected by the leaderboard')
        frozen_path, snapshot_info = snapshot(source, dest / 'translation_snapshots')
        patches = patches_at(frozen_path)
        info = dict(dataset=key, translations=snapshot_info, translated_records=len(patches))
        start_retained, start_rejected = len(records), len(rejected)
        if source.get('kind') == 'mbeir':
            name = key.removeprefix('mbeir_')
            plan = read_json(ROOT / 'reports/absence_extension_plan.json')
            captions, links = build_mbeir_links(name, plan)
            revision = '023f7cc5425e15e3c93f6146f7295c28eae767dc'
            info['source_revision'] = revision
            for p in patches:
                if not requested(p):
                    continue
                if p['metadata']['source_revision'] != revision:
                    raise ValueError('M-BEIR revision mismatch')
                did = p['metadata']['corpus_id']
                en = p['source']['text']
                if captions.get(did) != en:
                    raise ValueError(f'{p["id"]}: M-BEIR caption mismatch')
                positives = sorted(links.get(did, []))
                if not positives:
                    reject(p['id'], ['no_verified_positive_image_mapping'])
                    continue
                for index, relative in enumerate(positives):
                    uid = p['id'] if len(positives) == 1 else p['id'] + f':positive:{index}'
                    parents = parent_ids(relative)
                    if name == 'visualnews':
                        parents = ['visualnews:' + relative]
                    base = dict(id=uid, translation_id=p['id'], dataset=key, source_revision=revision,
                                text_en=en, text_pl=p['translated']['text'], parent_ids=parents,
                                lineage_verified=bool(parents), source_image=relative,
                                sha256='', pixels_sha256='')
                    reasons = guard.reasons(base)
                    if reasons:
                        reject(uid, reasons, parent_ids=parents)
                        continue
                    path = (cache_root / 'coco_images' / Path(relative).name if name == 'mscoco'
                            else cache_root / relative)
                    if not path.is_file():
                        if source.get('missing_images') != 'report':
                            raise FileNotFoundError(f'Missing training image: {path}; run training/acquire_coco_images.py')
                        reject(uid, ['missing_image_asset'], source_image=relative)
                        continue
                    base['image'] = str(path)
                    accept(base)
        else:
            adapter = DATASETS[key].factory()
            by_index = {p['metadata']['row_index']: p for p in patches}
            if len(by_index) != len(patches):
                raise ValueError('Duplicate translation row index')
            positive_field = 'caption' if key == 'sugarcrepe' else 'true_caption'
            offset = 0
            info.update(source_revision=adapter.revision, shards=[])
            for shard_index, shard in enumerate(source['shards']):
                if f'{adapter.split}-{shard_index:05d}-of-' not in shard:
                    raise ValueError('Source shards must be a contiguous prefix')
                path = hf_hub_download(adapter.path, shard, revision=adapter.revision,
                                       repo_type='dataset', local_files_only=True)
                info['shards'].append(dict(path=shard, sha256=file_sha(path)))
                for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
                    for row in batch.to_pylist():
                        p = by_index.get(offset)
                        offset += 1
                        if p is None:
                            raise ValueError('Incomplete source translations')
                        if p['metadata']['source_revision'] != adapter.revision or p['source'] != {f: row[f] for f in adapter.text_fields}:
                            raise ValueError(f'{p["id"]}: translation/source mismatch')
                        if not requested(p):
                            continue
                        raw = row[adapter.image_fields[0]]
                        if isinstance(raw, list):
                            if len(raw) != 1:
                                raise ValueError('Expected one image per positive caption')
                            raw = raw[0]
                        image_name = raw.get('path') or row.get('filename') or ''
                        name = Path(image_name).stem
                        if key == 'sugarcrepe':
                            parents = parent_ids(image_name)
                            if not parents and re.fullmatch(r'\d+', name):
                                parents = ['coco:' + str(int(name))]
                            if not parents and 'image_id' in row:
                                parents = ['coco:' + str(int(row['image_id']))]
                        else:
                            parent = vg.get(name)
                            parents = ['vg:' + str(parent['image_id'])] if parent else []
                            if parent and parent.get('coco_id') is not None:
                                parents.append('coco:' + str(parent['coco_id']))
                            if parent and parent.get('flickr_id') is not None:
                                parents.append('flickr:' + str(parent['flickr_id']))
                        base = dict(id=p['id'], translation_id=p['id'], dataset=key,
                                    text_en=row[positive_field], text_pl=p['translated'][positive_field],
                                    source_revision=adapter.revision, source_image=image_name,
                                    parent_ids=parents, lineage_verified=bool(parents))
                        accept(base, payload=raw.get('bytes') or Path(raw['path']).read_bytes())
            info['source_rows_checked'] = offset
            if offset != len(patches):
                raise ValueError(f'{key}: selected shards do not cover every translated row')
        info.update(retained_before_selection=len(records) - start_retained, excluded=len(rejected) - start_rejected)
        sources.append(info)
        print(f'Full data {key}: translated={len(patches)}, usable={info["retained_before_selection"]}, excluded={info["excluded"]}', flush=True)
    if include is not None and not include <= requested_seen:
        raise ValueError('Unknown requested translation IDs')
    records.sort(key=lambda r: r['id'])
    random.Random(config['data']['selection_seed']).shuffle(records)
    safe_before_limit = len(records)
    maximum = config['data']['max_pairs']
    if maximum is not None:
        records = records[:maximum]
    train, val = grouped_split(records, config['data']['validation_fraction'], config['data']['split_seed'])
    for split, rows in (('train', train), ('validation', val)):
        for r in rows:
            r['split'] = split
    audit = dict(sources=sources, rejected=len(rejected), rejected_records=rejected,
                 reasons=dict(Counter(reason for r in rejected for reason in r['reasons'])),
                 translated_records=sum(s['translated_records'] for s in sources),
                 retained=len(records), safe_before_limit=safe_before_limit, selection_limit=maximum,
                 train=len(train), validation=len(val), unique_targets=len({r['target_id'] for r in records}),
                 blacklist_sha256=blacklist['sha256'], protected_datasets=sorted(protected),
                 post_filter_overlap=sum(bool(guard.reasons(r)) for r in records))
    if audit['post_filter_overlap']:
        raise ValueError('Leakage remained after filtering')
    return train + val, audit
