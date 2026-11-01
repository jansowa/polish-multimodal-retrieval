import json
import random
from collections import Counter
from pathlib import Path

from pl_mmb_benchmark.data import load, translation_path, clean
from pl_mmb_benchmark.protocol import PREFIXES, PROTOCOL
from .common import ROOT, TASKS, file_sha, read_json
from .leakage import LeakageGuard, image_identity, grouped_split


def load_evaluation():
    extra = published_datasets() - set(TASKS)
    if extra:
        raise ValueError(f'Extend the E0 blacklist adapter before using new leaderboard datasets: {sorted(extra)}')
    datasets, provenance, english = {}, {}, {}
    for key in TASKS:
        datasets[key], sha = load(key)
        patches = [json.loads(line) for line in translation_path(key).read_text(encoding='utf-8').splitlines() if line.strip()]
        prefix = PREFIXES[key][0]
        for p in patches:
            text = clean(p['source']['qry_text'])
            if not text.startswith(prefix):
                raise ValueError('Unrecognized English instruction')
            english[p['id']] = text[len(prefix):].strip()
        provenance[key] = dict(translations_sha256=sha, examples=len(datasets[key]))
    return datasets, provenance, english


def published_datasets():
    validation = ROOT / 'benchmark/results/validation.json'
    invalid = {r['run_id'] for r in read_json(validation).get('excluded', [])} if validation.exists() else set()
    results = [read_json(p) for p in (ROOT / 'benchmark/results').glob('*/result.json')]
    results = [r for r in results if r.get('run_id') not in invalid]
    return {r['dataset'] for r in results if r.get('protocol') == PROTOCOL and r.get('full_dataset')}


def prepare_aro(config, blacklist, cache_root):
    """Only explicitly selected contiguous leading shards; never guess row offsets."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from pl_mmb_translate.registry import DATASETS
    selected, rejected, provenance, used_ids = [], [], [], set()
    vg_file = cache_root / 'visual_genome_image_data.json'
    vg_rows = read_json(vg_file)
    vg = {str(r['image_id']): r for r in vg_rows}
    # ARO filenames use VG image IDs; also index URL basenames where different.
    for r in vg_rows:
        vg.setdefault(Path(r['url']).stem, r)
    guard = LeakageGuard(blacklist)
    protected = published_datasets() | set(TASKS)
    image_dir = cache_root / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    include = config['data']['include_ids']
    include = set(include) if include is not None else None
    exclude = set(config['data']['exclude_ids'])
    for source in config['data']['sources']:
        key = source['dataset']
        if key in protected:
            raise ValueError(f'{key} is an evaluation dataset in the current leaderboard')
        if key not in ('aro_visual_relation', 'aro_visual_attribution'):
            raise ValueError(f'No verified positive-pair training adapter for {key}')
        adapter = DATASETS[key].factory()
        path = ROOT / source['translations']
        patches = {}
        for line in path.read_text(encoding='utf-8').splitlines():
            p = json.loads(line)
            index = p['metadata']['row_index']
            if index in patches:
                raise ValueError(f'Duplicate translation row: {key}:{index}')
            patches[index] = p
        shards = source['shards']
        if not shards:
            raise ValueError('Explicit source shards are required')
        for i, shard in enumerate(shards):
            if f'test-{i:05d}-of-' not in shard:
                raise ValueError('ARO shards must be a contiguous prefix to preserve source row IDs')
        source_info = dict(dataset=key, revision=adapter.revision, translations_sha256=file_sha(path), shards=[])
        row_index = 0
        for shard in shards:
            parquet = hf_hub_download(adapter.path, shard, repo_type='dataset', revision=adapter.revision, local_files_only=True)
            source_info['shards'].append(dict(path=shard, sha256=file_sha(parquet)))
            for batch in pq.ParquetFile(parquet).iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    p = patches.get(row_index)
                    index = row_index
                    row_index += 1
                    if p is None:
                        raise ValueError(f'{key}:{index}: missing translation')
                    if p['metadata'].get('source_revision') != adapter.revision:
                        raise ValueError('Training translation revision mismatch')
                    if p['source'] != {f: row[f] for f in adapter.text_fields}:
                        raise ValueError(f'{key}:{index}: translation/source mismatch')
                    if p['id'] in used_ids:
                        raise ValueError(f'Duplicate source ID: {p["id"]}')
                    used_ids.add(p['id'])
                    if (include is not None and p['id'] not in include) or p['id'] in exclude:
                        continue
                    raw = row['image']
                    payload = raw.get('bytes') or Path(raw['path']).read_bytes()
                    identity = image_identity(payload=payload)
                    name = Path(raw.get('path') or '').stem
                    parent = vg.get(name)
                    parents = ['vg:' + str(parent['image_id'])] if parent else []
                    if parent and parent.get('coco_id') is not None:
                        parents.append('coco:' + str(parent['coco_id']))
                    if parent and parent.get('flickr_id') is not None:
                        parents.append('flickr:' + str(parent['flickr_id']))
                    record = dict(id=p['id'], dataset=key, row_index=index,
                        text_en=row['true_caption'], text_pl=p['translated']['true_caption'],
                        source_revision=adapter.revision, source_image=name, parent_ids=parents,
                        lineage_verified=parent is not None, **identity)
                    if not record['text_en'].strip() or not record['text_pl'].strip():
                        raise ValueError('Empty positive caption')
                    reasons = guard.reasons(record)
                    if reasons:
                        rejected.append(dict(id=p['id'], reasons=reasons, parent_ids=parents))
                        continue
                    image_path = image_dir / (identity['sha256'] + '.jpg')
                    if not image_path.exists():
                        image_path.write_bytes(payload)
                    record['image'] = str(image_path)
                    # Identical pixels, even under different filenames, are one positive target.
                    record['target_id'] = identity['pixels_sha256']
                    selected.append(record)
        source_info['rows_checked'] = row_index
        provenance.append(source_info)
    if include is not None and not include <= used_ids:
        raise ValueError(f'Unknown requested IDs: {sorted(include - used_ids)[:10]}')
    selected.sort(key=lambda r: r['id'])
    random.Random(config['data']['selection_seed']).shuffle(selected)
    maximum = config['data']['max_pairs']
    if maximum is not None:
        if maximum < 2:
            raise ValueError('max_pairs must be at least two')
        selected = selected[:maximum]
    train, validation = grouped_split(selected, config['data']['validation_fraction'], config['data']['split_seed'])
    for split, records in (('train', train), ('validation', validation)):
        for r in records:
            r['split'] = split
    audit = dict(sources=provenance, lineage_metadata_sha256=file_sha(vg_file),
                 rejected=len(rejected), reasons=dict(Counter(reason for r in rejected for reason in r['reasons'])),
                 retained=len(selected), train=len(train), validation=len(validation),
                 blacklist_sha256=blacklist['sha256'], post_filter_overlap=sum(bool(guard.reasons(r)) for r in selected),
                 rejected_records=rejected, protected_datasets=sorted(protected))
    if audit['post_filter_overlap']:
        raise ValueError('Leakage after filtering')
    return train + validation, audit
