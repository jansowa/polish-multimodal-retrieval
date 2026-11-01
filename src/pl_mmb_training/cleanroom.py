"""CLEAN-v1 audits reuse the frozen E0 guard, manifests and native feature caches."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .common import ROOT, TASKS, digest, file_sha, read_json, write_json
from .leakage import LeakageGuard, normalized_text
from .runner import check_e0, records_at, write_records

SOURCES = ('aro_visual_relation', 'aro_visual_attribution', 'sugarcrepe', 'mbeir_mscoco')
REASONS = ('eval_id', 'eval_parent_image', 'eval_image_content', 'eval_text', 'unknown_parent_lineage')
VARIANTS = ('FULL_CLEAN', 'NO_MBEIR_MSCOCO', 'NON_COCO_CLEANROOM')


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        if not rows:
            return
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def source_for(uid):
    prefixes = {'AROVisualRelation:': SOURCES[0], 'AROVisualAttribution:': SOURCES[1],
                'SugarCrepe:': SOURCES[2], 'MBeirmscoco:': SOURCES[3]}
    return next((value for prefix, value in prefixes.items() if uid.startswith(prefix)), None)


def counts(records):
    return dict(records=len(records), unique_texts=len({r['text_pl'] for r in records}),
                unique_image_text_pairs=len({(r['target_id'], r['text_pl']) for r in records}),
                unique_images=len({r['target_id'] for r in records}),
                sources=dict(Counter(r['dataset'] for r in records)))


def select_variant(records, variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown clean-room variant')
    if variant == 'FULL_CLEAN':
        return list(records)
    if variant == 'NO_MBEIR_MSCOCO':
        return [r for r in records if r['dataset'] != 'mbeir_mscoco']
    # Follow connected parent/pixel aliases, including transitive crop families.
    roots = {}
    def find(value):
        roots.setdefault(value, value)
        while roots[value] != value:
            roots[value] = roots[roots[value]]
            value = roots[value]
        return value
    for r in records:
        anchor = 'pixels:' + r['target_id']
        find(anchor)
        for parent in r['parent_ids']:
            roots[find(parent)] = find(anchor)
    coco_roots = {find('pixels:' + r['target_id']) for r in records
                  if r['dataset'] in ('mbeir_mscoco', 'sugarcrepe')
                  or any(p.startswith('coco:') for p in r['parent_ids'])}
    coco_images = {r['target_id'] for r in records if find('pixels:' + r['target_id']) in coco_roots}
    return [r for r in records if r['target_id'] not in coco_images
            and r['dataset'] not in ('mbeir_mscoco', 'sugarcrepe')]


def load_inputs():
    baseline = ROOT / 'training/runs/e1-full'
    base = read_json(baseline / 'config.json')
    e0, records = check_e0(base, baseline)
    blacklist = read_json(baseline / 'blacklist.json')
    manifest = read_json(baseline / 'cache.json')
    compact_path = ROOT / 'training/runs/vision-four/common/evaluation_compact.json'
    compact = read_json(compact_path)
    if compact['sha256'] != digest(compact['datasets']):
        raise ValueError('Compact benchmark membership changed')
    if compact['feature_manifest_sha256'] != file_sha(baseline / 'cache.json'):
        raise ValueError('Benchmark membership uses another feature manifest')
    if {k: len(rows) for k, rows in compact['datasets'].items()} != {k: v['examples'] for k, v in e0['evaluation'].items()}:
        raise ValueError('Incomplete benchmark membership')
    ownership, ids, texts = defaultdict(set), {}, defaultdict(set)
    english = read_json(baseline / 'eval_english.json')
    for task, rows in compact['datasets'].items():
        for row in rows:
            ids[row['id']] = task
            for index in set(row['images'] + ([row['query_image']] if row['query_image'] is not None else [])):
                ownership[manifest['images'][index]].add(task)
            for t in row['texts']:
                texts[normalized_text(manifest['texts'][t])].add(task)
            texts[normalized_text(english[row['id']])].add(task)
    if set(ownership) != set(blacklist['images']):
        raise ValueError('Membership does not cover all blacklisted query/candidate images')
    return baseline, base, records, blacklist, manifest, ownership, ids, texts


def audit(output):
    output = Path(output)
    baseline, base, records, blacklist, manifest, ownership, eval_ids, eval_texts = load_inputs()
    old = read_json(baseline / 'data_audit.json')
    parent_tasks, hash_tasks = defaultdict(set), defaultdict(set)
    for path, info in blacklist['images'].items():
        for p in info['parent_ids']:
            parent_tasks[p].update(ownership[path])
        for h in (info['sha256'], info['pixels_sha256']):
            hash_tasks[h].update(ownership[path])
    rejected = [r for r in old['rejected_records'] if source_for(r['id'])]
    needed = {r['id'].split(':positive:')[0] for r in rejected if 'eval_text' in r['reasons']}
    patch_texts = {}
    for source in SOURCES:
        with (baseline / f'translation_snapshots/{source}.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                p = json.loads(line)
                if p['id'] in needed:
                    field = 'true_caption' if source.startswith('aro_') else 'caption' if source == 'sugarcrepe' else 'text'
                    patch_texts[p['id']] = (p['source'][field], p['translated'][field])
    if set(patch_texts) != needed:
        raise ValueError('Cannot attribute all text exclusions to their frozen translations')
    totals = {s: Counter() for s in SOURCES}
    cells = {(s, t): Counter() for s in SOURCES for t in TASKS}
    evidence = []
    for r in rejected:
        source = source_for(r['id'])
        totals[source].update(r['reasons'])
        row = dict(source=source, id=r['id'], parent_ids=r.get('parent_ids', []), reasons=r['reasons'], attribution={})
        for reason in r['reasons']:
            tasks = set()
            if reason == 'eval_parent_image':
                tasks = set().union(*(parent_tasks[p] for p in r.get('parent_ids', [])))
            elif reason == 'eval_id' and r['id'] in eval_ids:
                tasks = {eval_ids[r['id']]}
            elif reason == 'eval_text':
                tasks = set().union(*(eval_texts[normalized_text(t)] for t in patch_texts[r['id'].split(':positive:')[0]]))
            elif reason == 'eval_image_content':
                tasks = hash_tasks[r.get('sha256', '')] | hash_tasks[r.get('pixels_sha256', '')]
            row['attribution'][reason] = sorted(tasks)
            for task in tasks:
                cells[source, task][reason] += 1
            if reason in REASONS and reason != 'unknown_parent_lineage' and not tasks:
                raise ValueError(f'Historical exclusion cannot be attributed: {r["id"]} {reason}')
        evidence.append(row)
    remaining = []
    for source in SOURCES:
        for task in TASKS:
            rows = [r for r in records if r['dataset'] == source]
            overlap = []
            for r in rows:
                reasons = []
                if any(task in parent_tasks[p] for p in r['parent_ids']):
                    reasons.append('source_parent')
                if any(task in hash_tasks[h] for h in (r['sha256'], r['pixels_sha256'])):
                    reasons.append('exact_hash')
                if reasons:
                    overlap.append(dict(id=r['id'], benchmark=task, reasons=reasons))
            remaining.extend(overlap)
    table = [dict(train_source=s, benchmark=t, **{reason: cells[s, t][reason] for reason in REASONS},
                  retained_train=sum(r['dataset'] == s and r['split'] == 'train' for r in records),
                  retained_validation=sum(r['dataset'] == s and r['split'] == 'validation' for r in records))
             for s in SOURCES for t in TASKS]
    source_table = [dict(source=s, rejected_records=sum(source_for(r['id']) == s for r in rejected),
                         **{reason: totals[s][reason] for reason in REASONS},
                         other_reasons={k: v for k, v in totals[s].items() if k not in REASONS},
                         retained=counts([r for r in records if r['dataset'] == s])) for s in SOURCES]
    output.mkdir(parents=True, exist_ok=True)
    write_records(output / 'rejection_evidence.jsonl', evidence)
    write_csv(output / 'leakage_by_source_benchmark.csv', table)
    write_json(output / 'known_remaining_overlaps.json', remaining)
    validation = [r for r in records if r['split'] == 'validation' and r['dataset'] in SOURCES]
    train = [r for r in records if r['split'] == 'train' and r['dataset'] in SOURCES]
    variants = {}
    for name in VARIANTS:
        selected = select_variant(train, name)
        if not selected:
            raise ValueError('Empty diagnostic training variant: ' + name)
        path = output / f'{name}.jsonl'
        write_records(path, selected)
        variants[name] = dict(**counts(selected), path=str(path), sha256=file_sha(path))
    write_records(output / 'validation.jsonl', validation)
    provenance = []
    for r in records:
        provenance.append(dict(dataset=r['dataset'], split=r['split'], source_record_id=r['id'],
                               source_image_id=r['source_image'], canonical_image_id=r['target_id'],
                               parent_ids=r['parent_ids'], sha256=r['sha256'], pixels_sha256=r['pixels_sha256'],
                               local_path=r['image'], benchmark_membership=[]))
    for path, info in blacklist['images'].items():
        provenance.append(dict(dataset='legacy_benchmark', split='test', source_record_id=None,
                               source_image_id=Path(path).name, canonical_image_id=info['pixels_sha256'],
                               parent_ids=info['parent_ids'], sha256=info['sha256'], pixels_sha256=info['pixels_sha256'],
                               local_path=path, benchmark_membership=sorted(ownership[path])))
    write_records(output / 'provenance.jsonl', provenance)
    report = dict(protocol='CLEAN-v1', status='complete', sources=source_table, by_source_benchmark=table,
                  known_remaining_overlap_count=len(remaining), historical_label='LEGACY / pre-audit',
                  historical_contamination_found=bool(remaining), variants=variants, validation=counts(validation),
                  inputs={str(baseline / p): file_sha(baseline / p) for p in ('data.jsonl', 'blacklist.json', 'data_audit.json', 'cache.json')},
                  attribution_note='Counts are non-exclusive across reasons and benchmarks. Retained counts repeat per benchmark. Unknown lineage cannot be assigned to a benchmark.',
                  non_coco_note='Excludes known COCO lineage and pixel aliases of known COCO images. Missing VG-to-COCO links are not a proof of absence of all possible COCO ancestry.',
                  mbeir_source_split='query/test + task3 test candidate pool in frozen original loader; internal train/validation split is separate',
                  near_duplicate_filter_applied=False, historical_artifacts_modified=False)
    write_json(output / 'audit.json', report)
    if remaining:
        raise ValueError('Known source/hash overlap remains; do not run CLEAN-v1 training')
    return report


def phash(path):
    """64-bit DCT pHash, 32x32 grayscale; EXIF-normalized like the exact guard."""
    import numpy as np
    from PIL import Image, ImageOps
    from scipy.fft import dctn
    with Image.open(path) as im:
        image = ImageOps.exif_transpose(im).convert('L').resize((32, 32), Image.Resampling.LANCZOS)
        coefficients = dctn(np.asarray(image, dtype=float), type=2)[:8, :8]
    bits = coefficients > np.median(coefficients)
    return int.from_bytes(np.packbits(bits.reshape(-1)).tobytes(), 'big')


def near_duplicates(output, threshold=6, top=200):
    import numpy as np
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from .features import validate_features
    output = Path(output)
    baseline, base, records, blacklist, manifest, ownership, _, _ = load_inputs()
    train_rows = [r for r in records if r['split'] == 'train']
    by_path = defaultdict(list)
    for r in train_rows:
        by_path[r['image']].append(r)
    train_paths, eval_paths = sorted(by_path), sorted(blacklist['images'])
    expected = {p: i['sha256'] for p, i in blacklist['images'].items()}
    expected.update({r['image']: r['sha256'] for r in records})
    identity = dict(algorithm='phash-dct32-low8-median-including-dc-exif-v1', files=expected)
    cache_path = output / 'perceptual_hashes.json'
    if cache_path.exists():
        cached = read_json(cache_path)
        if cached['identity_sha256'] != digest(identity):
            raise ValueError('Perceptual cache uses another image collection')
        hashes = {p: int(h, 16) for p, h in cached['hashes'].items()}
    else:
        def compute(path):
            if file_sha(path) != expected[path]:
                raise ValueError('Image differs from frozen provenance: ' + path)
            return path, phash(path)
        with ThreadPoolExecutor(max_workers=8) as pool:
            hashes = dict(pool.map(compute, sorted(expected)))
        write_json(cache_path, dict(identity_sha256=digest(identity),
                                   hashes={p: f'{h:016x}' for p, h in hashes.items()}))
    print(f'pHash ready: {len(train_paths)} training paths, {len(eval_paths)} benchmark paths', flush=True)
    gallery = np.array([hashes[p] for p in eval_paths], dtype=np.uint64)
    popcount = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    candidates, nearest = [], []
    for offset in range(0, len(train_paths), 32):
        paths = train_paths[offset:offset + 32]
        values = np.array([hashes[p] for p in paths], dtype=np.uint64)
        xor = values[:, None] ^ gallery[None, :]
        distances = popcount[xor.view(np.uint8).reshape(len(paths), len(gallery), 8)].sum(2)
        for i, path in enumerate(paths):
            best = int(np.argmin(distances[i]))
            nearest.append((int(distances[i, best]), path, eval_paths[best]))
            for j in np.flatnonzero(distances[i] <= threshold):
                candidates.append((int(distances[i, j]), path, eval_paths[j]))
    selected = sorted(set(candidates) | set(sorted(nearest)[:top]))
    meta = read_json(ROOT / 'training/runs/vision-commercial-five/models/tinyclip-39m/feature_cache.json')
    if file_sha(meta['path']) != meta['sha256']:
        raise ValueError('TinyCLIP features changed')
    features = validate_features(manifest['images'], torch.load(meta['path'], weights_only=True, map_location='cpu'))
    features = torch.nn.functional.normalize(features, dim=1)
    ii = {p: i for i, p in enumerate(manifest['images'])}
    rows = []
    for distance, train, test in selected:
        parents = set().union(*(set(r['parent_ids']) for r in by_path[train]))
        shared = sorted(parents.intersection(blacklist['images'][test]['parent_ids']))
        rows.append(dict(train_image=train, eval_image=test,
                         source_dataset=sorted({r['dataset'] for r in by_path[train]}),
                         benchmark=sorted(ownership[test]), phash_distance=distance,
                         vision_cosine=float(features[ii[train]] @ features[ii[test]]),
                         known_parent_relation=bool(shared), shared_parent_ids=shared,
                         review_status='candidate_not_confirmed', below_threshold=distance <= threshold))
    rows.sort(key=lambda r: (r['phash_distance'], -r['vision_cosine'], r['train_image'], r['eval_image']))
    write_csv(output / 'near_duplicate_candidates.csv', rows)
    write_json(output / 'near_duplicate_candidates.json', rows)
    result = dict(protocol='CLEAN-v1', status='complete', train_image_paths=len(train_paths),
                  benchmark_image_paths=len(eval_paths), unique_train_pixels=len({r['target_id'] for r in train_rows}),
                  pairs_exhaustively_compared=len(train_paths) * len(eval_paths),
                  threshold=threshold, threshold_pairs=len(candidates), reported_pairs=len(rows),
                  minimum_distance=rows[0]['phash_distance'] if rows else None,
                  automatic_filter_applied=False, confirmed_new_overlaps=None,
                  limitation='pHash can miss crops and significant transformations. TinyCLIP cosine annotates pHash candidates; no exhaustive embedding search is performed.',
                  image_content_verified=True, features_sha256=meta['sha256'])
    write_json(output / 'near_duplicate_summary.json', result)
    # Contact sheet is a review aid, never training or automated evidence of identity.
    from PIL import Image, ImageDraw, ImageOps
    for page, start in enumerate(range(0, min(48, len(rows)), 12)):
        canvas = Image.new('RGB', (960, 6 * 184), 'white')
        draw = ImageDraw.Draw(canvas)
        for k, row in enumerate(rows[start:start + 12]):
            x, y = (k % 2) * 480, (k // 2) * 184
            for j, key in enumerate(('train_image', 'eval_image')):
                with Image.open(row[key]) as im:
                    thumb = ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'), (230, 146))
                    canvas.paste(thumb, (x + j * 240, y + 28))
            draw.text((x, y), f"#{start+k} pH={row['phash_distance']} cos={row['vision_cosine']:.3f}", fill='black')
        canvas.save(output / f'near_duplicate_review_{page}.jpg')
    return result
