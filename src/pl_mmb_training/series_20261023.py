"""CPU preparation for the consultation series. No model loads or live DB reads.

Frozen outputs use exclusive creation and can only be reused byte-for-byte.
Candidate manifests are deliberately distinct from evaluation-ready datasets.
"""
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .common import ROOT, file_sha
from .leakage import normalized_text

OUT = ROOT / 'reports/series_2026-10-23'
VN = ROOT / 'training/runs/tinyclip-visualnews-v1/data'
WIKI = ROOT / 'training/runs/tinyclip-wiki-pilot-v1/data'
RESERVE = 4 * 2**30
REVISION = 'd77ff16148468c84287f3efcbf602a015c5e4954'


def freeze(path, value, jsonl=False):
    path = Path(path)
    payload = (''.join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n' for r in value)
               if jsonl else json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f'Immutable output already exists with different content: {path}')
    else:
        with path.open('xb') as stream:
            stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def local_path(value):
    value = str(value)
    if re.match(r'^/mnt/[a-z]/', value) and Path('C:/').exists():
        return Path(value[5].upper() + ':' + value[6:])
    if re.match(r'^[A-Za-z]:[\\/]', value) and not Path('C:/').exists() and Path('/mnt/c').exists():
        return Path('/mnt/' + value[0].lower() + value[2:].replace('\\', '/'))
    return Path(value)


def disk_inventory():
    paths = ['C:/', 'D:/'] if Path('C:/').exists() else ['/mnt/c', '/mnt/d', '/']
    return {p: dict(zip(('total', 'used', 'free'), shutil.disk_usage(p))) for p in paths}


def require_space(paths, expected_bytes=0):
    for path in paths:
        if shutil.disk_usage(path).free - expected_bytes < RESERVE:
            raise RuntimeError(f'Disk reserve below 4 GiB: {path}; no new cache/download permitted')


def native_original(row):
    stages = row.get('stages', {})
    judge = stages.get('judge', {})
    selected = row.get('selected_candidate', {})
    text = row.get('text_pl')
    return bool(row.get('route') == 'ready_native_original'
                and row.get('source_language') == selected.get('language') == 'pl'
                and judge.get('visual_grounding') in (3, 4)
                and judge.get('image_caption_mismatch') is False
                and not set(stages).intersection(('recycle', 'recheck', 'translate'))
                and text and text == row.get('accepted_text') == row.get('source_caption') == selected.get('text'))


def deterministic_images(records, count, salt):
    by_image = defaultdict(list)
    for row in records:
        by_image[row['target_id']].append(row)
    if len(by_image) < count:
        raise ValueError(f'Need {count} distinct images, found {len(by_image)}')
    ordered = sorted(by_image, key=lambda key: hashlib.sha256((salt + ':' + key).encode()).hexdigest())
    return [min(by_image[key], key=lambda r: r['id']) for key in ordered[:count]]


def protected():
    paths = [ROOT / 'training/runs/clean-v1/FULL_CLEAN.jsonl',
             ROOT / 'training/runs/clean-v1/validation.jsonl', VN / 'vn_train.jsonl', VN / 'vn_dev.jsonl']
    groups = {p.name: rows(p) for p in paths}
    blacklists = [ROOT / 'training/runs/e1-full/blacklist.json', ROOT / 'training/runs/xm3600-pl/blacklist.json']
    for p in blacklists:
        black = read(p)
        groups['XM_blacklist' if 'xm3600' in str(p) else 'legacy_eval'] = list(black['images'].values())
    inputs = {str(p.relative_to(ROOT)): file_sha(p) for p in paths + blacklists}
    texts = {normalized_text(r[f]) for rs in groups.values() for r in rs for f in ('text_en', 'text_pl') if r.get(f)}
    for p in blacklists:
        texts.update(read(p)['texts'])
    return groups, texts, inputs


def audit_wiki(groups, protected_texts):
    existing = OUT / 'wiki_preflight.json'
    if existing.exists():
        saved = read(existing)
        if file_sha(OUT / 'wiki_native_candidates.jsonl') != saved['candidate_manifest_sha256']:
            raise ValueError('Frozen candidate manifest changed')
        return saved
    # Read the atomically published export once; SHA describes exactly the bytes parsed.
    source = ROOT / 'reports/wikimedia_pipeline/v1/final_pairs.jsonl'
    payload = source.read_bytes()
    export_sha = hashlib.sha256(payload).hexdigest()
    data = [json.loads(line) for line in payload.splitlines() if line.strip()]
    natives = [r for r in data if native_original(r)]
    known = {r['id']: r for r in rows(WIKI / 'candidates.jsonl')}
    old_train = {r['id'] for r in rows(WIKI / 'wiki_train.jsonl')}
    candidates = []
    for r in natives:
        s = r['source_record']
        meta = known.get(r['id'], {})
        candidates.append(dict(id=r['id'], dataset='wikimedia', text_pl=r['text_pl'], text_en=r.get('text_en'),
            source_caption=r['source_caption'], source_language='pl', route=r['route'],
            source_class=r['selected_candidate'].get('quality'), selected_candidate=r['selected_candidate'],
            judge=r['stages']['judge'], source_record=s,
            local_identity={k: meta[k] for k in ('sha256', 'pixels_sha256', 'phash', 'parent_ids', 'cache_index', 'preview') if k in meta},
            historical_wiki_train=r['id'] in old_train))
    # Early exact-text audit only. Missing original image identities must not pass.
    exact_conflicts = [r['id'] for r in candidates if normalized_text(r['text_pl']) in protected_texts]
    manifest = OUT / 'wiki_native_candidates.jsonl'
    candidate_sha = freeze(manifest, candidates, jsonl=True)
    result = dict(status='stopped_below_300' if len(natives) < 300 else 'pending_full_image_component_audit',
        source=str(source.relative_to(ROOT)), source_export_sha256=export_sha, export_rows=len(data),
        direct_native_rows=len(natives), source_classes=dict(Counter(r['source_class'] for r in candidates)),
        candidate_manifest_sha256=candidate_sha, preflight_exact_text_conflicts=len(exact_conflicts),
        local_identity_records=sum(bool(r['local_identity']) for r in candidates),
        historical_pilot_train_id_overlap=sum(r['historical_wiki_train'] for r in candidates),
        ready_for_retrieval=False, frozen_dev_created=False,
        blockers=['D volume below required reserve; original assets not acquired',
                  'Canonical-pixel, pHash<=6 and transitive source/page/text component audit incomplete',
                  'Re-freeze disjoint Wiki train into a NEW manifest only after dev is frozen'],
        native_provenance_rule='route native_original; PL provenance; judge 3/4 no mismatch; no recycle/recheck/translate; exact source/accepted/final text equality',
        source_class_definition='quality of selected provenance candidate, A-D; no C available in native PL export',
        candidate_manifest_is_not_dev=True)
    freeze(OUT / 'wiki_preflight.json', result)
    return result


def prepare_xtd(groups, inputs):
    folder = OUT / 'xtd10_source'
    upstream = read(folder / 'upstream.json')
    entries = {r['name']: r for r in upstream['tree']}
    texts = {}
    source_hashes = {}
    for name in ('test_image_names.txt', 'test_1kcaptions_en.txt', 'test_1kcaptions_pl.txt'):
        raw = (folder / name).read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if actual != entries[name]['sha']:
            raise ValueError(f'Upstream Git blob mismatch: {name}: {actual}')
        texts[name] = raw.decode('utf-8').splitlines()
        source_hashes[name] = hashlib.sha256(raw).hexdigest()
    if {len(v) for v in texts.values()} != {1000}:
        raise ValueError('XTD10 must contain 1000 aligned EN/PL/image rows')
    parent_sets = {name: {p for r in rs for p in r.get('parent_ids', [])} for name, rs in groups.items() if name != 'XM_blacklist'}
    hash_sets = {name: {r[k] for r in rs for k in ('sha256', 'pixels_sha256') if r.get(k)} for name, rs in groups.items() if name != 'XM_blacklist'}
    by_parent = defaultdict(list)
    for group_name, rs in groups.items():
        if group_name == 'XM_blacklist':
            continue  # XM is used exclusively for leakage detection, never asset resolution.
        for r in rs:
            for p in r.get('parent_ids', []):
                if p.startswith('coco:'):
                    by_parent[p].append(r)
    gallery = []
    for i, (name, en, pl) in enumerate(zip(texts['test_image_names.txt'], texts['test_1kcaptions_en.txt'], texts['test_1kcaptions_pl.txt'])):
        match = re.fullmatch(r'(?:.*/)?COCO_(?:train|val|test)\d{4}_(\d{12})\.(?:jpg|png)', name)
        if not match:
            raise ValueError(f'No explicit upstream COCO identity for row {i}: {name!r}')
        parent = 'coco:' + str(int(match[1]))
        matches = by_parent[parent]
        # A parent match is ancestry evidence, never evidence that a crop is the full image.
        assets = [r for r in matches if Path(str(r.get('image', '')).replace('\\', '/')).name == name and local_path(r['image']).is_file()]
        identities = {(r['sha256'], r['pixels_sha256']) for r in assets}
        if len(identities) > 1:
            raise ValueError(f'Ambiguous full-image content: {name}')
        asset = min(assets, key=lambda r: r['image']) if assets else None
        hashes = set(identities.pop()) if identities else set()
        overlap = {split: dict(parent=parent in parents, hashes=bool(hashes & hash_sets[split]))
                   for split, parents in parent_sets.items()}
        gallery.append(dict(id=f'xtd10:{i:04d}', upstream_image_name=name, parent_ids=[parent],
            text_en=en, text_pl=pl, overlap=overlap, image=asset.get('image') if asset else None,
            sha256=asset.get('sha256') if asset else None, pixels_sha256=asset.get('pixels_sha256') if asset else None,
            hash_audit='pending_missing_full_image' if asset is None else 'known_manifest_identity_requires_file_reverification',
            component=parent, role='translation_style_control_not_holdout'))
    if len({r['parent_ids'][0] for r in gallery}) != 1000:
        raise ValueError('XTD10 gallery is not 1000 distinct official IDs')
    annotation_sha = freeze(OUT / 'xtd10_annotation_manifest.jsonl', gallery, jsonl=True)
    clean = [r['id'] for r in gallery if not any(v['parent'] or v['hashes'] for v in r['overlap'].values())]
    result = dict(status='annotations_pinned_image_audit_pending', revision=REVISION,
        source_hashes=source_hashes, annotation_manifest_sha256=annotation_sha, images=1000,
        local_full_image_assets=sum(r['image'] is not None for r in gallery),
        parent_overlap_images={split: sum(r['overlap'][split]['parent'] for r in gallery) for split in parent_sets},
        preliminary_no_parent_overlap_ids=clean, preliminary_clean_count=len(clean),
        clean_subset_frozen=False, ready_for_retrieval=False,
        full_gallery='1000 images; reveal training overlap counts alongside scores',
        clean_gallery='exclude any parent or exact/pixel hash overlap; pending full-image verification',
        style_pairing='exact normalized original EN match only, never positional or image-only match; Qwen cache not yet audited',
        license_file='xtd10_source/LICENSE.md', role='style_control_not_holdout',
        license_url=f'https://github.com/adobe-research/Cross-lingual-Test-Dataset-XTD10/blob/{REVISION}/LICENSE.md',
        protected_inputs=inputs, xm3600_scored=False)
    freeze(OUT / 'xtd10_preparation.json', result)
    return result


def registration():
    # models.py has no heavyweight imports at module scope. This does not load weights.
    from pl_mmb_benchmark.models import MODELS, REVISIONS, MODEL_METADATA, CLIP_IMAGE_REPO, CLIP_IMAGE_REVISION
    prior = {}
    for p in (ROOT / 'benchmark/results').glob('*/result.json'):
        r = read(p)
        if r.get('model') in MODELS.values() and r.get('task_instructions') is False:
            prior.setdefault(r['model'], []).append((p, r))
    models = []
    for alias, repo in MODELS.items():
        candidates = prior.get(repo, [])
        policies = sorted({r.get('precision', '') for _, r in candidates})
        control = alias in ('mobileclip2-s2', 'pe-core-b16')
        models.append(dict(alias=alias, model=repo, revision=REVISIONS[repo],
            total_parameters=MODEL_METADATA[repo].get('parameters'),
            same_size_class_le_100M=MODEL_METADATA[repo].get('parameters', 10**10) <= 100_000_000,
            polish_adapted=repo.startswith('alphaedge-ai/'), english_text_only_control=control,
            polish_retrieval_competitor=not control, historical_precision_policies=policies,
            historical_results=[dict(path=str(p.relative_to(ROOT)), sha256=file_sha(p)) for p, _ in candidates],
            author_prompt_status='requires_primary_source_audit_before_sensitivity_execution'))
    assert len(models) == 20
    pin = dict(image_encoder=CLIP_IMAGE_REPO, image_revision=CLIP_IMAGE_REVISION)
    value = dict(schema=1, series='2026-10-23', status='model_list_frozen_execution_blocked',
        reference_models=models, optional_models=[], linked_CLIP_components=pin,
        source_hashes={p: file_sha(ROOT / p) for p in ('src/pl_mmb_benchmark/models.py', 'src/pl_mmb_benchmark/special_models.py', 'reports/benchmark_twenty_models.md')},
        our_models=[dict(name='VN_FULL', seeds=[42, 43, 44], seed42='training/runs/tinyclip-source-ablation-v1/VN_FULL/best.pt',
                         report='mean and sample SD across all three seeds; selected median-VN-dev seed for legacy test'),
                    dict(name='VN_only', checkpoint='training/runs/tinyclip-source-ablation-v1/VN_only/best.pt', role='without COCO'),
                    dict(name='historical_A1', role='historical reference; checkpoint identity must be verified before execution')],
        our_text_prefix='[query]: ', our_inference_parameters=dict(text=31883136, vision=38587392, head=196992, total=70667520),
        datasets=[dict(name='wiki_native_dev', gallery='full', status='pending', domain='zero-shot only for never-Wiki-trained checkpoints; historical Wiki pilot already exists'),
                  dict(name='xtd10_full_1k', role='translation-style control; disclose overlap'),
                  dict(name='xtd10_no_overlap', role='translation-style control, clean subset after hash audit'),
                  dict(name='VN_dev', labels='model-generated', domain='in-domain for our VN-trained models'),
                  dict(name='local_CIRR_FashionIQ_RefCOCO', scope='selected finalist; existing competitors'),
                  dict(name='human_short_queries', status='pending annotations')],
        harness='shared v2 native encoders; full-gallery evaluator; our adapter required, not training-script scores',
        metrics=['t2i/recall@1', 't2i/recall@5', 't2i/recall@10', 't2i/mrr', 't2i/ndcg@10',
                 'i2t/recall@1', 'i2t/recall@5', 'i2t/recall@10', 'i2t/mrr', 'i2t/ndcg@10'],
        positives='explicit multi-positive qrels; deduplicate normalized captions and canonical images; never inferred by model',
        paired_bootstrap=dict(unit='transitive leakage component', replicates=10000, seed=20261023,
            ci='percentile 2.5/97.5%', delta='our minus reference, percentage points',
            estimator='query-weighted metric after component resampling, common resample indices',
            pairing='identical dataset SHA, ordered query IDs, qrels SHA, component IDs and galleries; fail on mismatch',
            conditional='CI conditional on fixed gallery, available qrels and sampled components; not pretraining uncertainty'),
        prompt_modes=[dict(name='v2_no_instructions', primary=True),
                      dict(name='author_recommended', primary=False, stop_until='all model-specific prompts and primary sources are pinned')],
        efficiency=dict(cpu=dict(batch_size=1, dtype='fp32', threads=2), gpu=dict(batch_size=64, dtype='bf16'),
            warmup_batches=10, measured_batches=100, timings='synchronize GPU around timed encode; median and p95; no download/cache time',
            report=['text_parameters', 'vision_parameters', 'head_parameters', 'dimension', 'query_ms', 'image_ms'],
            same_machine=True, historical_precision_deviations='record per result; BF16 efficiency is separate from model-specific retrieval precision'),
        cache_precision='FP32 weights + BF16 autocast, FP32 cache for our encoders; reference models reuse registered historical policies',
        parity_gate=dict(models=['siglip2-base-256', 'metaclip-s16', 'clip-b32-pl'],
            source='existing local MSCOCO raw embedding caches only; reproduce every historical rank without model encoding',
            status='not_run', on_mismatch='stop series'),
        wiki_symmetry=dict(immutable_key='wiki_dev_zero_shot_pre_wiki_training',
            overwrite=False, later_results='separate in-domain column; historical Wiki pilot must be disclosed',
            new_wiki_training_requires='frozen disjoint dev guard, immutable baseline, user decision'),
        no_model_add_remove_rerun_after_scores=True, changes_require='new registration and rationale before execution',
        XM3600='FINAL_BLIND; excluded from this execution registration',
        remaining_execution_blockers=['disk reserve', 'frozen image-verified dataset', 'author prompt primary-source audit',
                                      'our harness adapter', 'historical cache parity gate'])
    freeze(OUT / 'reference_eval_registration.json', value)
    decision = dict(series='2026-10-23', phase='P1.A', status='registered_not_executed',
        recipe='tinyclip_source_ablation VN_FULL, exact original immutable 35918-pair snapshot',
        seed42_checkpoint_sha256=file_sha(ROOT / 'training/runs/tinyclip-source-ablation-v1/VN_FULL/best.pt'),
        recipe_config_sha256=file_sha(ROOT / 'training/runs/tinyclip-source-ablation-v1/VN_FULL/config.json'),
        seeds=[42, 43, 44], reuse=[42], train=[43, 44], steps=5500,
        projector='fresh Linear(512,384), bias', frozen_encoders=True,
        checkpoint_selection=['VN dev t2i R@1', 'mean t2i/i2t R@1', 'earliest update'],
        test_seed_selection=dict(rule='median of three selected VN dev t2i R@1 values', ties='ascending seed'),
        COCO_validation='post hoc after all checkpoints closed; mean and sample SD',
        legacy_test='only median seed after decision materializes checkpoint SHA; no tests now',
        GPU_order=['P1.A', 'P1.B0', 'P1.B extraction', 'P1.B', 'P1.C', 'P1.D'],
        prerequisites=['one GPU worker lock', 'C and D and native WSL volume >=4 GiB after planned write', 'checkpoint reload/metrics/SHA verification'])
    freeze(OUT / 'vn_full_seeds_decision.json', decision)
    freeze(OUT / 'xm3600_gate_registration.json', dict(status='DRAFT_REQUIRES_USER_DECISION', execution_allowed=False,
        protocol='same registered v2 full gallery for all participants in one jointly registered run',
        reference_models=[dict(model=r['model'], revision=r['revision'], role='EN control' if r['english_text_only_control'] else 'reference') for r in models],
        our_model='median VN_FULL seed; checkpoint identity pending P1.A',
        proposed_gate=dict(primary='t2i R@1 paired delta vs best Polish-capable <=100M reference',
            minimum_point_gain_pp=2.0, lower_95CI_above_pp=0.0,
            caveat='proposal only; freeze comparator selection from non-XM dev before requesting approval'),
        user_approval_received=False, xm3600_scored=False))
    return value


def annotation_manifests():
    train, dev = rows(VN / 'vn_train.jsonl'), rows(VN / 'vn_dev.jsonl')
    query_rows = deterministic_images(dev, 150, '20261023:short-query:VN')
    query_targets = {r['target_id'] for r in query_rows}
    # Prevent the same annotator from seeing query-image captions in audit mode.
    audit = []
    for publisher in ('washington_post', 'usa_today'):
        for split, pool in (('train', train), ('dev', dev)):
            eligible = [r for r in pool if r['source'] == publisher and r['target_id'] not in query_targets]
            audit.extend(deterministic_images(eligible, 25, f'20261023:audit:{publisher}:{split}'))
    def materialize(row, caption):
        result = {k: row[k] for k in ('id', 'target_id', 'component', 'source', 'split', 'cache_index')}
        result.update(dataset='VN', asset='vn_native_rgb', asset_view='native 224x224 frozen preprocessing; not original resolution')
        if caption:
            result['text_pl'] = row['text_pl']
        return result
    query_sha = freeze(OUT / 'annotation/query_tasks.jsonl', [materialize(r, False) for r in query_rows], jsonl=True)
    audit_sha = freeze(OUT / 'annotation/audit_tasks.jsonl', [materialize(r, True) for r in audit], jsonl=True)
    result = dict(status='VN_ready_Wiki_pending', seed_rule='SHA256(salt:canonical_pixels) ascending; minimum row ID per image',
        query=dict(VN=150, Wiki=0, manifest_sha256=query_sha), audit=dict(VN=100, Wiki=0, manifest_sha256=audit_sha,
            publisher_split_counts=dict(Counter(r['source'] + ':' + r['split'] for r in audit))),
        wiki_blocker='Cannot sample new native Wiki dev or its complement before image/component audit and freeze',
        audit_query_images_disjoint=True, human_annotations_created=0,
        inputs={name: file_sha(VN / name) for name in ('vn_train.jsonl', 'vn_dev.jsonl', 'snapshot.json')},
        asset=str((VN / 'native_rgb.npy').relative_to(ROOT)),
        server='python experiments/series_annotation.py --port 8768 (127.0.0.1 only)')
    freeze(OUT / 'annotation/sampling.json', result)
    return result


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'preparation_state.json').exists():
        for name, sha in read(OUT / 'preservation_after.json').items():
            if file_sha(ROOT / name) != sha:
                raise RuntimeError(f'Protected snapshot changed: {name}')
        print('CPU preparation already frozen; preserved input SHA256 values verified. No live export reread.')
        return
    before = disk_inventory()
    inputs = ['training/runs/clean-v1/FULL_CLEAN.jsonl', 'training/runs/clean-v1/validation.jsonl',
              'training/runs/tinyclip-visualnews-v1/data/pipeline_pairs_snapshot.jsonl',
              'training/runs/tinyclip-visualnews-v1/data/vn_train.jsonl', 'training/runs/tinyclip-visualnews-v1/data/vn_dev.jsonl',
              'training/runs/tinyclip-wiki-pilot-v1/data/wiki_pairs_snapshot.jsonl',
              'training/runs/tinyclip-wiki-pilot-v1/data/wiki_train.jsonl', 'training/runs/tinyclip-wiki-pilot-v1/data/wiki_dev.jsonl']
    preservation = {p: file_sha(ROOT / p) for p in inputs}
    freeze(OUT / 'preservation_before.json', preservation)
    registration()
    groups, texts, hashes = protected()
    wiki = audit_wiki(groups, texts)
    xtd = prepare_xtd(groups, hashes)
    annotation = annotation_manifests()
    after_hashes = {p: file_sha(ROOT / p) for p in inputs}
    if preservation != after_hashes:
        raise RuntimeError('Protected snapshot changed during CPU preparation')
    freeze(OUT / 'preservation_after.json', after_hashes)
    freeze(OUT / 'preparation_state.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        disk_before=before, disk_after=disk_inventory(), wiki=wiki, xtd=xtd, annotation=annotation,
        snapshots_unchanged=True, GPU_runs=[], live_pipeline_databases_read=False, xm3600_scored=False))
    print(json.dumps(dict(wiki_native=wiki['direct_native_rows'], xtd_overlap=xtd['parent_overlap_images'],
        xtd_local_assets=xtd['local_full_image_assets'], annotation=annotation['status'], snapshots_unchanged=True), indent=2))
