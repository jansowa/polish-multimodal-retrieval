"""Audit candidate originals against the SHA-verified complete A universe on CPU."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from collections import Counter
import json
from pathlib import Path
import time

from prepare import ROOT, OUT, COORD, read, write, load_common, verify_marker, local_path, sha, committed_bytes, status
from acquire import ASSETS
from leakage import ProtectedIndex, filter_components


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def flickr(common):
    from vizwiz_assets import fingerprints
    import pyarrow.parquet as pq
    target = ASSETS / 'flickr30k/candidate_images.json'
    source = ASSETS / 'flickr30k/test.parquet'
    download = read(OUT / 'flickr30k_download.json')
    if sha(source) != download['sha256']:
        raise ValueError('Pinned Flickr shard changed')
    metadata = read(OUT / 'deep_research_v2.json')['sources']
    split = next(r for r in metadata if 'test_2016_flickr.txt' in r['url'])
    if split['status'] != 'fetched':
        raise ValueError('Original Flickr-ID split list not available')
    expected = set(split['text'].splitlines())
    common.reserve('B', 'candidate-audit', {'C': 32 * 1024**2}, committed_bytes=committed_bytes())
    if target.exists():
        records = read(target)
    else:
        records = []
        for batch in pq.ParquetFile(source).iter_batches(batch_size=16):
            for offset, row in enumerate(batch.to_pylist()):
                name = row['image']['path']
                if name not in expected or not name.removesuffix('.jpg').isdigit():
                    raise ValueError('Flickr original ID not in source test split')
                raw = row['image']['bytes']
                identity = fingerprints(raw)
                en, pl = row['captions_en'], row['captions_pl']
                if len(en) != len(pl) or not en or any(not s.strip() for s in en + pl):
                    raise ValueError('Missing or misaligned original caption translations')
                records.append(dict(identity, id='flickr30k:' + name, filename=name,
                    parent_ids=['flickr:' + name.removesuffix('.jpg')], dataset='flickr30k_pl_local_test',
                    split='test', lineage_verified=True, original_row=len(records),
                    image=f'parquet-original://flickr30k/test/{len(records)}',
                    verified_location=dict(path=common.windows_path(source), row=len(records)),
                    bytes=len(raw), captions=[dict(id=f'{name}#{i}', text_en=a, text_pl=b)
                                             for i, (a, b) in enumerate(zip(en, pl))],
                    normalized_texts=en + pl,
                    translation='NLLB-200-1.3B, repack-provided; original translator revision unavailable'))
        common.atomic_json(target, records)
    if {r['filename'] for r in records} != expected:
        raise ValueError('Flickr pool differs from original source test IDs')
    return records, dict(source_test_list_sha256=split['sha256'], exact_ID_set_match=True,
                         official_flickr30k1k_claim=False, parquet_originals_not_copied=True)


def main():
    started = time.monotonic()
    common = load_common()
    universe = verify_marker('A_UNIVERSE_FROZEN')
    common.verify_marker(COORD / 'A_UNIVERSE_FROZEN.json')
    index = ProtectedIndex()
    for artifact in universe['artifacts']:
        if artifact['role'] in ('training_universe', 'protected_index'):
            for row in rows(local_path(artifact['path'])):
                index.add(row)
    for artifact in universe.get('normalized_text_guard_artifacts', []):
        # verify_marker checks these additional dependencies as well.
        value = read(local_path(artifact['path']))
        for text in value.get('texts', []):
            index.add(dict(text_en=text))
        for identifier in value.get('ids', []):
            index.add(dict(id=identifier))
        for image in value.get('images', {}).values():
            index.add(image)
        for parent in value.get('parent_ids', []):
            index.add(dict(parent_ids=[parent]))
    # Prepare Flickr without duplicating embedded originals. VizWiz acquisition
    # runs separately; if unfinished, audit the full pool only on a later call.
    records, split_proof = flickr(common)
    vizwiz_path = ASSETS / 'vizwiz/candidate_images.json'
    if vizwiz_path.exists():
        vizwiz = read(vizwiz_path)
        translation_summary = OUT / 'translation_summary.json'
        translated = {}
        if translation_summary.exists():
            translation_meta = read(translation_summary)
            translation_path = local_path(translation_meta['artifact']['path'])
            if sha(translation_path) != translation_meta['artifact']['sha256']:
                raise ValueError('Final translation cache changed')
            translated_rows=list(rows(translation_path))
            translated = {r['text_en']:r['text_pl'] for r in translated_rows}
            segment_maxima=[];last=0.0
            for row in translated_rows:
                elapsed=row.get('elapsed_run_seconds',0.0)
                if elapsed+1e-6<last:
                    segment_maxima.append(last);last=0.0
                last=max(last,elapsed)
            segment_maxima.append(last)
            translation_meta.update(translation_loop_wall_seconds_across_resumes=sum(segment_maxima),
                observed_translation_loop_segments=len(segment_maxima),
                time_cost_definition='Sum of observed monotonic translation-loop wall segments in durable cache; excludes downloads/model initialization and unrecorded interrupted work. CPU financial/energy cost unpriced.',
                maximum_CPU_workers=2,maximum_Torch_threads_per_worker=1)
            write(translation_summary,translation_meta)
        for row in vizwiz:
            row.update(id='vizwiz:' + row['filename'], dataset='vizwiz_pl_local_validation',
                       lineage_verified=True, normalized_texts=[c['caption'] for c in row['captions']] +
                       [translated[c['caption']] for c in row['captions'] if c['caption'] in translated])
        records.extend(vizwiz)
    result = filter_components(records, index)
    target = ASSETS / 'candidate_overlap_audit.json'
    common.atomic_json(target, result)
    common.reserve('B', 'candidate-audit', {'C': 0}, committed_bytes=committed_bytes())
    final_translations = (OUT / 'translation_summary.json').exists()
    summary = dict(status='CPU_originals_and_final_translations_audited' if final_translations else 'CPU_image_and_EN_PL_overlap_audited_translation_reaudit_required',
        A_UNIVERSE_FROZEN_sha256=sha(COORD / 'A_UNIVERSE_FROZEN.json'),
        implementation_sha256={name: sha(Path(__file__).parent / name) for name in ('audit_candidates.py','leakage.py','vizwiz_assets.py')},
        protected_index_rows=index.rows, protected_unique_phashes=len(index.hashes),
        candidates=dict(Counter(row['dataset'] for row in result)),
        accepted=dict(Counter(row['dataset'] for row in result if not row['excluded'])),
        rejected=dict(Counter(row['dataset'] for row in result if row['excluded'])),
        rejection_reasons=dict(Counter(reason for row in result for reason in row['exclusion_reasons'])),
        flickr_source_split=split_proof, dataset_selection_not_final=True, GPU_operations=0,
        final_translation_summary_sha256=sha(OUT/'translation_summary.json') if final_translations else None,
        artifact=common.artifact('candidate_overlap_audit', target), elapsed_seconds=time.monotonic()-started)
    write(OUT / 'candidate_overlap_audit.json', summary)
    status()
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
