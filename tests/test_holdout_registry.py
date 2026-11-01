import hashlib
import json

import pytest

from pl_mmb_training import holdouts
from pl_mmb_training.holdouts import GUARD, Components, Registry, canonical_path


def candidate(**changes):
    row = dict(id='new:1', sha256='new-raw', pixels_sha256='new-pixels',
               phash='ffffffffffffffff', lineage_verified=True)
    row.update(changes)
    return row


def registry(tmp_path, protected, caches=()):
    inputs = []
    for i, (fmt, value) in enumerate([('json_array', protected), *caches]):
        path = tmp_path / f'input-{i}.json'
        if fmt == 'fingerprint_cache':
            path.write_text('\n'.join(json.dumps(r) for r in value), encoding='utf-8')
        else:
            path.write_text(json.dumps(value), encoding='utf-8')
        inputs.append(dict(key=f'test-{i}', path=str(path), format=fmt,
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    path = tmp_path / 'registry.json'
    path.write_text(json.dumps(dict(schema=1, guard_sha256=hashlib.sha256(GUARD.read_bytes()).hexdigest(), inputs=inputs)), encoding='utf-8')
    return path


def test_image_only_candidate_is_protected(tmp_path):
    path = registry(tmp_path, [dict(pixels_sha256='held-image-only', phash='0000000000000000')])
    result = Registry(path).audit([candidate(pixels_sha256='held-image-only')])
    assert result['status'] == 'blocked'
    assert result['rejected_ids'] == ['new:1']


def test_protection_propagates_across_whole_candidate_component(tmp_path):
    path = registry(tmp_path, [dict(parent_ids=['reserved'], phash='0000000000000000')])
    rows = [candidate(id='bridge', parent_ids=['reserved', 'same-article']),
            candidate(id='indirect', sha256='other', pixels_sha256='other', parent_ids=['same-article'])]
    assert Registry(path).audit(rows)['rejected_ids'] == ['bridge', 'indirect']


def test_phash_distance_six_is_excluded_but_seven_alone_is_not(tmp_path):
    path = registry(tmp_path, [dict(phash='0000000000000000')])
    assert Registry(path).audit([candidate(phash='000000000000003f')])['rejected_rows'] == 1
    assert Registry(path).audit([candidate(phash='000000000000007f')])['rejected_rows'] == 0


def test_native_and_translated_text_pool_is_protected(tmp_path):
    path = registry(tmp_path, [dict(text_en='An old title', text_pl='Polski podpis')])
    assert Registry(path).audit([candidate(normalized_texts=['  POLSKI  podpis  '])])['rejected_rows'] == 1


def test_flickr_url_alias_reaches_namespaced_parent(tmp_path):
    path = registry(tmp_path, [dict(parent_ids=['flickr:123456789'])])
    row = candidate(source_record=dict(source_page_url='https://www.flickr.com/photos/someone/123456789?utm_source=wiki'))
    assert Registry(path).audit([row])['rejected_rows'] == 1


def test_shared_license_author_or_collection_url_is_not_an_image_identity(tmp_path):
    metadata = dict(license_url='https://creativecommons.org/licenses/by/4.0/',
                    source_author_url='https://example.com/uploader', source_url='https://commons.wikimedia.org/')
    path = registry(tmp_path, [dict(source_record=metadata, id='held')])
    assert Registry(path).audit([candidate(source_record=metadata)])['status'] == 'passed'


def test_same_source_sha1_hex_and_base36_is_protected(tmp_path):
    path = registry(tmp_path, [dict(source_sha1_hex='000000000000000000000000000000000000007b')])
    assert Registry(path).audit([candidate(source_record=dict(source_sha1='3f'))])['rejected_rows'] == 1


def test_changed_holdout_file_fails_closed(tmp_path):
    path = registry(tmp_path, [dict(text_pl='held')])
    (tmp_path / 'input-0.json').write_text('[]', encoding='utf-8')
    with pytest.raises(RuntimeError, match='Changed holdout input'):
        Registry(path)


def test_missing_original_fingerprint_fails_closed(tmp_path):
    path = registry(tmp_path, [dict(pixels_sha256='held')])
    with pytest.raises(RuntimeError, match='Missing holdout original-image pHash'):
        list(Registry(path).rows())


def test_fingerprint_cache_does_not_reserve_its_training_rows(tmp_path):
    path = registry(tmp_path, [dict(pixels_sha256='held')], caches=[('fingerprint_cache', [
        dict(pixels_sha256='held', phash='0000000000000000'), candidate()])])
    guard = Registry(path)
    assert list(guard.rows())[0][0]['phash'] == '0000000000000000'
    assert guard.audit([candidate()])['status'] == 'passed'


def test_unverified_admission_fails_closed(tmp_path):
    path = registry(tmp_path, [dict(text_pl='held')])
    assert Registry(path).audit([candidate(lineage_verified=False)])['unverified_rows'] == 1


def test_windows_wsl_path_aliases_connect():
    graph = Components()
    graph.add(dict(image='D:/work/a.jpg'), ['held'])
    assert graph.reasons(graph.add(dict(image='/mnt/d/work/a.jpg'))) == ['held']
    assert canonical_path('//wsl.localhost/Ubuntu/home/jsowa/a.jpg') == '/home/jsowa/a.jpg'


def release_inputs(tmp_path):
    path = registry(tmp_path, [dict(text_pl='held')])
    store, report = tmp_path / 'store', tmp_path / 'report'
    store.mkdir()
    report.mkdir()
    (store / 'Wiki_train.jsonl').write_text(json.dumps(candidate()) + '\n', encoding='utf-8')
    (store / 'new_VN_train.jsonl').write_text('', encoding='utf-8')
    binding = dict(registry_sha256=holdouts.digest(path), guard_sha256=holdouts.digest(holdouts.__file__),
                   manifests={name: holdouts.digest(store / name) for name in ('Wiki_train.jsonl', 'new_VN_train.jsonl')})
    (report / 'holdout_registry_binding.json').write_text(json.dumps(binding), encoding='utf-8')
    return path, store, report


def test_release_binds_training_manifests_before_gpu(tmp_path):
    path, store, report = release_inputs(tmp_path)
    proof = holdouts.require_training_release(store, report, path)
    assert proof['status'] == 'passed'
    holdouts.verify_release_proof(proof, store, path)
    (store / 'Wiki_train.jsonl').write_text('', encoding='utf-8')
    with pytest.raises(RuntimeError, match='Training admissions changed'):
        holdouts.require_training_release(store, report, path)


def test_changed_holdout_after_gpu_wait_invalidates_release(tmp_path):
    path, store, report = release_inputs(tmp_path)
    proof = holdouts.require_training_release(store, report, path)
    (tmp_path / 'input-0.json').write_text('[]', encoding='utf-8')
    with pytest.raises(RuntimeError, match='Holdout input changed while waiting'):
        holdouts.verify_release_proof(proof, store, path)


def test_old_binding_cannot_silently_accept_new_registry(tmp_path):
    path, store, report = release_inputs(tmp_path)
    data = json.loads(path.read_text(encoding='utf-8'))
    data['new_version'] = 2
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(RuntimeError, match='Holdout registry/guard changed after data freeze'):
        holdouts.require_training_release(store, report, path)
