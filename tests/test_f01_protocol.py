"""F01-a data integrity checks and rejection of corrupted split artifacts."""
import copy
import importlib.util
import json
from pathlib import Path

import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('f01_protocol', ROOT / 'experiments/f_series/f01_protocol.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def inputs():
    assignments = json.loads((ROOT / module.OUT / 'assignments.json').read_text())
    source = json.loads(module.local_path(assignments['source_protocol']['path']).read_text())
    artifacts = {str(module.OUT / f'{h}.json'): (ROOT / module.OUT / f'{h}.json').read_bytes() for h in ('a', 'b')}
    return artifacts, source


def check_corrupt_split_is_rejected(corruption):
    artifacts, source = copy.deepcopy(inputs())
    path = str(module.OUT / 'a.json')
    a = json.loads(artifacts[path])
    b = json.loads(artifacts[str(module.OUT / 'b.json')])
    if corruption == 'drop_qrel':
        a['edges'].pop()
    elif corruption == 'duplicate_qrel':
        a['edges'].append(a['edges'][0])
    elif corruption == 'cross_component':
        a['image_components'][0] = b['image_components'][0]
    elif corruption == 'alter_text':
        a['texts'][0] += ' changed'
    elif corruption == 'drop_image':
        a['source_image_indices'].pop()
    artifacts[path] = module.encoded(a)
    with unittest.TestCase().assertRaises((AssertionError, IndexError)):
        module.validate(artifacts, source)


def test_protocol_sha_and_normative_source():
    sums = json.loads((ROOT / 'protocols/protocol-v3-f.SHA256SUMS.json').read_text())
    for path, sha in sums.items():
        assert module.digest(ROOT / path) == sha
    protocol = json.loads((ROOT / 'protocols/protocol-v3-f.json').read_text())
    normative = (ROOT / module.SOURCE).read_text()
    assert module.digest(ROOT / protocol['normative_source']['path']) == protocol['normative_source']['sha256']
    assert normative in (ROOT / 'protocols/protocol-v3-f.md').read_text()
    assert len(protocol['normative_sections']) == 9
    assert protocol['catalog']['sha256'] == module.digest(ROOT / protocol['catalog']['path'])
    assert protocol['normative_markdown']['sha256'] == sums['protocols/protocol-v3-f.md']


def test_catalog_halves_bound_to_protected_parent():
    catalog = json.loads((ROOT / 'retrieval_benchmark/catalog-f-v3.json').read_text())
    assignments = json.loads((ROOT / module.OUT / 'assignments.json').read_text())
    assert catalog['holdout_registry']['sha256'] == module.digest(module.DEFAULT_REGISTRY)
    assert assignments['registry_sha256'] == catalog['holdout_registry']['sha256']
    assert module.digest(module.local_path(assignments['source_manifest']['path'])) == assignments['source_manifest']['sha256']
    for half, role in [('a', 'SEL-OOD'), ('b', 'GUARD')]:
        dataset = catalog['datasets']['vizwiz_pl_' + half]
        assert dataset['f_role'] == role
        assert dataset['protocol_sha256'] == module.digest(ROOT / dataset['protocol'])
        assert dataset['assignments_sha256'] == module.digest(ROOT / dataset['assignments'])
        assert dataset['parent'] == 'vizwiz_pl_local_validation'


class TestF01Protocol(unittest.TestCase):
    def test_frozen_split_conserves_source(self):
        module.validate(*inputs())

    def test_protocol_sha_and_normative_source(self):
        test_protocol_sha_and_normative_source()

    def test_catalog_halves_bound_to_protected_parent(self):
        test_catalog_halves_bound_to_protected_parent()


def rejection_test(corruption):
    def test(self):
        check_corrupt_split_is_rejected(corruption)
    return test


for corruption in ('drop_qrel', 'duplicate_qrel', 'cross_component', 'alter_text', 'drop_image'):
    setattr(TestF01Protocol, 'test_reject_' + corruption, rejection_test(corruption))


if __name__ == '__main__':
    unittest.main()
