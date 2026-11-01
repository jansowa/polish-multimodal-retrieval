"""CPU checks and read-only HTTP smoke test. Never submits real annotations."""
import importlib.util
import io
import json
import subprocess
import sys
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'tests'))
from pl_mmb_training.series_20261023 import OUT, VN, file_sha, freeze, read


def main():
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_series_20261023.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    subprocess.run(['node','--check',str(ROOT/'experiments/series_annotation/app.js')], check=True)
    expected = read(VN/'extraction.json')['files']['native_rgb.npy']
    actual = file_sha(VN/'native_rgb.npy')
    if actual != expected:
        raise ValueError('Existing native VN RGB array changed')
    freeze(OUT/'annotation/asset_verification.json', dict(native_rgb_sha256=actual,
        source_extraction_sha256=file_sha(VN/'extraction.json'), bytes=(VN/'native_rgb.npy').stat().st_size,
        role='Existing frozen native RGB asset; no copies, no cache writes'))
    spec = importlib.util.spec_from_file_location('series_annotation_smoke',ROOT/'experiments/series_annotation.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    collection = module.Collection()
    server = ThreadingHTTPServer(('127.0.0.1',0), module.handler(collection))
    thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    codes = {}
    try:
        for path in ('/','/app.js','/style.css','/api/state','/api/tasks?mode=query','/api/tasks?mode=audit','/asset/query/0','/asset/audit/0'):
            with urllib.request.urlopen(base+path, timeout=60) as response:
                payload = response.read(); codes[path] = response.status
            if path.startswith('/asset/'):
                from PIL import Image
                with Image.open(io.BytesIO(payload)) as image:
                    if image.size != (224,224):
                        raise ValueError('Unexpected native preview dimensions')
            if path == '/api/tasks?mode=query':
                tasks = json.loads(payload)
                if len(tasks) != 150 or any('caption' in r for r in tasks):
                    raise ValueError('Incorrect query API or caption disclosure')
            if path == '/api/tasks?mode=audit' and len(json.loads(payload)) != 100:
                raise ValueError('Incorrect audit sample count')
    finally:
        server.shutdown(); server.server_close(); thread.join()
    for name, sha in read(OUT/'preservation_before.json').items():
        if file_sha(ROOT/name) != sha:
            raise ValueError(f'Protected input changed: {name}')
    if list((OUT/'annotation').glob('human_*.jsonl')):
        raise ValueError('Unexpected production human annotations during read-only validation')
    validation = dict(tests_total=result.testsRun, tests_passed=result.testsRun,
        scope='Provenance, immutable writes, deterministic image sampling, Wiki training gate, multi-positive NDCG, paired identities, query privacy, append-only writes and explicit audit fields.',
        javascript_syntax='passed (node --check)', http_smoke='Eight real loopback GET endpoints passed; both image previews decoded as 224x224; no POST to real annotation storage.',
        http_statuses=codes, existing_VN_asset_sha256=actual, protected_snapshot_SHA_verified=True,
        browser_visual_QA='not performed; no browser automation used', production_annotations_created=0)
    freeze(OUT/'validation.json',validation)
    print(json.dumps(validation,indent=2))


if __name__ == '__main__':
    main()
