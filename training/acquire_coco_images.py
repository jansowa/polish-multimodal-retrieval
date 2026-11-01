"""Acquire only the COCO images referenced by the translated M-BEIR source."""
import concurrent.futures
import json
import sys
import urllib.request
import threading
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pl_mmb_training.common import write_json


def main():
    from PIL import Image
    plan = json.loads((ROOT / 'reports/absence_extension_plan.json').read_text())
    info = plan['original_files']['query/test/mbeir_mscoco_task3_test.jsonl']
    rows = [json.loads(line) for line in Path(info['path']).read_text().splitlines()]
    names = sorted({Path(row['query_img_path']).name for row in rows})
    dest = Path.home() / '.cache/huggingface/pl_mmb_training/coco_images'
    dest.mkdir(parents=True, exist_ok=True)
    connections = threading.local()
    def acquire(name):
        path = dest / name
        if path.is_file():
            with Image.open(path) as image:
                image.verify()
            return name
        url = 'https://s3.amazonaws.com/images.cocodataset.org/val2014/' + name
        tmp = path.with_suffix('.tmp')
        import time
        for attempt in range(3):
            try:
                if not hasattr(connections, 'session'):
                    connections.session = requests.Session()
                response = connections.session.get(url, timeout=45)
                response.raise_for_status()
                payload = response.content
                import io
                with Image.open(io.BytesIO(payload)) as image:
                    image.verify()
                tmp.write_bytes(payload)
                tmp.replace(path)
                return name
            except (OSError, requests.RequestException):
                if attempt == 2:
                    raise
                time.sleep(1)
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        pending = {pool.submit(acquire, name): name for name in names}
        for index, future in enumerate(concurrent.futures.as_completed(pending), 1):
            try:
                future.result()
            except Exception as exc:
                errors.append(dict(image=pending[future], error=str(exc)))
            if index % 100 == 0 or index == len(names):
                print(f'COCO images {index}/{len(names)}; errors={len(errors)}', flush=True)
    write_json(dest / 'acquisition.json', dict(expected=len(names), errors=errors,
               source='official COCO S3 bucket', filename=info['path'], source_sha256=info['sha256']))
    if errors:
        raise RuntimeError(f'{len(errors)} COCO images could not be acquired; see acquisition.json')


if __name__ == '__main__':
    main()
