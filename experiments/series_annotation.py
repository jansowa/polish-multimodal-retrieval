"""Loopback-only human query/audit collection; no model-generated suggestions."""
import argparse
import io
import json
import mimetypes
import os
import secrets
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pl_mmb_training.series_20261023 import OUT, ROOT, VN, file_sha, read, rows, local_path

STATIC = Path(__file__).with_suffix('')


def validate_annotation(mode, value):
    if not isinstance(value, dict):
        raise ValueError('Oczekiwano obiektu JSON.')
    if mode == 'query':
        queries = value.get('queries')
        if not isinstance(queries, list) or not 1 <= len(queries) <= 2:
            raise ValueError('Wpisz jedno lub dwa krótkie zapytania.')
        if any(not isinstance(q, str) or not 1 <= len(q.strip()) <= 200 for q in queries):
            raise ValueError('Każde zapytanie musi mieć od 1 do 200 znaków.')
        return dict(queries=[q.strip() for q in queries])
    if mode == 'audit':
        score, mismatch, comment = value.get('grounding'), value.get('mismatch'), value.get('comment', '')
        if type(score) is not int or score not in range(5) or type(mismatch) is not bool:
            raise ValueError('Wybierz grounding 0–4 i mismatch tak/nie.')
        if not isinstance(comment, str) or len(comment) > 4000:
            raise ValueError('Komentarz może mieć do 4000 znaków.')
        return dict(grounding=score, mismatch=mismatch, comment=comment.strip())
    raise ValueError('Nieznany tryb.')


class Collection:
    def __init__(self, folder=None):
        self.folder = Path(folder) if folder else OUT/('annotation_v2' if (OUT/'annotation_v2/sampling.json').exists() else 'annotation')
        self.sampling = read(self.folder / 'sampling.json')
        self.tasks = {}
        self.hashes = {}
        for mode in ('query', 'audit'):
            path = self.folder / f'{mode}_tasks.jsonl'
            self.hashes[mode] = file_sha(path)
            if self.hashes[mode] != self.sampling[mode]['manifest_sha256']:
                raise ValueError('Task manifest SHA mismatch')
            self.tasks[mode] = rows(path)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.rgb = None
        self.asset_verified = False

    def public_tasks(self, mode):
        if mode not in self.tasks:
            raise ValueError('Nieznany tryb.')
        result = []
        completed = set()
        output = self.folder / f'human_{mode}.jsonl'
        with self.lock:
            if output.exists():
                completed = {r['task_id'] for r in rows(output)}
        for i, task in enumerate(self.tasks[mode]):
            public = dict(index=i, id=task['id'], dataset=task['dataset'], completed=task['id'] in completed,
                          image=f'/asset/{mode}/{i}')
            if mode == 'audit':
                public['caption'] = task['text_pl']
            result.append(public)
        return result

    def save(self, mode, index, value):
        if mode not in self.tasks or type(index) is not int or not 0 <= index < len(self.tasks[mode]):
            raise ValueError('Nieznany obraz.')
        annotation = validate_annotation(mode, value)
        task = self.tasks[mode][index]
        row = dict(schema=1, timestamp=datetime.now(timezone.utc).isoformat(), human_entered=True,
                   task_id=task['id'], image_id=task['target_id'], dataset=task['dataset'],
                   component=task['component'], mode=mode, task_manifest_sha256=self.hashes[mode], **annotation)
        with self.lock:
            with (self.folder / f'human_{mode}.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        return row

    def image(self, mode, index):
        if mode not in self.tasks or not 0 <= index < len(self.tasks[mode]):
            raise ValueError('Nieznany obraz.')
        import numpy as np
        from PIL import Image
        task = self.tasks[mode][index]
        if task['asset'] in ('wiki_original','wiki_pilot_preview'):
            path=local_path(task['image'])
            if file_sha(path)!=task['sha256']:raise ValueError('Wiki asset changed')
            with Image.open(path) as original:
                from PIL import ImageOps
                prepared=ImageOps.exif_transpose(original).convert('RGB');prepared.thumbnail((960,960))
                stream=io.BytesIO();prepared.save(stream,format='JPEG',quality=94)
                return stream.getvalue()
        if task['asset'] != 'vn_native_rgb':
            raise ValueError('Nieobsługiwany asset.')
        with self.lock:
            if self.rgb is None:
                verification = read(self.folder / 'asset_verification.json')
                if file_sha(VN / 'native_rgb.npy') != verification['native_rgb_sha256']:
                    raise ValueError('Frozen native RGB SHA mismatch')
                self.asset_verified = True
                self.rgb = np.load(VN / 'native_rgb.npy', mmap_mode='r')
            array = np.array(self.rgb[task['cache_index']], copy=True)
        if array.ndim == 3 and array.shape[0] == 3:
            array = array.transpose(1, 2, 0)
        stream = io.BytesIO()
        Image.fromarray(array).save(stream, format='JPEG', quality=94)
        return stream.getvalue()


def handler(collection):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, body, content_type='application/json; charset=utf-8'):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def trusted(self):
            expected = f'127.0.0.1:{self.server.server_port}'
            return self.headers.get('Host') == expected and self.headers.get('Origin', f'http://{expected}') == f'http://{expected}'

        def do_GET(self):
            if not self.trusted():
                return self.respond(403, dict(error='Loopback origin required'))
            url = urlparse(self.path)
            try:
                if url.path == '/api/state':
                    return self.respond(200, dict(token=collection.token, sampling=collection.sampling))
                if url.path == '/api/tasks':
                    mode = parse_qs(url.query).get('mode', ['query'])[0]
                    return self.respond(200, collection.public_tasks(mode))
                if url.path.startswith('/asset/'):
                    _, _, mode, index = url.path.split('/')
                    return self.respond(200, collection.image(mode, int(index)), 'image/jpeg')
                name = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}.get(url.path)
                if name:
                    return self.respond(200, (STATIC / name).read_bytes(), mimetypes.guess_type(name)[0] + '; charset=utf-8')
                return self.respond(404, dict(error='Nie znaleziono.'))
            except (ValueError, KeyError, IndexError):
                return self.respond(400, dict(error='Nieprawidłowe żądanie.'))

        def do_POST(self):
            # Consume bounded request bodies before rejecting credentials. Otherwise
            # Windows can reset the connection while the client is still sending.
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 16384:
                    raise ValueError('Nieprawidłowy rozmiar żądania.')
            except ValueError:
                return self.respond(400, dict(error='Nieprawidłowy rozmiar żądania.'))
            payload = self.rfile.read(size)
            if not self.trusted() or self.headers.get('X-Annotation-Token') != collection.token:
                return self.respond(403, dict(error='Invalid local annotation token'))
            if self.path != '/api/save' or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self.respond(400, dict(error='Expected JSON save request'))
            try:
                value = json.loads(payload)
                if not isinstance(value, dict):
                    raise ValueError('Oczekiwano obiektu JSON.')
                collection.save(value.get('mode'), value.get('index'), value)
                return self.respond(200, dict(saved=True))
            except (ValueError, KeyError, TypeError) as error:
                return self.respond(400, dict(error=str(error)))

        def log_message(self, fmt, *args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8768)
    args = parser.parse_args()
    collection = Collection()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(collection))
    print(f'Adnotacje: http://127.0.0.1:{args.port} — {collection.sampling["status"]}.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
