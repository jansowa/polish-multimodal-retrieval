from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .data import BrowserStore, Dataset, Query, default_datasets

WEB = Path(__file__).parent / "web"
LOG = logging.getLogger(__name__)


def make_server(store: BrowserStore, host: str = "127.0.0.1", port: int = 8766) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def send(self, status: int, body: bytes, content_type: str = "application/json; charset=utf-8") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def json(self, status: int, value) -> None:
            self.send(status, json.dumps(value, ensure_ascii=False).encode())

        def do_GET(self) -> None:
            url = urlparse(self.path)
            args = {key: values[-1] for key, values in parse_qs(url.query, keep_blank_values=True).items()}
            try:
                if url.path == "/app-config.json":
                    self.json(200, dict(mode="live"))
                elif url.path == "/api/catalog":
                    self.json(200, store.catalog())
                elif url.path == "/api/records":
                    query = Query(dataset=args.get("dataset", ""), page=int(args.get("page", "1")),
                        size=int(args.get("size", "24")), q=args.get("q", ""), language=args.get("language", "any"),
                        quality=args.get("quality", ""), license=args.get("license", ""),
                        sort=args.get("sort", "newest"), seed=args.get("seed", "42"))
                    self.json(200, store.records(query))
                elif url.path == "/api/record":
                    self.json(200, store.record(args.get("dataset", ""), args.get("id", "")))
                elif url.path in {"/", "/index.html", "/app.js", "/style.css"}:
                    path = WEB / ("index.html" if url.path == "/" else url.path[1:])
                    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                    self.send(200, path.read_bytes(), mime + "; charset=utf-8")
                else:
                    self.json(404, dict(error="Nie znaleziono strony"))
            except ValueError as exc:
                self.json(400, dict(error=str(exc)))
            except KeyError as exc:
                self.json(404, dict(error=str(exc).strip("'")))
            except sqlite3.Error:
                LOG.exception("Could not read dataset")
                self.json(503, dict(error="Zbiór jest chwilowo niedostępny. Spróbuj odświeżyć."))
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, format: str, *args) -> None:
            LOG.info(format, *args)

    return ThreadingHTTPServer((host, port), Handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only Polish culture dataset browser")
    parser.add_argument("command", choices=["serve", "export"], nargs="?", default="serve")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", type=Path, action="append", help="Explicit dataset directories (repeatable)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--site-dir", type=Path, default=Path("data_collection/browser_site"))
    args = parser.parse_args(argv)
    datasets = ([Dataset(f"dataset-{i}", path.name, path.resolve()) for i, path in enumerate(args.dataset)]
                if args.dataset else default_datasets(args.root.resolve()))
    store = BrowserStore(datasets)
    if args.command == "export":
        from .snapshot import export_snapshot
        print(export_snapshot(store, args.site_dir.resolve()))
        return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    server = make_server(store, args.host, args.port)
    LOG.info("Dataset browser: http://%s:%s", args.host, server.server_port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
