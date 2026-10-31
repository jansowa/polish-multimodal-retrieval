"""Local EN/PL/image review workspace. Uses Python's standard library only."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
import random
import re
from pathlib import Path
import sqlite3
import signal
import subprocess
import sys
import time
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
STATUSES = {"unreviewed", "ok", "fix", "unclear"}
TEXT_FIELDS = ("qry_text", "text", "caption", "negative_caption", "true_caption", "false_caption")
FIELD_LABELS = {"qry_text": "Zapytanie", "text": "Opis", "caption": "Opis pozytywny",
                "negative_caption": "Opis negatywny", "true_caption": "Opis pozytywny", "false_caption": "Opis negatywny"}


def text_projection(column):
    return "json_object(" + ",".join(f"'{k}',json_extract({column},'$.{k}')" for k in TEXT_FIELDS) + ")"


def text_values(raw):
    return {k: v for k, v in json.loads(raw).items() if isinstance(v, str)}


def text_join(raw):
    return "\n\n".join(text_values(raw).values())


def text_hash(source, translated):
    return digest(text_join(source) + "\0" + text_join(translated))


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Store:
    def __init__(self, root: Path = ROOT):
        self.root = root
        self._images_cache = None
        self.annotation_db = root / "reviewer/annotations.sqlite3"
        self.annotation_db.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.annotation_db)) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS reviews (
                run TEXT NOT NULL, unit_id TEXT NOT NULL, text_hash TEXT NOT NULL,
                status TEXT NOT NULL, note TEXT NOT NULL, correction TEXT NOT NULL,
                updated_at TEXT NOT NULL, PRIMARY KEY (run, unit_id, text_hash))""")
            conn.commit()

    def catalog(self) -> dict:
        result = {}
        roots = {"": self.root / "outputs"}
        for plan in (self.root / "reports").glob("*plan.json"):
            try:
                value = json.loads(plan.read_text(encoding="utf-8")).get("output_dir")
                if value:
                    canonical = str(value).replace("\\", "/")
                    if re.match(r"^[A-Za-z]:/", canonical):
                        canonical = "/mnt/" + canonical[0].lower() + canonical[2:]
                    directory = canonical
                    if os.name == "nt" and re.match(r"^/mnt/[a-z]/", directory):
                        directory = directory[5].upper() + ":/" + directory[7:]
                    roots["external-" + digest(canonical)[:10] + "/"] = Path(directory)
            except (OSError, ValueError):
                continue
        for prefix, root in roots.items():
            for db in root.glob("*/*/*/checkpoint.sqlite3"):
                if db.resolve().is_relative_to(root.resolve()):
                    result[prefix + db.parent.relative_to(root).as_posix()] = db
        return result

    def connect(self, run: str):
        if not isinstance(run, str):
            raise ValueError("Nieprawidłowy przebieg")
        db = self.catalog().get(run)
        if db is None:
            raise ValueError("Nieznany przebieg")
        conn = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
        conn.create_function("review_hash", 2, text_hash, deterministic=True)
        conn.create_function("casefold", 1, lambda s: (s or "").casefold(), deterministic=True)
        return conn

    def runs(self) -> list:
        result = []
        for key in self.catalog():
            try:
                with closing(self.connect(key)) as conn:
                    n = conn.execute("SELECT COUNT(*) FROM completed").fetchone()[0]
                run_name, dataset, mode = key.split("/")[-3:]
                result.append({"id": key, "label": dataset + " · " + run_name + (" · tekst" if mode == "text_only" else ""),
                               "count": n, "dataset": dataset, "pilot": "pilot" in run_name})
            except sqlite3.Error:
                continue
        return sorted(result, key=lambda x: (x["pilot"], x["dataset"], x["label"]))

    def image_index(self) -> dict:
        path = self.root / "reviewer/image_index.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"records": {}}

    def records(self, run: str, *, offset=0, limit=None, search="", status="all", random_record=False, uid=None):
        source_sql, translated_sql = text_projection("c.source_json"), text_projection("c.translated_json")
        where, params = [], []
        if uid is not None:
            where.append("c.unit_id=?"); params.append(uid)
        if search:
            where.append(f"instr(casefold(c.unit_id || {source_sql} || {translated_sql}),?)>0")
            params.append(search.casefold())
        if status == "flagged":
            where.append("COALESCE(json_array_length(json_extract(c.metadata_json,'$.translation_quality.flags')),0)>0")
        elif status == "annotated":
            where.append(f"EXISTS (SELECT 1 FROM annotations.reviews r WHERE r.run=? AND r.unit_id=c.unit_id AND r.text_hash=review_hash({source_sql},{translated_sql}) AND (r.status!='unreviewed' OR r.note!='' OR r.correction!=''))")
            params.append(run)
        elif status in STATUSES:
            where.append(f"COALESCE((SELECT r.status FROM annotations.reviews r WHERE r.run=? AND r.unit_id=c.unit_id AND r.text_hash=review_hash({source_sql},{translated_sql})), 'unreviewed')=?")
            params.extend((run, status))
        elif status != "all":
            raise ValueError("Nieprawidłowy filtr")
        clause = " WHERE " + " AND ".join(where) if where else ""
        with closing(self.connect(run)) as conn:
            conn.execute("ATTACH DATABASE ? AS annotations", (self.annotation_db.resolve().as_uri() + "?mode=ro",))
            total = conn.execute("SELECT COUNT(*) FROM completed c" + clause, params).fetchone()[0]
            all_count = conn.execute("SELECT COUNT(*) FROM completed").fetchone()[0]
            reviewed_count = conn.execute(f"SELECT COUNT(*) FROM annotations.reviews r JOIN completed c ON c.unit_id=r.unit_id WHERE r.run=? AND r.status!='unreviewed' AND r.text_hash=review_hash({source_sql},{translated_sql})", (run,)).fetchone()[0]
            if random_record:
                offset = random.randrange(total) if total else 0
            offset = min(max(0, offset), max(0, total - 1))
            paging = " LIMIT ? OFFSET ?" if limit is not None else ""
            rows = conn.execute(f"SELECT c.unit_id, {source_sql}, {translated_sql}, json_extract(c.metadata_json,'$.translation_quality.flags') FROM completed c" + clause + " ORDER BY c.rowid" + paging,
                                params + ([limit, offset] if limit is not None else [])).fetchall()
        with closing(sqlite3.connect(self.annotation_db)) as conn:
            reviews = {(uid, hashed): {"status": status, "note": note, "correction": correction, "updated_at": updated}
                       for uid, hashed, status, note, correction, updated in conn.execute(
                           "SELECT unit_id,text_hash,status,note,correction,updated_at FROM reviews WHERE run=?", (run,))}
        index = self.image_index()
        source_images = self.source_images()
        result = []
        for uid, source, translated, flags in rows:
            en, pl = text_values(source), text_values(translated)
            fields = [{"key": k, "label": FIELD_LABELS[k], "source": v, "translated": pl[k]} for k, v in en.items() if k in pl]
            if not fields:
                continue
            source_text, translated_text = text_join(source), text_join(translated)
            hashed = text_hash(source, translated)
            entry = index["records"].get(uid)
            images = []
            mapped = source_images.get(uid)
            if mapped and mapped["source_sha256"] == digest(en.get("qry_text", "")):
                images = [{"label": im["label"], "slot": n} for n, im in enumerate(mapped["images"]) if self.local_image(im).is_file()]
            if not images and entry and entry["source_sha256"] == digest(source_text):
                images = [{"label": "Obraz docelowy", "slot": 0}]
            result.append({"id": uid, "source": source_text, "translated": translated_text, "text_hash": hashed, "fields": fields,
                           "flags": json.loads(flags) if flags else [],
                           "image": bool(images), "images": images,
                           "review": reviews.get((uid, hashed), {"status": "unreviewed", "note": "", "correction": ""})})
        if limit is None:
            return result
        return {"records": result, "total": total, "all_count": all_count, "reviewed_count": reviewed_count, "offset": offset}

    def source_images(self):
        paths = [self.root / ".cache/absence-v3" / (name + ".jsonl") for name in ("cirr", "fashioniq", "refcoco")]
        stamp = tuple((str(p), p.stat().st_mtime_ns) for p in paths if p.exists())
        if self._images_cache and self._images_cache[0] == stamp:
            return self._images_cache[1]
        result = {}
        for path in paths:
            if not path.exists(): continue
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    unit = json.loads(line)
                    result[unit["id"]] = {"source_sha256": digest(unit["texts"].get("qry_text", "")), "images": unit.get("images", [])}
        self._images_cache = (stamp, result)
        return result

    def local_image(self, image):
        raw = str(image.get("root", "")).replace("\\", "/")
        marker = "/.cache/"
        if marker in raw:
            root = self.root / ".cache" / raw.split(marker, 1)[1]
        else:
            root = self.root / raw
        root = root.resolve()
        path = (root / image["value"]).resolve()
        if not root.is_relative_to(self.root.resolve()) or not path.is_relative_to(root) or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise ValueError("Nieprawidłowa ścieżka obrazu")
        return path

    def image_path(self, uid: str, run="", slot=0) -> Path:
        if run:
            rows = self.records(run, uid=uid, limit=1)["records"]
            if not rows or not any(i["slot"] == slot for i in rows[0]["images"]):
                raise ValueError("Brak zgodnego obrazu")
            entry = self.source_images().get(uid)
            if entry:
                return self.local_image(entry["images"][slot])
        index = self.image_index()
        entry = index["records"].get(uid)
        if not entry:
            raise ValueError("Brak obrazu")
        root = (self.root / index["image_root"]).resolve()
        path = (root / entry["path"]).resolve()
        if not root.is_relative_to(self.root.resolve()) or not path.is_relative_to(root) or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise ValueError("Nieprawidłowa ścieżka obrazu")
        return path

    def save(self, payload: dict) -> dict:
        run, uid = payload.get("run"), payload.get("id")
        record = next(iter(self.records(run, uid=uid, limit=1)["records"]), None)
        if not record or record["text_hash"] != payload.get("text_hash"):
            raise ValueError("Tekst zmienił się lub rekord nie istnieje. Odśwież dane.")
        status, note, correction = payload.get("status"), payload.get("note", ""), payload.get("correction", "")
        if status not in STATUSES or not isinstance(note, str) or not isinstance(correction, str) or max(len(note), len(correction)) > 4000:
            raise ValueError("Nieprawidłowa ocena lub zbyt długa notatka")
        updated = datetime.now(timezone.utc).isoformat()
        with closing(sqlite3.connect(self.annotation_db)) as conn:
            conn.execute("INSERT OR REPLACE INTO reviews VALUES (?,?,?,?,?,?,?)",
                         (run, uid, record["text_hash"], status, note, correction, updated))
            conn.commit()
        return {"status": status, "note": note, "correction": correction, "updated_at": updated}


def make_handler(store: Store):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, body, *, status=200, mime="application/json; charset=utf-8"):
            raw = json.dumps(body, ensure_ascii=False).encode("utf-8") if not isinstance(body, bytes) else body
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            try:
                if url.path == "/api/health":
                    return self.respond({"app": "pl-translation-review", "indexed_images": len(store.image_index()["records"])})
                if url.path == "/api/runs":
                    return self.respond(store.runs())
                if url.path == "/api/records":
                    return self.respond(store.records(query.get("run", [""])[0]))
                if url.path == "/api/page":
                    return self.respond(store.records(query.get("run", [""])[0],
                        offset=max(0, int(query.get("offset", ["0"])[0])), limit=min(200, max(1, int(query.get("limit", ["100"])[0]))),
                        search=query.get("search", [""])[0], status=query.get("filter", ["all"])[0], random_record=query.get("random", [""])[0] == "1"))
                if url.path == "/api/image":
                    path = store.image_path(query.get("id", [""])[0], query.get("run", [""])[0], int(query.get("slot", ["0"])[0]))
                    return self.respond(path.read_bytes(), mime=mimetypes.guess_type(path.name)[0] or "image/jpeg")
                if url.path == "/api/export":
                    run = query.get("run", [""])[0]
                    return self.respond([{"run": run, **r} for r in store.records(run, status="annotated")])
                static = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}.get(url.path)
                if static:
                    return self.respond((HERE / static).read_bytes(), mime={"index.html": "text/html; charset=utf-8",
                        "app.js": "text/javascript; charset=utf-8", "style.css": "text/css; charset=utf-8"}[static])
                return self.respond({"error": "Nie znaleziono"}, status=404)
            except (ValueError, OSError, sqlite3.Error) as exc:
                self.respond({"error": str(exc)}, status=400)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if urlparse(self.path).path != "/api/annotation" or not 0 < length <= 32768 or self.headers.get_content_type() != "application/json":
                    raise ValueError("Nieprawidłowe żądanie")
                raw = self.rfile.read(length)
                origin = self.headers.get("Origin")
                allowed = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
                if origin and origin not in allowed:
                    return self.respond({"error": "Niedozwolone źródło żądania"}, status=403)
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("Oczekiwano obiektu JSON")
                return self.respond(store.save(payload))
            except (ValueError, OSError, sqlite3.Error) as exc:
                self.respond({"error": str(exc)}, status=400)

        def log_message(self, format, *args):
            if len(args) > 1 and str(args[1]) != "200":
                super().log_message(format, *args)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--restart", action="store_true", help="Restart this local server in WSL, keeping annotations")
    args = parser.parse_args()
    (ROOT / "reports").mkdir(exist_ok=True)
    if args.detach:
        if args.restart:
            if sys.platform != "linux":
                parser.error("--restart is supported in WSL/Linux")
            # Stop only this exact workspace's background server on this port.
            script = str(Path(__file__).resolve()).encode()
            for entry in Path("/proc").iterdir():
                if not entry.name.isdigit() or int(entry.name) == os.getpid():
                    continue
                try:
                    command = (entry / "cmdline").read_bytes().split(b"\0")
                    if (entry.stat().st_uid == os.getuid() and script in command and b"--detach" not in command
                            and b"--port" in command and command[command.index(b"--port") + 1] == str(args.port).encode()):
                        os.kill(int(entry.name), signal.SIGTERM)
                except (OSError, ValueError, IndexError):
                    continue
            time.sleep(0.5)
        try:
            with urlopen(f"http://127.0.0.1:{args.port}/api/health", timeout=2) as response:
                if json.load(response).get("app") == "pl-translation-review":
                    print(f"Already running: http://localhost:{args.port}")
                    return
        except (OSError, ValueError):
            pass
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--port", str(args.port)]
        if args.build_index:
            command.append("--build-index")
        with (ROOT / "reports/reviewer_server.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        for _ in range(40):
            if child.poll() is not None:
                raise RuntimeError("Server exited; see reports/reviewer_server.log")
            try:
                with urlopen(f"http://127.0.0.1:{args.port}/api/health", timeout=1) as response:
                    if json.load(response).get("app") == "pl-translation-review":
                        print(f"Ready: http://localhost:{args.port} (PID {child.pid})")
                        return
            except OSError:
                time.sleep(0.25)
        raise RuntimeError("Server did not become ready; see reports/reviewer_server.log")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(Store()))
    if args.build_index and not (HERE / "image_index.json").exists():
        with (ROOT / "reports/reviewer_index.log").open("a", encoding="utf-8") as log:
            subprocess.Popen([sys.executable, "-u", str(HERE / "build_image_index.py")], cwd=ROOT,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True, close_fds=True)
    print(f"Review workspace: http://localhost:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
