import hashlib
from contextlib import closing
import importlib.util
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

spec = importlib.util.spec_from_file_location("review_server", Path(__file__).resolve().parents[1] / "reviewer/server.py")
reviewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reviewer)


class ReviewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = "test-run/mscoco_t2i/with_image"
        directory = self.root / "outputs" / self.run
        directory.mkdir(parents=True)
        self.db = directory / "checkpoint.sqlite3"
        with closing(sqlite3.connect(self.db)) as conn:
            conn.execute("CREATE TABLE completed(unit_id TEXT, source_json TEXT, translated_json TEXT, metadata_json TEXT)")
            conn.execute("INSERT INTO completed VALUES(?,?,?,?)", ("MSCOCO_t2i:0", json.dumps({"qry_text": "A dog."}),
                json.dumps({"qry_text": "Pies."}), json.dumps({"translation_quality": {"flags": []}})))
            conn.commit()
        self.store = reviewer.Store(self.root)
        (self.root / "images").mkdir()
        (self.root / "images/dog.jpg").write_bytes(b"fake-jpeg-content")
        self.index = {"image_root": "images", "records": {"MSCOCO_t2i:0": {
            "path": "dog.jpg", "source_sha256": reviewer.digest("A dog.")}}}
        self.write_index()

    def tearDown(self):
        self.temp.cleanup()

    def write_index(self):
        (self.root / "reviewer/image_index.json").write_text(json.dumps(self.index), encoding="utf-8")

    def test_mapping_and_stale_source(self):
        self.assertTrue(self.store.records(self.run)[0]["image"])
        self.index["records"]["MSCOCO_t2i:0"]["source_sha256"] = reviewer.digest("A cat.")
        self.write_index()
        self.assertFalse(self.store.records(self.run)[0]["image"])

    def test_annotations_persist_without_changing_translation(self):
        before = self.db.read_bytes()
        record = self.store.records(self.run)[0]
        payload = {"run": self.run, "id": record["id"], "text_hash": record["text_hash"],
                   "status": "fix", "note": "Uwaga", "correction": "Propozycja"}
        self.store.save(payload)
        again = reviewer.Store(self.root).records(self.run)[0]
        self.assertEqual(again["review"]["correction"], "Propozycja")
        self.assertEqual(again["review"]["status"], "fix")
        self.assertEqual(before, self.db.read_bytes())
        with self.assertRaises(ValueError):
            self.store.save({**payload, "text_hash": "old"})

    def test_image_paths_stay_inside_cache(self):
        self.assertEqual(self.store.image_path("MSCOCO_t2i:0"), self.root / "images/dog.jpg")
        self.index["records"]["MSCOCO_t2i:0"]["path"] = "../../outside.jpg"
        self.write_index()
        with self.assertRaises(ValueError):
            self.store.image_path("MSCOCO_t2i:0")

    def add_run(self, run, records, output_root=None):
        directory = (output_root or self.root / "outputs") / run
        directory.mkdir(parents=True)
        with closing(sqlite3.connect(directory / "checkpoint.sqlite3")) as conn:
            conn.execute("CREATE TABLE completed(unit_id TEXT PRIMARY KEY, source_json TEXT, translated_json TEXT, metadata_json TEXT)")
            for uid, en, pl in records:
                conn.execute("INSERT INTO completed VALUES(?,?,?,?)", (uid, json.dumps(en), json.dumps(pl), '{}'))
            conn.commit()

    def test_contrastive_pairs_and_hash_cover_both_captions(self):
        run = "new/aro_visual_relation/text_only"
        self.add_run(run, [("ARO:0", {"true_caption": "dog left", "false_caption": "dog right"},
                           {"true_caption": "pies po lewej", "false_caption": "pies po prawej"})])
        record = self.store.records(run)[0]
        self.assertEqual([f["key"] for f in record["fields"]], ["true_caption", "false_caption"])
        self.assertIn("pies po prawej", record["translated"])
        payload = {"run": run, "id": record["id"], "text_hash": record["text_hash"], "status": "ok"}
        self.store.save(payload)
        db = self.store.catalog()[run]
        with closing(sqlite3.connect(db)) as conn:
            conn.execute("UPDATE completed SET translated_json=?", (json.dumps({"true_caption": "pies po lewej", "false_caption": "kot po prawej"}),))
            conn.commit()
        self.assertEqual(self.store.records(run)[0]["review"]["status"], "unreviewed")
        self.assertEqual(self.store.records(run, limit=10, status="unreviewed")["reviewed_count"], 0)

    def test_external_outputs_and_retrieval_text(self):
        external = self.root / "external-output"
        self.add_run("new/mbeir_visualnews/text_only", [("MBeir:0:9", {"text": "A caption"}, {"text": "Opis"})], external)
        (self.root / "reports").mkdir()
        (self.root / "reports/extension_plan.json").write_text(json.dumps({"output_dir": str(external)}), encoding="utf-8")
        run = next(k for k in self.store.catalog() if "mbeir_visualnews" in k)
        self.assertEqual(self.store.records(run)[0]["translated"], "Opis")
        self.assertIn("mbeir_visualnews", next(r["label"] for r in self.store.runs() if r["id"] == run))

    def test_page_search_random_and_annotations(self):
        run = "new/sugarcrepe/text_only"
        self.add_run(run, [(f"Sugar:{i}", {"caption": f"dog {i}", "negative_caption": f"cat {i}"},
                                  {"caption": f"żółw {i}", "negative_caption": f"kot {i}"}) for i in range(205)])
        page = self.store.records(run, offset=100, limit=100)
        self.assertEqual(page["total"], 205)
        self.assertEqual(len(page["records"]), 100)
        self.assertEqual(page["records"][0]["id"], "Sugar:100")
        record = page["records"][0]
        self.store.save({"run": run, "id": record["id"], "text_hash": record["text_hash"], "status": "fix"})
        self.assertEqual(self.store.records(run, limit=100, status="fix")["total"], 1)
        self.assertEqual(self.store.records(run, limit=100, status="unreviewed")["total"], 204)
        self.assertEqual(self.store.records(run, limit=100, search="ŻÓŁW 204")["records"][0]["id"], "Sugar:204")
        random_page = self.store.records(run, limit=1, random_record=True)
        self.assertEqual(len(random_page["records"]), 1)
        self.assertLess(random_page["offset"], 205)
        self.assertEqual(len(self.store.records(run, status="annotated")), 1)

    def test_query_and_target_images_and_stale_mapping(self):
        run = "new/cirr/with_image"
        source = "<|image_1|> Change the bottle"
        self.add_run(run, [("CIRR:0", {"qry_text": source}, {"qry_text": "Zmień butelkę"})])
        cache = self.root / ".cache/absence-v3"
        cache.mkdir(parents=True)
        (cache / "cirr.jsonl").write_text(json.dumps({"id": "CIRR:0", "texts": {"qry_text": source}, "images": [
            {"label": "query_image[0]", "root": "images", "value": "dog.jpg"},
            {"label": "groundtruth_target_image", "root": "images", "value": "dog.jpg"}]}), encoding="utf-8")
        self.assertEqual(len(self.store.records(run)[0]["images"]), 2)
        self.assertEqual(self.store.image_path("CIRR:0", run, 1), self.root / "images/dog.jpg")
        with self.assertRaises(ValueError):
            self.store.image_path("CIRR:0", run, -1)
        with closing(sqlite3.connect(self.store.catalog()[run])) as conn:
            conn.execute("UPDATE completed SET source_json=?", (json.dumps({"qry_text": "different"}),))
            conn.commit()
        self.assertFalse(self.store.records(run)[0]["image"])
        with self.assertRaises(ValueError):
            self.store.image_path("CIRR:0", run, 0)

    def test_http_read_save_export_and_origin(self):
        server = reviewer.ThreadingHTTPServer(("127.0.0.1", 0), reviewer.make_handler(self.store))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        def get(path):
            with urlopen(base + path) as response:
                return json.load(response)
        try:
            self.assertEqual(get("/api/health")["indexed_images"], 1)
            self.assertEqual(get("/api/runs")[0]["count"], 1)
            rows = get("/api/records?run=" + self.run)
            self.assertEqual(get("/api/page?run=" + self.run + "&limit=1")["records"], rows)
            payload = {"run": self.run, "id": rows[0]["id"], "text_hash": rows[0]["text_hash"], "status": "ok"}
            request = Request(base + "/api/annotation", data=json.dumps(payload).encode(),
                              headers={"Content-Type": "application/json", "Origin": base}, method="POST")
            with urlopen(request) as response:
                self.assertEqual(json.load(response)["status"], "ok")
            self.assertEqual(get("/api/export?run=" + self.run)[0]["review"]["status"], "ok")
            request.add_header("Origin", "https://unrelated.example")
            with self.assertRaises(HTTPError) as caught:
                urlopen(request)
            self.assertEqual(caught.exception.code, 403)
            with self.assertRaises(HTTPError):
                urlopen(base + "/.env")
            with urlopen(base + "/api/image?id=MSCOCO_t2i:0") as response:
                self.assertEqual(response.read(), b"fake-jpeg-content")
        finally:
            server.shutdown();server.server_close();thread.join()


if __name__ == "__main__":
    unittest.main()
