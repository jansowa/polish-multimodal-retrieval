from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import replace
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from pl_mmb_collection.browser.data import BrowserStore, Dataset, Query, summary
from pl_mmb_collection.browser.server import make_server
from pl_mmb_collection.browser.snapshot import export_snapshot
from pl_mmb_collection.config import Config
from pl_mmb_collection.models import DatasetRecord
from pl_mmb_collection.storage.checkpoint import CheckpointStore


@pytest.fixture
def browser(tmp_path):
    config = Config()
    with CheckpointStore(tmp_path / "dataset", config) as store:
        for index in range(31):
            record = DatasetRecord(id=f"test:{index:02d}", source="wikimedia",
                source_page_url=f"https://commons.wikimedia.org/wiki/File:Kraków_{index}.jpg",
                image_url=f"https://upload.wikimedia.org/{index}.jpg",
                image_original_url=f"https://upload.wikimedia.org/{index}.jpg",
                image_filename=f"File:Kraków {index}.jpg", source_file_id=str(index),
                entity_label_pl="Kraków" if index < 20 else "Łódź", wikidata_qid="Q31487",
                caption_pl=f"Widok miasta Kraków {index}.", caption_en=f"City view {index}" if index % 2 == 0 else None,
                caption_quality={"pl": "A" if index < 10 else "D", "en": "B" if index % 2 == 0 else None},
                caption_source={"pl": "structured_caption", "en": "image_description"},
                license="CC0" if index < 15 else "CC BY-SA 4.0", width=1000 + index, height=1000,
                retrieved_at=f"2026-10-20T12:00:{index:02d}+00:00", categories=["Category:Życie codzienne"])
            store.accept(record, record.image_filename)
    return BrowserStore([Dataset("test", "Zbiór testowy", tmp_path / "dataset")])


def test_pages_are_stable_disjoint_complete_and_clamped(browser):
    first = browser.records(Query("test", size=12))
    second = browser.records(Query("test", page=2, size=12))
    last = browser.records(Query("test", page=999, size=12))
    assert (first["total"], first["pages"]) == (31, 3)
    assert last["page"] == 3 and len(last["items"]) == 7
    ids = [row["id"] for page in [first, second, last] for row in page["items"]]
    assert len(ids) == len(set(ids)) == 31
    assert first["items"][0]["id"] == "test:30"
    assert browser.records(Query("test", size=12)) == first


def test_unicode_search_quality_language_and_license_filters(browser):
    assert browser.records(Query("test", q="łÓDŹ"))["total"] == 11
    assert browser.records(Query("test", q="Życie"))["total"] == 31
    assert browser.records(Query("test", language="both"))["total"] == 16
    assert browser.records(Query("test", quality="A", language="pl"))["total"] == 10
    assert browser.records(Query("test", quality="A", language="en"))["total"] == 0
    assert browser.records(Query("test", quality="B", language="en", license="CC0"))["total"] == 8
    assert browser.records(Query("test", q="' OR 1=1 --"))["total"] == 0


def test_random_sort_is_stable_across_pages(browser):
    query = Query("test", sort="random", seed="sample")
    first = browser.records(query)
    assert browser.records(query) == first
    assert browser.records(replace(query, seed="different"))["items"] != first["items"]
    second = browser.records(replace(query, page=2))
    assert not ({r["id"] for r in first["items"]} & {r["id"] for r in second["items"]})


def test_detail_and_catalog_do_not_change_checkpoint(browser):
    path = browser.datasets["test"].directory / "checkpoint.sqlite3"
    before = path.read_bytes()
    catalog = browser.catalog()
    assert catalog["datasets"][0]["direct_count"] == 21
    assert catalog["datasets"][0]["count"] == 31
    assert browser.record("test", "test:00")["caption_pl"] == "Widok miasta Kraków 0."
    assert browser.records(Query("test"))["total"] == 31
    with browser.connect("test") as conn:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM records")
    assert path.read_bytes() == before
    with pytest.raises(KeyError):
        browser.record("test", "missing")


@pytest.mark.parametrize("patch", [{"size": 10000}, {"page": 0}, {"sort": "id; DROP TABLE records"},
                                   {"quality": "Z"}, {"q": "x" * 201}])
def test_invalid_page_filters_rejected(browser, patch):
    with pytest.raises(ValueError):
        browser.records(Query("test", **patch))


def test_snapshot_exports_every_record_without_database(browser, tmp_path_factory):
    # Keep nested archive paths below Windows' legacy path-length limit.
    tmp_path = tmp_path_factory.mktemp("s")
    dist = export_snapshot(browser, tmp_path / "site")
    catalog = json.loads((dist / "site-data/catalog.json").read_text(encoding="utf-8"))
    assert catalog["mode"] == "snapshot"
    index = json.loads((dist / "site-data/test/index.json").read_text(encoding="utf-8"))
    assert len(index) == 31
    assert "życie" in index[0]["search_text"]
    for row in index:
        detail = dist / "site-data/test/records" / (row["detail_key"] + ".json")
        assert json.loads(detail.read_text(encoding="utf-8"))["id"] == row["id"]
    assert not list(dist.rglob("*.sqlite3"))
    assert (dist / "index.html").is_file()


def test_browser_http_routes_errors_and_path_boundaries(browser):
    server = make_server(browser, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/records?dataset=test&page=2&size=12") as response:
            assert json.load(response)["page"] == 2
        with urlopen(base + "/") as response:
            assert b"<!doctype html>" in response.read()
        with urlopen(base + "/app-config.json") as response:
            assert json.load(response)["mode"] == "live"
        for path, status in [("/api/records?dataset=test&size=999", 400),
                             ("/api/records?dataset=../../private", 404), ("/../checkpoint.sqlite3", 404)]:
            with pytest.raises(HTTPError) as error:
                urlopen(base + path)
            assert error.value.code == status
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_summary_does_not_allow_script_urls():
    row = summary(dict(id="x", source="other", image_url="javascript:alert(1)",
                       source_page_url="javascript:alert(1)", license_url="data:text/html,evil"))
    assert row["thumbnail_url"] is None
    assert row["license_url"] is None
    assert row["source_page_url"] is None
