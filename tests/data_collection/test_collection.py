from __future__ import annotations

import copy
import io
import json
from dataclasses import replace
from pathlib import Path
from urllib.error import HTTPError

import pyarrow.parquet as pq
import pytest

from pl_mmb_collection.cli import main
from pl_mmb_collection.config import Config, DiscoveryConfig, HttpConfig, WikimediaConfig, load_config
from pl_mmb_collection.models import Entity, ImageCandidate
from pl_mmb_collection.pipeline.collect import collect
from pl_mmb_collection.pipeline.dedup import deduplicate, file_hash
from pl_mmb_collection.pipeline.discovery import discover
from pl_mmb_collection.pipeline.download import download_images
from pl_mmb_collection.pipeline.filters import license_is_allowed, rejection_reasons
from pl_mmb_collection.pipeline.reporting import statistics, validate
from pl_mmb_collection.preview import write_preview
from pl_mmb_collection.sources.http import ApiError, HttpClient
from pl_mmb_collection.sources.normalization import description_languages, normalize
from pl_mmb_collection.sources.wikidata import claim_values, parse_entity
from pl_mmb_collection.sources.wikimedia import WikimediaSource
from pl_mmb_collection.storage.checkpoint import CheckpointStore
from pl_mmb_collection.storage.parquet import export, record_schema


@pytest.fixture
def payload():
    return json.loads((Path(__file__).parent / "fixtures/commons.json").read_text(encoding="utf-8"))


@pytest.fixture
def entity(payload):
    item = parse_entity(payload["entity"], "Q36", 1, ["P150"])
    item.raw_request = "fixture_entity_request"
    return item


@pytest.fixture
def record(payload, entity):
    return normalize(payload["page"], payload["structured"], entity,
                     ImageCandidate("File:Wawel Castle test.jpg", "wikidata_P18"),
                     ["fixture_request"], "2026-10-20T12:00:00+00:00")


def test_entity_parsing(payload, entity):
    assert entity.label_pl == "Kraków"
    assert entity.images == ["Wawel Castle test.jpg"]
    assert entity.categories == ["Kraków"]
    assert entity.links == ["Q36"]
    assert entity.revision == 123
    deprecated = {"claims": {"P18": [{"rank": "deprecated", "mainsnak": {"snaktype": "value", "datavalue": {"value": "old"}}}]}}
    assert claim_values(deprecated, "P18") == []


def test_normalization_provenance_and_bilingual_captions(record):
    assert record.id == "wikimedia:12345"
    assert record.caption_pl == "Widok zamku na Wawelu od strony Wisły."
    assert record.caption_source == {"pl": "structured_caption", "en": "structured_caption"}
    assert record.caption_quality == {"pl": "A", "en": "A"}
    assert record.creator == "Fixture Photographer"
    assert record.depicts == ["Q189698"]
    assert record.associations[0]["seed_entity"] == "Q36"
    assert record.structured_revision == 98765
    assert record.location is None
    assert "lang" not in record.description


def test_caption_fallback_does_not_infer_language(payload, entity):
    payload["page"]["imageinfo"][0]["extmetadata"]["ImageDescription"]["value"] = "Nieoznaczony język"
    result = normalize(payload["page"], {}, entity, ImageCandidate("file", "commons_category"), [], None)
    assert result.caption_source["pl"] == "entity_description"
    assert result.caption_quality["pl"] == "D"
    assert any(t["language"] == "und" and t["source"] == "image_description" for t in result.text_candidates)
    assert result.caption_pl == "miasto w Polsce"


def test_localized_description_is_quality_b(payload, entity):
    result = normalize(payload["page"], {}, entity, ImageCandidate("file", "wikidata_P18"), [], None)
    assert result.caption_pl == "Zamek na Wawelu."
    assert result.caption_quality["pl"] == "B"
    _, languages = description_languages('<div lang="pl">Zamek <b>królewski</b></div><div lang="en">Castle</div>')
    assert languages == {"pl": "Zamek królewski", "en": "Castle"}


@pytest.mark.parametrize("name,url,expected", [
    ("CC BY-SA 4.0", "https://creativecommons.org/licenses/by-sa/4.0/", True),
    ("CC BY-SA 4.0", "https://creativecommons.org/licenses/by/4.0/", False),
    ("CC BY-SA 4.0", "https://creativecommons.org.evil.example/licenses/by-sa/4.0/", False),
    ("CC0", "https://creativecommons.org/publicdomain/zero/1.0/", True),
    ("Public domain", None, False),
    (None, "https://creativecommons.org/licenses/by/4.0/", False),
    ("CC BY-NC 4.0", "https://creativecommons.org/licenses/by-nc/4.0/", False),
    ("CC BY-SA 4.0", "https://creativecommons.org/licenses/by-sa/4.0/pl/", False),
])
def test_license_pairs(name, url, expected):
    assert license_is_allowed(name, url, Config().filters.allowed_licenses) is expected


def test_quality_filters_and_optional_categories(record):
    config = Config().filters
    assert rejection_reasons(record, config) == []
    record.width, record.height = 100, 50
    assert {"min_width", "min_height", "min_area"} <= set(rejection_reasons(record, config))
    record.width, record.height = 1800, 1200
    record.image_url += "?thumbnail"
    record.image_width, record.image_height = 80, 60
    assert "small_thumbnail" in rejection_reasons(record, config)
    record.image_url = record.image_original_url
    record.mime_type = "image/svg+xml"
    assert "svg" in rejection_reasons(record, config)
    record.mime_type = "image/jpeg"
    config = replace(config, excluded_category_patterns=["Architecture"])
    assert "excluded_category:Architecture" in rejection_reasons(record, config)
    assert not rejection_reasons(record, replace(config, excluded_category_patterns=[]))


def test_exact_dedup_merges_entity_provenance(tmp_path, record):
    with CheckpointStore(tmp_path, Config()) as store:
        store.accept(record, "first")
        alias = copy.deepcopy(record)
        alias.id = "wikimedia:54321"
        alias.source_file_id = "54321"
        alias.image_original_url += "-alias"
        alias.associations[0]["wikidata_qid"] = "Q270"
        store.accept(alias, "second")
        assert store.counts()["records"] == 1
        assert len(store.get_record(record.id).associations) == 2
        assert store.file_result("second")[0] == record.id
        assert statistics(store)["unique_qids"] == 2


def test_url_identity_dedup_without_hash(tmp_path, record):
    record.source_sha1 = None
    with CheckpointStore(tmp_path, Config()) as store:
        store.accept(record, "one")
        alias = copy.deepcopy(record)
        alias.id, alias.source_file_id = "wikimedia:99", "99"
        store.accept(alias, "two")
        assert store.counts()["records"] == 1


def test_empty_and_nonempty_parquet_schema(tmp_path, record):
    with CheckpointStore(tmp_path, Config()) as store:
        path = export(store, True)
        assert pq.read_table(path).num_rows == 0
        assert pq.read_schema(path).equals(record_schema(), check_metadata=True)
        store.accept(record, "one")
        export(store, True)
        assert pq.read_table(path).to_pylist() == [record.to_dict()]
        assert validate(store, Config())["valid"]
        assert json.loads((path.parent / "records.jsonl").read_text(encoding="utf-8"))["id"] == record.id


def test_manifest_rejects_changed_policy(tmp_path):
    with CheckpointStore(tmp_path, Config()):
        pass
    with pytest.raises(ValueError, match="Configuration/version"):
        CheckpointStore(tmp_path, replace(Config(), filters=replace(Config().filters, min_width=999)))


def test_resume_file_failure_and_no_refetch(tmp_path, record, entity):
    class Source:
        calls = 0
        fail = True
        def candidates(self, entity):
            yield ImageCandidate("good", "wikidata_P18")
            yield ImageCandidate("transient", "commons_category")
        def fetch(self, candidate, entity):
            self.calls += 1
            if candidate.title == "transient" and self.fail:
                raise TimeoutError("fixture timeout")
            return copy.deepcopy(record)
    source = Source()
    config = Config(jsonl=True)
    with CheckpointStore(tmp_path, config) as store:
        store.add_entity(entity, discovered=True)
        collect(config, store, source)
        assert store.counts()["failures"] == 1
        assert len(store.entities("collect")) == 1
        assert pq.read_table(tmp_path / "metadata/records.parquet").num_rows == 1
    with CheckpointStore(tmp_path, config) as store:
        source.fail = False
        collect(config, store, source)
        assert source.calls == 3  # Completed file was not fetched on resume.
        assert store.counts()["failures"] == 0
        assert store.entities("collect") == []
        collect(config, store, source)
        assert source.calls == 3


def test_rejection_is_persisted_and_counted_once(tmp_path, record, entity):
    record.width = 1
    class Source:
        calls = 0
        def candidates(self, entity):
            yield ImageCandidate("tiny", "wikidata_P18")
        def fetch(self, candidate, entity):
            self.calls += 1
            return record
    source = Source()
    with CheckpointStore(tmp_path, Config()) as store:
        store.add_entity(entity, discovered=True)
        collect(Config(), store, source)
        store.add_entity(replace(entity, qid="Q270"), discovered=True)
        collect(Config(), store, source)
        assert source.calls == 1
        assert statistics(store)["rejected_files"] == 1


def test_controlled_graph_expansion_and_resume(tmp_path):
    config = replace(Config(), discovery=DiscoveryConfig(seeds=["Q36"], max_entities=3, max_depth=1))
    class Wikidata:
        calls = []
        def entity(self, qid, seed, depth):
            self.calls.append(qid)
            return Entity(qid=qid, seed=seed, depth=depth, links=["Q270", "Q268", "Q31487"])
    source = Wikidata()
    with CheckpointStore(tmp_path, config) as store:
        discover(config, store, source)
        assert store.entity_count() == 3
        assert set(source.calls) == {"Q36", "Q270", "Q268"}
        discover(config, store, source)
        assert len(source.calls) == 3
        assert all(e.depth <= 1 and e.seed == "Q36" for _, e in store.entities())


def test_category_pagination_cycles_depth_and_limits(tmp_path):
    class Client:
        calls = []
        def get(self, endpoint, **params):
            self.calls.append(params)
            if params["cmtitle"] == "Category:Root" and "cmcontinue" not in params:
                return {"query": {"categorymembers": [{"ns": 14, "title": "Category:Sub"}, {"ns": 6, "title": "File:A.jpg"}]}, "continue": {"cmcontinue": "page2", "continue": "-||"}}
            if params["cmtitle"] == "Category:Root":
                return {"query": {"categorymembers": [{"ns": 6, "title": "File:B.jpg"}]}}
            return {"query": {"categorymembers": [{"ns": 14, "title": "Category:Root"}, {"ns": 6, "title": "File:C.jpg"}]}}
    config = replace(Config(), wikimedia=replace(WikimediaConfig(), category_depth=1, max_files_per_entity=3))
    client = Client()
    candidates = list(WikimediaSource(client, config).candidates(Entity(None, "Root", categories=["Root"])))
    assert [c.title for c in candidates] == ["File:A.jpg", "File:B.jpg", "File:C.jpg"]
    assert candidates[-1].category_depth == 1
    assert len(client.calls) == 3


def test_api_parsing_and_missing_structured_data(payload, entity):
    class Client:
        def get(self, endpoint, **params):
            if params["action"] == "query":
                return {"query": {"pages": [payload["page"]]}, "_request_key": "info"}
            return {"entities": {"M12345": {"missing": ""}}, "_request_key": "media"}
    record = WikimediaSource(Client(), Config()).fetch(ImageCandidate("File:test.jpg", "wikidata_P18"), entity)
    assert record.caption_quality["pl"] == "B"
    assert record.depicts == []


def test_image_history_is_not_followed_as_category_continuation(payload, entity):
    class Client:
        calls = []
        def get(self, endpoint, **params):
            self.calls.append(params)
            if params["action"] == "query":
                return {"query": {"pages": [payload["page"]]},
                        "continue": {"iistart": "old-version", "continue": "||categories"}}
            return {"entities": {"M12345": payload["structured"]}}
    client = Client()
    WikimediaSource(client, Config()).fetch(ImageCandidate("file", "wikidata_P18"), entity)
    assert len(client.calls) == 2


def test_http_cache_and_maxlag_retry(tmp_path, monkeypatch):
    client = HttpClient(HttpConfig(interval=0, retries=1, backoff=0), tmp_path)
    calls = []
    def request(url):
        calls.append(url)
        payload = {"error": {"code": "maxlag"}} if len(calls) == 1 else {"entities": {}}
        return io.BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr(client, "_request", request)
    first = client.get("https://example.org/api", action="wbgetentities", ids="Q36")
    second = client.get("https://example.org/api", ids="Q36", action="wbgetentities")
    assert first == second
    assert len(calls) == 2
    assert "maxlag=5" in calls[0]
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_http_retry_after_and_nonretryable_error(tmp_path, monkeypatch):
    client = HttpClient(HttpConfig(interval=0, retries=1, backoff=0), tmp_path)
    sleeps = []
    monkeypatch.setattr("pl_mmb_collection.sources.http.time.sleep", sleeps.append)
    attempts = []
    def operation():
        attempts.append(1)
        if len(attempts) == 1:
            raise HTTPError("https://example.org", 429, "slow down", {"Retry-After": "3"}, None)
        return "success"
    assert client._retry(operation) == "success"
    assert sleeps == [3]
    def forbidden():
        raise HTTPError("https://example.org", 403, "forbidden", {}, None)
    with pytest.raises(HTTPError):
        client._retry(forbidden)


def test_download_resumes_verifies_hash_and_dedup(tmp_path, record):
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", (10, 10), "red").save(buffer, format="JPEG")
    data = buffer.getvalue()
    import hashlib
    record.source_sha1 = hashlib.sha1(data).hexdigest()
    class Client:
        calls = 0
        def download(self, url, path, limit):
            self.calls += 1
            path.write_bytes(data)
    client = Client()
    with CheckpointStore(tmp_path, Config()) as store:
        store.accept(record, "one")
        download_images(Config(), store, client)
        assert store.get_record(record.id).sha256 == hashlib.sha256(data).hexdigest()
        download_images(Config(), store, client)
        assert client.calls == 1
        assert store.counts()["failures"] == 0
        assert deduplicate(store)["count"] == 0


def test_perceptual_dedup_is_optional_and_keeps_records(tmp_path, record):
    with CheckpointStore(tmp_path, Config()) as store:
        for index in range(3):
            item = copy.deepcopy(record)
            item.id = f"wikimedia:{index}"
            item.source_file_id, item.source_sha1 = str(index), None
            item.image_original_url += str(index)
            item.local_path = f"{index}.jpg"
            (tmp_path / item.local_path).write_bytes(str(index).encode())
            store.accept(item, str(index))
        class Hasher:
            def fingerprint(self, path):
                return "a"
            def distance(self, left, right):
                return 0
        assert deduplicate(store, use_phash=False)["count"] == 0
        result = deduplicate(store, use_phash=True, hasher=Hasher())
        assert result["count"] == 2
        assert store.counts()["records"] == 3


def test_cli_offline_end_to_end_20_entities(tmp_path, record, monkeypatch, capsys):
    class Source:
        def discover(self, store):
            for index in range(20):
                store.add_entity(Entity(f"Q{index + 1}", "Q36"), discovered=True)
        def candidates(self, entity):
            yield ImageCandidate("File:" + entity.qid, "wikidata_P18")
        def fetch(self, candidate, entity):
            item = copy.deepcopy(record)
            item.id = "fixture:" + entity.qid
            item.source_file_id = entity.qid
            item.source_sha1 = None
            item.image_original_url += entity.qid
            item.associations[0]["wikidata_qid"] = entity.qid
            return item
    monkeypatch.setattr("pl_mmb_collection.cli.make_source", lambda *args: Source())
    config_path = tmp_path / "config.json"
    output = tmp_path / "run"
    config_path.write_text(json.dumps({"output_dir": str(output), "jsonl": True}), encoding="utf-8")
    for command in ["collect", "collect", "validate", "stats"]:
        assert main([command, "--config", str(config_path)]) == 0
    assert pq.read_table(output / "metadata/records.parquet").num_rows == 20
    stats = json.loads((output / "metadata/stats.json").read_text(encoding="utf-8"))
    assert stats["records"] == stats["unique_qids"] == 20
    assert stats["caption_languages"] == {"pl": 20, "en": 20}


@pytest.mark.parametrize("data", [
    {"discovery": {"seeds": ["Q36 } UNION {}"]}},
    {"discovery": {"max_depth": -1}},
    {"http": {"interval": -1}},
    {"unexpected": True},
])
def test_invalid_config_is_rejected(tmp_path, data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises((ValueError, TypeError)):
        load_config(path)


def test_html_preview_samples_reproducibly_and_escapes_source_text(tmp_path, record):
    record.caption_pl = '</script><script>alert("untrusted source")</script>'
    with CheckpointStore(tmp_path, Config()) as store:
        for index in range(5):
            item = copy.deepcopy(record)
            item.id, item.source_file_id, item.source_sha1 = str(index), str(index), None
            item.image_original_url += str(index)
            store.accept(item, str(index))
        preview = write_preview(store, tmp_path / "preview.html", 3, 17)
        document = preview.read_text(encoding="utf-8")
        assert record.caption_pl not in document
        data = document.split('<script type="application/json" id="records">')[1].split('</script>')[0]
        rows = json.loads(data)
        assert len(rows) == 3
        assert rows[0]["caption_pl"] == record.caption_pl
        assert "Special:Redirect/file/" in rows[0]["preview_url"]
        assert write_preview(store, preview, 3, 17).read_text(encoding="utf-8") == document


def test_context_descriptions_have_explicit_weak_provenance(payload):
    candidate = ImageCandidate("file", "commons_category", category="Category:Poland",
                               raw_requests=("category-request",),
                               context_description={"pl": "Opis kolekcji ze źródła."},
                               context_source="collection_description")
    payload["page"]["imageinfo"][0]["extmetadata"].pop("ImageDescription")
    record = normalize(payload["page"], {}, Entity(None, "Category:Poland"), candidate, [], None)
    assert record.caption_pl == "Opis kolekcji ze źródła."
    assert record.caption_source["pl"] == "collection_description"
    assert record.caption_quality["pl"] == "D"
    assert "category-request" in record.raw_requests
