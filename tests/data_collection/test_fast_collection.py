from __future__ import annotations

import copy
import gzip
import io
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from urllib.error import HTTPError

import pytest

from pl_mmb_collection.config import Config, DiscoveryConfig, HttpConfig, WikimediaConfig
from pl_mmb_collection.config import PipelineConfig
from pl_mmb_collection.models import Entity, ImageCandidate
from pl_mmb_collection.pipeline.collect import collect
from pl_mmb_collection.pipeline.discovery import discover
from pl_mmb_collection.pipeline.hybrid import run
from pl_mmb_collection.pipeline.imports import import_seeds
from pl_mmb_collection.pipeline.expansion import expand
from pl_mmb_collection.sources.batching import batches
from pl_mmb_collection.sources.http import HttpClient
from pl_mmb_collection.sources.rate_limit import RateLimiter, RetryLater, scope_for
from pl_mmb_collection.sources.wikidata import WikidataSource
from pl_mmb_collection.sources.wikimedia import WikimediaSource
from pl_mmb_collection.sources.normalization import normalize
from pl_mmb_collection.storage.checkpoint import CheckpointStore, atomic_json
from pl_mmb_collection.storage.lease import CollectorLease
from pl_mmb_collection.pipeline.reporting import validate
from pl_mmb_collection.cli import main


class FakeClient:
    """Realistic, batched API fixtures; counts only simulated network requests."""
    def __init__(self, count=50):
        fixture = json.loads((Path(__file__).parent / "fixtures/commons.json").read_text(encoding="utf-8"))
        self.pages = {}
        for index in range(count):
            page = copy.deepcopy(fixture["page"])
            page["pageid"] = 1000 + index
            page["title"] = f"File:Culture {index}.jpg"
            page["imageinfo"][0]["url"] += str(index)
            page["imageinfo"][0]["sha1"] = str(index)
            self.pages[page["title"]] = page
        self.calls = []
        self.items = {}
        self.error = None
        self.metrics = {"requests": 0}

    def save_metrics(self):
        pass

    def cached(self, endpoint, **params):
        return None

    def item_get(self, namespace, key):
        return copy.deepcopy(self.items.get((namespace, key)))

    def item_put(self, namespace, key, value):
        self.items[namespace, key] = copy.deepcopy(value)

    def get(self, endpoint, **params):
        self.calls.append((endpoint, params))
        self.metrics["requests"] += 1
        if self.error:
            raise self.error
        if params["action"] == "query":
            pages = []
            for title in params["titles"].split("|"):
                if title not in self.pages:
                    pages.append(dict(title=title, missing=True))
                    continue
                page = copy.deepcopy(self.pages[title])
                if "extmetadata" not in params["iiprop"]:
                    page["imageinfo"][0].pop("extmetadata", None)
                pages.append(page)
            result = {"query": {"pages": pages}}
        else:
            entities = {}
            for entity_id in params["ids"].split("|"):
                if entity_id.startswith("M"):
                    entities[entity_id] = dict(id=entity_id, labels={"pl": {"value": "Caption " + entity_id}}, statements={})
                else:
                    entities[entity_id] = dict(id=entity_id, labels={"pl": {"value": "Polska " + entity_id}},
                                              claims={}, sitelinks={})
            result = {"entities": entities}
        return {**result, "_request_key": f"fixture-{len(self.calls)}", "_retrieved_at": "2026-10-21T12:00:00+00:00"}


def work(count=50):
    return [(ImageCandidate(f"File:Culture {i}.jpg", "wikidata_P18"), Entity("Q36", "Q36")) for i in range(count)]


def test_expansion_keeps_records_dedup_rejections_evidence_and_source(tmp_path):
    original = replace(Config(), output_dir=str(tmp_path / "original"))
    enlarged = replace(original, output_dir=str(tmp_path / "enlarged"),
        pipeline=replace(original.pipeline, export_every_records=5000, export_interval=900),
        discovery=replace(original.discovery, max_depth=2),
        wikimedia=replace(original.wikimedia, max_files_per_entity=100))
    entity = Entity("Q36", "Q36", depth=1)
    fixture = json.loads((Path(__file__).parent / "fixtures/commons.json").read_text(encoding="utf-8"))
    record = normalize(fixture["page"], fixture["structured"], entity,
                       ImageCandidate("File:Known.jpg", "wikidata_P18"), ["proof"], "2026-10-21T12:00:00Z")
    with CheckpointStore(Path(original.output_dir), original) as store:
        store.add_entity(entity, True)
        store.complete_entity("Q36")
        store.accept(record, "File:Known.jpg")
        store.reject("File:Rejected.jpg", ["too_small"])
    raw = Path(original.output_dir) / "raw"
    raw.mkdir()
    (raw / "proof.json").write_text('{"response": {}}', encoding="utf-8")
    report = expand(Path(original.output_dir), enlarged)
    assert report["counts"]["records"] == 1
    with CheckpointStore(Path(enlarged.output_dir), enlarged) as store:
        assert store.get_record(record.id) == record
        assert store.file_result("File:Known.jpg")[0] == record.id
        assert store.file_result("File:Rejected.jpg")[1]
        assert store.counts()["pending_discovery"] == 1
        assert store.counts()["pending_collection"] == 0  # Hydrate claims before collecting.
    with CheckpointStore(Path(original.output_dir), original) as store:
        assert store.counts()["pending_discovery"] == 0
        assert store.counts()["pending_collection"] == 0
    assert (Path(enlarged.output_dir) / "raw/proof.json").read_bytes() == (raw / "proof.json").read_bytes()


@pytest.mark.parametrize("change", ["filters", "smaller", "existing"])
def test_expansion_rejects_changed_policy_reduced_bounds_or_existing_target(tmp_path, change):
    original = replace(Config(), output_dir=str(tmp_path / "original"))
    with CheckpointStore(Path(original.output_dir), original):
        pass
    enlarged = replace(original, output_dir=str(tmp_path / "new"),
                       discovery=replace(original.discovery, max_depth=2))
    if change == "filters":
        enlarged = replace(enlarged, filters=replace(original.filters, require_license=False))
    elif change == "smaller":
        enlarged = replace(enlarged, discovery=replace(enlarged.discovery, max_entities=1))
    else:
        Path(enlarged.output_dir).mkdir()
    with pytest.raises(ValueError):
        expand(Path(original.output_dir), enlarged)


def test_atomic_json_retries_transient_replacement_lock_without_losing_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    atomic_json(path, {"old": True})
    original_replace = Path.replace
    attempts = []

    def locked_once(temporary, target):
        attempts.append(temporary)
        if len(attempts) == 1:
            assert json.loads(path.read_text()) == {"old": True}
            raise PermissionError("Reader temporarily holds destination")
        return original_replace(temporary, target)

    monkeypatch.setattr(Path, "replace", locked_once)
    monkeypatch.setattr("pl_mmb_collection.storage.checkpoint.time.sleep", lambda _: None)
    atomic_json(path, {"new": True})
    assert len(attempts) == 2
    assert json.loads(path.read_text()) == {"new": True}


def test_50_images_use_seven_requests_and_preserve_per_file_captions():
    client = FakeClient()
    records = WikimediaSource(client, Config()).fetch_many(work())
    assert len(records) == 50 and all(r.record and not r.error for r in records)
    assert len(client.calls) == 7  # 1 lightweight + 5 small extmetadata + 1 MediaInfo.
    assert records[0].record.caption_pl == "Caption M1000"
    assert records[-1].record.caption_pl == "Caption M1049"
    assert records[0].record.caption_quality["pl"] == "A"
    assert records[0].record.raw_requests
    assert all(len(p["titles"].split("|")) <= 10 for _, p in client.calls if "extmetadata" in p.get("iiprop", ""))


def test_parsed_cache_avoids_refetch_when_batch_order_or_size_changes():
    client = FakeClient()
    source = WikimediaSource(client, Config())
    source.fetch_many(work())
    calls = len(client.calls)
    assert all(r.record for r in source.fetch_many(list(reversed(work(13)))))
    assert len(client.calls) == calls


def test_cheap_preflight_and_license_filter_skip_unnecessary_requests():
    client = FakeClient(4)
    client.pages["File:Culture 0.jpg"]["imageinfo"][0]["mime"] = "image/svg+xml"
    client.pages["File:Culture 1.jpg"]["imageinfo"][0]["width"] = 10
    client.pages["File:Culture 2.jpg"]["imageinfo"][0]["extmetadata"].pop("LicenseUrl")
    results = WikimediaSource(client, Config()).fetch_many(work(4))
    metadata = [p for _, p in client.calls if "extmetadata" in p.get("iiprop", "")]
    assert metadata[0]["titles"] == "File:Culture 2.jpg|File:Culture 3.jpg"
    media = [p for _, p in client.calls if p["action"] == "wbgetentities"]
    assert media[0]["ids"] == "M1003"
    assert results[-1].record.caption_quality["pl"] == "A"
    assert results[0].record.mime_type == "image/svg+xml"


def test_missing_file_does_not_discard_other_batch_results():
    client = FakeClient(3)
    client.pages.pop("File:Culture 1.jpg")
    results = WikimediaSource(client, Config()).fetch_many(work(3))
    assert results[0].record and results[2].record
    assert isinstance(results[1].error, ValueError)


def test_duplicate_file_contexts_are_normalized_and_merged_independently(tmp_path):
    client = FakeClient(1)
    rows = [(work(1)[0][0], Entity(qid, qid, label_pl=label))
            for qid, label in [("Q36", "Polska"), ("Q270", "Warszawa")]]
    results = WikimediaSource(client, Config()).fetch_many(rows)
    assert results[0].record.entity_label_pl == "Polska"
    assert results[1].record.entity_label_pl == "Warszawa"
    with CheckpointStore(tmp_path, Config()) as store:
        for row in results:
            store.accept(row.record, row.candidate.title)
        assert len(next(store.records()).associations) == 2
        assert store.counts()["records"] == 1


def test_wikidata_50_entities_are_one_network_request():
    client = FakeClient()
    source = WikidataSource(client, Config())
    result = source.entities_many([Entity(f"Q{i+1}", "Q36") for i in range(50)])
    assert len(client.calls) == 1
    assert len(result) == 50 and all(isinstance(v, Entity) for v in result.values())


def test_batch_get_length_is_bounded():
    values = ["ą" * 100 + str(i) for i in range(50)]
    chunks = list(batches(values, 50, 2000))
    assert len(chunks) > 1
    assert [x for chunk in chunks for x in chunk] == values


def test_known_entities_hydrate_before_sparql_and_sparql_can_be_disabled(tmp_path):
    events = []
    class Source:
        def entity(self, qid, seed, depth):
            events.append("entity")
            return Entity(qid, seed, depth)
        def by_class(self, qid):
            events.append("sparql")
            return []
    config = replace(Config(), discovery=DiscoveryConfig(seeds=["Q36"], sparql_classes=["Q5"], enable_sparql=False))
    with CheckpointStore(tmp_path, config) as store:
        discover(config, store, Source())
    assert events == ["entity"]


def test_retry_after_is_persistent_and_sparql_does_not_block_commons(tmp_path, monkeypatch):
    config = HttpConfig(interval=0, requests_per_minute=180, contact="test@example.org",
                        state_path=str(tmp_path / "limits.json"), max_inline_wait=5, retries=0)
    client = HttpClient(config, tmp_path / "raw")
    def blocked(url):
        raise HTTPError(url, 429, "slow down", {"Retry-After": "1000"}, None)
    monkeypatch.setattr(client, "_request", blocked)
    before = time.time()
    with pytest.raises(RetryLater) as error:
        client.get("https://query.wikidata.org/sparql", query="SELECT")
    assert error.value.retry_at >= before + 999
    restored = RateLimiter(config, tmp_path / "limits.json")
    with pytest.raises(RetryLater):
        restored.reserve("https://query.wikidata.org/sparql")
    # This reserves a permitted request to a different service; no network call.
    restored.reserve("https://commons.wikimedia.org/w/api.php")


def test_action_api_429_blocks_both_projects(tmp_path):
    config = HttpConfig(interval=0, keep_alive=False)
    limiter = RateLimiter(config, tmp_path / "limits.json")
    limiter.cooldown("https://commons.wikimedia.org/w/api.php", 1000, "429", True)
    with pytest.raises(RetryLater):
        limiter.reserve("https://www.wikidata.org/w/api.php")
    assert scope_for("https://commons.wikimedia.org/w/api.php") == scope_for("https://www.wikidata.org/w/api.php")


def test_budget_is_shared_across_hosts_and_has_no_burst(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("pl_mmb_collection.sources.rate_limit.time.time", lambda: clock[0])
    monkeypatch.setattr("pl_mmb_collection.sources.rate_limit.time.sleep", lambda delay: clock.__setitem__(0, clock[0]+delay))
    config = HttpConfig(interval=0, requests_per_minute=180, contact="test@example.org")
    limiter = RateLimiter(config, tmp_path / "limits.json")
    starts = []
    for index in range(181):
        limiter.reserve("https://commons.wikimedia.org/w/api.php" if index % 2 else "https://www.wikidata.org/w/api.php")
        starts.append(clock[0])
    assert starts[-1] - starts[0] >= 62.9
    assert max(sum(t <= stamp < t+60 for stamp in starts) for t in starts) <= 172


def test_contact_required_to_enable_higher_budget():
    with pytest.raises(ValueError, match="contact"):
        replace(Config(), http=HttpConfig(requests_per_minute=180)).validate()
    replace(Config(), http=HttpConfig(contact="test@example.org", requests_per_minute=180)).validate()


def test_shared_budget_waits_for_other_process_without_reading_locked_byte(tmp_path):
    path = tmp_path / "shared.json"
    child_code = """
import sys, time
from pathlib import Path
from pl_mmb_collection.config import HttpConfig
from pl_mmb_collection.sources.rate_limit import RateLimiter
limiter = RateLimiter(HttpConfig(), Path(sys.argv[1]))
with limiter.shared_state():
    print('locked', flush=True)
    time.sleep(1)
"""
    child = subprocess.Popen([sys.executable, "-c", child_code, str(path)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "locked"
        limiter = RateLimiter(HttpConfig(), path)
        with limiter.shared_state():
            assert limiter.state["cooldowns"] == {}
        _, stderr = child.communicate(timeout=10)
        assert child.returncode == 0, stderr
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_deferred_metadata_is_pending_and_resumes_without_marking_complete(tmp_path):
    config = Config()
    client = FakeClient(2)
    source = WikimediaSource(client, config)
    source.candidates = lambda entity: iter([w[0] for w in work(2)])
    client.error = RetryLater("wikimedia-api", time.time()+1000, "fixture pause")
    with CheckpointStore(tmp_path, config) as store:
        store.add_entity(Entity("Q36", "Q36"), True)
        collect(config, store, source)
        assert store.counts()["records"] == 0
        assert len(store.entities("collect")) == 1
        assert store.counts()["failures"] == 0
        client.error = None
        with store.conn:
            store.conn.execute("UPDATE deferred SET retry_at=0")
        collect(config, store, source)
        assert store.counts()["records"] == 2
        assert store.entities("collect") == []


def test_local_dump_and_previous_checkpoint_import_are_resumable(tmp_path):
    previous = tmp_path / "previous"
    with CheckpointStore(previous, Config()) as old:
        old.add_entity(Entity("Q36", "Q36"))
    path = tmp_path / "entities.jsonl"
    path.write_text('Q270\n'+json.dumps(dict(id="Q268", claims={}, labels={"pl":{"value":"Poznań"}})), encoding="utf-8")
    config = replace(Config(), discovery=DiscoveryConfig(seeds=[], entity_files=[str(path)], seed_checkpoints=[str(previous)]))
    client = HttpClient(config.http, tmp_path / "run/raw")
    with CheckpointStore(tmp_path / "run", config) as store:
        import_seeds(config, store, client)
        import_seeds(config, store, client)
        assert store.entity_count() == 3
        discovered = [e for _, e in store.entities() if e.qid == "Q268"][0]
        assert discovered.label_pl == "Poznań"
        assert (client.raw_dir / (discovered.raw_request + ".json")).is_file()


def test_hybrid_end_to_end_51_entities_does_not_need_sparql(tmp_path):
    class Client(FakeClient):
        def get(self, endpoint, **params):
            result = super().get(endpoint, **params)
            if params["action"] == "wbgetentities" and params["ids"].startswith("Q"):
                for qid, entity in result["entities"].items():
                    title = f"Culture {int(qid[1:])-1}.jpg"
                    entity["claims"] = {"P18": [{"mainsnak": {"snaktype": "value", "datavalue": {"value": title}}}]}
            return result
    config = replace(Config(jsonl=True), discovery=DiscoveryConfig(seeds=[f"Q{i+1}" for i in range(51)],
                     max_entities=51, max_depth=0, enable_sparql=False, sparql_classes=["Q5"]))
    client = Client(51)
    source = WikimediaSource(client, config)
    with CheckpointStore(tmp_path, config) as store:
        run(config, store, source, client)
        assert store.counts()["records"] == 51
        assert store.counts()["pending_collection"] == 0
        assert validate(store, config)["valid"]
        calls = len(client.calls)
        run(config, store, source, client)
        assert len(client.calls) == calls
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "complete"
    assert not any("query.wikidata.org" in endpoint for endpoint, _ in client.calls)


def test_empty_api_metadata_list_is_missing_license_not_a_parser_failure():
    fixture = json.loads((Path(__file__).parent / "fixtures/commons.json").read_text(encoding="utf-8"))
    fixture["page"]["imageinfo"][0]["extmetadata"] = []
    record = normalize(fixture["page"], fixture["structured"], Entity("Q36", "Q36"),
                       ImageCandidate("File:Empty.jpg", "wikidata_P18"), [], None)
    from pl_mmb_collection.pipeline.filters import rejection_reasons
    assert record.license is None
    assert "missing_license" in rejection_reasons(record, Config().filters)


def test_storage_guard_preserves_export_and_stops_before_network(tmp_path, monkeypatch):
    from types import SimpleNamespace
    config = Config()
    client = FakeClient(1)
    source = WikimediaSource(client, config)
    record = source.fetch_many(work(1))[0].record
    calls = len(client.calls)
    monkeypatch.setattr("pl_mmb_collection.pipeline.hybrid.shutil.disk_usage", lambda _: SimpleNamespace(free=0))
    with CheckpointStore(tmp_path, config) as store:
        store.accept(record, "File:Culture 0.jpg")
        run(config, store, source, client, min_free_bytes=1024)
        assert store.counts()["records"] == 1
    assert len(client.calls) == calls
    assert json.loads((tmp_path / "progress.json").read_text())["status"] == "storage_limit_reached"
    assert (tmp_path / "metadata/records.parquet").is_file()


def test_record_ceiling_stops_mid_batch_counts_unique_and_survives_resume(tmp_path):
    import pyarrow.parquet as pq
    config = replace(Config(jsonl=True), discovery=DiscoveryConfig(seeds=[], categories=["Test"], enable_sparql=False))
    client = FakeClient(12)
    duplicate = copy.deepcopy(client.pages["File:Culture 0.jpg"])
    duplicate["title"] = "File:Culture 1.jpg"
    client.pages[duplicate["title"]] = duplicate
    source = WikimediaSource(client, config)
    source.candidates = lambda entity: iter(candidate for candidate, _ in work(12))
    with CheckpointStore(tmp_path, config) as store:
        run(config, store, source, client, max_records=7)
        assert store.counts()["records"] == 7
        assert store.counts()["pending_collection"] == 1
        assert store.counts()["failures"] == 0
        assert store.file_result("File:Culture 0.jpg")[0] == store.file_result("File:Culture 1.jpg")[0]
        assert pq.read_metadata(tmp_path / "metadata/records.parquet").num_rows == 7
        calls = len(client.calls)
        run(config, store, source, client)
        assert store.counts()["records"] == 7
        assert len(client.calls) == calls
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["status"] == "record_limit_reached"
    assert progress["max_records"] == 7


def test_transport_has_no_implicit_retries_or_redirects(tmp_path, monkeypatch):
    client = HttpClient(HttpConfig(contact="test@example.org", requests_per_minute=180), tmp_path / "raw")
    calls = []
    class Raw:
        def read(self, size=None, decode_content=False):
            return b'{"ok":true}'
    class Response:
        status_code=200
        headers={}
        raw=Raw()
        def close(self):
            pass
    def get(url, **kwargs):
        calls.append(kwargs)
        return Response()
    monkeypatch.setattr(client.transport.session, "get", get)
    assert client.get("https://commons.wikimedia.org/w/api.php", action="query")["ok"]
    assert calls[0]["allow_redirects"] is False
    assert "test@example.org" in calls[0]["headers"]["User-Agent"]
    assert client.metrics["requests"] == 1
    client.close()


def test_maxlag_response_retry_after_is_honored(tmp_path, monkeypatch):
    client = HttpClient(HttpConfig(retries=0, max_inline_wait=5), tmp_path / "raw")
    class Response(io.BytesIO):
        headers={"Retry-After": "120"}
    monkeypatch.setattr(client, "_request", lambda url: Response(b'{"error":{"code":"maxlag"}}'))
    with pytest.raises(RetryLater) as error:
        client.get("https://commons.wikimedia.org/w/api.php", action="query")
    assert error.value.retry_at >= time.time()+119


def test_collector_lease_prevents_duplicate_workers(tmp_path):
    lock = tmp_path / "budget.lock"
    with CollectorLease(lock):
        with pytest.raises(ValueError, match="Another collector"):
            with CollectorLease(lock):
                pass
    with CollectorLease(lock):
        pass


def test_read_only_stats_does_not_reset_network_measurements(tmp_path):
    config = Config(output_dir=str(tmp_path))
    with CheckpointStore(tmp_path, config):
        pass
    metrics = tmp_path / "http_metrics.json"
    metrics.write_text('{"requests":123}', encoding="utf-8")
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"output_dir": str(tmp_path)}), encoding="utf-8")
    assert main(["stats", "--config", str(path)]) == 0
    assert json.loads(metrics.read_text()) == {"requests": 123}
