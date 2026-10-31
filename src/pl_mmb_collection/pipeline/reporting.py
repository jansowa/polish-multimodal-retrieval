from __future__ import annotations

import json
from collections import Counter
from typing import Any

from ..config import Config
from ..models import DatasetRecord
from ..storage.checkpoint import CheckpointStore, atomic_json
from .filters import rejection_reasons


def statistics(store: CheckpointStore) -> dict[str, Any]:
    languages: Counter = Counter()
    sources: Counter = Counter()
    quality: Counter = Counter()
    licenses: Counter = Counter()
    resolutions: Counter = Counter()
    entities: Counter = Counter()
    categories: Counter = Counter()
    qids: set[str] = set()
    urls: set[str] = set()
    for record in store.records():
        urls.add(record.image_original_url)
        for lang in ("pl", "en"):
            if getattr(record, "caption_" + lang):
                languages[lang] += 1
                sources[record.caption_source[lang] or "unknown"] += 1
                quality[record.caption_quality[lang] or "unknown"] += 1
        licenses[record.license or "unknown"] += 1
        area = (record.width or 0) * (record.height or 0)
        resolutions["<1MP" if area < 1_000_000 else "1-4MP" if area < 4_000_000 else ">=4MP"] += 1
        entity_ids = {a["wikidata_qid"] for a in record.associations if a.get("wikidata_qid")}
        qids.update(entity_ids)
        entities.update(entity_ids)
        categories.update(record.categories)
    reasons: Counter = Counter()
    for (raw,) in store.conn.execute("SELECT rejection FROM files WHERE rejection IS NOT NULL"):
        reasons.update(json.loads(raw))
    result = {**store.counts(), "unique_images": len(urls), "unique_qids": len(qids),
              "caption_languages": dict(languages), "caption_sources": dict(sources),
              "caption_quality": dict(quality), "licenses": dict(licenses),
              "resolution_buckets": dict(resolutions), "rejection_reasons": dict(reasons),
              "top_entities": entities.most_common(20), "top_categories": categories.most_common(20)}
    atomic_json(store.directory / "metadata/stats.json", result)
    return result


def validate(store: CheckpointStore, config: Config) -> dict[str, Any]:
    import pyarrow.parquet as pq
    from ..storage.parquet import record_schema
    errors = []
    seen: set[str] = set()
    count = 0
    for record in store.records():
        count += 1
        issues = rejection_reasons(record, config.filters)
        for name in ("source_page_url", "image_original_url", "retrieved_at"):
            if not getattr(record, name):
                issues.append("missing_" + name)
        if record.id in seen:
            issues.append("duplicate_id")
        seen.add(record.id)
        if not record.associations or not record.raw_requests:
            issues.append("missing_provenance")
        if issues:
            errors.append({"id": record.id, "issues": issues})
    path = store.directory / "metadata/records.parquet"
    if not path.exists():
        errors.append({"issues": ["missing_parquet_export"]})
    else:
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != count:
            errors.append({"issues": ["parquet_checkpoint_count_mismatch"]})
        if not parquet.schema_arrow.equals(record_schema(), check_metadata=True):
            errors.append({"issues": ["parquet_schema_mismatch"]})
        # Read every batch to detect truncated/corrupted exports and parse records.
        for batch in parquet.iter_batches(batch_size=1000):
            for row in batch.to_pylist():
                exported = DatasetRecord(**row)
                if exported.to_dict() != store.get_record(exported.id).to_dict():
                    errors.append({"id": exported.id, "issues": ["parquet_checkpoint_content_mismatch"]})
    result = {"valid": not errors, "records": count, "errors": errors}
    atomic_json(store.directory / "metadata/validation.json", result)
    return result
