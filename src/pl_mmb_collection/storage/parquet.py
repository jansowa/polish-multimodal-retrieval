from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from ..models import DatasetRecord
from .checkpoint import CheckpointStore, atomic_replace


def record_schema():
    import pyarrow as pa
    integers = {"width", "height", "image_width", "image_height", "structured_revision"}
    lists = {"depicts", "categories", "raw_requests"}
    language_fields = {"caption_source", "caption_quality"}
    association = pa.struct([
        *[pa.field(k, pa.string()) for k in ["wikidata_qid", "seed_entity", "entity_label_pl",
                                           "entity_label_en", "retrieval_method", "category", "candidate_title"]],
        pa.field("category_depth", pa.int64()), pa.field("entity_revision", pa.int64()),
        pa.field("entity_raw_request", pa.string())])
    def list_of(value):
        # Parquet's compliant LIST encoding names the repeated child "element".
        return pa.list_(pa.field("element", value))
    types = {"associations": list_of(association),
             "text_candidates": list_of(pa.struct([pa.field(k, pa.string())
                 for k in ["language", "text", "source", "quality"]]))}
    result = []
    for f in fields(DatasetRecord):
        dtype = (pa.int64() if f.name in integers else list_of(pa.string()) if f.name in lists
                 else pa.struct([pa.field("pl", pa.string()), pa.field("en", pa.string())])
                 if f.name in language_fields else types.get(f.name, pa.string()))
        result.append(pa.field(f.name, dtype))
    return pa.schema(result, metadata={b"schema_version": b"1", b"pipeline": b"pl_mmb_collection"})


def export(store: CheckpointStore, jsonl: bool = False) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq
    directory = store.directory / "metadata"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "records.parquet"
    tmp = directory / "records.parquet.tmp"
    debug = directory / "records.jsonl.tmp"
    schema = record_schema()
    try:
        with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
            with debug.open("w", encoding="utf-8") as output:
                batch = []
                for record in store.records():
                    row = record.to_dict()
                    batch.append(row)
                    if jsonl:
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if len(batch) >= 1000:
                        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                        batch.clear()
                if batch:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        atomic_replace(tmp, target)
        if jsonl:
            atomic_replace(debug, directory / "records.jsonl")
    finally:
        tmp.unlink(missing_ok=True)
        debug.unlink(missing_ok=True)
    return target
