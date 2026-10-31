from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse


@dataclass(frozen=True)
class Dataset:
    id: str
    label: str
    directory: Path


def safe_url(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlparse(value)
    return value if parsed.scheme in {"http", "https"} and parsed.hostname else None


def summary(record: dict[str, Any]) -> dict[str, Any]:
    keys = ["id", "source", "image_filename", "wikidata_qid", "entity_label_pl", "entity_label_en",
            "caption_source", "caption_quality", "creator", "license", "width", "height", "retrieved_at"]
    row = {k: record.get(k) for k in keys}
    row["caption_source"] = row["caption_source"] or {}
    row["caption_quality"] = row["caption_quality"] or {}
    for lang in ("pl", "en"):
        text = record.get("caption_" + lang)
        row["caption_" + lang] = text[:650] + ("…" if len(text) > 650 else "") if text else None
    for key in ("source_page_url", "image_original_url", "license_url"):
        row[key] = safe_url(record.get(key))
    filename = str(record.get("image_filename") or "").removeprefix("File:")
    row["thumbnail_url"] = ("https://commons.wikimedia.org/wiki/Special:Redirect/file/" + quote(filename, safe="")
                            + "?width=960") if record.get("source") == "wikimedia" else safe_url(record.get("image_url"))
    return row


@dataclass(frozen=True)
class Query:
    dataset: str
    page: int = 1
    size: int = 24
    q: str = ""
    language: str = "any"
    quality: str = ""
    license: str = ""
    sort: str = "newest"
    seed: str = "42"

    def validate(self) -> None:
        if self.page < 1 or self.size not in {12, 24, 48, 96}:
            raise ValueError("Nieprawidłowy numer lub rozmiar strony")
        if self.language not in {"any", "pl", "en", "both"}:
            raise ValueError("Nieznany język")
        if self.quality not in {"", "A", "B", "C", "D"}:
            raise ValueError("Nieznana ocena jakości")
        if self.sort not in {"newest", "oldest", "entity", "resolution", "random"}:
            raise ValueError("Nieznane sortowanie")
        if len(self.q) > 200 or len(self.license) > 200 or len(self.seed) > 100:
            raise ValueError("Filtr jest zbyt długi")


class BrowserStore:
    """Short read transactions never change the collector checkpoint or manifest."""

    def __init__(self, datasets: list[Dataset]):
        self.datasets = {d.id: d for d in datasets}
        if len(self.datasets) != len(datasets):
            raise ValueError("Dataset IDs must be unique")

    def connect(self, dataset: str) -> sqlite3.Connection:
        item = self.datasets.get(dataset)
        if item is None:
            raise KeyError("Nieznany zbiór")
        path = (item.directory / "checkpoint.sqlite3").resolve()
        if not path.is_file():
            raise KeyError("Zbiór jeszcze nie powstał")
        conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5)
        conn.execute("PRAGMA query_only=ON")
        conn.create_function("casefold", 1, lambda value: str(value or "").casefold(), deterministic=True)
        conn.create_function("search_text", 1, lambda raw: json.dumps(
            json.loads(raw), ensure_ascii=False).casefold(), deterministic=True)
        conn.create_function("random_key", 2, lambda value, seed: hashlib.sha256(
            (str(seed) + ":" + str(value)).encode()).hexdigest(), deterministic=True)
        return conn

    def catalog(self) -> dict[str, Any]:
        datasets = []
        for item in self.datasets.values():
            try:
                with closing(self.connect(item.id)) as conn:
                    conn.execute("BEGIN")
                    count = conn.execute("SELECT count(*) FROM records").fetchone()[0]
                    total, discovered, collected = conn.execute(
                        "SELECT count(*),coalesce(sum(discovered),0),coalesce(sum(collected),0) FROM entities").fetchone()
                    rejected = conn.execute("SELECT count(*) FROM files WHERE rejection IS NOT NULL").fetchone()[0]
                    failures = conn.execute("SELECT count(*) FROM failures").fetchone()[0]
                    licenses = [r[0] for r in conn.execute("SELECT DISTINCT json_extract(json,'$.license') FROM records ORDER BY 1") if r[0]]
                    latest = conn.execute("SELECT max(json_extract(json,'$.retrieved_at')) FROM records").fetchone()[0]
                    quality = {q: conn.execute("""SELECT count(*) FROM records WHERE
                        json_extract(json,'$.caption_quality.pl')=? OR json_extract(json,'$.caption_quality.en')=?""", (q, q)).fetchone()[0]
                               for q in "ABCD"}
                    direct = conn.execute("""SELECT count(*) FROM records WHERE
                        json_extract(json,'$.caption_quality.pl') IN ('A','B') OR
                        json_extract(json,'$.caption_quality.en') IN ('A','B')""").fetchone()[0]
                phase = "Odkrywanie tematów" if discovered < total else "Zbieranie metadanych" if collected < total else "Zbiór gotowy"
                datasets.append(dict(id=item.id, label=item.label, count=count, entities=total,
                    discovered=discovered, collected=collected, rejected=rejected, failures=failures,
                    licenses=licenses, quality=quality, direct_count=direct, latest=latest, phase=phase))
            except (KeyError, sqlite3.Error):
                continue
        default = next((d["id"] for d in datasets if d["count"]), datasets[0]["id"] if datasets else None)
        return dict(mode="live", generated_at=datetime.now(timezone.utc).isoformat(),
                    datasets=datasets, default_dataset=default)

    def records(self, query: Query) -> dict[str, Any]:
        query.validate()
        where, params = [], []
        if query.q:
            # Search actual text, including long descriptions, categories and QIDs.
            where.append("instr(search_text(json),?)>0")
            params.append(query.q.casefold())
        if query.language in {"pl", "en", "both"}:
            languages = ("pl", "en") if query.language == "both" else (query.language,)
            for lang in languages:
                where.append(f"coalesce(json_extract(json,'$.caption_{lang}'),'')<>''")
        if query.quality:
            languages = (query.language,) if query.language in {"pl", "en"} else ("pl", "en")
            where.append("(" + " OR ".join(f"json_extract(json,'$.caption_quality.{lang}')=?" for lang in languages) + ")")
            params.extend([query.quality] * len(languages))
        if query.license:
            where.append("json_extract(json,'$.license')=?")
            params.append(query.license)
        clause = " WHERE " + " AND ".join(where) if where else ""
        orders = {"newest": "json_extract(json,'$.retrieved_at') DESC,id ASC",
                  "oldest": "json_extract(json,'$.retrieved_at') ASC,id ASC",
                  "entity": "casefold(coalesce(json_extract(json,'$.entity_label_pl'),json_extract(json,'$.entity_label_en'),'')) ASC,id ASC",
                  "resolution": "coalesce(json_extract(json,'$.width'),0)*coalesce(json_extract(json,'$.height'),0) DESC,id ASC",
                  "random": "random_key(id,?) ASC,id ASC"}
        with closing(self.connect(query.dataset)) as conn:
            conn.execute("BEGIN")
            total = conn.execute("SELECT count(*) FROM records" + clause, params).fetchone()[0]
            pages = max(1, math.ceil(total / query.size))
            page = min(query.page, pages)
            ordering_params = [query.seed] if query.sort == "random" else []
            rows = [summary(json.loads(raw)) for (raw,) in conn.execute(
                "SELECT json FROM records" + clause + " ORDER BY " + orders[query.sort] + " LIMIT ? OFFSET ?",
                [*params, *ordering_params, query.size, (page - 1) * query.size])]
        return dict(items=rows, total=total, page=page, pages=pages, size=query.size)

    def record(self, dataset: str, record_id: str) -> dict[str, Any]:
        with closing(self.connect(dataset)) as conn:
            row = conn.execute("SELECT json FROM records WHERE id=?", (record_id,)).fetchone()
        if row is None:
            raise KeyError("Nie znaleziono rekordu")
        return json.loads(row[0])

    def iter_records(self, dataset: str):
        with closing(self.connect(dataset)) as conn:
            conn.execute("BEGIN")
            for (raw,) in conn.execute("SELECT json FROM records ORDER BY id"):
                yield json.loads(raw)


def default_datasets(root: Path) -> list[Dataset]:
    current = root / "outputs/polish_culture_expanded"
    if not (current / "expansion.json").is_file():
        current = root / "outputs/polish_culture_fast"
    pointer = root / "outputs/collection_browser/active_dataset.json"
    if pointer.is_file():
        selected = Path(json.loads(pointer.read_text(encoding="utf-8-sig"))["directory"])
        if (selected / "expansion.json").is_file():
            current = selected
    return [Dataset("fast", "Zbiór główny · szybki", current),
            Dataset("production", "Poprzedni przebieg", root / "outputs/polish_culture"),
            Dataset("pilot", "Pilot · 20 encji", root / "outputs/polish_culture_pilot")]
