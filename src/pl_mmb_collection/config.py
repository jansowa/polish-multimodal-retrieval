from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HttpConfig:
    user_agent: str = "pl-mmb-collection/1.0 (Polish culture research; configure contact in YAML)"
    interval: float = 1.0
    timeout: float = 45.0
    retries: int = 3
    backoff: float = 2.0
    cache: bool = True
    contact: str | None = None
    requests_per_minute: int = 10
    max_inline_wait: float = 5.0
    state_path: str | None = None
    cache_dirs: list[str] = field(default_factory=list)
    keep_alive: bool = True


@dataclass(frozen=True)
class DiscoveryConfig:
    seeds: list[str] = field(default_factory=lambda: ["Q36"])
    categories: list[str] = field(default_factory=list)
    max_entities: int = 100
    max_depth: int = 1
    max_links_per_entity: int = 20
    # Outgoing edges only. No unbounded transitive graph queries.
    properties: list[str] = field(default_factory=lambda: ["P150", "P527"])
    sparql_classes: list[str] = field(default_factory=list)
    country_qid: str = "Q36"
    enable_sparql: bool = True
    sparql_limit_per_class: int = 50
    seed_checkpoints: list[str] = field(default_factory=list)
    entity_files: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WikimediaConfig:
    wikidata_api: str = "https://www.wikidata.org/w/api.php"
    commons_api: str = "https://commons.wikimedia.org/w/api.php"
    sparql_endpoint: str = "https://query.wikidata.org/sparql"
    max_files_per_entity: int = 10
    category_depth: int = 0
    max_categories: int = 10
    category_member_limit: int = 100
    use_depicts_search: bool = False
    thumbnail_width: int | None = None
    entity_batch_size: int = 50
    metadata_batch_size: int = 10
    structured_batch_size: int = 50
    preflight_images: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    discovery_batches_per_round: int = 1
    collect_entities_per_round: int = 20
    export_every_records: int = 100
    export_interval: float = 60.0


@dataclass(frozen=True)
class FilterConfig:
    min_width: int = 400
    min_height: int = 400
    min_area: int = 250_000
    allowed_mime_types: list[str] = field(default_factory=lambda: ["image/jpeg", "image/png", "image/webp"])
    reject_svg: bool = True
    require_text: bool = True
    require_license: bool = True
    allowed_licenses: list[str] = field(default_factory=lambda: [
        "CC0", "CC BY 4.0", "CC BY-SA 4.0", "CC BY 3.0", "CC BY-SA 3.0",
        "CC BY 2.0", "CC BY-SA 2.0", "CC BY 2.5", "CC BY-SA 2.5",
    ])
    excluded_category_patterns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DownloadConfig:
    max_bytes: int = 50_000_000
    phash: bool = False
    phash_distance: int = 6


@dataclass(frozen=True)
class Config:
    output_dir: str = "outputs/polish_culture"
    source: str = "wikimedia"
    jsonl: bool = False
    http: HttpConfig = field(default_factory=HttpConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    wikimedia: WikimediaConfig = field(default_factory=WikimediaConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    def validate(self) -> None:
        for value in [*self.discovery.seeds, self.discovery.country_qid, *self.discovery.sparql_classes]:
            if not re.fullmatch(r"Q[1-9][0-9]*", value):
                raise ValueError(f"Invalid Wikidata QID: {value}")
        for value in self.discovery.properties:
            if not re.fullmatch(r"P[1-9][0-9]*", value):
                raise ValueError(f"Invalid Wikidata property: {value}")
        positive = [self.discovery.max_entities, self.wikimedia.max_files_per_entity,
                    self.wikimedia.max_categories, self.wikimedia.category_member_limit,
                    self.http.timeout, self.download.max_bytes]
        nonnegative = [self.discovery.max_depth, self.discovery.max_links_per_entity,
                       self.wikimedia.category_depth, self.http.interval, self.http.retries,
                       self.http.backoff, self.filters.min_width, self.filters.min_height,
                       self.filters.min_area]
        if any(v <= 0 for v in positive) or any(v < 0 for v in nonnegative):
            raise ValueError("Limits must be positive; depths, retries and filter minima nonnegative")
        if not 0 <= self.download.phash_distance <= 64:
            raise ValueError("phash_distance must be in 0..64")
        if len(self.http.user_agent.strip()) < 15:
            raise ValueError("Provide a descriptive User-Agent with contact information")
        contact = self.http.contact
        if contact and not (re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", contact)
                            or re.fullmatch(r"https?://[^\s]+", contact)):
            raise ValueError("http.contact must be an email or full project/contact URL")
        if not 1 <= self.http.requests_per_minute <= (200 if contact else 10):
            raise ValueError("Use at most 10 requests/min without contact, or 200 with a compliant contact")
        if self.http.max_inline_wait < 0:
            raise ValueError("max_inline_wait must be nonnegative")
        if not (1 <= self.wikimedia.entity_batch_size <= 50
                and 1 <= self.wikimedia.structured_batch_size <= 50
                and 1 <= self.wikimedia.metadata_batch_size <= 10):
            raise ValueError("Entity/MediaInfo batches: 1..50; expensive extmetadata batches: 1..10")
        if min(self.discovery.sparql_limit_per_class, self.pipeline.discovery_batches_per_round,
               self.pipeline.collect_entities_per_round, self.pipeline.export_every_records,
               self.pipeline.export_interval) <= 0:
            raise ValueError("Discovery, export and scheduling limits must be positive")
        if self.wikimedia.thumbnail_width is not None and self.wikimedia.thumbnail_width < 1:
            raise ValueError("thumbnail_width must be positive")
        for pattern in self.filters.excluded_category_patterns:
            re.compile(pattern)

    def manifest(self) -> dict[str, Any]:
        from . import __version__
        return {"pipeline_version": __version__, "config": asdict(self)}


def load_config(path: Path) -> Config:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a mapping")
    data = dict(data)
    try:
        for key, cls in [("http", HttpConfig), ("discovery", DiscoveryConfig),
                         ("wikimedia", WikimediaConfig), ("filters", FilterConfig),
                         ("download", DownloadConfig)]:
            if key in data:
                data[key] = cls(**data[key])
        if "pipeline" in data:
            data["pipeline"] = PipelineConfig(**data["pipeline"])
        config = Config(**data)
    except TypeError as exc:
        raise ValueError(f"Invalid configuration: {exc}") from exc
    config.validate()
    return config
