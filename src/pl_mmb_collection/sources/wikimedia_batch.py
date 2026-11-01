from __future__ import annotations

import copy
from typing import Any

from ..config import Config
from ..models import Entity, ImageCandidate
from ..models.acquisition import FetchResult
from ..pipeline.filters import rejection_reasons
from .batching import batches
from .http import HttpClient
from .normalization import normalize


class WikimediaBatch:
    """Small extmetadata batches, up to 50 lightweight/MediaInfo items per call."""

    def __init__(self, client: HttpClient, config: Config):
        self.client, self.config = client, config
        self.namespace = "commons-page-v1-" + str(config.wikimedia.thumbnail_width)

    def _params(self, titles: list[str], metadata: bool) -> dict[str, Any]:
        params = dict(action="query", formatversion=2, titles="|".join(titles), redirects=1,
            prop="imageinfo|categories", iiprop="url|size|mime|sha1" + ("|extmetadata" if metadata else ""),
            iiurlwidth=self.config.wikimedia.thumbnail_width, cllimit="max")
        if metadata:
            params.update(iiextmetadatamultilang=1,
                iiextmetadatafilter="ImageDescription|Artist|Attribution|DateTimeOriginal|LicenseShortName|LicenseUrl|GPSLatitude|GPSLongitude")
        return params

    def _legacy(self, title: str) -> dict[str, Any] | None:
        params = self._params([title], True)
        params.pop("iiextmetadatafilter", None)
        response = self.client.cached(self.config.wikimedia.commons_api, **params)
        if response and "clcontinue" not in response.get("continue", {}):
            pages = response.get("query", {}).get("pages", [])
            if pages and "missing" not in pages[0]:
                return {"page": pages[0], "metadata": True, "requests": [response.get("_request_key")],
                        "retrieved_at": response.get("_retrieved_at")}
        return None

    def _load(self, titles: list[str], pages: dict[str, dict], errors: dict[str, Exception], metadata: bool) -> None:
        size = self.config.wikimedia.metadata_batch_size if metadata else 50
        for group in batches(titles, size):
            fragments: dict[str, dict] = {}
            aliases: dict[str, str] = {}
            requests: list[str] = []
            retrieved_at = None
            continuation: dict[str, Any] = {}
            try:
                while True:
                    response = self.client.get(self.config.wikimedia.commons_api, **self._params(group, metadata), **continuation)
                    requests.append(response.get("_request_key"))
                    retrieved_at = retrieved_at or response.get("_retrieved_at")
                    query = response.get("query", {})
                    for mapping in [*query.get("normalized", []), *query.get("redirects", [])]:
                        aliases[mapping["from"]] = mapping["to"]
                    for current in query.get("pages", []):
                        title = current["title"]
                        if title not in fragments:
                            fragments[title] = copy.deepcopy(current)
                        else:
                            target = fragments[title]
                            if current.get("imageinfo"):
                                target.setdefault("imageinfo", [{}])[0].update(current["imageinfo"][0])
                            target.setdefault("categories", []).extend(current.get("categories", []))
                    more = response.get("continue", {})
                    continuation = {"continue": more.get("continue", ""), "clcontinue": more["clcontinue"]} if "clcontinue" in more else {}
                    if not continuation:
                        break
                for original in group:
                    canonical = original
                    for _ in range(len(aliases) + 1):
                        if canonical not in aliases:
                            break
                        canonical = aliases[canonical]
                    page = fragments.get(canonical) or fragments.get(canonical.replace("_", " "))
                    if not page or "missing" in page or not page.get("imageinfo"):
                        errors[original] = ValueError("Commons file missing: " + original)
                        continue
                    categories = {c["title"]: c for c in page.get("categories", [])}
                    page["categories"] = list(categories.values())
                    entry = {"page": page, "metadata": metadata, "requests": requests, "retrieved_at": retrieved_at}
                    pages[original] = entry
                    self.client.item_put(self.namespace, original, entry)
            except Exception as exc:
                for title in group:
                    errors[title] = exc

    def fetch(self, work: list[tuple[ImageCandidate, Entity]]) -> list[FetchResult]:
        pages: dict[str, dict] = {}
        errors: dict[str, Exception] = {}
        unique = dict.fromkeys(candidate.title for candidate, _ in work)
        for title in unique:
            entry = self.client.item_get(self.namespace, title) or self._legacy(title)
            if entry:
                pages[title] = entry
        missing = [title for title in unique if title not in pages]
        if self.config.wikimedia.preflight_images:
            self._load(missing, pages, errors, metadata=False)
        needed = []
        preliminary: dict[str, Any] = {}
        for candidate, entity in work:
            title = candidate.title
            if title in errors:
                continue
            if title not in pages:
                needed.append(title)
                continue
            entry = pages[title]
            try:
                record = normalize(entry["page"], {}, entity, candidate,
                                   [entity.raw_request, *entry["requests"]], entry["retrieved_at"])
                structural = [r for r in rejection_reasons(record, self.config.filters)
                              if r not in {"missing_text", "missing_license", "license_not_allowed"}]
                if structural:
                    preliminary[title] = record
                elif not entry["metadata"]:
                    needed.append(title)
            except Exception as exc:
                errors[title] = exc
        self._load(list(dict.fromkeys(needed)), pages, errors, metadata=True)
        media_ids: dict[str, list[str]] = {}
        # Only files which pass size/MIME/license/category checks need MediaInfo.
        for candidate, entity in work:
            title = candidate.title
            if title in errors or title in preliminary:
                continue
            entry = pages[title]
            try:
                record = normalize(entry["page"], {}, entity, candidate,
                                   [entity.raw_request, *entry["requests"]], entry["retrieved_at"])
                if any(r != "missing_text" for r in rejection_reasons(record, self.config.filters)):
                    preliminary[title] = record
                else:
                    media_ids.setdefault("M" + str(entry["page"]["pageid"]), []).append(title)
            except Exception as exc:
                errors[title] = exc
        structured: dict[str, dict] = {}
        for media_id in media_ids:
            entry = self.client.item_get("mediainfo-v1", media_id)
            if not entry:
                cached = self.client.cached(self.config.wikimedia.commons_api, action="wbgetentities", ids=media_id,
                                            props="labels|claims", languages="pl|en")
                if cached:
                    entry = {"entity": cached.get("entities", {}).get(media_id, {}), "request": cached.get("_request_key")}
            if entry:
                structured[media_id] = entry
        for group in batches((v for v in media_ids if v not in structured), self.config.wikimedia.structured_batch_size):
            try:
                response = self.client.get(self.config.wikimedia.commons_api, action="wbgetentities", ids="|".join(group),
                                            props="labels|claims", languages="pl|en")
                for media_id in group:
                    entry = {"entity": response.get("entities", {}).get(media_id, {}), "request": response.get("_request_key")}
                    structured[media_id] = entry
                    self.client.item_put("mediainfo-v1", media_id, entry)
            except Exception as exc:
                for media_id in group:
                    for title in media_ids[media_id]:
                        errors[title] = exc
        results = []
        for candidate, entity in work:
            title = candidate.title
            if title in errors:
                results.append(FetchResult(candidate, entity, error=errors[title]))
                continue
            entry = pages[title]
            media = structured.get("M" + str(entry["page"]["pageid"]), {"entity": {}, "request": None})
            try:
                record = normalize(entry["page"], media["entity"], entity, candidate,
                    [entity.raw_request, *entry["requests"], media["request"]], entry["retrieved_at"])
                results.append(FetchResult(candidate, entity, record=record))
            except Exception as exc:
                results.append(FetchResult(candidate, entity, error=exc))
        return results
