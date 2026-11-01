from __future__ import annotations

from typing import Any, Iterator

from ..config import Config
from ..models import DatasetRecord, Entity, ImageCandidate
from ..storage.checkpoint import CheckpointStore
from .http import HttpClient
from .normalization import normalize
from .wikidata import WikidataSource


class WikimediaSource:
    def __init__(self, client: HttpClient, config: Config):
        self.client, self.config = client, config
        self.wikidata = WikidataSource(client, config)

    def fetch_many(self, work: list[tuple[ImageCandidate, Entity]]):
        from .wikimedia_batch import WikimediaBatch
        return WikimediaBatch(self.client, self.config).fetch(work)

    def discover(self, store: CheckpointStore) -> None:
        from ..pipeline.discovery import discover
        discover(self.config, store, self.wikidata)

    def _query(self, **params: Any) -> dict[str, Any]:
        return self.client.get(self.config.wikimedia.commons_api, action="query", formatversion=2, **params)

    def _category(self, title: str) -> Iterator[dict[str, Any]]:
        continuation: dict[str, Any] = {}
        remaining = self.config.wikimedia.category_member_limit
        while remaining > 0:
            response = self._query(list="categorymembers", cmtitle=title, cmtype="file|subcat",
                                   cmlimit=min(remaining, 500), **continuation)
            members = response.get("query", {}).get("categorymembers", [])
            for member in members:
                yield {**member, "_request_key": response.get("_request_key")}
            remaining -= len(members)
            continuation = response.get("continue", {})
            if not continuation or not members:
                break

    def candidates(self, entity: Entity) -> Iterator[ImageCandidate]:
        config = self.config.wikimedia
        seen: set[str] = set()
        count = 0
        def emit(title: str, method: str, category: str | None = None,
                 depth: int | None = None, raw_request: str | None = None) -> ImageCandidate | None:
            nonlocal count
            title = title.replace("_", " ")
            if not title.startswith("File:"):
                title = "File:" + title
            if title in seen or count >= config.max_files_per_entity:
                return None
            count += 1
            seen.add(title)
            return ImageCandidate(title, method, category, depth, (raw_request,) if raw_request else ())
        for image in entity.images:
            if candidate := emit(image, "wikidata_P18"):
                yield candidate
        if config.use_depicts_search and entity.qid and count < config.max_files_per_entity:
            response = self._query(list="search", srsearch=f"haswbstatement:P180={entity.qid}", srnamespace=6,
                                   srlimit=min(50, config.max_files_per_entity - count))
            for row in response.get("query", {}).get("search", []):
                if candidate := emit(row["title"], "commons_depicts", raw_request=response.get("_request_key")):
                    yield candidate
        queue = [(c if c.startswith("Category:") else "Category:" + c, 0) for c in entity.categories]
        visited: set[str] = set()
        while queue and count < config.max_files_per_entity and len(visited) < config.max_categories:
            title, depth = queue.pop(0)
            if title in visited:
                continue
            visited.add(title)
            for member in self._category(title):
                if member.get("ns") == 14 and depth < config.category_depth:
                    queue.append((member["title"], depth + 1))
                elif member.get("ns") == 6:
                    if candidate := emit(member["title"], "commons_category", title, depth, member.get("_request_key")):
                        yield candidate
                if count >= config.max_files_per_entity:
                    break

    def fetch(self, candidate: ImageCandidate, entity: Entity) -> DatasetRecord:
        continuation: dict[str, Any] = {}
        page: dict[str, Any] | None = None
        raw_requests = [entity.raw_request]
        retrieved_at = None
        # Categories can have continuation even for a single file.
        while True:
            response = self._query(titles=candidate.title, redirects=1, prop="imageinfo|categories",
                iiprop="url|size|mime|sha1|extmetadata", iiextmetadatamultilang=1,
                iiurlwidth=self.config.wikimedia.thumbnail_width, cllimit="max", **continuation)
            raw_requests.append(response.get("_request_key"))
            retrieved_at = retrieved_at or response.get("_retrieved_at")
            pages = response.get("query", {}).get("pages", [])
            if not pages or pages[0].get("missing") is not None:
                raise ValueError(f"Commons file missing: {candidate.title}")
            current = pages[0]
            if page is None:
                page = current
            else:
                page.setdefault("categories", []).extend(current.get("categories", []))
            more = response.get("continue", {})
            # iistart continues *old file revisions*, not the category list.
            # Only current image metadata belongs in this dataset.
            continuation = {"continue": more.get("continue", ""), "clcontinue": more["clcontinue"]} if "clcontinue" in more else {}
            if not continuation:
                break
        media_id = "M" + str(page["pageid"])
        response = self.client.get(self.config.wikimedia.commons_api, action="wbgetentities", ids=media_id,
                                   props="labels|claims", languages="pl|en")
        raw_requests.append(response.get("_request_key"))
        structured = response.get("entities", {}).get(media_id, {})
        return normalize(page, structured, entity, candidate, raw_requests, retrieved_at)
