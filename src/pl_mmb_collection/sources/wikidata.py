from __future__ import annotations

from typing import Any

from ..config import Config
from ..models import Entity
from .http import HttpClient
from .batching import batches


def claim_values(item: dict[str, Any], prop: str) -> list[Any]:
    result = []
    for claim in item.get("claims", item.get("statements", {})).get(prop, []):
        if claim.get("rank") == "deprecated":
            continue
        snak = claim.get("mainsnak", {})
        if snak.get("snaktype") == "value" and "datavalue" in snak:
            result.append(snak["datavalue"]["value"])
    return result


def localized(item: dict[str, Any], field: str, lang: str) -> str | None:
    return item.get(field, {}).get(lang, {}).get("value") or None


def parse_entity(item: dict[str, Any], seed: str, depth: int, properties: list[str]) -> Entity:
    links = []
    for prop in properties:
        for value in claim_values(item, prop):
            if isinstance(value, dict) and str(value.get("id", "")).startswith("Q"):
                links.append(value["id"])
    categories = [str(c) for c in claim_values(item, "P373")]
    return Entity(qid=item["id"], seed=seed, depth=depth,
                  label_pl=localized(item, "labels", "pl"), label_en=localized(item, "labels", "en"),
                  description_pl=localized(item, "descriptions", "pl"),
                  description_en=localized(item, "descriptions", "en"),
                  categories=categories, images=[str(v) for v in claim_values(item, "P18")],
                  links=list(dict.fromkeys(links)), revision=item.get("lastrevid"))


class WikidataSource:
    def __init__(self, client: HttpClient, config: Config):
        self.client, self.config = client, config

    def entity(self, qid: str, seed: str, depth: int) -> Entity:
        result = self.client.get(self.config.wikimedia.wikidata_api, action="wbgetentities", ids=qid,
                                 props="labels|descriptions|claims|sitelinks", languages="pl|en", redirects="yes")
        return self._entity_from_response(result, qid, seed, depth)

    def _entity_from_response(self, result: dict[str, Any], qid: str, seed: str, depth: int) -> Entity:
        items = result.get("entities", {})
        item = items.get(qid)
        if item is None:
            redirects = {r["from"]: r["to"] for r in result.get("redirects", [])}
            item = items.get(redirects.get(qid))
        if not item or "missing" in item:
            raise ValueError(f"Wikidata entity missing: {qid}")
        entity = parse_entity(item, seed, depth, self.config.discovery.properties)
        entity.raw_request = result.get("_request_key")
        commons = item.get("sitelinks", {}).get("commonswiki", {}).get("title", "")
        if commons.startswith("Category:") and commons not in entity.categories:
            entity.categories.append(commons)
        return entity

    def entities_many(self, entities: list[Entity]) -> dict[str, Entity | Exception]:
        results: dict[str, Entity | Exception] = {}
        pending: dict[str, Entity] = {}
        params = dict(action="wbgetentities", props="labels|descriptions|claims|sitelinks",
                      languages="pl|en", redirects="yes")
        for entity in entities:
            cached = self.client.cached(self.config.wikimedia.wikidata_api, ids=entity.qid, **params)
            if cached:
                try:
                    results[entity.qid] = self._entity_from_response(cached, entity.qid, entity.seed, entity.depth)
                except Exception as exc:
                    results[entity.qid] = exc
            else:
                pending[entity.qid] = entity
        for qids in batches(pending, self.config.wikimedia.entity_batch_size):
            try:
                response = self.client.get(self.config.wikimedia.wikidata_api, ids="|".join(qids), **params)
            except Exception as exc:
                results.update({qid: exc for qid in qids})
                continue
            for qid in qids:
                previous = pending[qid]
                try:
                    results[qid] = self._entity_from_response(response, qid, previous.seed, previous.depth)
                except Exception as exc:
                    results[qid] = exc
        return results

    def by_class(self, class_qid: str) -> list[str]:
        discovery = self.config.discovery
        # Identifiers and limits are validated before interpolation.
        query = f"""SELECT DISTINCT ?item WHERE {{
          {{ ?item wdt:P17 wd:{discovery.country_qid} }} UNION
          {{ ?item wdt:P27 wd:{discovery.country_qid} }} UNION
          {{ ?item wdt:P495 wd:{discovery.country_qid} }}
          ?item wdt:P31/wdt:P279* wd:{class_qid} .
        }} LIMIT {min(discovery.max_entities, discovery.sparql_limit_per_class)}"""
        response = self.client.get(self.config.wikimedia.sparql_endpoint, query=query)
        return [row["item"]["value"].rsplit("/", 1)[-1]
                for row in response.get("results", {}).get("bindings", [])]
