from __future__ import annotations

import logging

from ..config import Config
from ..models import Entity
from ..sources.rate_limit import RetryLater
from ..sources.wikidata import WikidataSource
from ..storage.checkpoint import CheckpointStore

LOG = logging.getLogger(__name__)


def seed(config: Config, store: CheckpointStore) -> None:
    limit = config.discovery.max_entities
    def enqueue(entity: Entity, discovered: bool = False) -> None:
        if store.entity_count() < limit:
            store.add_entity(entity, discovered)

    for qid in config.discovery.seeds:
        enqueue(Entity(qid=qid, seed=qid))
    for category in config.discovery.categories:
        title = category if category.startswith("Category:") else "Category:" + category
        enqueue(Entity(qid=None, seed=title, categories=[title]), discovered=True)


def discover(config: Config, store: CheckpointStore, wikidata: WikidataSource,
             max_batches: int | None = None, allow_sparql: bool = True) -> None:
    """Hydrate known IDs first. SPARQL is optional and never a prerequisite."""
    seed(config, store)
    attempted: set[str] = set()
    batch_count = 0

    def hydrate() -> None:
        nonlocal batch_count
        while max_batches is None or batch_count < max_batches:
            pending = [(k, e) for k, e in store.entities("discover")
                       if k not in attempted and store.ready("discover:" + k)]
            if not pending:
                return
            pending = pending[:config.wikimedia.entity_batch_size]
            batch_count += 1
            attempted.update(k for k, _ in pending)
            if hasattr(wikidata, "entities_many"):
                results = wikidata.entities_many([e for _, e in pending])
            else:
                results = {}
                for key, previous in pending:
                    try:
                        results[previous.qid] = wikidata.entity(previous.qid, previous.seed, previous.depth)
                    except Exception as exc:
                        results[previous.qid] = exc
            for key, previous in pending:
                result = results.get(previous.qid, ValueError("Entity missing from batch response"))
                task = "discover:" + key
                if isinstance(result, RetryLater):
                    store.defer(task, result)
                elif isinstance(result, Exception):
                    LOG.warning("Discovery failed for %s: %s", key, result)
                    store.failure(task, result)
                else:
                    if result.depth < config.discovery.max_depth:
                        for qid in result.links[:config.discovery.max_links_per_entity]:
                            if store.entity_count() < config.discovery.max_entities:
                                store.add_entity(Entity(qid=qid, seed=result.seed, depth=result.depth + 1))
                    store.save_entity(key, result)
                    store.clear_failure(task)
                    LOG.info("Discovered %s: %s", key, result.label_pl or result.label_en)

    hydrate()
    if not allow_sparql or not config.discovery.enable_sparql:
        return
    for class_qid in config.discovery.sparql_classes:
        task = "query:" + class_qid
        if store.entity_count() >= config.discovery.max_entities:
            break
        if store.task_completed(task) or not store.ready(task):
            continue
        try:
            for qid in wikidata.by_class(class_qid):
                if store.entity_count() < config.discovery.max_entities:
                    store.add_entity(Entity(qid=qid, seed=class_qid))
            store.complete_task(task)
        except RetryLater as exc:
            store.defer(task, exc)
            break
        except Exception as exc:
            LOG.warning("Discovery query failed for %s: %s", class_qid, exc)
            store.failure(task, exc)
    hydrate()
