from __future__ import annotations

from collections.abc import Iterator

from datasets import load_dataset

from ..config import Settings
from ..models import ImageRef, TranslationUnit
from .base import DatasetAdapter


class ImageCoDeAdapter(DatasetAdapter):
    PATH = "mteb/imagecode-multi"
    REVISION = "0a27b28e13841897d490a40fb88a003f04d7e314"

    def iter_units(self, settings: Settings, with_image: bool) -> Iterator[TranslationUnit]:
        common = {
            "path": self.PATH,
            "revision": self.REVISION,
            "cache_dir": str(settings.hf_cache_dir),
            "token": settings.hf_token,
        }
        query = load_dataset(**common, name="query", split="test")
        correct_by_qid: dict[str, str] = {}
        corpus_by_id = None
        if with_image:
            qrels = load_dataset(**common, name="qrels", split="test")
            for row in qrels:
                if int(row["score"]) == 1:
                    correct_by_qid[str(row["query-id"])] = str(row["corpus-id"])
            wanted = set(correct_by_qid.values())
            corpus = load_dataset(**common, name="corpus", split="corpus")
            corpus_by_id = {}
            for corpus_row in corpus:
                cid = str(corpus_row["id"])
                if cid in wanted:
                    corpus_by_id[cid] = corpus_row["image"]
                    if len(corpus_by_id) == len(wanted):
                        break

        for index, row in enumerate(query):
            qid = str(row["id"])
            images = []
            if with_image:
                cid = correct_by_qid.get(qid)
                if cid is None or corpus_by_id is None or cid not in corpus_by_id:
                    raise RuntimeError(f"ImageCoDe: no positive image found for query {qid}")
                images = [ImageRef(label="correct_image", value=corpus_by_id[cid])]
            yield TranslationUnit(
                unit_id=f"ImageCoDe:{qid}",
                texts={"text": row["text"]},
                images=images,
                metadata={
                    "dataset": "ImageCoDe",
                    "hf_path": self.PATH,
                    "split": "test",
                    "row_index": index,
                    "query_id": qid,
                },
                prompt_profile="retrieval",
            )
