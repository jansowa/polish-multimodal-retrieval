from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from datasets import load_dataset

from ..config import Settings
from ..models import TranslationUnit
from .base import DatasetAdapter
from .hf_generic import HFGenericAdapter


class MBEIRQueryAdapter(HFGenericAdapter):
    def __init__(self):
        super().__init__(
            dataset_name="mbeir_queries",
            path="TIGER-Lab/M-BEIR",
            config="query",
            split="test",
            text_fields=("query_txt",),
            image_fields=("query_img_path",),
            id_field="qid",
            prompt_profile="retrieval",
            image_root_env="MBEIR_IMAGE_ROOT",
        )


class MBEIRCandidateAdapter(HFGenericAdapter):
    def __init__(self, global_pool: bool = False):
        split = "mbeir_global" if global_pool else "mbeir_local"
        name = "mbeir_candidates_global" if global_pool else "mbeir_candidates"
        super().__init__(
            dataset_name=name,
            path="TIGER-Lab/M-BEIR",
            config="cand_pool",
            split=split,
            text_fields=("txt",),
            image_fields=("img_path",),
            id_field="did",
            prompt_profile="retrieval",
            image_root_env="MBEIR_IMAGE_ROOT",
        )


class MBEIRInstructionsAdapter(DatasetAdapter):
    """Translate both query and candidate instruction TSVs exposed by the HF `instructions` config."""

    def iter_units(self, settings: Settings, with_image: bool) -> Iterator[TranslationUnit]:
        ds = load_dataset(
            "TIGER-Lab/M-BEIR",
            "instructions",
            split="instructions",
            cache_dir=str(settings.hf_cache_dir),
            token=settings.hf_token,
            streaming=settings.hf_streaming,
        )
        for index, row in enumerate(ds):
            # The repository has changed TSV details over time. Select instruction-like string columns
            # dynamically instead of coupling the translator to one header revision.
            texts: dict[str, Any] = {
                key: value
                for key, value in row.items()
                if "instruction" in key.lower() and isinstance(value, str) and value.strip()
            }
            if not texts:
                # Conservative fallback: translate long natural-language strings, never IDs/codes.
                texts = {
                    key: value
                    for key, value in row.items()
                    if isinstance(value, str)
                    and len(value.split()) >= 3
                    and not key.lower().endswith("id")
                }
            if not texts:
                continue
            yield TranslationUnit(
                unit_id=f"mbeir_instructions:{index}",
                texts=texts,
                images=[],
                metadata={
                    "dataset": "mbeir_instructions",
                    "hf_path": "TIGER-Lab/M-BEIR",
                    "hf_config": "instructions",
                    "split": "instructions",
                    "row_index": index,
                },
                prompt_profile="retrieval",
            )
