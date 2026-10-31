from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .adapters.base import DatasetAdapter
from .adapters.hf_generic import HFGenericAdapter
from .adapters.imagecode import ImageCoDeAdapter
from .adapters.mbeir import MBEIRCandidateAdapter, MBEIRInstructionsAdapter, MBEIRQueryAdapter
from .adapters.mmeb import MMEBAdapter


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    description: str
    factory: Callable[[], DatasetAdapter]
    expected_examples: int | None = None


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


MMEB_CONFIGS = [
    "A-OKVQA", "CIFAR-100", "CIRR", "ChartQA", "Country211", "DocVQA", "EDIS",
    "FashionIQ", "GQA", "HatefulMemes", "ImageNet-1K", "ImageNet-A", "ImageNet-R",
    "InfographicsVQA", "MSCOCO", "MSCOCO_i2t", "MSCOCO_t2i", "N24News", "NIGHTS",
    "OK-VQA", "OVEN", "ObjectNet", "Place365", "RefCOCO", "RefCOCO-Matching", "SUN397",
    "ScienceQA", "TextVQA", "VOC2007", "VisDial", "Visual7W-Pointing", "Visual7W",
    "VisualNews_i2t", "VisualNews_t2i", "VizWiz", "WebQA", "Wiki-SS-NQ",
]

QA_CONFIGS = {"A-OKVQA", "GQA", "OK-VQA", "ScienceQA", "TextVQA", "Visual7W", "VizWiz", "WebQA"}
RETRIEVAL_CONFIGS = {
    "CIRR", "EDIS", "FashionIQ", "MSCOCO", "MSCOCO_i2t", "MSCOCO_t2i", "NIGHTS",
    "RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing", "VisualNews_i2t", "VisualNews_t2i",
    "VisDial", "Wiki-SS-NQ",
}

DATASETS: dict[str, DatasetSpec] = {}
for config_name in MMEB_CONFIGS:
    key = _key(config_name)
    profile = "qa" if config_name in QA_CONFIGS else "retrieval" if config_name in RETRIEVAL_CONFIGS else "faithful"
    DATASETS[key] = DatasetSpec(
        key=key,
        description=f"MMEB-eval / {config_name}",
        factory=lambda c=config_name, p=profile: MMEBAdapter(c, prompt_profile=p),
        expected_examples=1000,
    )

DATASETS.update(
    {
        "sugarcrepe": DatasetSpec(
            "sugarcrepe",
            "SugarCrepe compositional caption pairs",
            lambda: HFGenericAdapter(
                dataset_name="SugarCrepe",
                path="mteb/SUGARCREPE_fmt",
                revision="e9e0e3ce40c94023c819d3413edb5d090ec9f0d8",
                split="train",
                text_fields=("caption", "negative_caption"),
                image_fields=("images",),
                prompt_profile="contrastive",
            ),
            expected_examples=7511,
        ),
        "winoground": DatasetSpec(
            "winoground",
            "Winoground paired images and captions",
            lambda: HFGenericAdapter(
                dataset_name="Winoground",
                path="facebook/winoground",
                revision="b400e173549071916ad1b3d449293bc8d8b4b763",
                split="test",
                text_fields=("caption_0", "caption_1"),
                image_fields=("image_0", "image_1"),
                id_field="id",
                prompt_profile="contrastive",
            ),
            expected_examples=400,
        ),
        "aro_visual_relation": DatasetSpec(
            "aro_visual_relation",
            "ARO Visual Relation contrastive captions",
            lambda: HFGenericAdapter(
                dataset_name="AROVisualRelation",
                path="mteb/ARO-Visual-Relation",
                revision="53c1115ab5871ab33b09d5bb8856236d46bdaf2d",
                split="test",
                text_fields=("true_caption", "false_caption"),
                image_fields=("image",),
                prompt_profile="contrastive",
            ),
        ),
        "aro_visual_attribution": DatasetSpec(
            "aro_visual_attribution",
            "ARO Visual Attribution contrastive captions",
            lambda: HFGenericAdapter(
                dataset_name="AROVisualAttribution",
                path="mteb/ARO-Visual-Attribution",
                revision="25fe9b460426ff67cdb0608a737c94f8f49a901e",
                split="test",
                text_fields=("true_caption", "false_caption"),
                image_fields=("image",),
                prompt_profile="contrastive",
            ),
        ),
        "imagecode": DatasetSpec("imagecode", "ImageCoDe", ImageCoDeAdapter),
        "mbeir_queries": DatasetSpec("mbeir_queries", "M-BEIR test queries", MBEIRQueryAdapter),
        "mbeir_candidates": DatasetSpec(
            "mbeir_candidates", "M-BEIR local evaluation candidate pool", MBEIRCandidateAdapter
        ),
        "mbeir_candidates_global": DatasetSpec(
            "mbeir_candidates_global",
            "M-BEIR global test candidate pool (very large)",
            lambda: MBEIRCandidateAdapter(global_pool=True),
        ),
        "mbeir_instructions": DatasetSpec(
            "mbeir_instructions", "M-BEIR query/candidate retrieval instructions", MBEIRInstructionsAdapter
        ),
    }
)

MMEB_CORE = [
    "cirr", "fashioniq", "mscoco_i2t", "mscoco_t2i", "visualnews_i2t", "visualnews_t2i",
    "edis", "gqa", "visual7w", "a_okvqa", "ok_vqa", "vizwiz", "refcoco",
    "visual7w_pointing", "webqa",
]
COMPOSITIONALITY = [
    "sugarcrepe", "winoground", "aro_visual_relation", "aro_visual_attribution", "imagecode"
]

BENCHMARKS: dict[str, list[str]] = {
    "mmeb": [_key(x) for x in MMEB_CONFIGS],
    "mmeb_all": [_key(x) for x in MMEB_CONFIGS],
    "mmeb_core": MMEB_CORE,
    "compositionality": COMPOSITIONALITY,
    "mieb": COMPOSITIONALITY,
    "mieb_curated": COMPOSITIONALITY,
    "mbeir": ["mbeir_instructions", "mbeir_queries", "mbeir_candidates"],
    "mbeir_local": ["mbeir_instructions", "mbeir_queries", "mbeir_candidates"],
    "mbeir_global": ["mbeir_instructions", "mbeir_queries", "mbeir_candidates_global"],
    "pl_v1": list(dict.fromkeys(MMEB_CORE + COMPOSITIONALITY)),
}


def resolve_target(target: str) -> list[str]:
    normalized = _key(target)
    if normalized in DATASETS:
        return [normalized]
    if normalized in BENCHMARKS:
        return BENCHMARKS[normalized]
    raise KeyError(target)
