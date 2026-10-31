from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any

from ..models import DatasetRecord, Entity, ImageCandidate
from ..storage.checkpoint import association
from .wikidata import claim_values, localized


class TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.languages: dict[str, list[str]] = {}
        self.stack: list[str | None] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"br", "hr", "img", "input", "meta", "link", "wbr"}:
            self.parts.append(" ")
            return
        lang = dict(attrs).get("lang") or (self.stack[-1] if self.stack else None)
        self.stack.append(lang)
        self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if self.stack:
            self.stack.pop()
        self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)
        if self.stack and self.stack[-1]:
            self.languages.setdefault(self.stack[-1], []).append(data)


def clean_text(value: Any) -> str | None:
    if not isinstance(value, (str, int, float)):
        return None
    parser = TextParser()
    parser.feed(str(value))
    return re.sub(r"\s+", " ", "".join(parser.parts)).strip() or None


def metadata_value(metadata: dict[str, Any], key: str) -> Any:
    # MediaWiki serializes an empty extmetadata map as [] in some responses.
    if not isinstance(metadata, dict):
        return None
    value = metadata.get(key, {})
    return value.get("value") if isinstance(value, dict) else value


def description_languages(value: Any) -> tuple[str | None, dict[str, str]]:
    if isinstance(value, dict):
        translations = {lang: text for lang, raw in value.items()
                        if not lang.startswith("_") and (text := clean_text(raw))}
        parts = list(translations.values())
        if untagged := clean_text(value.get("_")):
            parts.append(untagged)
        return " | ".join(dict.fromkeys(parts)) or None, translations
    parser = TextParser()
    parser.feed(value if isinstance(value, str) else "")
    return clean_text(value), {lang: re.sub(r"\s+", " ", " ".join(parts)).strip()
                               for lang, parts in parser.languages.items()}


def normalize(page: dict[str, Any], structured: dict[str, Any], entity: Entity,
              candidate: ImageCandidate, raw_requests: list[str], retrieved_at: str | None) -> DatasetRecord:
    info = page.get("imageinfo", [{}])[0]
    if not info.get("url") or not info.get("descriptionurl") or not page.get("pageid"):
        raise ValueError(f"Missing source image information: {candidate.title}")
    metadata = info.get("extmetadata", {})
    description, translations = description_languages(metadata_value(metadata, "ImageDescription"))
    depicts = [v["id"] for v in claim_values(structured, "P180") if isinstance(v, dict) and "id" in v]
    texts: list[dict[str, str | None]] = []
    for lang in ("pl", "en"):
        if caption := localized(structured, "labels", lang):
            texts.append(dict(language=lang, text=caption, source="structured_caption", quality="A"))
        if caption := translations.get(lang):
            texts.append(dict(language=lang, text=caption, source="image_description", quality="B"))
    if description and not translations:
        # A UI language parameter does not establish the language of free text.
        texts.append(dict(language="und", text=description, source="image_description", quality="B"))
    title = page.get("title", candidate.title)
    texts.append(dict(language="und", text=title.removeprefix("File:").rsplit(".", 1)[0].replace("_", " "),
                      source="image_title", quality="B"))
    if depicts:
        texts.append(dict(language="zxx", text="; ".join(depicts), source="depicts", quality="C"))
    for lang in ("pl", "en"):
        if text := getattr(entity, "description_" + lang):
            texts.append(dict(language=lang, text=text, source="entity_description", quality="D"))
        if text := getattr(entity, "label_" + lang):
            texts.append(dict(language=lang, text=text, source="entity_title", quality="D"))
    for lang, raw_text in candidate.context_description.items():
        if text := clean_text(raw_text):
            texts.append(dict(language=lang, text=text, source=candidate.context_source, quality="D"))
    if candidate.category:
        texts.append(dict(language="und", text=candidate.category, source="category_title", quality="D"))
    selected = {lang: next((t for t in texts if t["language"] == lang), None) for lang in ("pl", "en")}
    source = {lang: t["source"] if t else None for lang, t in selected.items()}
    quality = {lang: t["quality"] if t else None for lang, t in selected.items()}
    lat, lon = metadata_value(metadata, "GPSLatitude"), metadata_value(metadata, "GPSLongitude")
    location = f"{lat}, {lon}" if lat is not None and lon is not None else None
    # Wikibase coordinates describe the media file only when supplied on its MediaInfo.
    if not location:
        coords = claim_values(structured, "P1259")
        if coords and isinstance(coords[0], dict):
            location = f"{coords[0].get('latitude')}, {coords[0].get('longitude')}"
    return DatasetRecord(
        id=f"wikimedia:{page['pageid']}", source="wikimedia", source_page_url=info["descriptionurl"],
        image_url=info.get("thumburl") or info["url"], image_original_url=info["url"], image_filename=title,
        wikidata_qid=entity.qid, entity_label_pl=entity.label_pl, entity_label_en=entity.label_en,
        entity_description_pl=entity.description_pl, entity_description_en=entity.description_en,
        caption_pl=selected["pl"]["text"] if selected["pl"] else None,
        caption_en=selected["en"]["text"] if selected["en"] else None,
        description=description, depicts=depicts,
        categories=[c["title"] for c in page.get("categories", [])],
        creator=clean_text(metadata_value(metadata, "Artist")),
        attribution=clean_text(metadata_value(metadata, "Attribution")),
        date=clean_text(metadata_value(metadata, "DateTimeOriginal")), location=location,
        license=clean_text(metadata_value(metadata, "LicenseShortName")),
        license_url=clean_text(metadata_value(metadata, "LicenseUrl")),
        width=info.get("width"), height=info.get("height"), mime_type=info.get("mime"),
        image_width=info.get("thumbwidth", info.get("width")),
        image_height=info.get("thumbheight", info.get("height")),
        retrieved_at=retrieved_at, caption_source=source, caption_quality=quality,
        retrieval_method=candidate.method, seed_entity=entity.seed, source_file_id=str(page["pageid"]),
        source_sha1=info.get("sha1"), structured_revision=structured.get("lastrevid"),
        raw_requests=list(dict.fromkeys(v for v in [*raw_requests, *candidate.raw_requests] if v)), text_candidates=texts,
        associations=[association(entity, candidate)])
