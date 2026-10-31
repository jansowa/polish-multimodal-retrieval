from __future__ import annotations

import asyncio
import json
import re
import time
import unicodedata
from typing import Any

from openai import AsyncOpenAI

from .config import Settings
from .image_utils import image_to_data_uri
from .models import JsonText, TranslationUnit
from .prompts import REVIEW_SYSTEM_PROMPT, SYSTEM_PROMPT, review_payload, user_payload
from .text_templates import prepare_templates, restore_templates


class TranslationError(RuntimeError):
    pass


CONTRASTIVE_VALIDATION_VERSION = "distinct-captions-v1"


def _validate_contrastive(source: JsonText, translated: JsonText) -> None:
    """Reject literal collapse of distinct captions; this is not a semantic score."""
    def normalize(text: str) -> str:
        return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())

    seen: dict[str, tuple[str, str]] = {}

    def visit(a, b, path):
        if isinstance(a, str):
            # English articles have no direct Polish equivalent.
            original = normalize(re.sub(r"\b(?:a|an|the)\b", "", a, flags=re.IGNORECASE))
            target = normalize(b)
            if original and target:
                previous = seen.get(target)
                if previous and previous[0] != original:
                    raise TranslationError(
                        f"Distinct source captions collapsed into identical Polish at {previous[1]} and {path}. "
                        "Preserve their differing objects, roles, attributes or relations."
                    )
                seen[target] = (original, path)
        elif isinstance(a, dict):
            for key in a:
                visit(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, list):
            for index, (left, right) in enumerate(zip(a, b, strict=True)):
                visit(left, right, f"{path}[{index}]")

    visit(source, translated, "texts")


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError as exc:
            raise TranslationError(
                f"Model returned invalid JSON: {exc}; response={text[:1000]!r}"
            ) from exc
    raise TranslationError(f"Model returned no JSON object: {text[:1000]!r}")


def _validate_same_shape(source: JsonText, translated: Any, path: str = "texts") -> None:
    if isinstance(source, str):
        if not isinstance(translated, str):
            raise TranslationError(
                f"Shape mismatch at {path}: expected string, got {type(translated).__name__}"
            )
        if not source.strip() and translated != source:
            raise TranslationError(f"Empty/whitespace string changed at {path}")
        if source.strip() and not translated.strip():
            raise TranslationError(f"Nonempty source translated as empty at {path}")
        # Cover image/control tokens as well as common named template placeholders.
        token_pattern = r"<\|[^<>\r\n]+\|>|</?image(?:[_ -]?\d+)?>|\{\{[^{}\r\n]+\}\}|(?<!\{)\{[A-Za-z_][A-Za-z_0-9]*\}(?!\})"
        if re.findall(token_pattern, source) != re.findall(token_pattern, translated):
            raise TranslationError(f"Placeholder sequence changed at {path}")
        return
    if isinstance(source, list):
        if not isinstance(translated, list) or len(source) != len(translated):
            raise TranslationError(
                f"Shape mismatch at {path}: expected list of len {len(source)}"
            )
        for i, (a, b) in enumerate(zip(source, translated, strict=True)):
            _validate_same_shape(a, b, f"{path}[{i}]")
        return
    if isinstance(source, dict):
        if not isinstance(translated, dict) or set(source) != set(translated):
            raise TranslationError(f"Shape/key mismatch at {path}")
        for key in source:
            _validate_same_shape(source[key], translated[key], f"{path}.{key}")
        return
    raise TranslationError(
        f"Unsupported source text shape at {path}: {type(source).__name__}"
    )


def _compact_json_text(value: JsonText) -> tuple[JsonText, Any]:
    """Compact repeated string lists before sending them to the model.

    MMEB retrieval rows can contain hundreds/thousands of candidate entries whose text is
    identical (for example ``<|image_1|>\\nRepresent the given image.`` for image-only
    candidates). Asking the model to emit the same translation hundreds of times wastes
    output tokens and can truncate otherwise valid JSON.

    The returned expansion spec is local-only metadata. The model still sees one logical
    benchmark example, and the original list length/order is restored after translation.
    """
    if isinstance(value, str):
        return value, None

    if isinstance(value, dict):
        compacted: dict[str, JsonText] = {}
        specs: dict[str, Any] = {}
        changed = False
        for key, child in value.items():
            compact_child, child_spec = _compact_json_text(child)
            compacted[key] = compact_child
            specs[key] = child_spec
            changed = changed or child_spec is not None
        return compacted, {"kind": "dict", "children": specs} if changed else None

    if isinstance(value, list):
        # The important MMEB case: long candidate lists with identical/repeated strings.
        if all(isinstance(item, str) for item in value):
            unique: list[str] = []
            positions: dict[str, int] = {}
            indices: list[int] = []
            for item in value:
                idx = positions.get(item)
                if idx is None:
                    idx = len(unique)
                    positions[item] = idx
                    unique.append(item)
                indices.append(idx)
            if len(unique) < len(value):
                return unique, {
                    "kind": "dedupe_string_list",
                    "indices": indices,
                    "original_len": len(value),
                }
            return value, None

        compacted_list: list[JsonText] = []
        child_specs: list[Any] = []
        changed = False
        for child in value:
            compact_child, child_spec = _compact_json_text(child)
            compacted_list.append(compact_child)
            child_specs.append(child_spec)
            changed = changed or child_spec is not None
        return (
            compacted_list,
            {"kind": "list", "children": child_specs} if changed else None,
        )

    raise TranslationError(f"Unsupported source text type: {type(value).__name__}")


def _expand_json_text(value: JsonText, spec: Any) -> JsonText:
    if spec is None:
        return value

    kind = spec.get("kind")
    if kind == "dedupe_string_list":
        if not isinstance(value, list):
            raise TranslationError("Compacted translation is not a list")
        indices = spec["indices"]
        try:
            return [value[index] for index in indices]
        except (IndexError, TypeError) as exc:
            raise TranslationError("Model changed compacted list shape") from exc

    if kind == "dict":
        if not isinstance(value, dict):
            raise TranslationError("Compacted translation is not an object")
        return {
            key: _expand_json_text(value[key], child_spec)
            for key, child_spec in spec["children"].items()
        }

    if kind == "list":
        if not isinstance(value, list) or len(value) != len(spec["children"]):
            raise TranslationError("Model changed compacted nested-list shape")
        return [
            _expand_json_text(child, child_spec)
            for child, child_spec in zip(value, spec["children"], strict=True)
        ]

    raise TranslationError(f"Unknown expansion spec: {kind!r}")


def _compact_texts(texts: dict[str, JsonText]) -> tuple[dict[str, JsonText], dict[str, Any]]:
    compacted: dict[str, JsonText] = {}
    specs: dict[str, Any] = {}
    for key, value in texts.items():
        compacted[key], specs[key] = _compact_json_text(value)
    return compacted, specs


def _exception_chain(exc: BaseException) -> str:
    parts: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return " <- ".join(parts)


class OpenAITranslator:
    """Keep exactly one benchmark example in each translation or review request."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.request_timeout,
            max_retries=0,  # retries are controlled here for predictable resume behavior
        )

    async def close(self) -> None:
        await self.client.close()

    def _build_content(
        self, unit: TranslationUnit, texts_for_model: dict[str, JsonText], with_image: bool
    ) -> list[dict[str, Any]]:
        # IMPORTANT: one TranslationUnit == one benchmark example == one API request.
        # Repeated values inside the example may be compacted, but examples are never mixed.
        record = {"id": unit.unit_id, "texts": texts_for_model}
        content: list[dict[str, Any]] = [
            {"type": "text", "text": user_payload([record], unit.prompt_profile)}
        ]

        if not with_image:
            return content

        images = unit.images[: self.settings.max_images_per_record]
        if len(images) > self.settings.max_images_per_request:
            raise TranslationError(
                f"Example {unit.unit_id!r} needs {len(images)} images after "
                f"MAX_IMAGES_PER_RECORD, but MAX_IMAGES_PER_REQUEST="
                f"{self.settings.max_images_per_request}. A benchmark example is never "
                "split across requests; lower MAX_IMAGES_PER_RECORD or raise the server limit."
            )

        for image in images:
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Context image for this benchmark example, label={image.label!r}. "
                        "Use it only to disambiguate the translation; never repair benchmark semantics."
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_uri(image, self.settings)},
                }
            )
        return content

    async def _request_translation(
        self, unit: TranslationUnit, source: dict[str, JsonText], content: list[dict],
        *, review: bool = False,
    ) -> tuple[dict[str, JsonText], list[dict[str, str]]]:
        messages = [
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT if review else SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        last_error: Exception | None = None
        started = time.monotonic()
        for attempt in range(self.settings.max_retries + 1):
            text = ""
            finish_reason = None
            try:
                response = await self.client.chat.completions.create(
                    model=self.settings.model,
                    messages=messages,
                    temperature=self.settings.temperature,
                    max_tokens=self.settings.max_output_tokens,
                )
                choice = response.choices[0]
                text = choice.message.content or ""
                finish_reason = getattr(choice, "finish_reason", None)
                if not text.strip():
                    raise TranslationError(
                        f"Model returned empty content (finish_reason={finish_reason!r})"
                    )
                if finish_reason not in (None, "stop"):
                    raise TranslationError(
                        f"Model response ended with finish_reason={finish_reason!r}; "
                        f"response_prefix={text[:500]!r}"
                    )

                obj = _extract_json(text)
                items = obj.get("translations")
                if not isinstance(items, list) or len(items) != 1:
                    raise TranslationError(
                        "JSON must contain exactly one item in top-level 'translations' list"
                    )
                item = items[0]
                if not isinstance(item, dict) or item.get("id") != unit.unit_id:
                    got_id = item.get("id") if isinstance(item, dict) else None
                    raise TranslationError(
                        f"Expected translation id={unit.unit_id!r}, got {got_id!r}"
                    )
                translated_compact = item.get("texts")
                if not isinstance(translated_compact, dict):
                    raise TranslationError("Translation must contain object 'texts'")
                if set(translated_compact) != set(source):
                    raise TranslationError(
                        f"Text keys changed for {unit.unit_id}: "
                        f"expected={sorted(source)} got={sorted(translated_compact)}"
                    )
                _validate_same_shape(source, translated_compact, f"{unit.unit_id}.texts")
                if unit.prompt_profile == "contrastive":
                    _validate_contrastive(source, translated_compact)
                flags = obj.get("quality_flags", [])
                if review and (
                    "quality_flags" not in obj or not isinstance(flags, list)
                    or any(
                        not isinstance(flag, dict) or set(flag) != {"path", "reason"}
                        or any(not isinstance(flag.get(k), str) or not flag[k].strip()
                               for k in ("path", "reason"))
                        for flag in flags
                    )
                ):
                    raise TranslationError("Review must include quality_flags: [{path: string, reason: string}]")
                if self.settings.save_drafts:
                    usage = getattr(response, "usage", None)
                    unit.metadata.setdefault("translation_audit", {}).setdefault("stages", {})[
                        "review" if review else "draft"
                    ] = {
                        "seconds": round(time.monotonic() - started, 3),
                        "attempts": attempt + 1,
                        "usage": {
                            name: getattr(usage, name, None)
                            for name in ("prompt_tokens", "completion_tokens", "total_tokens")
                        },
                    }
                return translated_compact, flags if review else []
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # API errors + malformed model output
                last_error = exc
                if attempt >= self.settings.max_retries:
                    break
                if isinstance(exc, TranslationError):
                    # Feedback makes a retry useful even with temperature=0. Keep only
                    # the most recent failed response, rather than growing the context.
                    feedback = (
                        "The previous attempt exhausted its output budget without a translation. "
                        "Choose the closest faithful rendering of the supplied text, including its defects; "
                        "do not invent a missing word or explore alternative reconstructions. Return concise JSON. "
                        if not text and finish_reason == "length" else ""
                    )
                    messages = messages[:2] + (
                        [{"role": "assistant", "content": text}] if text else []
                    ) + [{"role": "user", "content": (
                            feedback +
                            f"Validation failed: {exc}. Correct this error using the original "
                            "source. Return the complete required JSON object only."
                        )}]
                await asyncio.sleep(min(2**attempt, 8))

        if last_error is not None:
            raise TranslationError(_exception_chain(last_error)) from last_error
        raise TranslationError("Unknown translation failure")

    async def translate_one(
        self, unit: TranslationUnit, *, with_image: bool
    ) -> dict[str, Any]:
        # Limit boilerplate matching to its known dataset family.
        use_templates = unit.metadata.get("hf_path") == "TIGER-Lab/MMEB-eval"
        pending, template_spec = (
            prepare_templates(unit.texts) if use_templates else (unit.texts, None)
        )
        flags: list[dict[str, str]] = []
        if pending is None:
            translated = restore_templates(None, template_spec)
        else:
            texts_for_model, expansion_specs = _compact_texts(pending)
            # Image work stays off the event loop.
            content = await asyncio.to_thread(
                self._build_content, unit, texts_for_model, with_image
            )
            draft, _ = await self._request_translation(unit, texts_for_model, content)
            if self.settings.save_drafts:
                unit.metadata["translation_audit"].update(
                    source_compact=texts_for_model, draft_compact=draft,
                )
            if self.settings.quality_review:
                review_content = [{"type": "text", "text": review_payload(
                    unit.unit_id, texts_for_model, draft, unit.prompt_profile
                )}]
                draft, flags = await self._request_translation(
                    unit, texts_for_model, review_content, review=True
                )
            if self.settings.save_drafts:
                audit = unit.metadata["translation_audit"]
                audit["reviewed_compact"] = draft if self.settings.quality_review else None
                audit["review_changed"] = (
                    draft != audit["draft_compact"] if self.settings.quality_review else None
                )
            expanded = {
                key: _expand_json_text(draft[key], expansion_specs[key])
                for key in texts_for_model
            }
            translated = restore_templates(expanded, template_spec)
        _validate_same_shape(unit.texts, translated, f"{unit.unit_id}.texts")
        # Quality flags stay outside benchmark text and are exported with the record.
        unit.metadata["translation_quality"] = {
            "reviewed": self.settings.quality_review and pending is not None,
            "flags": flags,
            "flag_paths": "compact_source",
            "templates_applied": use_templates,
        }
        if flags:
            unit.metadata["translation_quality"]["review_source"] = texts_for_model
        return translated
