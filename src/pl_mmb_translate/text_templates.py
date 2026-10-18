"""Translate known MMEB boilerplate locally, leaving only variable text for the model."""
from __future__ import annotations

import re
from typing import Any

from .models import JsonText


# These constants are also included in the run fingerprint.
QUERY_PREFIX_EN = "Find me an everyday image that matches the given caption: "
QUERY_PREFIX_PL = "Znajdź obraz z życia codziennego pasujący do podanego opisu: "
IMAGE_INSTRUCTION_EN = "Represent the given image."
IMAGE_INSTRUCTION_PL = "Przedstaw podany obraz."
QUERY_TEMPLATES = {
    QUERY_PREFIX_EN: QUERY_PREFIX_PL,
    "Given an image, find a similar everyday image with the described changes: ":
        "Na podstawie podanego obrazu znajdź podobny obraz z życia codziennego z następującymi zmianami: ",
    "Find an image to match the fashion image and style note: ":
        "Znajdź obraz pasujący do zdjęcia ubrania i opisu stylu: ",
    "Select the portion of the image that follows the language expressions. ":
        "Wybierz fragment obrazu odpowiadający opisowi. ",
}
IMAGE_TEMPLATES = {
    IMAGE_INSTRUCTION_EN: IMAGE_INSTRUCTION_PL,
    "Represent the given cropped image of the object": "Przedstaw podany wycinek obrazu obiektu",
}
TEMPLATE_VERSION = "mmeb-boilerplate-v2"


def prepare_templates(value: JsonText) -> tuple[JsonText | None, Any]:
    """Return a reduced tree and a local restoration spec. None means fully local.

    Called only for MMEB records. Exact matching avoids interpreting arbitrary
    caption content as a template; lists retain a spec for every original position.
    """
    if isinstance(value, str):
        for english, polish in IMAGE_TEMPLATES.items():
            match = re.fullmatch(
                r"(\s*(?:<\|image_\d+\|>\s*)?)" + re.escape(english) + r"(\s*)", value,
            )
            if match:
                return None, {"kind": "fixed", "value": match[1] + polish + match[2]}
        leading = re.match(r"(\s*(?:<\|image_\d+\|>\s*)?)(.*)", value, flags=re.DOTALL)
        for english, polish in QUERY_TEMPLATES.items():
            if leading[2].startswith(english):
                suffix = leading[2][len(english):]
                prefix = leading[1] + polish
                if not suffix.strip():
                    return None, {"kind": "fixed", "value": prefix + suffix}
                return suffix, {"kind": "prefix", "value": prefix}
        if not value.strip():
            return None, {"kind": "fixed", "value": value}
        return value, None
    if isinstance(value, dict):
        reduced = {}
        children = {}
        for key, child in value.items():
            pending, spec = prepare_templates(child)
            children[key] = spec
            if pending is not None:
                reduced[key] = pending
        return (reduced if reduced else None), {"kind": "dict", "children": children}
    if isinstance(value, list):
        pending_list = []
        children = []
        for child in value:
            pending, spec = prepare_templates(child)
            children.append({"spec": spec, "pending": pending is not None})
            if pending is not None:
                pending_list.append(pending)
        return (pending_list if pending_list else None), {"kind": "list", "children": children}
    raise TypeError(f"Unsupported text type: {type(value).__name__}")


def restore_templates(value: JsonText | None, spec: Any) -> JsonText:
    if spec is None:
        return value
    kind = spec["kind"]
    if kind == "fixed":
        return spec["value"]
    if kind == "prefix":
        return spec["value"] + value
    if kind == "dict":
        return {
            key: restore_templates(value.get(key) if value is not None else None, child)
            for key, child in spec["children"].items()
        }
    if kind == "list":
        pending = iter(value or [])
        return [
            restore_templates(next(pending) if child["pending"] else None, child["spec"])
            for child in spec["children"]
        ]
    raise ValueError(f"Unknown template spec: {kind!r}")
