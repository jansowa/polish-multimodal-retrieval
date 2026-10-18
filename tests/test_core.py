from pl_mmb_translate.cli import parse_bool
from pl_mmb_translate.openai_translator import (
    _compact_json_text,
    _expand_json_text,
    _extract_json,
    _validate_same_shape,
)
from pl_mmb_translate.registry import resolve_target


def test_bool():
    assert parse_bool("True") is True
    assert parse_bool("false") is False


def test_json_extract_fence():
    obj = _extract_json('```json\n{"translations": []}\n```')
    assert obj == {"translations": []}


def test_shape():
    _validate_same_shape({"a": ["x", "y"]}, {"a": ["xx", "yy"]})


def test_registry():
    assert resolve_target("CIRR") == ["cirr"]
    assert "sugarcrepe" in resolve_target("compositionality")


def test_compact_repeated_string_list_roundtrip():
    source = ["<|image_1|>\
Represent the given image.\
"] * 1000
    compacted, spec = _compact_json_text(source)
    assert compacted == ["<|image_1|>\
Represent the given image.\
"]
    translated = ["<|image_1|>\
Przedstaw podany obraz.\
"]
    expanded = _expand_json_text(translated, spec)
    assert len(expanded) == 1000
    assert expanded[0] == translated[0]
    assert expanded[-1] == translated[0]


def test_compact_preserves_distinct_string_order():
    source = ["a", "b", "a", "c", "b"]
    compacted, spec = _compact_json_text(source)
    assert compacted == ["a", "b", "c"]
    assert _expand_json_text(["A", "B", "C"], spec) == ["A", "B", "A", "C", "B"]
