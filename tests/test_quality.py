"""Offline regression tests: no model server or Hugging Face access required."""
import copy
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pl_mmb_translate.config import Settings
from pl_mmb_translate.checkpoint import CheckpointStore
from pl_mmb_translate.models import TranslationUnit
from pl_mmb_translate.openai_translator import OpenAITranslator, TranslationError, _validate_same_shape
from pl_mmb_translate.runner import _write_manifest, run_dataset
from pl_mmb_translate.text_templates import (
    QUERY_PREFIX_EN, QUERY_PREFIX_PL, prepare_templates, restore_templates,
)


def settings(**overrides):
    defaults = dict(
        base_url="http://localhost:9999/v1", api_key="test", model="test-model",
        batch_size=2, request_timeout=10, max_retries=0, max_output_tokens=1024,
        temperature=0, output_dir=Path("outputs"), hf_cache_dir=Path(".cache"),
        hf_token=None, hf_streaming=True, image_max_side=1280, image_jpeg_quality=88,
        max_images_per_record=2, max_images_per_request=4, mbeir_image_root=None,
        run_name="test", quality_review=True,
    )
    defaults.update(overrides)
    return Settings(**defaults)


class TemplateTests(unittest.TestCase):
    def test_composed_retrieval_prefix_preserves_image_token_and_variable_text(self):
        source = {"qry_text": "<|image_1|>\nGiven an image, find a similar everyday image with the described changes: Remove a dog.\n",
                  "tgt_text": ["<|image_1|>\nRepresent the given cropped image of the object"] * 3}
        pending, spec = prepare_templates(source)
        self.assertEqual(pending, {"qry_text": "Remove a dog.\n"})
        restored = restore_templates({"qry_text": "Usuń psa.\n"}, spec)
        self.assertTrue(restored["qry_text"].startswith("<|image_1|>\nNa podstawie podanego obrazu"))
        self.assertTrue(restored["qry_text"].endswith("Usuń psa.\n"))
        _validate_same_shape(source, restored)
        self.assertEqual(len(restored["tgt_text"]), 3)

    def test_mixed_tree_preserves_order_and_fixed_values(self):
        source = {
            "qry_text": QUERY_PREFIX_EN + "A dog.\n",
            "tgt_text": ["<|image_1|>\nRepresent the given image.\n", "A cat.", "", "A cat."],
            "nested": {"blank": "  ", "empty_list": [], "empty_dict": {}},
        }
        original = copy.deepcopy(source)
        pending, spec = prepare_templates(source)
        self.assertEqual(pending, {"qry_text": "A dog.\n", "tgt_text": ["A cat.", "A cat."]})
        restored = restore_templates({"qry_text": "Pies.\n", "tgt_text": ["Kot.", "Kot."]}, spec)
        self.assertEqual(restored["qry_text"], QUERY_PREFIX_PL + "Pies.\n")
        self.assertEqual(restored["tgt_text"], ["<|image_1|>\nPrzedstaw podany obraz.\n", "Kot.", "", "Kot."])
        self.assertEqual(restored["nested"], source["nested"])
        self.assertEqual(source, original)

    def test_no_substring_replacement(self):
        value = "A sign says Represent the given image."
        pending, spec = prepare_templates(value)
        self.assertEqual((pending, spec), (value, None))


class ValidationTests(unittest.TestCase):
    def test_placeholder_loss_order_and_duplicates(self):
        for source, target in [
            ("<|image_1|> dog", "pies"),
            ("<image> <|image_2|>", "<|image_2|> <image>"),
            ("{{name}} {{name}}", "{{name}}"),
            ("Hello {user}", "Witaj {uzytkownik}"),
        ]:
            with self.subTest(source=source), self.assertRaises(TranslationError):
                _validate_same_shape(source, target)

    def test_empty_values(self):
        for source, target in [("", "tekst"), ("\n", ""), ("dog", " ")]:
            with self.subTest(source=source), self.assertRaises(TranslationError):
                _validate_same_shape(source, target)
        _validate_same_shape({"text": ["", "<|image_1|> dog"]}, {"text": ["", "<|image_1|> pies"]})


class ManifestTests(unittest.TestCase):
    def test_capture_and_sample_are_resume_invariants(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            config = settings(save_drafts=True)
            ids = ("row:1", "row:2")
            _write_manifest(path, dataset_key="mscoco_t2i", settings=config, with_image=True, unit_ids=ids)
            _write_manifest(path, dataset_key="mscoco_t2i", settings=config, with_image=True, unit_ids=ids[::-1])
            with self.assertRaises(RuntimeError):
                _write_manifest(path, dataset_key="mscoco_t2i", settings=config, with_image=True, unit_ids=("row:1",))
            with self.assertRaises(RuntimeError):
                _write_manifest(path, dataset_key="mscoco_t2i", settings=replace(config, save_drafts=False), with_image=True, unit_ids=ids)

    def test_resume_and_changed_policy(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            config = settings()
            _write_manifest(path, dataset_key="mscoco_t2i", settings=config, with_image=True)
            # Concurrency does not change translation semantics.
            _write_manifest(path, dataset_key="mscoco_t2i", settings=replace(config, batch_size=4), with_image=True)
            for changed in [replace(config, temperature=0.1), replace(config, quality_review=False),
                            replace(config, image_max_side=640)]:
                with self.subTest(changed=changed), self.assertRaises(RuntimeError):
                    _write_manifest(path, dataset_key="mscoco_t2i", settings=changed, with_image=True)
            with patch("pl_mmb_translate.runner.SYSTEM_PROMPT", "Changed prompt"):
                with self.assertRaises(RuntimeError):
                    _write_manifest(path, dataset_key="mscoco_t2i", settings=config, with_image=True)
            manifest = json.loads(path.read_text())
            del manifest["translation_config"]
            path.write_text(json.dumps(manifest))
            with self.assertRaises(RuntimeError):
                _write_manifest(path, dataset_key="mscoco_t2i", settings=config, with_image=True)


class TranslatorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.translator = OpenAITranslator(settings())
        await self.translator.close()
        self.requests = []
        self.responses = []

        async def create(**kwargs):
            self.requests.append(copy.deepcopy(kwargs))
            obj = self.responses.pop(0)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(obj)), finish_reason="stop",
            )])

        self.translator.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    def response(self, texts, flags=None):
        result = {"translations": [{"id": "row:0", "texts": texts}]}
        if flags is not None:
            result["quality_flags"] = flags
        return result

    async def test_paired_export_keeps_original_draft_and_compact_source(self):
        self.translator.settings = settings(save_drafts=True)
        unit = TranslationUnit("row:0", {
            "qry_text": QUERY_PREFIX_EN + "Two zebras.",
            "tgt_text": ["<|image_1|>\nRepresent the given image.\n"] * 1000,
        }, metadata={"hf_path": "TIGER-Lab/MMEB-eval"})
        self.responses = [self.response({"qry_text": "Dwa zebry."}),
                          self.response({"qry_text": "Dwie zebry."}, [])]
        result = await self.translator.translate_one(unit, with_image=False)
        with TemporaryDirectory() as tmp:
            store = CheckpointStore(Path(tmp))
            try:
                store.put_completed(unit, result, model="test", with_image=False)
                rows = [json.loads(line) for line in store.export_review_comparison().read_text(encoding="utf-8").splitlines()]
            finally:
                store.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_compact"], {"qry_text": "Two zebras."})
        self.assertEqual(rows[0]["draft_compact"], {"qry_text": "Dwa zebry."})
        self.assertEqual(rows[0]["reviewed_compact"], {"qry_text": "Dwie zebry."})
        self.assertTrue(rows[0]["review_changed"])
        self.assertEqual(set(rows[0]["stages"]), {"draft", "review"})

    async def test_review_then_restore_1000_candidates(self):
        unit = TranslationUnit("row:0", {
            "qry_text": QUERY_PREFIX_EN + "Two zebras.",
            "tgt_text": ["<|image_1|>\nRepresent the given image.\n"] * 1000,
        }, metadata={"hf_path": "TIGER-Lab/MMEB-eval"}, prompt_profile="retrieval")
        self.responses = [self.response({"qry_text": "Dwa zebry."}),
                          self.response({"qry_text": "Dwie zebry."}, [])]
        result = await self.translator.translate_one(unit, with_image=False)
        self.assertEqual(result["qry_text"], QUERY_PREFIX_PL + "Dwie zebry.")
        self.assertEqual(result["tgt_text"], ["<|image_1|>\nPrzedstaw podany obraz.\n"] * 1000)
        self.assertEqual(len(self.requests), 2)
        self.assertNotIn("Represent the given image", self.requests[0]["messages"][1]["content"][0]["text"])
        self.assertTrue(unit.metadata["translation_quality"]["reviewed"])

    async def test_flags_survive_outside_text(self):
        unit = TranslationUnit("row:0", {"caption": "Two swam swimming."})
        flags = [{"path": "texts.caption", "reason": "Ambiguous source typo: swam."}]
        self.responses = [self.response({"caption": "Dwa pływały."}),
                          self.response({"caption": "Dwa pływały."}, flags)]
        result = await self.translator.translate_one(unit, with_image=False)
        self.assertEqual(result, {"caption": "Dwa pływały."})
        quality = unit.metadata["translation_quality"]
        self.assertEqual(quality["flags"], flags)
        self.assertEqual(quality["review_source"], unit.texts)

    async def test_fully_fixed_record_needs_no_request(self):
        unit = TranslationUnit("row:0", {"tgt_text": ["Represent the given image."]},
                               metadata={"hf_path": "TIGER-Lab/MMEB-eval"})
        result = await self.translator.translate_one(unit, with_image=True)
        self.assertEqual(result, {"tgt_text": ["Przedstaw podany obraz."]})
        self.assertEqual(self.requests, [])

    async def test_templates_are_scoped_to_mmeb(self):
        self.translator.settings = settings(quality_review=False)
        unit = TranslationUnit("row:0", {"caption": "Represent the given image."})
        self.responses = [self.response({"caption": "Przedstaw obraz."})]
        self.assertEqual(await self.translator.translate_one(unit, with_image=False),
                         {"caption": "Przedstaw obraz."})
        self.assertEqual(len(self.requests), 1)

    async def test_review_cannot_drop_placeholder(self):
        unit = TranslationUnit("row:0", {"caption": "<|image_1|> dog"})
        self.responses = [self.response({"caption": "<|image_1|> pies"}),
                          self.response({"caption": "pies"}, [])]
        with self.assertRaises(TranslationError):
            await self.translator.translate_one(unit, with_image=False)
        self.assertNotIn("translation_quality", unit.metadata)

    async def test_review_must_return_valid_flags(self):
        for flags in [None, ["uncertain"], [{"path": "texts.caption", "reason": ""}]]:
            unit = TranslationUnit("row:0", {"caption": "A dog."})
            self.responses = [self.response({"caption": "Pies."}),
                              self.response({"caption": "Pies."}, flags)]
            with self.subTest(flags=flags), self.assertRaises(TranslationError):
                await self.translator.translate_one(unit, with_image=False)

    async def test_retry_receives_validation_feedback(self):
        self.translator.settings = settings(quality_review=False, max_retries=1)
        unit = TranslationUnit("row:0", {"caption": "<image> dog"})
        self.responses = [self.response({"caption": "pies"}), self.response({"caption": "<image> pies"})]
        result = await self.translator.translate_one(unit, with_image=False)
        self.assertEqual(result, {"caption": "<image> pies"})
        self.assertIn("Placeholder sequence changed", self.requests[1]["messages"][-1]["content"])

    async def test_contrastive_collapse_is_retried_but_repetitions_are_allowed(self):
        self.translator.settings = settings(quality_review=False, max_retries=1)
        unit = TranslationUnit("row:0", {"pair": ["dog chasing cat", "cat chasing dog", "dog chasing cat"]},
                               prompt_profile="contrastive")
        collapsed = {"pair": ["Pies goni kota.", "pies goni kota!"]}
        correct = {"pair": ["Pies goni kota.", "Kot goni psa."]}
        self.responses = [self.response(collapsed), self.response(correct)]
        self.assertEqual(await self.translator.translate_one(unit, with_image=False),
                         {"pair": correct["pair"] + [correct["pair"][0]]})
        self.assertIn("Distinct source captions collapsed", self.requests[1]["messages"][-1]["content"])

    async def test_contrastive_review_cannot_collapse_pair(self):
        unit = TranslationUnit("row:0", {"a": "red cup", "b": "blue cup"}, prompt_profile="contrastive")
        self.responses = [self.response({"a": "Czerwony kubek.", "b": "Niebieski kubek."}),
                          self.response({"a": "Kubek.", "b": "kubek!"}, [])]
        with self.assertRaises(TranslationError):
            await self.translator.translate_one(unit, with_image=False)

    async def test_empty_length_response_gets_feedback_on_retry(self):
        self.translator.settings = settings(quality_review=False, max_retries=1)
        async def create(**kwargs):
            self.requests.append(copy.deepcopy(kwargs))
            content = None if len(self.requests) == 1 else json.dumps(self.response({"caption": "Pies."}))
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="length" if content is None else "stop")])
        self.translator.client.chat.completions.create = create
        result = await self.translator.translate_one(TranslationUnit("row:0", {"caption": "A dog."}), with_image=False)
        self.assertEqual(result, {"caption": "Pies."})
        messages = self.requests[1]["messages"]
        self.assertIn("exhausted its output budget", messages[-1]["content"])
        self.assertFalse(any(m["role"] == "assistant" and not m["content"] for m in messages))


class SelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_selected_ids_and_resume(self):
        units = [TranslationUnit(f"row:{i}", {"caption": f"caption {i}"}) for i in range(5)]
        seen = []
        class Adapter:
            def iter_units(self, settings, with_image):
                yield from units
        class Translator:
            def __init__(self, settings):
                pass
            async def close(self):
                pass
        async def fake_run(pending, **kwargs):
            batch = list(pending)
            seen.append([u.unit_id for u in batch])
            self.assertEqual(kwargs["expected_total"], 2)
            for unit in batch:
                kwargs["store"].put_completed(unit, unit.texts, model="test", with_image=False)
        spec = SimpleNamespace(factory=Adapter, description="fake", expected_examples=5)
        with TemporaryDirectory() as tmp, patch.dict("pl_mmb_translate.runner.DATASETS", {"fake": spec}), \
                patch("pl_mmb_translate.runner.OpenAITranslator", Translator), \
                patch("pl_mmb_translate.runner._run_concurrently", fake_run):
            config = settings(output_dir=Path(tmp))
            for _ in range(2):
                await run_dataset("fake", config, with_image=False, unit_ids=("row:1", "row:3"))
        self.assertEqual(seen, [["row:1", "row:3"], []])

    async def test_missing_selection_is_reported(self):
        class Adapter:
            def iter_units(self, settings, with_image):
                yield TranslationUnit("row:0", {"caption": "x"})
        class Translator:
            def __init__(self, settings):
                pass
            async def close(self):
                pass
        async def fake_run(pending, **kwargs):
            list(pending)
        spec = SimpleNamespace(factory=Adapter, description="fake", expected_examples=1)
        with TemporaryDirectory() as tmp, patch.dict("pl_mmb_translate.runner.DATASETS", {"fake": spec}), \
                patch("pl_mmb_translate.runner.OpenAITranslator", Translator), \
                patch("pl_mmb_translate.runner._run_concurrently", fake_run):
            with self.assertRaisesRegex(RuntimeError, "Selected IDs not found"):
                await run_dataset("fake", settings(output_dir=Path(tmp)), with_image=False, unit_ids=("row:99",))


def load_tests(loader, tests, pattern):
    # The original core tests are plain functions; include them in the stdlib runner
    # so local checks need no extra pytest installation.
    import test_core
    for name in sorted(dir(test_core)):
        if name.startswith("test_") and callable(getattr(test_core, name)):
            tests.addTest(unittest.FunctionTestCase(getattr(test_core, name)))
    return tests
