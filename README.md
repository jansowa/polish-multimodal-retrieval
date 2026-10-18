# Polish Multimodal Benchmark Translator

Resumable English → Polish translation pipeline for multimodal embedding benchmarks. Translation is performed through the **OpenAI Python library** and therefore works with OpenAI-compatible servers such as vLLM/SGLang as long as they implement `chat.completions` with image inputs.

The project deliberately contains **no GPU / CUDA / PyTorch dependency**. Image preprocessing uses Pillow on CPU. Hugging Face `datasets` is used only for dataset I/O.

## Supported targets

The registry includes:

- **MMEB-eval** datasets (all configs), including CIRR, FashionIQ, MSCOCO i2t/t2i, VisualNews i2t/t2i, EDIS, GQA, Visual7W, A-OKVQA, OK-VQA, VizWiz, RefCOCO, Visual7W-Pointing, WebQA, etc.
- **SugarCrepe**
- **Winoground**
- **ARO Visual Relation**
- **ARO Visual Attribution**
- **ImageCoDe**
- **M-BEIR** retrieval instructions + test queries + local or global candidate pool

Predefined benchmark aliases:

- `mmeb` / `mmeb_all` — every MMEB-eval config
- `mmeb_core` — curated translation-friendly MMEB subset
- `compositionality` / `mieb` / `mieb_curated` — selected MIEB compositionality tasks: SugarCrepe + Winoground + ARO Relation + ARO Attribution + ImageCoDe
- `mbeir` / `mbeir_local` — M-BEIR instructions + queries + local candidate pool
- `mbeir_global` — same, but translates the 5.6M-scale global candidate pool; expect a very large job
- `pl_v1` — recommended first Polish benchmark: `mmeb_core + compositionality`

List everything:

```bash
python translate.py --list
```

## Installation

Minimal setup (does not install the project package itself):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional editable install, if you want the `pl-mmb-translate` console command:

```bash
pip install -e .
```

The archive contains a placeholder `.env` ready to edit; `.env.example` is kept as the version-control template.

Example for local vLLM:

```dotenv
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_API_KEY=not-needed
OPENAI_MODEL=your-vllm-model-name
BATCH_SIZE=8
```

`BATCH_SIZE` is the maximum number of concurrent requests. Each translation request contains exactly one benchmark example; optional quality review makes a second request for that same example.

## CLI

The two main CLI parameters are positional:

```bash
python translate.py <dataset-or-benchmark> <True|False>
```

Examples:

```bash
# One dataset, text only
python translate.py cirr false

# One dataset, include the related image(s)
python translate.py sugarcrepe true

# Whole benchmark; datasets are processed sequentially
python translate.py pl_v1 true

# Smoke test: first 20 records of each selected dataset
python translate.py compositionality true --limit 20
```

Installed entrypoint works too:

```bash
pl-mmb-translate mmeb_core false
```

## Resume / crash safety

Every dataset+mode gets a SQLite checkpoint:

```text
outputs/<RUN_NAME>/<dataset>/text_only/checkpoint.sqlite3
outputs/<RUN_NAME>/<dataset>/with_image/checkpoint.sqlite3
```

A record is marked complete only after a validated translation has been returned. Rerunning the exact same command skips completed IDs and continues with the rest.

At the end of every run (including `Ctrl+C`) SQLite is exported to:

```text
translations.jsonl
```

SQLite is the source of truth, so a crash cannot corrupt resume state by leaving a half-written JSONL file.

Requests are retried individually. A record that still fails validation or the API request is saved in the `failures` table; other examples continue.

## Output format

Each line of `translations.jsonl` contains a translation patch, not duplicated image bytes. One MMEB row is one unit containing its query and candidate texts. Repeated candidate strings are compacted for the API and expanded locally to their original list length and order:

```json
{
  "id": "SugarCrepe:17",
  "source": {
    "caption": "A dog is behind a man.",
    "negative_caption": "A man is behind a dog."
  },
  "translated": {
    "caption": "Pies znajduje się za mężczyzną.",
    "negative_caption": "Mężczyzna znajduje się za psem."
  },
  "metadata": {
    "dataset": "SugarCrepe",
    "split": "train",
    "row_index": 17
  },
  "model": "...",
  "with_image": true
}
```

The original dataset remains the authoritative source for images, labels, qrels and other non-text fields. This avoids making large copies of image datasets and makes it easy to audit only the changed text.

## Why contrastive datasets are translated jointly

SugarCrepe, Winoground and ARO contain intentionally wrong / minimally different captions. A vision-language translator could otherwise use the image to silently “fix” a hard negative.

For these datasets the positive and negative captions are put in the **same translation record** and the system prompt explicitly requires preservation of:

- negation;
- subject ↔ object direction;
- spatial relations;
- counts;
- attributes;
- semantic inconsistencies with the image;
- distinctions between positive and negative captions.

The image is described to the model as **disambiguation context only**, never as ground truth to which the text should be corrected.

## Images

### MMEB

When `WITH_IMAGE=true`, the MMEB adapter downloads the per-dataset image ZIP from a pinned MMEB-eval revision and extracts it into the local HF cache. This avoids downloading the repository's single multi-gigabyte all-images archive.

Images are resized on CPU to `IMAGE_MAX_SIDE` and encoded as JPEG/base64 before being sent to the OpenAI-compatible endpoint.

### M-BEIR

M-BEIR's full image archive is enormous and is intentionally **not auto-downloaded**. For multimodal M-BEIR translation, extract the official images yourself and set:

```dotenv
MBEIR_IMAGE_ROOT=/data/M-BEIR/images
```

Text-only M-BEIR translation does not need this setting.

## Important benchmark-design note

For OCR-centric datasets such as DocVQA, TextVQA, ChartQA and InfographicsVQA, translating only the query leaves English text inside the pixels. They are included under `mmeb_all` because the adapter can technically translate them, but they are intentionally **not part of `mmeb_core`**. Treat them as a separate cross-lingual visual-OCR track.

## Adding another Hugging Face dataset

Most datasets only require a `DatasetSpec` in `registry.py` using `HFGenericAdapter`:

```python
"my_dataset": DatasetSpec(
    "my_dataset",
    "description",
    lambda: HFGenericAdapter(
        dataset_name="MyDataset",
        path="org/hf-dataset",
        split="test",
        text_fields=("caption", "negative_caption"),
        image_fields=("image",),
        prompt_profile="contrastive",
    ),
),
```

If multiple text fields form a semantic contrast, keep them in the same record as above rather than translating each field independently.

## Licenses / gated datasets

The code does not bypass dataset terms. Some Hugging Face datasets (notably Winoground) can require accepting their license or authenticating with `HF_TOKEN`. Put your token in `.env` when needed.


### MMEB repeated-candidate compaction and progress

For MMEB retrieval rows, repeated candidate text values are compacted inside a single benchmark example before the API request and expanded back after translation. This avoids asking the model to emit the same translated instruction hundreds of times while preserving the original output list length/order. Examples are never mixed across requests.

For datasets with a known size (MMEB: 1000 examples), the runner displays completed/total, failures in the current run, requests in flight, throughput, and ETA. `BATCH_SIZE` remains the maximum number of concurrent one-example requests.

## Translation quality

Known MMEB boilerplate is translated locally using exact matches. The prefix
`Find me an everyday image that matches the given caption: ` becomes
`Znajdź obraz z życia codziennego pasujący do podanego opisu: `, and
`Represent the given image.` becomes `Przedstaw podany obraz.`. The model receives
the variable text only; original keys, list order, placeholders and whitespace
around the image instruction are restored locally. Other dataset families do not
use these templates.

The prompt distinguishes unambiguous language errors from deliberately unusual
facts and requires grammatical Polish. Validation rejects altered placeholder
sequences, changed empty strings, empty translations of nonempty text, and changed
JSON structure. Validation failures are fed back to the model on retry.

Set `QUALITY_REVIEW=true` to make a second, text-only EN/PL review request for each
example with variable text. It corrects the draft and flags unresolved source
ambiguity or translation uncertainty. This roughly doubles model calls; it is an
additional check by the same model, not a guarantee of correctness. Flags are
stored under `metadata.translation_quality.flags`, outside benchmark text. Flagged
records also include `review_source`, the compact source tree to which flag paths
refer. These records remain in the export and need human assessment before use.

The manifest stores the pipeline version, prompt/template policy hash, temperature,
output token limit, review mode and image preprocessing settings. Resume rejects
incompatible configurations, including old manifests. Use a new `RUN_NAME` for a
new policy; keep the old run for comparison.

From the project directory in WSL, using the existing `.env` and virtualenv:

```bash
source .venv/bin/activate
RUN_NAME=qwen3.8-27b-v3-review QUALITY_REVIEW=true TEMPERATURE=0 \
  python translate.py mscoco_t2i true
```

Results go to `outputs/qwen3.8-27b-v3-review/mscoco_t2i/with_image/` when using the
default `OUTPUT_DIR`. Repeat the same command to resume. To compare without image
context as a separate experiment, use the same command with `false` instead of
`true`; results go to its separate `text_only/` directory.

Offline regression checks (no model or dataset downloads):

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

To measure the effect of review directly, use `SAVE_DRAFTS=true` and optionally
`--ids-file` to select exact examples. A reproducible 100-example experiment and
its command are documented in [experiments/README.md](experiments/README.md).

## Manual EN/PL/image review

The local review workspace shows the source caption, Polish translation and
verified ground-truth image together. It reads completed records from the live
checkpoints, supports searching and rating examples, and saves notes/corrections
separately from benchmark data.

```bash
.venv/bin/python -u reviewer/server.py --detach --build-index
```

Open **http://localhost:8765** on the same computer. See
[reviewer/README.md](reviewer/README.md) for controls, saved ratings and exports.

## Unattended full-dataset suite

See [experiments/ABSENCE.md](experiments/ABSENCE.md) for the prepared six-dataset
suite, separate pilots, resumable background worker and exact source-ID checks.
Pipeline v3 also retries contrastive examples when distinct captions collapse
into identical Polish text. This check does not establish semantic correctness.

Current continuation, output locations and resume commands:
[reports/absence_work_status.md](reports/absence_work_status.md).
