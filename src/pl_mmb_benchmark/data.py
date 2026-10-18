from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re

REVISION = 'eb3a8af230a66be5c9308cd2be5779b9d6de4cb3'
ROOT = Path(__file__).resolve().parents[2]
SPECS = {
    'mscoco_t2i': ('MSCOCO_t2i', 'qwen3.8-27b-full-v2-single', 'with_image', 1000),
    'cirr': ('CIRR', 'qwen3.8-27b-absence-v3b-single', 'with_image', 1000),
    'fashioniq': ('FashionIQ', 'qwen3.8-27b-absence-v3b-single', 'with_image', 1000),
    'refcoco': ('RefCOCO', 'qwen3.8-27b-absence-v3b-single', 'with_image', 1000),
    'aro_visual_relation': ('AROVisualRelation', 'qwen3.8-27b-absence-v3b-single', 'text_only', 23937),
    'aro_visual_attribution': ('AROVisualAttribution', 'qwen3.8-27b-absence-v3b-single', 'text_only', 28748),
}


@dataclass
class Example:
    id: str
    texts: list[str]
    images: list[str]
    query_image: str | None = None


def translation_path(key):
    _, run, mode, _ = SPECS[key]
    return ROOT / 'outputs' / run / key / mode / 'translations.jsonl'


def clean(text):
    return re.sub(r'<\|image_\d+\|>', '', text).strip()


def load(key):
    """Join frozen translations to pinned source rows, never to export order."""
    from datasets import load_dataset, Image
    name, _, _, expected = SPECS[key]
    path = translation_path(key)
    payload = path.read_bytes()
    patches = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    by_index = {p['metadata']['row_index']: p for p in patches}
    if len(patches) != expected or len(by_index) != expected or set(by_index) != set(range(expected)):
        raise ValueError(f'{key}: incomplete or duplicate translations; expected {expected}')
    examples = []
    if key.startswith('aro_'):
        from pl_mmb_translate.registry import DATASETS
        a = DATASETS[key].factory()
        ds = load_dataset(a.path, split=a.split, revision=a.revision)
        ds = ds.cast_column('image', Image(decode=False))
        image_dir = Path.home() / '.cache/huggingface/pl_mmb_benchmark/images' / key
        image_dir.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(ds):
            p = by_index[i]
            if p['source'] != {f: row[f] for f in a.text_fields}:
                raise ValueError(f'Source mismatch: {p["id"]}')
            raw = row['image']
            if raw.get('bytes'):
                image = image_dir / f'{i}.jpg'
                if not image.exists():
                    image.write_bytes(raw['bytes'])
            else:
                image = Path(raw['path'])
            examples.append(Example(p['id'], [p['translated'][f] for f in a.text_fields], [str(image)]))
    else:
        from huggingface_hub import hf_hub_download
        from datasets import Dataset
        print(f'{key}: loading pinned source parquet', flush=True)
        parquet = hf_hub_download('TIGER-Lab/MMEB-eval', f'{name}/test-00000-of-00001.parquet',
                                  repo_type='dataset', revision=REVISION)
        ds = Dataset.from_parquet(parquet)
        image_root = ROOT / '.cache/huggingface/pl_mmb_images/mmeb' / name
        for i, row in enumerate(ds):
            p = by_index[i]
            if p['source'] != {k: row[k] for k in p['source']}:
                raise ValueError(f'Source mismatch: {p["id"]}')
            images = [str(image_root / x) for x in row['tgt_img_path']]
            query = str(image_root / row['qry_img_path']) if row['qry_img_path'] else None
            examples.append(Example(p['id'], [clean(p['translated']['qry_text'])], images, query))
    if len(examples) != expected:
        raise ValueError('Source length mismatch')
    unique_images = {image for e in examples for image in e.images + ([e.query_image] if e.query_image else [])}
    for image in unique_images:
        if not Path(image).is_file():
            raise FileNotFoundError(image)
    return examples, hashlib.sha256(payload).hexdigest()
