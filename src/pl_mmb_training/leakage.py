"""Fail closed on unknown image lineage; reject all crops from eval parent images."""
import hashlib
import io
import re
from collections import defaultdict
from pathlib import Path

from .common import digest


def image_identity(path=None, payload=None):
    from PIL import Image, ImageOps
    raw = payload if payload is not None else Path(path).read_bytes()
    with Image.open(io.BytesIO(raw)) as im:
        rgb = ImageOps.exif_transpose(im).convert('RGB')
        pixel = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'pixels_sha256': pixel}


def parent_ids(path, annotation_to_image=None):
    """Namespaced IDs keep unrelated numeric filenames distinct."""
    path = str(path).replace('\\', '/')
    ids = []
    coco = re.search(r'COCO_(?:train|val|test)\d{4}_(\d+)', path, re.I)
    if coco:
        ids.append('coco:' + str(int(coco[1])))
    # RefCOCO crops retain the COCO parent in the filename.
    if 'RefCOCO' in path and not coco:
        annotation = re.search(r'/image_(\d+)(?:_crop)?\.jpg$', path)
        if annotation:
            if annotation_to_image is None or annotation[1] not in annotation_to_image:
                raise ValueError(f'Missing RefCOCO annotation-to-image lineage: {path}')
            ids.extend(['coco-ann:' + annotation[1], 'coco:' + str(annotation_to_image[annotation[1]])])
        match = re.search(r'(?:^|/)(\d{12})(?:[_./])', path)
        if match:
            ids.append('coco:' + str(int(match[1])))
    if '/CIRR/' in path:
        ids.append('cirr:' + Path(path).stem)
    if '/FashionIQ/' in path:
        ids.append('fashioniq:' + Path(path).stem)
    return ids


def normalized_text(value):
    return ' '.join(value.casefold().split())


def make_blacklist(datasets, english=None, annotation_to_image=None):
    """All candidates and all query images, not just the positive at index zero."""
    images, ids, texts = {}, set(), set()
    image_owners = defaultdict(set)
    parent_owners = defaultdict(set)
    for key, examples in datasets.items():
        for e in examples:
            ids.add(e.id)
            texts.update(normalized_text(t) for t in e.texts if t.strip())
            if english:
                texts.add(normalized_text(english[e.id]))
            for path in set(e.images + ([e.query_image] if e.query_image else [])):
                if path not in images:
                    images[path] = dict(image_identity(path), parent_ids=parent_ids(path, annotation_to_image))
                image_owners[images[path]['pixels_sha256']].add(key)
                for parent in images[path]['parent_ids']:
                    parent_owners[parent].add(key)
    cross = {parent: sorted(owners) for parent, owners in parent_owners.items() if len(owners) > 1}
    content_cross = {h: sorted(owners) for h, owners in image_owners.items() if len(owners) > 1}
    value = dict(schema=1, datasets=sorted(datasets), ids=sorted(ids), texts=sorted(texts), images=images,
                 cross_dataset_parents=cross, cross_dataset_pixels=content_cross)
    value['sha256'] = digest(value)
    return value


class LeakageGuard:
    def __init__(self, blacklist, extra_holdout='workspace'):
        from .series_holdout import ExternalHoldoutGuard,workspace_guard
        self.extra_holdout=workspace_guard() if extra_holdout=='workspace' else (ExternalHoldoutGuard(extra_holdout) if extra_holdout else None)
        self.ids = set(blacklist['ids'])
        self.texts = set(blacklist['texts'])
        self.parents, self.hashes = set(), set()
        for image in blacklist['images'].values():
            self.parents.update(image['parent_ids'])
            self.hashes.update((image['sha256'], image['pixels_sha256']))

    def reasons(self, record):
        reasons = []
        if record['id'] in self.ids:
            reasons.append('eval_id')
        if not record.get('lineage_verified') or not record.get('parent_ids'):
            reasons.append('unknown_parent_lineage')
        if self.parents.intersection(record.get('parent_ids', [])):
            reasons.append('eval_parent_image')
        if self.hashes.intersection((record['sha256'], record['pixels_sha256'])):
            reasons.append('eval_image_content')
        if any(normalized_text(record[f]) in self.texts for f in ('text_en', 'text_pl') if record.get(f)):
            reasons.append('eval_text')
        if self.extra_holdout:
            reasons.extend(self.extra_holdout.reasons(record))
        return reasons


def grouped_split(records, fraction, seed):
    """Connected parent groups never cross train/validation, including aliases."""
    roots = {}
    def find(x):
        roots.setdefault(x, x)
        if roots[x] != x:
            roots[x] = find(roots[x])
        return roots[x]
    for r in records:
        parents = r['parent_ids'] + ['pixels:' + r['pixels_sha256']]
        for p in parents[1:]:
            roots[find(p)] = find(parents[0])
    train, validation = [], []
    for r in records:
        group = find(r['parent_ids'][0])
        h = int(digest([seed, group])[:16], 16) / 2**64
        (validation if h < fraction else train).append(r)
    if not train or not validation:
        raise ValueError('Insufficient independent image groups for train/validation')
    return train, validation
