"""CPU graph preparation, independent of models. Not a leakage-audit substitute."""
from collections import defaultdict
import hashlib
from pl_mmb_training.leakage import normalized_text


class UnionFind:
    def __init__(self):
        self.parents = {}

    def find(self, key):
        self.parents.setdefault(key, key)
        root = key
        while self.parents[root] != root:
            root = self.parents[root]
        while self.parents[key] != key:
            key, self.parents[key] = self.parents[key], root
        return root

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parents[max(a, b)] = min(a, b)


def graph(records, dataset_id, manifest_sha256):
    """Build a draft shared-scorer graph from already audited caption records.

    This deliberately returns draft status: only the separate complete leakage
    audit, translation QA and immutable handoff can promote it to frozen status.
    Each row carries original caption ID, EN/PL, exact pixel identity, pHash and
    image path. Near photographs remain distinct candidates with distinct qrels.
    """
    if not records or 'xm3600' in dataset_id.casefold():
        raise ValueError('Empty or forbidden gallery')
    rows = sorted(records, key=lambda row: row['id'])
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate original caption IDs')
    for row in rows:
        for field in ('text_en', 'text_pl', 'pixels_sha256', 'image', 'phash'):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError('Missing nonempty ' + field)
        if len(row['phash']) != 16 or len(row['pixels_sha256']) != 64:
            raise ValueError('Expected 64-bit pHash and SHA256 pixel identity')
        int(row['phash'], 16)
        int(row['pixels_sha256'], 16)
    images = defaultdict(list)
    for row in rows:
        images[row['pixels_sha256']].append(row)
    for group in images.values():
        if len({r['phash'] for r in group}) != 1:
            raise ValueError('Same exact pixels have inconsistent pHash')
    image_ids = sorted(images)
    texts = UnionFind()
    known_en, known_pl = {}, {}
    for row in rows:
        key = row['id']
        texts.find(key)
        for index, value in ((known_en, row['text_en']), (known_pl, row['text_pl'])):
            normalized = normalized_text(value)
            if normalized in index:
                texts.join(key, index[normalized])
            index[normalized] = key
    query_rows = defaultdict(list)
    for row in rows:
        query_rows[texts.find(row['id'])].append(row)
    query_groups = sorted(query_rows.values(), key=lambda group: min(r['id'] for r in group))
    query_texts = [min(group, key=lambda r: (normalized_text(r['text_pl']), r['id']))['text_pl'] for group in query_groups]
    # Identical texts and near images affect resampling components; only original
    # edges determine relevance. No pHash-derived positive labels are introduced.
    components = UnionFind()
    for image in image_ids:
        components.find('image:' + image)
    for i, left in enumerate(image_ids):
        a = int(images[left][0]['phash'], 16)
        for right in image_ids[i + 1:]:
            if (a ^ int(images[right][0]['phash'], 16)).bit_count() <= 6:
                components.join('image:' + left, 'image:' + right)
    image_index = {key: index for index, key in enumerate(image_ids)}
    edges = set()
    for qi, group in enumerate(query_groups):
        for row in group:
            edges.add((qi, image_index[row['pixels_sha256']]))
            components.join('query:' + str(qi), 'image:' + row['pixels_sha256'])
    query_ids = [hashlib.sha256(('pl:' + normalized_text(text)).encode()).hexdigest() for text in query_texts]
    return dict(status='draft_requires_universe_audit_and_freeze', id=dataset_id, manifest_sha256=manifest_sha256,
        images=[min(images[key], key=lambda row: row['image'])['image'] for key in image_ids],
        image_ids=image_ids, texts=query_texts, query_ids=query_ids, edges=sorted(edges),
        image_components=[components.find('image:' + key) for key in image_ids],
        text_components=[components.find('query:' + str(i)) for i in range(len(query_groups))],
        original_caption_ids=[[row['id'] for row in group] for group in query_groups],
        canonicalization='exact pixel identity; normalized identical EN or PL union known positives; pHash only components')


def choose_components(protocol, target=1000, seed=20261023):
    """Whole components, deterministic hash order, final component may exceed N."""
    if target < 1:
        raise ValueError('Positive target required')
    groups = defaultdict(list)
    for image, component in zip(protocol['image_ids'], protocol['image_components']):
        groups[component].append(image)
    order = sorted(groups, key=lambda component: hashlib.sha256(
        (str(seed) + ':' + '|'.join(sorted(groups[component]))).encode()).hexdigest())
    selected = []
    for component in order:
        selected.extend(groups[component])
        if len(selected) >= target:
            break
    return sorted(selected)
