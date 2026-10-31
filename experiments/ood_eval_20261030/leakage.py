"""CPU index of all A/protected identities, normalized texts and 64-bit pHash."""
from collections import defaultdict
import hashlib
import re
from urllib.parse import unquote, urlsplit, urlunsplit
from pl_mmb_training.leakage import normalized_text


def tokens(row):
    result = set()
    for field in ('sha256', 'pixels_sha256'):
        if row.get(field):
            result.add('hash:' + row[field])
    for field in ('text_en', 'text_pl', 'original_en', 'source_caption', 'accepted_text'):
        if row.get(field):
            result.add('text:' + normalized_text(row[field]))
    result.update('text:' + normalized_text(value) for value in row.get('normalized_texts', []) if value)
    for field in ('id', 'image_did', 'source_image'):
        if row.get(field):
            result.add(field + ':' + str(row[field]))
    for field in ('parent_ids', 'article_ids'):
        result.update(field + ':' + str(value) for value in row.get(field, []))
    for identity in list(result):
        found = re.search(r'flickr(?:30k|8k)?[:/-](\d{5,})', identity, re.I)
        if found:
            result.add('flickr-id:' + str(int(found[1])))
    def source(value):
        if isinstance(value, dict):
            for key, content in value.items():
                if key in ('source_record', 'associations') or any(word in key.lower() for word in ('url', 'parent', 'source', 'flickr')):
                    source(content)
        elif isinstance(value, list):
            for item in value:
                source(item)
        elif isinstance(value, str) and value.startswith(('http://', 'https://')):
            url = urlsplit(value)
            result.add('url:' + urlunsplit(('', url.netloc.lower(), unquote(url.path), url.query, '')))
            found = re.search(r'flickr.com/photos/[^/]+/(\d+)', value)
            found = found or re.search(r'(?:staticflickr|static.flickr)\.com/[^?]*/(\d+)_', value)
            if found:
                result.add('flickr-id:' + str(int(found[1])))
    source(row)
    return {hashlib.sha256(token.encode()).digest() for token in result}


class ProtectedIndex:
    def __init__(self):
        self.tokens = set()
        self.hashes = set()
        self.buckets = defaultdict(set)
        self.rows = 0

    @staticmethod
    def bands(value):
        shift = 0
        for i, width in enumerate((10, 9, 9, 9, 9, 9, 9)):
            yield i, (value >> shift) & ((1 << width) - 1)
            shift += width

    def add(self, row):
        self.rows += 1
        self.tokens.update(tokens(row))
        if row.get('phash'):
            value = int(row['phash'], 16)
            if value not in self.hashes:
                self.hashes.add(value)
                for band in self.bands(value):
                    self.buckets[band].add(value)

    def reasons(self, row):
        reasons = []
        if self.tokens.intersection(tokens(row)):
            reasons.append('protected_source_hash_or_normalized_text')
        value = int(row['phash'], 16)
        matches = {h for band in self.bands(value) for h in self.buckets.get(band, ())}
        if any((value ^ other).bit_count() <= 6 for other in matches):
            reasons.append('protected_phash_le6')
        return reasons


def filter_components(rows, index):
    """Propagate protection across all B identity/text/near-image bridges."""
    from protocol import UnionFind
    graph = UnionFind()
    owners = {}
    buckets = defaultdict(list)
    hashes = []
    direct = {};cross_source=set()
    for i, row in enumerate(rows):
        graph.find(i)
        for token in tokens(row):
            if token in owners:
                graph.join(i, owners[token])
            else:
                owners[token] = i
        h = int(row['phash'], 16)
        for j in {j for band in index.bands(h) for j in buckets[band]}:
            if (h ^ hashes[j]).bit_count() <= 6:
                graph.join(i, j)
                if row.get('dataset') and rows[j].get('dataset') and row['dataset']!=rows[j]['dataset']:
                    cross_source.update((i,j))
        for band in index.bands(h):
            buckets[band].append(i)
        hashes.append(h)
        direct[i] = index.reasons(row)
    contaminated = {graph.find(i) for i, reasons in direct.items() if reasons} | {graph.find(i) for i in cross_source}
    return [dict(row, leakage_component=str(graph.find(i)),
                 excluded=graph.find(i) in contaminated,
                 exclusion_reasons=direct[i] or (['cross_source_near_original'] if i in cross_source else
                    ['transitive_B_component'] if graph.find(i) in contaminated else []))
            for i, row in enumerate(rows)]
