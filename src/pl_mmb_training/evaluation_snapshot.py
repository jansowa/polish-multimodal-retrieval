"""Stream the pinned, pretty-printed snapshot instead of materializing 800MB strings."""
import json
from types import SimpleNamespace

from .common import digest, read_json, write_json


def compact_snapshot(source, target, image_values, text_values, source_sha, manifest_sha, expected_counts):
    if target.exists():
        value = read_json(target)
        if value['source_sha256'] != source_sha or value['feature_manifest_sha256'] != manifest_sha or value['sha256'] != digest(value['datasets']):
            raise ValueError('Compact evaluation snapshot changed')
    else:
        ii, ti = {p: i for i, p in enumerate(image_values)}, {t: i for i, t in enumerate(text_values)}
        datasets, key, buffered = {}, None, None
        # The input is the SHA-pinned output of common.write_json(indent=2).
        # Boundaries at these indentation levels cannot occur inside JSON strings.
        with source.open(encoding='utf-8') as stream:
            for line in stream:
                if line.startswith('    "') and line.strip().endswith(': ['):
                    key = json.loads(line.strip()[:-3])
                    if key not in expected_counts:
                        raise ValueError('Unknown dataset in evaluation snapshot')
                    datasets[key] = []
                elif line.rstrip() == '      {':
                    buffered = [line]
                elif buffered is not None:
                    buffered.append(line)
                    if line.startswith('      }'):
                        row = json.loads(''.join(buffered).rstrip().removesuffix(','))
                        datasets[key].append(dict(id=row['id'], texts=[ti[t] for t in row['texts']],
                            images=[ii[p] for p in row['images']],
                            query_image=ii[row['query_image']] if row['query_image'] is not None else None))
                        buffered = None
        if {k: len(rows) for k, rows in datasets.items()} != expected_counts:
            raise ValueError('Incomplete streamed evaluation snapshot')
        value = dict(source_sha256=source_sha, feature_manifest_sha256=manifest_sha,
                     datasets=datasets, sha256=digest(datasets))
        write_json(target, value)
    result = {key: [SimpleNamespace(id=row['id'], texts=[text_values[i] for i in row['texts']],
                  images=[image_values[i] for i in row['images']],
                  query_image=image_values[row['query_image']] if row['query_image'] is not None else None)
                   for row in rows] for key, rows in value['datasets'].items()}
    if {k: len(rows) for k, rows in result.items()} != expected_counts:
        raise ValueError('Compact evaluation row counts mismatch')
    return result
