"""Two-worker CPU metadata research needed for reproducible acquisition/MT."""
from concurrent.futures import ThreadPoolExecutor
import json
from prepare import OUT, fetch, write, read


def main():
    urls = [
        'https://huggingface.co/api/models/Helsinki-NLP/opus-mt-en-zlw',
        'https://huggingface.co/api/models/Helsinki-NLP/opus-mt-en-zlw/tree/main',
        'https://huggingface.co/Helsinki-NLP/opus-mt-en-zlw/raw/main/README.md',
        'https://api.github.com/repos/multi30k/dataset/commits/master',
        'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/image_splits/test_2016_flickr.txt',
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(fetch, urls))
    write(OUT / 'deep_research_v2.json', dict(sources=results, workers=2))
    for row in results:
        print(row['url'], row['status'])
        if row['status']=='fetched':
            if '/tree/' in row['url']:
                print([(r['path'], r.get('size'), r.get('lfs', {}).get('oid')) for r in json.loads(row['text'])])
            elif '/api/models/' in row['url']:
                print('revision', json.loads(row['text'])['sha'])
            elif 'raw.githubusercontent' in row['url']:
                print('lines', len(row['text'].splitlines()), 'first', row['text'].splitlines()[:5])
    for name in ('source_capacity_checks.json', 'candidate_caption_inventory.json'):
        print(name, json.dumps(read(OUT / name)))


if __name__=='__main__':
    main()
