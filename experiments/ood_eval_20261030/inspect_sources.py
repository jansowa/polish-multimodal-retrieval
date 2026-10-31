"""Print selected cached evidence without inspecting images or model outputs."""
from html.parser import HTMLParser
import json
from prepare import OUT, read


class Links(HTMLParser):
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            href = dict(attrs).get('href', '')
            if any(word in href.lower() for word in ('zip', 'tar', 'json', 'download', 'flickr', 'dataset', 'aide', 'jpg')):
                print(href)


for key, sources in read(OUT / 'source_metadata.json')['sources'].items():
    if key in ('flickr30k', 'docci'):
        continue
    print('\nSOURCE ' + key)
    for source in sources:
        if source['status'] != 'fetched':
            print(source)
            continue
        parser = Links()
        parser.feed(source['text'])
        if key in ('aide', 'flickr30k_original'):
            print(' '.join(source['text'].split())[-12000:])
