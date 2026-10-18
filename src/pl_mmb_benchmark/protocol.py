"""Pinned MMEB-v1 CLIP multimodal baseline: no task boilerplate, raw feature sum."""
PROTOCOL = 'mmeb-v1-no-instructions-raw-sum-cosine-v2'
REFERENCE_REVISION = '28fdbdd68a66374ec1b53207ad7a24b9979f1ab6'

PREFIXES = {
    'mscoco_t2i': ('Find me an everyday image that matches the given caption: ',
                   'Znajdź obraz z życia codziennego pasujący do podanego opisu: '),
    'cirr': ('Given an image, find a similar everyday image with the described changes: ',
             'Na podstawie podanego obrazu znajdź podobny obraz z życia codziennego z następującymi zmianami: '),
    'fashioniq': ('Find an image to match the fashion image and style note: ',
                  'Znajdź obraz pasujący do zdjęcia ubrania i opisu stylu: '),
    'refcoco': ('Select the portion of the image that follows the language expressions. ',
                'Wybierz fragment obrazu odpowiadający opisowi. '),
}


def without_instruction(key, source, translated):
    from .data import clean
    en_prefix, pl_prefix = PREFIXES[key]
    en, pl = clean(source), clean(translated)
    if not en.startswith(en_prefix) or not pl.startswith(pl_prefix):
        raise ValueError(f'{key}: unrecognized instruction; refusing to guess caption boundary')
    content = pl[len(pl_prefix):].strip()
    if not content:
        raise ValueError(f'{key}: empty content after removing instruction')
    return content
