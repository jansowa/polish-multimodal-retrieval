"""Prepare official XM3600-PL as FINAL BLIND; evaluation requires explicit release."""
import argparse
import csv
import json
import os
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pl_mmb_training.common import digest, file_sha, read_json, write_json
from pl_mmb_training.leakage import image_identity, normalized_text
from pl_mmb_training.runner import write_records

URLS = dict(captions='https://google.github.io/crossmodal-3600/web-data/captions.zip',
            images='https://open-images-dataset.s3.amazonaws.com/crossmodal-3600/images.tgz',
            attributions='https://google.github.io/crossmodal-3600/web-data/image_attributions.csv')


def download(url, path):
    if not path.exists():
        part = path.with_suffix(path.suffix + '.part')
        with urlopen(url, timeout=120) as response, part.open('wb') as target:
            while block := response.read(1024 * 1024):
                target.write(block)
        part.replace(path)
    return dict(url=url, path=str(path), sha256=file_sha(path), bytes=path.stat().st_size)


def captions_from_row(row):
    captions = row['pl']['caption']
    if not isinstance(captions, list) or not captions or any(not isinstance(c, str) or not c.strip() for c in captions):
        raise ValueError('Invalid original Polish captions')
    return list(dict.fromkeys(c.strip() for c in captions))


def prepare(root, output):
    root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    assets = {key: download(url, root / Path(url).name) for key, url in URLS.items()}
    if (output / 'manifest.json').exists():
        pinned = read_json(output / 'manifest.json')['assets']
        if any(assets[key]['sha256'] != pinned[key]['sha256'] for key in assets):
            raise ValueError('Official downloaded assets differ from the pinned preparation')
    with zipfile.ZipFile(assets['captions']['path']) as archive:
        with archive.open('captions.jsonl') as stream:
            annotations = [json.loads(line) for line in stream]
    if len(annotations) != 3600 or len({r['image/key'] for r in annotations}) != 3600:
        raise ValueError('Expected the complete official 3600-image corpus')
    image_root = root / 'images'
    image_root.mkdir(exist_ok=True)
    expected = {r['image/key'] for r in annotations}
    # Read only regular expected images; archive paths are never used as output paths.
    with tarfile.open(assets['images']['path'], 'r:gz') as archive:
        for member in archive:
            name = Path(member.name).name
            if not member.isfile() or Path(name).stem not in expected or Path(name).suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            target = image_root / name
            if not target.exists():
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError('Missing image member')
                with stream, target.open('wb') as dest:
                    import shutil
                    shutil.copyfileobj(stream, dest)
    images = {p.stem: p for p in image_root.iterdir() if p.is_file()}
    if set(images) != expected:
        raise ValueError('Missing or unexpected XM3600 images')
    with Path(assets['attributions']['path']).open(encoding='utf-8-sig', newline='') as stream:
        attributions = list(csv.DictReader(stream))
    attribution_by_id = {}
    for row in attributions:
        for value in row.values():
            if value in expected:
                attribution_by_id[value] = row
    if set(attribution_by_id) != expected:
        raise ValueError('Attributions must cover every official image key')
    from pl_mmb_training.cleanroom import load_inputs
    baseline, base, training, old_blacklist, _, _, _, _ = load_inputs()
    occupied_hashes = {h for r in training for h in (r['sha256'], r['pixels_sha256'])}
    occupied_hashes.update(h for i in old_blacklist['images'].values() for h in (i['sha256'], i['pixels_sha256']))
    occupied_parents = {p for r in training for p in r['parent_ids']}
    occupied_parents.update(p for i in old_blacklist['images'].values() for p in i['parent_ids'])
    vg_path = Path.home() / '.cache/huggingface/pl_mmb_training/visual_genome_image_data.json'
    if file_sha(vg_path) != old_blacklist['lineage_metadata']['visual_genome']:
        raise ValueError('VG-to-COCO/Flickr metadata differ from the frozen guard')
    vg_rows = read_json(vg_path)
    flickr_coco = {}
    for row in vg_rows:
        if row.get('flickr_id') is not None and row.get('coco_id') is not None:
            flickr_coco.setdefault(str(row['flickr_id']), set()).add(str(row['coco_id']))
    pairs, protected_images, overlaps, texts = [], {}, [], set()
    for row in annotations:
        key = row['image/key']
        path = images[key]
        identity = image_identity(path)
        attribution = attribution_by_id[key]
        # Open Images comes from Flickr. Link known Flickr identities rather than assuming disjoint datasets.
        flickr_ids = set(re.findall(r'flickr\.com/(?:photos/[^/]+/|photo\.gne\?id=)(\d+)', json.dumps(attribution)))
        parents = ['openimages:' + key] + ['flickr:' + x for x in sorted(flickr_ids)]
        coco_ids = sorted({c for f in flickr_ids for c in flickr_coco.get(f, ())})
        parents += ['coco:' + c for c in coco_ids]
        hash_overlap = sorted(occupied_hashes.intersection(identity.values()))
        parent_overlap = sorted(occupied_parents.intersection(parents))
        conflict = bool(hash_overlap or parent_overlap or coco_ids)
        if conflict:
            overlaps.append(dict(image_id=key, parent_ids=parents, exact_hash_overlap=hash_overlap,
                                 shared_parent_ids=parent_overlap, known_coco_ids=coco_ids,
                                 training_matches=[dict(id=r['id'], dataset=r['dataset'], split=r['split']) for r in training
                                                   if set(r['parent_ids']).intersection(parents)
                                                   or identity['pixels_sha256'] == r['pixels_sha256']],
                                 action='excluded_entire_image_from_independent_benchmark'))
        if conflict:
            continue
        protected_images[str(path)] = dict(**identity, parent_ids=parents)
        texts.update(normalized_text(t) for t in [*captions_from_row(row), *row['en']['caption']])
        for n, text in enumerate(captions_from_row(row)):
            pairs.append(dict(id=f'XM3600-PL:{key}:{n}', dataset='xm3600_pl', split='final_blind',
                              image=str(path), image_id=identity['pixels_sha256'], source_image_id=key,
                              parent_ids=parents, text=text, language='pl',
                              caption_origin='original-human-Polish', attribution=attribution))
    protected = dict(schema=1, datasets=['xm3600_pl'], ids=[p['id'] for p in pairs],
                     texts=sorted(texts), images=protected_images,
                     cross_dataset_parents={}, cross_dataset_pixels={})
    protected['sha256'] = digest(protected)
    write_json(output / 'blacklist.json', protected)
    write_records(output / 'pairs.jsonl', pairs)
    write_json(output / 'overlap_check.json', dict(overlaps=overlaps, checked_against='frozen train+validation and all legacy query/candidate images',
                                                 methods=['sha256', 'pixels_sha256', 'known Flickr source IDs'],
                                                 limitations='Unknown crops and unparsed source-URL variants are not certified.'))
    if not pairs:
        raise ValueError('Independent benchmark is empty after known-source exclusions')
    manifest = dict(protocol='xm3600-pl-clean-subset-full-gallery-v1', role='FINAL_BLIND',
                    original_images=len(images), images=len({p['source_image_id'] for p in pairs}), captions=len(pairs),
                    unique_texts=len({p['text'] for p in pairs}), assets=assets,
                    pairs_sha256=file_sha(output / 'pairs.jsonl'), blacklist_sha256=file_sha(output / 'blacklist.json'),
                    source='https://google.github.io/crossmodal-3600/',
                    annotations_license='CC BY 4.0; individual image attribution/license is preserved separately',
                    lineage='Open Images / known Flickr source IDs; no COCO-specific source pool',
                    excluded_known_overlap_or_coco_images=len(overlaps), known_overlap_count=0, scores_computed=False,
                    vg_mapping_sha256=file_sha(vg_path),
                    exclusion_policy='Entire images with exact hash/source matches to frozen train/val/legacy eval, or Flickr-to-COCO mapping, are excluded. This is a documented subset, not the unchanged official 3600-image gallery.',
                    note='Do not add this dataset to per-run evaluation or select hyperparameters on it. Future ingestion must merge its blacklist with the existing guard.')
    write_json(output / 'manifest.json', manifest)
    write_json(ROOT / 'reports/xm3600_pl_preparation.json', manifest)
    write_json(ROOT / 'reports/xm3600_pl_exclusions.json', overlaps)
    return manifest


def evaluate(output, checkpoint, release):
    if not release:
        raise ValueError('XM3600-PL is FINAL BLIND. Use --release-final-blind only for a preselected finalist.')
    manifest = read_json(output / 'manifest.json')
    if manifest['known_overlap_count'] or file_sha(output / 'pairs.jsonl') != manifest['pairs_sha256']:
        raise ValueError('Independent benchmark integrity/overlap check failed')
    import torch
    from pl_mmb_training.backbones import cache_backbone
    from pl_mmb_training.commercial_backbones import load_backbone
    from pl_mmb_training.features import encode_cached
    from pl_mmb_training.retrieval import make_graph, evaluate_graph
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    pairs = [json.loads(line) for line in (output / 'pairs.jsonl').read_text(encoding='utf-8').splitlines()]
    images, texts = sorted({p['image'] for p in pairs}), sorted({p['text'] for p in pairs})
    blacklist = read_json(output / 'blacklist.json')
    if file_sha(output / 'blacklist.json') != manifest['blacklist_sha256']:
        raise ValueError('Independent image identities changed')
    cache_root = Path.home() / '.cache/huggingface/pl_mmb_training/xm3600/features'
    lock = read_json(ROOT / 'training/runs/vision-commercial-five/model_lock.json')['tinyclip-39m']
    x, vision_meta = cache_backbone('tinyclip-39m', lock, images,
        {p: v['sha256'] for p, v in blacklist['images'].items()}, cache_root, 32, loader=load_backbone)
    base = read_json(ROOT / 'training/runs/e1-full/config.json')
    model_lock = read_json(ROOT / 'training/runs/e1-full/model_lock.json')
    t, text_meta = encode_cached(texts, 'texts', base, model_lock, cache_root)
    saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
    model = torch.nn.Linear(saved['input_dim'], saved['output_dim'])
    model.load_state_dict(saved['state_dict'])
    with torch.inference_mode():
        projected = model(x)
    graph = make_graph(pairs, images, texts)
    metrics = evaluate_graph(graph, projected, t, include_ranks=True)
    result = dict(protocol=manifest['protocol'], role='FINAL_BLIND', checkpoint_sha256=file_sha(checkpoint),
                  checkpoint=str(checkpoint), manifest_sha256=file_sha(output / 'manifest.json'),
                  metrics=metrics, vision_cache=vision_meta, text_cache=text_meta,
                  selection_uses_xm3600=False, interpretation='Cross-dataset full-gallery retrieval, not directly comparable to legacy local candidate pools.')
    write_json(output / f"finalist-{result['checkpoint_sha256']}.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'evaluate'))
    parser.add_argument('--cache', type=Path, default=Path.home() / '.cache/huggingface/pl_mmb_training/xm3600')
    parser.add_argument('--output', type=Path, default=ROOT / 'training/runs/xm3600-pl')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--release-final-blind', action='store_true')
    args = parser.parse_args()
    if args.command == 'evaluate' and args.checkpoint is None:
        parser.error('--checkpoint is required for evaluation')
    result = prepare(args.cache, args.output) if args.command == 'prepare' else evaluate(args.output, args.checkpoint, args.release_final_blind)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
