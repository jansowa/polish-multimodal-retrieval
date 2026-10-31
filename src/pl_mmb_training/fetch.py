"""Explicit online acquisition; E0, extraction, and optimization work offline."""
import io
import json
import urllib.request
import zipfile
from pathlib import Path

from .common import ROOT, TASKS, digest, file_sha, read_json, write_json

VG_URL = 'https://visualgenome.org/static/data/dataset/image_data.json.zip'
VG_MIRROR = 'https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip'
VG_ARCHIVE_SHA256 = 'b87a94918cb2ff4d952cf1dfeca0b9cf6cd6fd204c2f8704645653be1163681a'
COCO_URLS = ('https://images.cocodataset.org/annotations/annotations_trainval2014.zip',
             'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip')


def fetch(config, dest, cache_root, data_only=False):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from pl_mmb_translate.registry import DATASETS
    from pl_mmb_benchmark.data import REVISION, SPECS
    cache_root.mkdir(parents=True, exist_ok=True)
    vg_path = cache_root / 'visual_genome_image_data.json'
    if vg_path.exists():
        provenance = read_json(cache_root / 'visual_genome_provenance.json')
        if provenance['extracted_sha256'] != file_sha(vg_path) or provenance['archive_sha256'] != VG_ARCHIVE_SHA256:
            raise ValueError('Visual Genome lineage cache checksum mismatch')
    if not vg_path.exists():
        for url in (VG_URL, VG_MIRROR):
            try:
                with urllib.request.urlopen(url, timeout=45) as response:
                    raw = response.read()
                break
            except (OSError, urllib.error.URLError):
                if url == VG_MIRROR:
                    raise
        import hashlib
        if hashlib.sha256(raw).hexdigest() != VG_ARCHIVE_SHA256:
            raise ValueError('Visual Genome lineage metadata checksum mismatch')
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            write_json(vg_path, json.loads(archive.read('image_data.json')))
        write_json(cache_root / 'visual_genome_provenance.json', dict(url=url, archive_sha256=VG_ARCHIVE_SHA256,
                                                                   extracted_sha256=file_sha(vg_path)))
    coco_path = cache_root / 'coco_annotation_parents.json'
    coco_provenance = cache_root / 'coco_annotation_provenance.json'
    if coco_path.exists():
        if read_json(coco_provenance)['mapping_sha256'] != file_sha(coco_path):
            raise ValueError('COCO annotation lineage cache checksum mismatch')
    else:
        archive_path = cache_root / 'annotations_trainval2014.zip'
        if not archive_path.exists():
            import shutil
            for url in COCO_URLS:
                try:
                    print(f'Fetch COCO annotation lineage: {url}', flush=True)
                    with urllib.request.urlopen(url, timeout=45) as response, archive_path.with_suffix('.tmp').open('wb') as stream:
                        shutil.copyfileobj(response, stream)
                    archive_path.with_suffix('.tmp').replace(archive_path)
                    break
                except (OSError, urllib.error.URLError):
                    if url == COCO_URLS[-1]:
                        raise
        else:
            url = 'existing-authoritative-archive'
        with zipfile.ZipFile(archive_path) as archive:
            coco = json.loads(archive.read('annotations/instances_train2014.json'))
        write_json(coco_path, {str(a['id']): a['image_id'] for a in coco['annotations']})
        write_json(coco_provenance, dict(url=url, archive_sha256=file_sha(archive_path), mapping_sha256=file_sha(coco_path)))
    def acquire(repo, filename, **kwargs):
        try:
            return hf_hub_download(repo, filename, local_files_only=True, **kwargs)
        except FileNotFoundError:
            return hf_hub_download(repo, filename, **kwargs)
    for key in TASKS:
        acquire('TIGER-Lab/MMEB-eval', f'{SPECS[key][0]}/test-00000-of-00001.parquet',
                repo_type='dataset', revision=REVISION)
    for source in config['data']['sources']:
        if source.get('kind') == 'mbeir':
            continue  # M-BEIR JSON sources were already pinned by the translation plan.
        adapter = DATASETS[source['dataset']].factory()
        for shard in source['shards']:
            acquire(adapter.path, shard, revision=adapter.revision, repo_type='dataset')
    if any(source['dataset'] == 'mbeir_mscoco' for source in config['data']['sources']):
        import subprocess
        import sys
        subprocess.run([sys.executable, str(ROOT / 'training/acquire_coco_images.py')], check=True)
    if data_only:
        return
    lock_path = dest / 'model_lock.json'
    lock = read_json(lock_path) if lock_path.exists() else {'config_sha256': digest(config)}
    if lock['config_sha256'] != digest(config):
        raise ValueError('Model lock belongs to another config')
    for kind in ('text', 'vision'):
        model = config[kind]['model']
        revision = lock.get(kind, {}).get('revision') or config[kind]['revision'] or HfApi().model_info(model, timeout=30).sha
        lock[kind] = dict(model=model, revision=revision)
        write_json(lock_path, lock)
        print(f'Fetch {model}@{revision}', flush=True)
        snapshot_download(model, revision=revision, max_workers=2,
            allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '1_Pooling/*', '2_Normalize/*'])
    print('Acquisition complete; all subsequent stages can run offline.', flush=True)
