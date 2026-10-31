"""Fail-closed gate for any further Wiki pilot in the consultation series."""
from pathlib import Path
from .common import ROOT, file_sha, read_json


class ExternalHoldoutGuard:
    """Reserve entire Wiki components, including image-only future training."""
    def __init__(self, value):
        self.ids=set(value.get('ids',[]));self.texts=set(value.get('texts',[]))
        self.parents=set(value.get('parent_ids',[]));self.hashes=set()
        self.source_sha1=set(value.get('source_sha1_hex',[]));self.phashes=set()
        self.computed={}
        for image in value.get('images',{}).values():
            self.parents.update(image.get('parent_ids',[]))
            self.hashes.update(image[k] for k in ('sha256','pixels_sha256') if image.get(k))
            if image.get('phash'):self.phashes.add(int(image['phash'],16))

    def reasons(self,record):
        from .leakage import normalized_text,image_identity
        reasons=[]
        if record.get('id') in self.ids:reasons.append('wiki_dev_id')
        if self.parents.intersection(record.get('parent_ids',[])):reasons.append('wiki_dev_source_or_page')
        if self.hashes.intersection(record.get(k) for k in ('sha256','pixels_sha256') if record.get(k)):
            reasons.append('wiki_dev_file_or_pixels')
        if any(normalized_text(record[f]) in self.texts for f in ('text_en','text_pl','source_caption') if record.get(f)):
            reasons.append('wiki_dev_text')
        sha1=record.get('source_sha1_hex') or record.get('source_record',{}).get('source_sha1')
        if sha1:
            try:
                normalized=f'{int(sha1,16 if len(sha1)==40 else 36):040x}'
                if normalized in self.source_sha1:reasons.append('wiki_dev_source_sha1')
            except (TypeError,ValueError):reasons.append('wiki_dev_invalid_source_identity')
        value=record.get('phash')
        if not value and self.phashes:
            sha=record.get('sha256')
            if sha in self.computed:value=self.computed[sha]
            else:
                path=Path(record.get('image',''))
                if path.is_file():
                    if image_identity(path)!=dict(sha256=record.get('sha256'),pixels_sha256=record.get('pixels_sha256')):
                        raise ValueError('Image changed during Wiki holdout verification')
                    import numpy as np
                    from PIL import Image,ImageOps
                    from scipy.fft import dctn
                    with Image.open(path) as image:
                        gray=ImageOps.exif_transpose(image).convert('RGB').convert('L').resize((32,32),Image.Resampling.LANCZOS)
                        coefficients=dctn(np.asarray(gray,dtype=float),type=2)[:8,:8]
                        numeric=int.from_bytes(np.packbits((coefficients>np.median(coefficients)).reshape(-1)).tobytes(),'big')
                    value=f'{numeric:016x}';self.computed[sha]=value
        if self.phashes:
            if not value:reasons.append('wiki_dev_unverified_phash')
            elif any((int(value,16)^h).bit_count()<=6 for h in self.phashes):reasons.append('wiki_dev_phash_le6')
        return reasons


def workspace_guard():
    folder=ROOT/'reports/series_2026-10-23'
    path=folder/'wiki_dev_guard_enforcement.json'
    if not path.exists():path=folder/'wiki_dev_guard.json'
    if not path.exists():return None
    value=read_json(path)
    if value.get('status')!='frozen':raise ValueError('Wiki holdout guard is not frozen')
    if file_sha(ROOT/value['manifest_path'])!=value['dev_manifest_sha256']:
        raise ValueError('Wiki holdout manifest differs from frozen guard')
    return ExternalHoldoutGuard(value)


def require_wiki_training_release(folder=None):
    folder = Path(folder) if folder else ROOT / 'reports/series_2026-10-23'
    gate = folder / 'wiki_training_release_decision.json'
    if not gate.exists():
        raise RuntimeError('Wiki training blocked: new native dev, disjoint-train guard, immutable pre-Wiki baseline and user decision are required')
    decision = read_json(gate)
    if decision.get('status') != 'approved' or decision.get('user_approved') is not True:
        raise RuntimeError('Wiki training release decision has not been approved')
    for key in ('dev_manifest', 'train_manifest', 'guard', 'pre_wiki_baseline'):
        item = decision[key]
        path = ROOT / item['path']
        if not path.is_file() or file_sha(path) != item['sha256']:
            raise RuntimeError(f'Wiki release artifact missing or changed: {key}')
    raise RuntimeError('A new disjoint Wiki data loader must be implemented and verified before releasing this historical pilot runner')
