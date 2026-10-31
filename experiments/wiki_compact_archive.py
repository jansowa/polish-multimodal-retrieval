"""WSL CPU archive helper: verified originals, immutable compact assets, SHA1 leases.

Uses the same pinned TinyCLIP image processor and pixel/pHash definitions as
data_study_20261030/wiki_recovery_20261031/image_assets.py. No network or GPU.
"""
from __future__ import annotations
import argparse
import base64
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import uuid
from datetime import datetime,timezone

from PIL import Image,ImageOps,ImageFile

PROCESSOR_PATH=Path('/home/jsowa/.cache/huggingface/hub/models--wkcn--TinyCLIP-ViT-39M-16-Text-19M-YFCC15M/snapshots/07a4b0bc751cb64fecd2b661c048c1dd98d69444')
STORE=Path('/mnt/c/Users/jsowa/benchmark-translations/data-ood-20261030/A/wiki-recovery-20261031')
PROCESSOR_CONFIG_SHA256='910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd'


def digest(payload):return hashlib.sha256(payload).hexdigest()


def sha1_hex(value):
    value=str(value or '').strip().lower()
    if not re.fullmatch(r'[0-9a-z]{1,40}',value):raise ValueError('Missing or invalid original SHA1')
    try:number=int(value,16 if len(value)==40 else 36)
    except ValueError:raise ValueError('Invalid original SHA1')
    if number>=2**160:raise ValueError('Invalid original SHA1 size')
    return f'{number:040x}'


def local_path(value):
    value=str(value)
    if re.match(r'^[A-Za-z]:[/\\]',value):return Path('/mnt/'+value[0].lower()+'/'+value[3:].replace('\\','/'))
    return Path(value)


def utc():return datetime.now(timezone.utc).isoformat()


def publish_absent(path,payload):
    """Hard-link commit is atomic create-if-absent; existing bytes are immutable."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=payload:raise ValueError('Existing archive artifact differs; preserved: '+str(path))
        return
    fd,temporary=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        try:os.link(temporary,path)
        except FileExistsError:
            if path.read_bytes()!=payload:raise ValueError('Concurrent archive artifact differs; preserved: '+str(path))
        directory=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        # Only this call's private staging file is removed; archive data is never deleted.
        if os.path.exists(temporary):os.unlink(temporary)


def json_bytes(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode('utf-8')


class Archive:
    def __init__(self,store=STORE,processor=None,processor_config_sha256=PROCESSOR_CONFIG_SHA256,external_catalogs=None):
        self.store=Path(store);self.processor=processor;self.config_hash=processor_config_sha256
        self.leases={};self.external=set();self.catalogs=external_catalogs
        if self.catalogs is None:self.catalogs=[self.store/'download_catalog.jsonl',self.store/'download_catalog_components.jsonl']
        for catalog in self.catalogs:
            if not Path(catalog).exists():continue
            for line in Path(catalog).open(encoding='utf-8'):
                if line.strip():self.external.add(sha1_hex(json.loads(line)['sha1']))

    def paths(self,h):
        return self.store/'masters1536'/(h+'.jpg'),self.store/'tinyclip224'/(h+'.png'),self.store/'compact_receipts'/(h+'.json')

    def get_processor(self):
        if self.processor is None:
            if digest((PROCESSOR_PATH/'preprocessor_config.json').read_bytes())!=self.config_hash:
                raise ValueError('Pinned TinyCLIP processor configuration changed')
            from transformers import CLIPImageProcessor
            self.processor=CLIPImageProcessor.from_pretrained(PROCESSOR_PATH,local_files_only=True)
        return self.processor

    def validate(self,h):
        master,native,path=self.paths(h)
        if not path.exists():return None
        receipt=json.loads(path.read_text(encoding='utf-8'))
        if receipt.get('status')=='failed':raise ValueError('Existing archive failure receipt preserved for review: '+str(path))
        if receipt.get('status')!='ready':return None
        if receipt.get('sha1')!=h or receipt.get('original_sha1')!=h or receipt.get('preprocessing_bit_exact') is not True:
            raise ValueError('Invalid existing original verification receipt: '+str(path))
        if receipt.get('processor_config_sha256')!=self.config_hash:raise ValueError('Existing processor fingerprint mismatch')
        for field,size in [('sha256',64),('pixels_sha256',64),('phash',16)]:
            if not re.fullmatch('[0-9a-f]{'+str(size)+'}',str(receipt.get(field,''))):raise ValueError('Invalid original fingerprint '+field)
        for target,key,size_key in [(master,'master','master_dimensions'),(native,'native','native_dimensions')]:
            if local_path(receipt.get(key+'_path','')).resolve()!=target.resolve():raise ValueError('Receipt path outside expected archive asset')
            if not target.exists():raise ValueError('Existing ready receipt lacks artifact: '+str(target))
            payload=target.read_bytes()
            if digest(payload)!=receipt.get(key+'_sha256'):raise ValueError('Existing archive artifact hash mismatch: '+str(target))
            with Image.open(io.BytesIO(payload)) as image:
                if list(image.size)!=receipt.get(size_key):raise ValueError('Archive dimensions mismatch')
                if key=='native' and (image.format!='PNG' or image.mode!='RGB' or image.size!=(224,224)):
                    raise ValueError('TinyCLIP artifact is not a lossless RGB 224 PNG')
                if key=='master' and (image.format!='JPEG' or image.mode!='RGB' or max(image.size)>1536):
                    raise ValueError('Invalid full-frame master')
                image.verify()
        original=receipt.get('original_dimensions')
        if not isinstance(original,list) or len(original)!=2 or any(type(x) is not int or x<1 for x in original):
            raise ValueError('Invalid original dimensions')
        if max(receipt['master_dimensions'])>max(original):raise ValueError('Master was enlarged')
        return receipt

    def source_metadata(self,h,record):
        keys=['id','source_file_id','source_sha1','image_original_url','source_page_url','image_filename',
              'width','height','license','license_url','creator','attribution','retrieved_at']
        value={key:record.get(key) for key in keys}
        value['original_sha1']=h
        payload=json_bytes(value)
        target=self.store/'compact_receipts/source_metadata'/h/(digest(payload)+'.json')
        # A companion preserves new attribution without editing another producer's receipt.
        publish_absent(target,payload)
        return target,value

    def ready(self,h,record,receipt):
        metadata,_=self.source_metadata(h,record)
        master,_,path=self.paths(h)
        return dict(action='ready',master_base64=base64.b64encode(master.read_bytes()).decode(),
            receipt=receipt,receipt_path=str(path),metadata_path=str(metadata))

    def begin(self,record):
        import fcntl
        h=sha1_hex(record.get('source_sha1'))
        lockpath=self.store/'sha1_locks'/(h+'.lock');lockpath.parent.mkdir(parents=True,exist_ok=True)
        lock=lockpath.open('a+b')
        try:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return dict(action='deferred',reason='Archive SHA1 is held by another producer',retry_at=time.time()+10)
            receipt=self.validate(h)
            if receipt:return self.ready(h,record,receipt)
            # The running legacy recovery process does not acquire SHA1 locks. Its
            # frozen catalog gives it exclusive ownership until a ready receipt exists.
            if h in self.external:return dict(action='deferred',reason='SHA1 belongs to the active recovery catalog; wait for its receipt',retry_at=time.time()+30)
            _,_,path=self.paths(h)
            if path.exists():return dict(action='deferred',reason='Existing non-ready receipt preserved for its owner',retry_at=time.time()+60)
            if shutil.disk_usage(self.store).free<4*1024**3+200*1024**2:
                return dict(action='deferred',reason='Archive disk reserve below 4 GiB + staging allowance',retry_at=time.time()+300)
            token=uuid.uuid4().hex;self.leases[token]=(h,record,lock);lock=None
            return dict(action='download',token=token,sha1=h)
        finally:
            if lock is not None:lock.close()

    def abort(self,token):
        lease=self.leases.pop(token,None)
        if lease:lease[2].close()

    def build(self,raw,h):
        import numpy as np
        from scipy.fft import dctn
        if hashlib.sha1(raw).hexdigest()!=h:raise ValueError('Original source SHA1 changed')
        raw_hash=digest(raw);processor=self.get_processor()
        with Image.open(io.BytesIO(raw)) as image:
            dimensions=list(image.size)
            canonical=ImageOps.exif_transpose(image).convert('RGB')
            pixels_hash=digest(str(canonical.size).encode()+canonical.tobytes())
            gray=canonical.convert('L').resize((32,32),Image.Resampling.LANCZOS)
            coefficients=dctn(np.asarray(gray,dtype=float),type=2)[:8,:8]
            phash=int.from_bytes(np.packbits((coefficients>np.median(coefficients)).reshape(-1)).tobytes(),'big')
            canonical.thumbnail((1536,1536),Image.Resampling.LANCZOS)
            master=io.BytesIO()
            # High-entropy 4:4:4 JPEGs can exceed Pillow's default optimizer buffer.
            previous_block=ImageFile.MAXBLOCK
            ImageFile.MAXBLOCK=max(previous_block,4*canonical.width*canonical.height)
            try:canonical.save(master,'JPEG',quality=92,subsampling=0,optimize=True)
            finally:ImageFile.MAXBLOCK=previous_block
            master_dimensions=list(canonical.size);canonical.close()
            # Deliberately match the pinned historical processor's EXIF policy:
            # RGB conversion of the original, not the EXIF-transposed JPEG master.
            rgb=image.convert('RGB')
            native=processor(images=rgb,do_normalize=False,do_rescale=False,return_tensors='np')['pixel_values'][0]
            native=np.transpose(native,(1,2,0)).astype(np.uint8)
            if native.shape!=(224,224,3):raise ValueError('Unexpected TinyCLIP input dimensions')
            native_image=Image.fromarray(native)
            expected=processor(images=rgb,return_tensors='np')['pixel_values']
            restored=processor(images=native_image,return_tensors='np')['pixel_values']
            if not np.array_equal(expected,restored):raise ValueError('TinyCLIP224 preprocessing is not bit-exact')
            encoded=io.BytesIO();native_image.save(encoded,'PNG');native_image.close();rgb.close()
        master_bytes=master.getvalue();native_bytes=encoded.getvalue();master_path,native_path,_=self.paths(h)
        receipt=dict(original_sha1=h,sha256=raw_hash,pixels_sha256=pixels_hash,phash=f'{phash:016x}',
            original_bytes=len(raw),original_dimensions=dimensions,
            master_path=str(master_path),master_sha256=digest(master_bytes),master_dimensions=master_dimensions,
            native_path=str(native_path),native_sha256=digest(native_bytes),native_dimensions=[224,224],
            stored_bytes=len(master_bytes)+len(native_bytes),preprocessing_bit_exact=True,
            original_fingerprint_evidence='Original downloaded/reused, SHA1 verified, decoded and hashed before derivative creation',
            processing_code_sha256=digest(Path(__file__).read_bytes()),processor_config_sha256=self.config_hash)
        return receipt,master_bytes,native_bytes

    def commit(self,token,raw):
        h,record,_=self.leases[token]
        try:
            receipt,master_bytes,native_bytes=self.build(raw,h)
            metadata,source=self.source_metadata(h,record)
            master,native,path=self.paths(h)
            publish_absent(master,master_bytes);publish_absent(native,native_bytes)
            receipt.update(status='ready',sha1=h,id=record['id'],url=record['image_original_url'],
                source_page_url=record.get('source_page_url'),retained_original_path=None,updated_at=utc(),
                producer='wikimedia_grounding_pipeline',source_metadata=source,source_metadata_path=str(metadata))
            publish_absent(path,json_bytes(receipt))
            verified=self.validate(h)
            return self.ready(h,record,verified)
        finally:self.abort(token)

    def close(self):
        for token in list(self.leases):self.abort(token)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--store',default=str(STORE));args=parser.parse_args()
    archive=Archive(local_path(args.store))
    try:
        for line in sys.stdin:
            try:
                request=json.loads(line);operation=request['op']
                if operation=='begin':response=archive.begin(request['record'])
                elif operation=='commit':response=archive.commit(request['token'],base64.b64decode(request['raw_base64'],validate=True))
                elif operation=='abort':archive.abort(request['token']);response=dict(action='aborted')
                elif operation=='close':break
                else:raise ValueError('Unknown archive operation')
                print(json.dumps(dict(ok=True,result=response),ensure_ascii=False),flush=True)
            except Exception as exc:
                print(json.dumps(dict(ok=False,error=type(exc).__name__+': '+str(exc)),ensure_ascii=False),flush=True)
    finally:archive.close()


if __name__=='__main__':main()
