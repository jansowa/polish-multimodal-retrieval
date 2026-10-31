"""Verified originals in RAM -> full-frame master1536 + exact TinyCLIP224."""
import hashlib
import io
import os
from config import *
import numpy as np
from PIL import Image,ImageOps
from scipy.fft import dctn

PROCESSOR_PATH=Path('/home/jsowa/.cache/huggingface/hub/models--wkcn--TinyCLIP-ViT-39M-16-Text-19M-YFCC15M/snapshots/07a4b0bc751cb64fecd2b661c048c1dd98d69444')
_processor=None


def processor():
    global _processor
    if _processor is None:
        from transformers import CLIPImageProcessor
        _processor=CLIPImageProcessor.from_pretrained(PROCESSOR_PATH,local_files_only=True)
    return _processor


def store_bytes(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.part')
    with tmp.open('wb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
    tmp.replace(path)


def create(raw,expected_sha1):
    if hashlib.sha1(raw).hexdigest()!=expected_sha1:raise ValueError('Original source SHA1 changed')
    raw_hash=hashlib.sha256(raw).hexdigest();p=processor()
    with Image.open(io.BytesIO(raw)) as image:
        size=list(image.size)
        canonical=ImageOps.exif_transpose(image).convert('RGB')
        pixels_hash=hashlib.sha256(str(canonical.size).encode()+canonical.tobytes()).hexdigest()
        gray=canonical.convert('L').resize((32,32),Image.Resampling.LANCZOS)
        coefficients=dctn(np.asarray(gray,dtype=float),type=2)[:8,:8]
        phash=int.from_bytes(np.packbits((coefficients>np.median(coefficients)).reshape(-1)).tobytes(),'big')
        canonical.thumbnail((1536,1536),Image.Resampling.LANCZOS)
        master=io.BytesIO();canonical.save(master,'JPEG',quality=92,subsampling=0,optimize=True)
        master_size=list(canonical.size);canonical.close()
        # Match historical TinyCLIP preprocessing: RGB conversion before the
        # pinned CLIP processor, without introducing an EXIF policy change.
        rgb=image.convert('RGB')
        native=p(images=rgb,do_normalize=False,do_rescale=False,return_tensors='np')['pixel_values'][0]
        native=np.transpose(native,(1,2,0)).astype(np.uint8)
        native_image=Image.fromarray(native)
        expected=p(images=rgb,return_tensors='np')['pixel_values']
        restored=p(images=native_image,return_tensors='np')['pixel_values']
        if not np.array_equal(expected,restored):raise ValueError('TinyCLIP224 preprocessing is not bit-exact')
        encoded=io.BytesIO();native_image.save(encoded,'PNG');native_image.close();rgb.close()
    master_bytes=master.getvalue();native_bytes=encoded.getvalue()
    master_path=STORE/'masters1536'/(expected_sha1+'.jpg');native_path=STORE/'tinyclip224'/(expected_sha1+'.png')
    store_bytes(master_path,master_bytes);store_bytes(native_path,native_bytes)
    return dict(original_sha1=expected_sha1,sha256=raw_hash,pixels_sha256=pixels_hash,phash=f'{phash:016x}',
        original_bytes=len(raw),original_dimensions=size,
        master_path=str(master_path),master_sha256=hashlib.sha256(master_bytes).hexdigest(),master_dimensions=master_size,
        native_path=str(native_path),native_sha256=hashlib.sha256(native_bytes).hexdigest(),native_dimensions=[224,224],
        stored_bytes=len(master_bytes)+len(native_bytes),preprocessing_bit_exact=True,
        original_fingerprint_evidence='Original downloaded/reused, SHA1 verified, decoded and hashed before derivative creation',
        processing_code_sha256=sha256(__file__),processor_config_sha256=sha256(PROCESSOR_PATH/'preprocessor_config.json'))
