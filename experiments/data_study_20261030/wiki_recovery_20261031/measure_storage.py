"""CPU-only empirical size comparison; originals unchanged; no network."""
import io
import statistics
from config import *
from PIL import Image,ImageOps


def main():
    records=[read(p) for p in sorted((STORE/'receipts').glob('*.json')) if read(p).get('status')=='ready']
    records=[r for r in records if not r.get('reused')][:20]
    sizes={n:[] for n in ('original','long768_q92','long1024_q92','long1280_q92','long1536_q92')}
    for r in records:
        raw=Path(r['path']).read_bytes();sizes['original'].append(len(raw))
        with Image.open(io.BytesIO(raw)) as im:
            rgb=ImageOps.exif_transpose(im).convert('RGB')
            for edge in (768,1024,1280,1536):
                copy=rgb.copy();copy.thumbnail((edge,edge),Image.Resampling.LANCZOS)
                out=io.BytesIO();copy.save(out,'JPEG',quality=92,subsampling=0,optimize=True)
                sizes[f'long{edge}_q92'].append(len(out.getvalue()));copy.close()
    summary={k:dict(sample_n=len(v),mean_bytes=statistics.mean(v),median_bytes=statistics.median(v),
        estimated_all_24125_GB=statistics.mean(v)*24125/1e9) for k,v in sizes.items()}
    atomic_json(REPORT/'storage_measurement.json',dict(status='measured_on_20_deterministic_originals_not_full_census',
        profiles=summary,preserve_aspect_ratio=True,no_crop=True,no_upscale=True,JPEG_quality=92,JPEG_subsampling=0))
    print(__import__('json').dumps(summary))


if __name__=='__main__':main()
