"""Contact sheet of native training crops; no benchmark images or model scores."""
import json
import textwrap
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'training/runs/tinyclip-visualnews-v1/data'

def main():
    pixels=np.load(FOLDER/'native_rgb.npy',mmap_mode='r')
    selected=[]
    for split in ('train','dev'):
        rows=[json.loads(line) for line in (FOLDER/f'vn_{split}.jsonl').open(encoding='utf-8')]
        for source in ('washington_post','usa_today'):
            pool=[r for r in rows if r['source']==source]
            order=np.random.default_rng(1701).permutation(len(pool))[:6]
            selected.extend((split,pool[i]) for i in order)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    canvas=Image.new('RGB',(1400,6*375),'white');draw=ImageDraw.Draw(canvas)
    manifest=[]
    for i,(split,r) in enumerate(selected):
        x,y=(i%4)*350,(i//4)*375
        im=Image.fromarray(np.array(pixels[r['cache_index']]).transpose(1,2,0)).resize((280,224))
        canvas.paste(im,(x,y+18))
        draw.text((x+3,y+2),f'{i+1}. {split} / {r["source"]}',fill='black',font=font)
        caption='\n'.join(textwrap.wrap(r['text_pl'],width=44))
        draw.multiline_text((x+3,y+247),caption,fill='black',font=font,spacing=2)
        manifest.append(dict(number=i+1,split=split,id=r['id'],caption=r['text_pl']))
    canvas.save(ROOT/'reports/tinyclip_visualnews_quality_sample.png')
    (ROOT/'reports/tinyclip_visualnews_quality_sample.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
