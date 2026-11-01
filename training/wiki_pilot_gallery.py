"""Small deterministic contact sheet of the actual wiki inputs and PL captions."""
import hashlib
import json
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'training/runs/tinyclip-wiki-pilot-v1/data'


def main():
    rgb=np.load(DATA/'wiki_native_rgb.npy',mmap_mode='r');selected=[]
    for split in ('train','dev'):
        rows=[json.loads(line) for line in (DATA/f'wiki_{split}.jsonl').read_text(encoding='utf-8').splitlines()]
        rows.sort(key=lambda r:hashlib.sha256(('wiki-qa-v1:'+r['id']).encode()).hexdigest())
        seen=set()
        for r in rows:
            if r['component'] not in seen:
                seen.add(r['component']);selected.append((split,r))
                if len(seen)==8:break
    width,height=380,440;canvas=Image.new('RGB',(width*4,height*4),'white');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    for i,(split,row) in enumerate(selected):
        left=(i%4)*width;top=(i//4)*height
        image=Image.fromarray(np.array(rgb[row['cache_index']]).transpose(1,2,0))
        canvas.paste(image,(left+78,top+24))
        draw.text((left+8,top+4),f'{split}: {row["id"]}',fill='black',font=font)
        caption='\n'.join(textwrap.wrap(row['text_pl'],width=49))
        draw.multiline_text((left+8,top+256),caption,fill='black',font=font,spacing=3)
    path=ROOT/'reports/tinyclip_wiki_pilot_gallery.jpg';canvas.save(path,quality=90)
    print(path,flush=True)


if __name__=='__main__':main()
