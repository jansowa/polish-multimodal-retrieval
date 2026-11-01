"""Check local label placement and exact model selection without inference."""
import hashlib
import json
import math
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from retrieval_benchmark.core import read, sha, write
from retrieval_benchmark.space.view import default_columns,default_tasks,organizations,overview,pareto_html
FOLDER=Path(__file__).parent
raw=(ROOT/'retrieval_benchmark/space/results.json').read_bytes()
doc=json.loads(raw)

if '--live' in sys.argv:
    from gradio_client import Client
    for attempt in range(20):
        try:
            with urllib.request.urlopen('http://127.0.0.1:7865/config',timeout=5) as stream:
                config=json.load(stream)
            break
        except urllib.error.URLError:
            if attempt==19: raise
            time.sleep(.5)
    pickers=[c for c in config['components'] if c['type']=='dropdown' and c['props'].get('label')=='Modele']
    assert len(pickers)==1 and pickers[0]['props']['multiselect']
    ids=[p['id'] for p in overview(doc)['model_data'] if p['name'] in ('R0_seed44','R1_seed44','R3_seed44')]
    assert len(ids)==3
    args=['recall@1','v2_no_instructions','historical-native-v2',0,False,False,
          default_columns(doc),default_tasks(doc),0,organizations(doc),ids,False,True]
    client=Client('http://127.0.0.1:7865',verbose=False)
    answer=client.predict(*args,api_name='/leaderboard')
    table=answer[0].get('value',answer[0])
    assert len(table['data'])==len(ids)==answer[2].count('class="model-label"')
    for mid in ids: assert 'data-model-id="'+mid+'"' in answer[2]
    args[10]=[]
    restored=client.predict(*args,api_name='/leaderboard')
    assert len(restored[0].get('value',restored[0])['data'])>len(ids)
    write(FOLDER/'live-validation.json',dict(url='http://localhost:7865/',status='passed',
          searchable_model_picker=True,selected_model_ids=ids,exact_table_and_graph=True,empty_restores_all=True))
    print('LIVE_MODEL_SELECTION_OK')
else:
    names={'R1_seed44','alphaedge-ai/metaclip-2-worldwide-m16-384-pol-32768',
           'alphaedge-ai/siglip2-base-patch16-512-pol-32768','timm/ViT-B-16-SigLIP-i18n-256','google/embeddinggemma-2'}
    selected=[p['id'] for p in overview(doc)['model_data'] if p['name'] in names]
    cluster=[p['id'] for p in overview(doc)['model_data'] if p['name'] in ('R0_seed44','R1_seed44','R3_seed44','R3_wiki2_seed43')]
    cases=[('all',{}),('under100',dict(max_parameters=100)),('selected',dict(selected_models=selected)),
           ('cluster',dict(selected_models=cluster)),('selected-log',dict(selected_models=selected,log_scale=True))]
    results=[]
    for name,options in cases:
        markup=pareto_html(doc,**options)
        svg=ET.fromstring(markup[markup.index('<svg '):markup.index('</svg>')+6])
        ns={'s':'http://www.w3.org/2000/svg'}
        boxes=[]; distances=[]
        for anchor in svg.findall('s:a',ns):
            label=anchor.find('s:g[@class="model-label"]',ns); rect=label.find('s:rect',ns)
            box={k:float(rect.attrib[k]) for k in ('x','y','width','height')}
            circle=anchor.find('s:circle',ns); px=float(circle.attrib['cx']); py=float(circle.attrib['cy'])
            distances.append(math.hypot(max(box['x']-px,0,px-box['x']-box['width']),
                                       max(box['y']-py,0,py-box['y']-box['height'])))
            boxes.append(box)
        for i,a in enumerate(boxes):
            for b in boxes[i+1:]:
                assert not (a['x']<b['x']+b['width'] and b['x']<a['x']+a['width'] and
                            a['y']<b['y']+b['height'] and b['y']<a['y']+a['height'])
        options_without_scale={k:v for k,v in options.items() if k!='log_scale'}
        assert len(boxes)==len(overview(doc,**options_without_scale)['rows'])
        if name.startswith('selected'): assert max(distances)<40
        if name=='cluster': assert max(distances)<100
        page='<!doctype html><html lang="pl"><meta charset="utf-8"><title>Compact labels</title>'
        page+='<body style="font:15px system-ui;color:#334155;background:white;margin:20px"><h1>Jakość a rozmiar — '+name+'</h1>'+markup
        page+='''<pre id="browser-check" style="display:none"></pre><script>
        addEventListener('load',()=>{const groups=[...document.querySelectorAll('.model-label')];
        const overflow=[];for(const group of groups){const t=group.querySelector('text').getBBox();const r=group.querySelector('rect').getBBox();
        if(t.x<r.x-1||t.y<r.y-1||t.x+t.width>r.x+r.width+1||t.y+t.height>r.y+r.height+1) overflow.push(group.dataset.labelFor);}
        document.querySelector('#browser-check').textContent=JSON.stringify({labels:groups.length,overflow});});
        </script></body></html>'''
        (FOLDER/(name+'.html')).write_text(page,encoding='utf-8')
        results.append(dict(case=name,models=len(boxes),overlaps=0,maximum_leader_pixels=max(distances,default=0)))
    protected=read(ROOT/'reports/leaderboard_requested_20261101/registration.json')['protected']
    assert {p:sha(ROOT/p) for p in protected}==protected
    write(FOLDER/'validation.json',dict(results_sha256=hashlib.sha256(raw).hexdigest(),cases=results,protected_sha256=protected,
        code_sha256={p:sha(ROOT/p) for p in ['retrieval_benchmark/space/view.py','retrieval_benchmark/space/app.py','retrieval_benchmark/tests/test_pareto.py']}))
    print('COMPACT_LABELS_OK',results)
