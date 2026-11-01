"""Preview and inspect actual SVG label boxes; optionally probe the live UI."""
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from retrieval_benchmark.core import read, sha, write
from retrieval_benchmark.space.view import default_columns, default_tasks, overview, pareto_html

FOLDER = Path(__file__).parent


if '--live' in sys.argv:
    import urllib.request
    import urllib.error
    import time
    from gradio_client import Client
    for attempt in range(20):
        try:
            with urllib.request.urlopen('http://127.0.0.1:7865/config',timeout=5) as response:
                config = json.load(response)
            break
        except urllib.error.URLError:
            if attempt==19:
                raise
            time.sleep(.5)
    toggles = [c for c in config['components'] if c['props'].get('label')=='Nazwy modeli zawsze widoczne']
    assert len(toggles)==1 and toggles[0]['props']['value'] is True
    doc = read(ROOT/'retrieval_benchmark/space/results.json')
    args = ['recall@1','v2_no_instructions','historical-native-v2',100,False,False,
            default_columns(doc),default_tasks(doc),0,['jansowa'],False,True]
    client = Client('http://127.0.0.1:7865',verbose=False)
    answer = client.predict(*args,api_name='/leaderboard')
    table = answer[0].get('value',answer[0])
    assert len(table['data'])==answer[2].count('class="model-label"')>0
    args[-1]=False
    hidden = client.predict(*args,api_name='/leaderboard')
    assert 'class="model-label"' not in hidden[2] and '<title>' in hidden[2]
    write(FOLDER/'live-validation.json',dict(url='http://localhost:7865/',status='passed',
          labels_default_on=True,labels_equal_visible_rows=True,hover_toggle=True))
    print('LIVE_LABELS_OK')
else:
    raw = (ROOT/'retrieval_benchmark/space/results.json').read_bytes()
    doc = json.loads(raw)
    protected = read(ROOT/'reports/leaderboard_requested_20261101/registration.json')['protected']
    assert {p:sha(ROOT/p) for p in protected}==protected
    results = []
    for name,options in [('all',{}),('under100',dict(max_parameters=100)),
                         ('under100-log',dict(max_parameters=100,log_scale=True))]:
        markup = pareto_html(doc,**options)
        svg = ET.fromstring(markup[markup.index('<svg '):markup.index('</svg>')+6])
        ns = {'s':'http://www.w3.org/2000/svg'}
        boxes = []
        for label in svg.findall('.//s:g[@class="model-label"]',ns):
            rect = label.find('s:rect',ns)
            boxes.append(dict(id=label.attrib['data-label-for'], **{k:float(rect.attrib[k]) for k in ('x','y','width','height')}))
        for i,a in enumerate(boxes):
            for b in boxes[i+1:]:
                assert not (a['x']<b['x']+b['width'] and b['x']<a['x']+a['width'] and
                            a['y']<b['y']+b['height'] and b['y']<a['y']+a['height']), (a,b)
        filtered = {k:v for k,v in options.items() if k!='log_scale'}
        assert len(boxes)==len(overview(doc,**filtered)['rows'])
        page = '<!doctype html><html lang="pl"><meta charset="utf-8"><title>Model labels</title>'
        page += '<body style="font:15px system-ui;color:#334155;background:white;padding:20px;margin:0">'
        page += '<h1>Jakość a rozmiar modelu — '+name+'</h1>'+markup
        # Browser text metrics validate actual glyph widths, not estimates alone.
        page += '''<pre id="browser-check" style="display:none"></pre><script>
        addEventListener('load',()=>{const labels=[...document.querySelectorAll('.model-label')];
        const errors=[];for(const group of labels){const t=group.querySelector('text').getBBox();
        const r=group.querySelector('rect').getBBox();if(t.x<r.x-1||t.y<r.y-1||
        t.x+t.width>r.x+r.width+1||t.y+t.height>r.y+r.height+1) errors.push({id:group.dataset.labelFor,
        text:{x:t.x,y:t.y,width:t.width,height:t.height},rect:{x:r.x,y:r.y,width:r.width,height:r.height},font:getComputedStyle(group.querySelector('text')).font});}
        document.querySelector('#browser-check').textContent=JSON.stringify({labels:labels.length,overflow:errors});});
        </script></body></html>'''
        (FOLDER/(name+'.html')).write_text(page,encoding='utf-8')
        results.append(dict(case=name,visible_models=len(boxes),labels=len(boxes),overlapping_label_boxes=0))
    write(FOLDER/'validation.json',dict(results_sha256=hashlib.sha256(raw).hexdigest(),cases=results,
        code_sha256={name:sha(ROOT/name) for name in ['retrieval_benchmark/space/view.py','retrieval_benchmark/space/app.py',
            'retrieval_benchmark/tests/test_pareto.py']},
        protected_sha256=protected))
    print('SVG_LABEL_BOXES_OK',results)
