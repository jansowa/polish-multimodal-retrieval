"""Summaries use matched records; raw failures and repeats remain auditable."""
from collections import Counter, defaultdict
import html
import itertools
import json
import math
from pathlib import Path
import random
import statistics

import visualnews_judge_pilot as p

KEYS = list(p.PROPS)[:8]

def read_results():
    result={}; attempts={}
    for path in sorted((p.OUT/'raw').glob('*.jsonl')):
        data=p.rows(path); latest={}
        for r in data:
            r['raw_json_valid']=r.get('json_valid',False)
            r['raw_schema_valid']=r.get('schema_valid',False)
            if not r.get('schema_valid') and 'raw_output' in r:
                # Only remove a complete enclosing markdown fence. Never extract arbitrary braces.
                raw=r['raw_output'].strip()
                if raw.startswith('```json\n') and raw.endswith('```'):
                    try:
                        parsed=json.loads(raw[len('```json\n'):-3].strip())
                        if p.validate(parsed,path.stem.startswith('T')):
                            r.update(parsed=parsed,schema_valid=True,recovered_fence=True)
                    except ValueError: pass
            key=(r['id'],r['repeat'])
            if key not in latest or r.get('schema_valid') or not latest[key].get('schema_valid'):
                latest[key]=r
        result[path.stem]=latest; attempts[path.stem]=data
    return result,attempts

def anomalies(o):
    flags=[]
    if o['visual_grounding']==4 and o['external_context_dependency']>0: flags.append('all_grounded_with_external_context')
    if o['visual_grounding']>=3 and o['caption_type']=='mostly_contextual': flags.append('high_grounding_mostly_contextual')
    if o['visual_grounding']==4 and o['unsupported_information']: flags.append('all_grounded_with_unsupported_claims')
    if o['image_caption_mismatch'] and o['visual_grounding']==4: flags.append('all_grounded_with_mismatch')
    return flags

def mean(xs): return statistics.mean(xs) if xs else None

def weighted_kappa(pairs):
    if not pairs:return None
    n=len(pairs); left=Counter(a['visual_grounding'] for a,b in pairs); right=Counter(b['visual_grounding'] for a,b in pairs)
    observed=sum((a['visual_grounding']-b['visual_grounding'])**2 for a,b in pairs)/n
    expected=sum(left[i]*right[j]*(i-j)**2 for i in range(5) for j in range(5))/(n*n)
    return 1-observed/expected if expected else None

def summarize():
    result,attempts=read_results()
    sample=json.loads((p.OUT/'sample.json').read_text(encoding='utf-8'))
    ids={r['id'] for r in sample}; dev={r['id'] for r in sample[:100]}; holdout=ids-dev
    summary={}
    manual=json.loads((p.OUT/'manual_reviews.json').read_text(encoding='utf-8')) if (p.OUT/'manual_reviews.json').exists() else []
    for name, rr in result.items():
        text_only=name.startswith('T')
        selected=[r for (uid,rep),r in rr.items() if uid in ids and rep==0]
        valid=[r for r in selected if r.get('schema_valid')]
        axis='potential_external_context_dependency' if text_only else 'visual_grounding'
        entry=dict(attempts=len(attempts[name]), requested_unique=len(selected), valid=len(valid),
            json_valid_rate=mean([r.get('raw_json_valid',False) for r in selected]), raw_schema_valid_rate=mean([r.get('raw_schema_valid',False) for r in selected]), schema_valid_rate=mean([r.get('schema_valid',False) for r in selected]),
            score_distribution=dict(Counter(r['parsed'][axis] for r in valid)),
            seconds_mean=mean([r['seconds'] for r in selected]),
            completion_tokens=sum((r.get('usage') or {}).get('completion_tokens',0) for r in attempts[name]),
            prompt_tokens=sum((r.get('usage') or {}).get('prompt_tokens',0) for r in attempts[name]))
        if not text_only:
            entry.update(caption_types=dict(Counter(r['parsed']['caption_type'] for r in valid)),
                external_context_distribution=dict(Counter(r['parsed']['external_context_dependency'] for r in valid)),
                mismatch_count=sum(r['parsed']['image_caption_mismatch'] for r in valid),
                logical_anomalies=dict(Counter(f for r in valid for f in anomalies(r['parsed']))),
                confidence_mean=mean([r['parsed']['confidence'] for r in valid]))
            entry['groups']={}
            groups=dict(random={r['id'] for r in sample if r['selection']=='random'},targeted={r['id'] for r in sample if r['selection']!='random'},development=dev,holdout=holdout)
            for tag in sorted({t for r in sample for t in r['tags']}): groups['tag:'+tag]={r['id'] for r in sample if tag in r['tags']}
            for group,group_ids in groups.items():
                vv=[r['parsed'] for r in valid if r['id'] in group_ids]
                entry['groups'][group]=dict(n=len(vv),grounding_mean=mean([o['visual_grounding'] for o in vv]),external_mean=mean([o['external_context_dependency'] for o in vv]),mismatch=sum(o['image_caption_mismatch'] for o in vv))
            pairs=[]
            for (uid,rep),r in rr.items():
                base=rr.get((uid,0))
                if uid in ids and rep>0 and r.get('schema_valid') and base and base.get('schema_valid'):
                    pairs.append((base['parsed'],r['parsed']))
            entry['stability']=dict(paired_repeats=len(pairs),
                quadratic_weighted_kappa=weighted_kappa(pairs),
                full_label_agreement=mean([all(a[k]==b[k] for k in KEYS) for a,b in pairs]),
                grounding_exact=mean([a['visual_grounding']==b['visual_grounding'] for a,b in pairs]),
                grounding_mae=mean([abs(a['visual_grounding']-b['visual_grounding']) for a,b in pairs]),
                caption_type_exact=mean([a['caption_type']==b['caption_type'] for a,b in pairs]),
                external_exact=mean([a['external_context_dependency']==b['external_context_dependency'] for a,b in pairs]))
            assessed=[]
            for review in manual:
                r=rr.get((review['id'],0),{})
                if not r.get('schema_valid'):continue
                value=r['parsed']['visual_grounding']
                distance=max(review['grounding_min']-value,value-review['grounding_max'],0)
                assessed.append(dict(id=review['id'],score=value,distance=distance,**{k:v for k,v in review.items() if k!='id'}))
            entry['manual_diagnostics']=dict(n=len(assessed),outside_interval=sum(a['distance']>0 for a in assessed),
                severe_distance_at_least_2=sum(a['distance']>=2 for a in assessed),mean_interval_distance=mean([a['distance'] for a in assessed]),cases=assessed)
        summary[name]=entry
    comparisons={}
    multi=[k for k in result if not k.startswith('T')]
    disagreements=[]
    for a,b in itertools.combinations(multi,2):
        pairs=[(result[a][(uid,0)]['parsed'],result[b][(uid,0)]['parsed']) for uid in sorted(ids)
            if result[a].get((uid,0),{}).get('schema_valid') and result[b].get((uid,0),{}).get('schema_valid')]
        comparisons[a+' vs '+b]=dict(n=len(pairs),grounding_mae=mean([abs(x['visual_grounding']-y['visual_grounding']) for x,y in pairs]),
            grounding_exact=mean([x['visual_grounding']==y['visual_grounding'] for x,y in pairs]),caption_type_exact=mean([x['caption_type']==y['caption_type'] for x,y in pairs]))
    for r in sample:
        outputs={k:result[k][(r['id'],0)]['parsed'] for k in multi if result[k].get((r['id'],0),{}).get('schema_valid')}
        if len(outputs)<2: continue
        spread=max(o['visual_grounding'] for o in outputs.values())-min(o['visual_grounding'] for o in outputs.values())
        types=len({o['caption_type'] for o in outputs.values()})
        mismatch=len({o['image_caption_mismatch'] for o in outputs.values()})>1
        external_spread=max(o['external_context_dependency'] for o in outputs.values())-min(o['external_context_dependency'] for o in outputs.values())
        disagreements.append(dict(**r,spread=spread,type_count=types,mismatch_disagreement=mismatch,external_spread=external_spread,
            priority=3*spread+2*int(mismatch)+external_spread+types-1,outputs=outputs))
    disagreements.sort(key=lambda r:(-r['priority'],r['id']))
    mismatch_path=p.OUT/'manual_mismatch_reviews.json'
    mismatch_reviews=json.loads(mismatch_path.read_text(encoding='utf-8')) if mismatch_path.exists() else []
    mandatory={r['id'] for r in mismatch_reviews}
    disagreements=([r for r in disagreements if r['id'] in mandatory]+[r for r in disagreements if r['id'] not in mandatory])
    for name,entry in summary.items():
        if name.startswith('T'):continue
        checks=[]
        for review in mismatch_reviews:
            value=result[name].get((review['id'],0),{})
            if review['expected_mismatch'] is None or not value.get('schema_valid'):continue
            checks.append(dict(id=review['id'],false_alarm=value['parsed']['image_caption_mismatch'],note=review['note']))
        entry['manual_mismatch_diagnostics']=dict(n=len(checks),false_alarms=sum(r['false_alarm'] for r in checks),cases=checks)
    controls_summary={}
    if (p.OUT/'controls.json').exists():
        controls=json.loads((p.OUT/'controls.json').read_text(encoding='utf-8'))
        for variant in multi:
            tests=[]
            for r in controls:
                if 'control_kind' not in r or r['control_kind'] in ['base','name']: continue
                a=result[variant].get((r['base_id'],0),{}); b=result[variant].get((r['id'],0),{})
                if not a.get('schema_valid') or not b.get('schema_valid'): continue
                a=a['parsed']; b=b['parsed']; kind=r['control_kind']
                if kind in ['date','event','precise_location','later_outcome']:
                    passed=b['external_context_dependency']>a['external_context_dependency']
                    no_grounding_increase=b['visual_grounding']<=a['visual_grounding']
                elif kind=='remove_name':
                    passed=b['visual_grounding']>=a['visual_grounding']; no_grounding_increase=None
                else:
                    passed=b['image_caption_mismatch'] and not a['image_caption_mismatch']; no_grounding_increase=None
                tests.append(dict(id=r['id'],kind=kind,passed=passed,no_grounding_increase=no_grounding_increase,base=a,changed=b))
            controls_summary[variant]=dict(n=len(tests),passed=sum(t['passed'] for t in tests),
                by_kind={k:dict(n=len(v),passed=sum(x['passed'] for x in v)) for k,v in _group(tests,'kind').items()},tests=tests)
    text_comparison={}
    for t in [k for k in result if k.startswith('T')]:
        for m in multi:
            pairs=[(r['id'],result[t][(r['id'],0)]['parsed'],result[m][(r['id'],0)]['parsed']) for r in sample if result[t].get((r['id'],0),{}).get('schema_valid') and result[m].get((r['id'],0),{}).get('schema_valid')]
            # This is disagreement against another model assessment, not ground truth FP.
            candidates=[uid for uid,a,b in pairs if a['potential_external_context_dependency']>=2 and b['visual_grounding']>=3]
            text_comparison[t+' vs '+m]=dict(n=len(pairs),external_mae=mean([abs(a['potential_external_context_dependency']-b['external_context_dependency']) for _,a,b in pairs]),
                potential_overfilter_candidates=candidates,good_visual_query_despite_entity=sum(a['named_entity_dependency']!='none' and a['potentially_good_visual_query'] for _,a,b in pairs))
    p.save(p.OUT/'summary.json',dict(variants=summary,pairwise=comparisons,controls=controls_summary,text_only_comparison=text_comparison,
        caveats=['No ground truth accuracy or calibrated confidence inferred from agreement.','Random and targeted sample estimates must be reported separately.','Preassigned remainder was included in exploratory diagnostics, so it is not a blinded holdout.','Natural mismatch candidates are not known negatives; synthetic contradictions are separately marked.']))
    p.save(p.OUT/'disagreements.json',disagreements[:50])
    rng=random.Random(p.SEED+1); random30=rng.sample(sample,30)
    for r in random30:
        r['outputs']={k:result[k][(r['id'],0)]['parsed'] for k in multi if result[k].get((r['id'],0),{}).get('schema_valid')}
    p.save(p.OUT/'random30.json',random30)
    gallery(disagreements[:50],random30)
    print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk in ['valid','requested_unique','schema_valid_rate','score_distribution','logical_anomalies','stability']} for k,v in summary.items()},indent=2),flush=True)

def _group(records,key):
    d=defaultdict(list)
    for r in records:d[r[key]].append(r)
    return d

def gallery(cases,random30):
    esc=lambda x:html.escape(str(x))
    out=['<!doctype html><meta charset="utf-8"><title>VisualNews judge pilot</title><style>body{font:16px system-ui;max-width:1350px;margin:auto;padding:24px;background:#f3f4f7}article{background:white;padding:20px;margin:18px 0;border-radius:12px}img{max-width:620px;max-height:420px}pre{white-space:pre-wrap;font-size:13px}.outputs{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}h3{margin:8px 0}details{border:1px solid #ddd;padding:8px}</style><h1>VisualNews: pilot judge</h1><p>360 pairs. Diagnostic benchmark subset. Confidence is model self-report, not calibrated probability. Natural pairs are official positives; contradiction controls are separately marked.</p>']
    for title,rr in [('Disagreements / error candidates (50)',cases),('Seeded random examples (30)',random30)]:
        out.append('<h2>'+title+'</h2>')
        for r in rr:
            review_path=p.OUT/'manual_reviews.json'
            reviews={a['id']:a for a in json.loads(review_path.read_text(encoding='utf-8'))} if review_path.exists() else {}
            note=reviews.get(r['id'],{}).get('note','Candidate for human review; no independent manual verdict recorded.')
            mismatch_path=p.OUT/'manual_mismatch_reviews.json'
            mismatch_reviews={a['id']:a for a in json.loads(mismatch_path.read_text(encoding='utf-8'))} if mismatch_path.exists() else {}
            if r['id'] in mismatch_reviews: note=mismatch_reviews[r['id']]['note']
            out.append('<article id="'+esc(r['id'])+'"><h3>'+esc(r['id'])+'</h3><img loading="lazy" src="images/'+esc(Path(r['image']).name)+'"><p>'+esc(r['caption'])+'</p><p>'+esc(r.get('selection',''))+' | '+esc(', '.join(r.get('tags',[])))+'</p><p><b>Ręczny komentarz Codex — osobna ocena audytowa, nie wynik Qwena:</b> '+esc(note)+'</p><div class="outputs">')
            ordered=sorted(r['outputs'],key=lambda k:(k!='B_taxonomy_v3',k))
            for k in ordered:
                o=r['outputs'][k]
                out.append('<details'+(' open' if k=='B_taxonomy_v3' else '')+'><summary>Wynik Qwena: '+esc(k)+' | VG '+str(o['visual_grounding'])+' | confidence '+str(o['confidence'])+'</summary><pre>'+esc(json.dumps(o,ensure_ascii=False,indent=2))+'</pre></details>')
            out.append('</div></article>')
    (p.OUT/'review.html').write_text('\n'.join(out),encoding='utf-8')

def sheet():
    from PIL import Image,ImageDraw
    sample=json.loads((p.OUT/'sample.json').read_text(encoding='utf-8'))
    for batch in range(3):
        rr=sample[batch*20:(batch+1)*20]
        im=Image.new('RGB',(1500,1000),'white'); d=ImageDraw.Draw(im)
        for i,r in enumerate(rr):
            x=(i%5)*300;y=(i//5)*250
            with Image.open(p.ROOT/r['image']) as pic:
                pic.thumbnail((290,210)); im.paste(pic,(x,y+25))
            d.text((x+4,y+3),str(batch*20+i)+' '+r['id'],fill='black')
        im.save(p.OUT/f'contact_{batch}.jpg')

if __name__=='__main__':
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='sheet':sheet()
    else:summarize()
