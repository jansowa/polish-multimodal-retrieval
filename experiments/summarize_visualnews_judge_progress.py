"""Read-only consistent snapshot; hypothetical filtering, never deletes data."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import sqlite3

ROOT=Path(__file__).resolve().parents[1]

def main():
    state=json.loads((ROOT/'reports/visualnews_judge_production_state.json').read_text(encoding='utf-8'))
    db=sqlite3.connect(Path(state['checkpoint']).resolve().as_uri()+'?mode=ro',uri=True,timeout=30)
    db.execute('BEGIN')
    records=[dict(id=uid,caption=caption,image_path=image_path,shard=shard,row_index=row_index,result=json.loads(raw))
        for uid,caption,image_path,shard,row_index,raw in db.execute(
          'SELECT r.id,i.caption,i.image_path,i.shard,i.row_index,r.result_json FROM results r JOIN items i ON i.id=r.id')]
    db.rollback();db.close()
    n=len(records)
    def count(fn):return sum(fn(r['result']) for r in records)
    grounding=Counter(r['result']['visual_grounding'] for r in records)
    groups={
        'keep_original_vg_at_least_3_no_mismatch':count(lambda o:o['visual_grounding']>=3 and not o['image_caption_mismatch']),
        'recycling_candidates_vg_1_2_no_mismatch':count(lambda o:1<=o['visual_grounding']<=2 and not o['image_caption_mismatch']),
        'vg_0_no_mismatch':count(lambda o:o['visual_grounding']==0 and not o['image_caption_mismatch']),
        'manual_review_mismatch':count(lambda o:o['image_caption_mismatch']),
    }
    policies={
        'strict_reject_original_vg_below_3_or_mismatch':count(lambda o:o['visual_grounding']<3 or o['image_caption_mismatch']),
        'moderate_reject_original_vg_below_2_or_mismatch':count(lambda o:o['visual_grounding']<2 or o['image_caption_mismatch']),
        'vg_0_only':grounding.get(0,0),
    }
    sources=Counter(r['image_path'].split('/')[2] for r in records)
    examples=[]
    for r in records:
        o=r['result']
        if len(examples)<12 and 1<=o['visual_grounding']<=2 and not o['image_caption_mismatch'] and len(r['caption'])>90:
            examples.append(r)
    result=dict(snapshot_at=datetime.now(timezone.utc).isoformat(),run_id=state['run_id'],completed=n,
        eligible_pairs=state['eligible_pairs'],total_captions=state['total_captions'],without_verified_image=state['without_verified_image'],
        progress_fraction=n/state['eligible_pairs'],grounding_counts=dict(sorted(grounding.items())),
        grounding_percent={k:100*v/n for k,v in sorted(grounding.items())},
        caption_types=dict(Counter(r['result']['caption_type'] for r in records)),
        external_context=dict(Counter(r['result']['external_context_dependency'] for r in records)),
        groups={k:dict(n=v,percent=100*v/n) for k,v in groups.items()},
        hypothetical_policies={k:dict(n=v,percent=100*v/n) for k,v in policies.items()},
        source_distribution=dict(sources),shard_distribution=dict(Counter(Path(r['shard']).name for r in records)),
        examples=examples,
        caveat='Queue ordered by image shard and row, not a random sample. Recycling eligibility is unmeasured; candidates are not confirmed salvageable. No filtering performed.')
    dest=ROOT/'reports/visualnews_judge_production/current_filtering_estimate.json'
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='examples'},ensure_ascii=False,indent=2))
    print(json.dumps(examples[:4],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
