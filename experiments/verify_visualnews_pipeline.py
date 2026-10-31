"""Verify a live read-only snapshot of routing, translation reuse and audit traces."""
import json
from pathlib import Path
import sqlite3

import visualnews_pipeline as p

def main():
    run=p.BASE/'v1'
    db=sqlite3.connect((run/'checkpoint.sqlite3').resolve().as_uri()+'?mode=ro',uri=True,timeout=30)
    db.row_factory=sqlite3.Row;db.execute('BEGIN')
    output_rows=db.execute('SELECT * FROM outputs').fetchall()
    outputs={(r['id'],r['stage']):json.loads(r['output_json']) for r in output_rows}
    ready=db.execute("SELECT * FROM items WHERE route LIKE 'ready%' AND status='completed'").fetchall()
    reused=0;new=0;recycled=0
    for item in ready:
        uid=item['id'];original=outputs[(uid,'judge')]
        assert p.judge.validate(original) and not original['image_caption_mismatch'],uid
        assert outputs[(uid,'translate')]['text']==item['final_pl'],uid
        if item['route'].endswith('recycled'):
            recycled+=1;rescue=outputs[(uid,'recycle')];recheck=outputs[(uid,'recheck')]
            assert 1<=original['visual_grounding']<=2,uid
            assert p.recycle_valid(rescue) and rescue['usable'],uid
            assert item['final_en']==rescue['recycled_caption'],uid
            assert p.judge.validate(recheck) and recheck['visual_grounding']>=3 and not recheck['image_caption_mismatch'],uid
        else:
            assert original['visual_grounding']>=3 and item['final_en']==item['caption'],uid
        assert not p.translation_issue(item['final_en'],{'text':item['final_pl']}),uid
        if item['route'].startswith('ready_reused_'):
            reused+=1
            row=next(r for r in output_rows if r['id']==uid and r['stage']=='translate')
            origin=json.loads(row['origin_json'])
            while 'previous_origin' in origin:origin=origin['previous_origin']
            matched=False
            for en,pl in p.paired_strings(origin['source_json'],origin['translated_json']):
                if en.removeprefix('Retrieve an image of this news caption. ').strip()==item['final_en'] and pl==item['final_pl']:
                    matched=True
            assert matched,uid
        else:new+=1
    def no_reasoning(value):
        if isinstance(value,dict):return not {'reasoning','reasoning_content','reasoning_details','analysis'}.intersection(value) and all(no_reasoning(v) for v in value.values())
        if isinstance(value,list):return all(no_reasoning(v) for v in value)
        return True
    requests=db.execute('SELECT * FROM requests').fetchall()
    for row in requests:
        inputs=json.loads(row['input_json'])
        assert inputs['with_image'] and row['image_sha256'] and row['request_image_sha256']
        if row['stage']=='translate':assert inputs['reasoning_enabled']
        if row['stage']=='recycle':assert inputs['initial_assessment']==outputs[(row['id'],'judge')]
        if row['response_json']:assert no_reasoning(json.loads(row['response_json']))
    count=db.execute('SELECT COUNT(*) FROM items').fetchone()[0]
    assert count==283012
    integrity=db.execute('PRAGMA quick_check').fetchone()[0];assert integrity=='ok'
    db.rollback();db.close()
    result=dict(verified_at=p.now(),items=count,ready_checked=len(ready),recycled_checked=recycled,
        reused_translation_sources_verified=reused,new_translations_checked=new,
        request_image_and_reasoning_controls_checked=len(requests),reasoning_text_absent=True,
        ready_routes_and_recheck_gates_valid=True,sqlite_quick_check=integrity)
    p.save(p.BASE/'verification.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
