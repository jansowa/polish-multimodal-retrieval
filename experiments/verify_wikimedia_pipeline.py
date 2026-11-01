"""Audit live Wiki eligibility, ready-pair gates and saved response redaction."""
import argparse
import json
from pathlib import Path
import sqlite3

import wikimedia_grounding_pipeline as w


def verify(run):
    db=sqlite3.connect((run/'checkpoint.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row;db.execute('BEGIN')
    ready=0;checked_requests=0
    for row in db.execute('SELECT * FROM items'):
        allowed=w.open_license(json.loads(row['record_json']))
        if row['route']=='excluded_license':
            assert row['final_pl'] is None
            continue
        assert allowed, row['id']
        if row['status']!='completed' or not (row['route'] or '').startswith('ready'):continue
        ready+=1
        outputs={r['stage']:json.loads(r['output_json']) for r in db.execute('SELECT * FROM outputs WHERE id=?',(row['id'],))}
        judge=outputs['judge']
        assert not judge['image_caption_mismatch']
        if 'recycled' in row['route']:
            assert judge['visual_grounding'] in [1,2]
            assert outputs['recycle']['usable']
            check=outputs['recheck']
            assert check['visual_grounding']>=3 and not check['image_caption_mismatch']
            assert row['accepted_text']==outputs['recycle']['recycled_caption']
        else:
            assert judge['visual_grounding']>=3
            assert row['accepted_text']==row['caption']
        assert row['final_pl']
        if row['language']=='pl':
            assert row['final_en'] is None and row['final_pl']==row['accepted_text']
        else:
            assert row['final_en']==row['accepted_text']
            assert outputs['translate']['text']==row['final_pl']
    for row in db.execute('SELECT response_json,image_sha256,request_image_sha256 FROM requests'):
        assert row['image_sha256'] and row['request_image_sha256']
        if row['response_json']:
            response=json.loads(row['response_json'])
            assert w.v.without_reasoning(response)==response
        checked_requests+=1
    result=dict(checked_at=w.v.now(),ready_pairs=ready,requests=checked_requests,
        checks=['open_license','grounding_and_mismatch','independent_recheck','source_language','reasoning_redaction','image_hashes'],status='passed')
    db.rollback();db.close()
    w.v.save(run/'verification.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=w.BASE/'v1')
    verify(parser.parse_args().run_dir)
