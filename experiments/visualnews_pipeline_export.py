"""Export a consistent read-only snapshot while the pipeline continues."""
import argparse
import json
from pathlib import Path
import sqlite3

ROOT=Path(__file__).resolve().parents[1]

def export(run,kind='pairs',target=None):
    target=target or run/('final_pairs.jsonl' if kind=='pairs' else 'stage_outputs.jsonl')
    db=sqlite3.connect((run/'checkpoint.sqlite3').resolve().as_uri()+'?mode=ro',uri=True,timeout=30)
    db.row_factory=sqlite3.Row;db.execute('BEGIN')
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    tmp=target.with_suffix('.tmp');count=0
    with tmp.open('w',encoding='utf-8') as stream:
        if kind=='pairs':
            query="SELECT * FROM items WHERE final_pl IS NOT NULL AND route LIKE 'ready%' ORDER BY id"
            for row in db.execute(query):
                item=dict(row)
                stages={x['stage']:dict(output=json.loads(x['output_json']),
                    origin=json.loads(x['origin_json']) if x['origin_json'] else None)
                    for x in db.execute('SELECT * FROM outputs WHERE id=?',(item['id'],))}
                record=dict(id=item['id'],original_en=item['caption'],text_en=item['final_en'],text_pl=item['final_pl'],
                    source_image=item['image_path'],image_did=item['image_did'],route=item['route'],stages=stages,
                    pipeline_fingerprint=manifest['fingerprint'])
                stream.write(json.dumps(record,ensure_ascii=False)+'\n');count+=1
        else:
            # Includes failures and retries: downstream training can explicitly select valid=1.
            for row in db.execute('SELECT * FROM requests ORDER BY request_id'):
                record=dict(row)
                for key in ['input_json','response_json']:
                    raw=record.pop(key);record[key.removesuffix('_json')]=json.loads(raw) if raw else None
                record['pipeline_fingerprint']=manifest['fingerprint']
                stream.write(json.dumps(record,ensure_ascii=False)+'\n');count+=1
            # Cache reuse has no new API request, but its earlier model output is still training data.
            for row in db.execute('SELECT o.*,i.caption,i.final_en,i.image_path FROM outputs o JOIN items i ON i.id=o.id '
              'WHERE NOT EXISTS (SELECT 1 FROM requests r WHERE r.id=o.id AND r.stage=o.stage AND r.valid=1) ORDER BY o.id,o.stage'):
                record=dict(id=row['id'],stage=row['stage'],valid=1,reused=True,
                  input=dict(caption=row['caption'] if row['stage']=='judge' else row['final_en'],image_path=row['image_path']),
                  output=json.loads(row['output_json']),origin=json.loads(row['origin_json']) if row['origin_json'] else None,
                  pipeline_fingerprint=manifest['fingerprint'])
                stream.write(json.dumps(record,ensure_ascii=False)+'\n');count+=1
    db.rollback();db.close();tmp.replace(target)
    print(json.dumps(dict(exported=count,kind=kind,path=str(target))),flush=True)
    return count

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=ROOT/'reports/visualnews_pipeline/v1')
    parser.add_argument('--kind',choices=['pairs','stages'],default='pairs');parser.add_argument('--output',type=Path)
    args=parser.parse_args();export(args.run_dir,args.kind,args.output)
