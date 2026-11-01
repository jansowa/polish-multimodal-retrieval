"""Change only Wiki source provenance after verifying all previous image identities."""
import json
from pathlib import Path
import sqlite3

import visualnews_pipeline as v


def migrate(run,source):
    run=Path(run);source=Path(source).resolve()
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    if manifest['config']['source']==str(source):return dict(status='already_migrated')
    old=sqlite3.connect((run/'checkpoint.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
    new=sqlite3.connect((source/'checkpoint.sqlite3').as_uri()+'?mode=ro',uri=True)
    checked=0
    try:
        for uid,raw in old.execute('SELECT id,record_json FROM items'):
            row=new.execute('SELECT json FROM records WHERE id=?',(uid,)).fetchone()
            if not row:raise RuntimeError('Expanded source lacks previous image '+uid)
            before=json.loads(raw);after=json.loads(row[0])
            for key in ['source_file_id','image_original_url','source_sha1']:
                if before.get(key)!=after.get(key):raise RuntimeError('Expanded source changed image '+uid+': '+key)
            checked+=1
        backup=run/'checkpoint.before_source_expansion.sqlite3'
        if backup.exists():raise RuntimeError('Migration backup already exists; inspect before migrating again')
        target=sqlite3.connect(backup)
        try:old.backup(target)
        finally:target.close()
    finally:old.close();new.close()
    v.save(run/'manifest.before_source_expansion.json',manifest)
    previous=manifest['fingerprint']
    event=dict(migrated_at=v.now(),previous_source=manifest['config']['source'],source=str(source),
        previous_fingerprint=previous,verified_existing_images=checked,backup=str(backup))
    manifest['config']['source']=str(source)
    manifest['fingerprint']=v.digest(manifest['config'])
    manifest.setdefault('source_history',[]).append(event)
    v.save(run/'manifest.json',manifest)
    v.save(run/'source_expansion.json',event)
    return dict(status='migrated',**event)
