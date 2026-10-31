"""Local read-only gallery of finished, open-license Wikimedia training pairs."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
from pathlib import Path
import sqlite3
import sys
from urllib.parse import parse_qs, quote, urlparse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pl_mmb_collection.config import FilterConfig
from pl_mmb_collection.pipeline.filters import license_is_allowed

WEB=Path(__file__).with_suffix('')
READY="status='completed' AND route LIKE 'ready%' AND final_pl IS NOT NULL AND length(trim(final_pl))>0 AND open_license(record_json)=1"
GRADE="json_extract(candidates_json,'$[' || candidate_index || '].quality')"
ALLOWED=FilterConfig().allowed_licenses


def safe_url(value):
    parsed=urlparse(value or '')
    return value if parsed.scheme=='https' and parsed.hostname else None


def allowed(raw):
    record=json.loads(raw)
    return int(license_is_allowed(record.get('license'),record.get('license_url'),ALLOWED))


def preview(record):
    url=record.get('image_url') or record.get('image_original_url')
    parsed=urlparse(url or '')
    if parsed.scheme!='https' or parsed.hostname!='upload.wikimedia.org':return None
    if '/thumb/' in parsed.path:return url
    prefix,_,filename=parsed.path.rpartition('/')
    if not filename or '/wikipedia/commons/' not in prefix:return url
    thumb=prefix.replace('/wikipedia/commons/','/wikipedia/commons/thumb/',1)
    return 'https://upload.wikimedia.org'+thumb+'/'+filename+'/640px-'+filename


class Store:
    def __init__(self,run):self.run=Path(run)

    def connect(self):
        db=sqlite3.connect((self.run/'checkpoint.sqlite3').resolve().as_uri()+'?mode=ro',uri=True,timeout=10)
        db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON')
        db.create_function('open_license',1,allowed,deterministic=True)
        db.create_function('casefold',1,lambda value:str(value or '').casefold(),deterministic=True)
        db.create_function('random_key',2,lambda uid,seed:hashlib.sha256((str(seed)+':'+uid).encode()).hexdigest(),deterministic=True)
        return db

    def stats(self):
        with closing(self.connect()) as db:
            db.execute('BEGIN')
            ready=db.execute('SELECT COUNT(*) FROM items WHERE '+READY).fetchone()[0]
            recycled=db.execute("SELECT COUNT(*) FROM items WHERE "+READY+" AND route LIKE '%recycled%'").fetchone()[0]
            grades=dict(db.execute('SELECT '+GRADE+',COUNT(*) FROM items WHERE '+READY+' GROUP BY 1'))
        return dict(ready=ready,recycled=recycled,original=ready-recycled,grades=grades,
            updated_at=datetime.now(timezone.utc).isoformat())

    @staticmethod
    def card(row):
        item=dict(row);source=json.loads(item.pop('record_json'))
        candidate=json.loads(item.pop('candidates_json'))[item['candidate_index']]
        item.update(source_caption=item.pop('caption'),source_language=item.pop('language'),
            grade=candidate.get('quality'),caption_source=candidate.get('source'),
            thumbnail_url=preview(source),image_url=safe_url(source.get('image_url')),
            source_page_url=safe_url(source.get('source_page_url')),license=source.get('license'),
            license_url=safe_url(source.get('license_url')),creator=source.get('creator'),
            title=source.get('entity_label_pl') or source.get('entity_label_en') or source.get('image_filename'),
            recycled='recycled' in item['route'])
        return item

    def records(self,args):
        try:page=int(args.get('page','1'));size=int(args.get('size','12'))
        except ValueError:raise ValueError('Nieprawidłowa strona')
        if page<1 or size not in {12,24,48}:raise ValueError('Nieprawidłowy rozmiar strony')
        grade=args.get('grade','');route=args.get('route','');sort=args.get('sort','newest');q=args.get('q','').strip()
        if grade not in {'','A','B','C','D'} or route not in {'','original','recycled'} or sort not in {'newest','oldest','random'} or len(q)>200:
            raise ValueError('Nieprawidłowy filtr')
        clauses=[READY];params=[]
        if grade:clauses.append(GRADE+'=?');params.append(grade)
        if route:clauses.append("route "+('LIKE' if route=='recycled' else 'NOT LIKE')+" '%recycled%'")
        if q:
            clauses.append("instr(casefold(final_pl || ' ' || caption || ' ' || record_json),?)>0")
            params.append(q.casefold())
        where=' AND '.join(clauses)
        order="(SELECT MAX(created_at) FROM outputs o WHERE o.id=items.id) "+('ASC' if sort=='oldest' else 'DESC')+',id'
        if sort=='random':order='random_key(id,?),id'
        seed=args.get('seed','42')[:100]
        with closing(self.connect()) as db:
            db.execute('BEGIN')
            total=db.execute('SELECT COUNT(*) FROM items WHERE '+where,params).fetchone()[0]
            pages=max(1,math.ceil(total/size));page=min(page,pages)
            columns='id,caption,language,final_pl,final_en,accepted_text,route,candidate_index,record_json,candidates_json,queue_class'
            rows=db.execute('SELECT '+columns+' FROM items WHERE '+where+' ORDER BY '+order+' LIMIT ? OFFSET ?',
                [*params,*([seed] if sort=='random' else []),size,(page-1)*size]).fetchall()
        return dict(items=[self.card(row) for row in rows],total=total,page=page,pages=pages,size=size)

    def record(self,uid):
        with closing(self.connect()) as db:
            db.execute('BEGIN')
            row=db.execute('SELECT * FROM items WHERE id=? AND '+READY,(uid,)).fetchone()
            if row is None:raise KeyError('Nie znaleziono gotowego przykładu')
            card=self.card(row)
            card['source_record']=json.loads(row['record_json'])
            card['stages']={r['stage']:dict(output=json.loads(r['output_json']),
                origin=json.loads(r['origin_json']) if r['origin_json'] else None)
                for r in db.execute('SELECT * FROM outputs WHERE id=?',(uid,))}
        return card


def make_server(store,host='127.0.0.1',port=8767):
    class Handler(BaseHTTPRequestHandler):
        def send(self,status,value,mime='application/json; charset=utf-8'):
            body=json.dumps(value,ensure_ascii=False).encode() if not isinstance(value,bytes) else value
            self.send_response(status)
            for key,val in {'Content-Type':mime,'Content-Length':str(len(body)),'Cache-Control':'no-store',
                    'X-Content-Type-Options':'nosniff','Referrer-Policy':'no-referrer',
                    'Content-Security-Policy':"default-src 'self'; img-src 'self' https: data:; script-src 'self'; style-src 'self'; frame-ancestors 'none'"}.items():
                self.send_header(key,val)
            self.end_headers();self.wfile.write(body)

        def do_GET(self):
            url=urlparse(self.path);args={key:values[-1] for key,values in parse_qs(url.query,keep_blank_values=True).items()}
            try:
                if url.path=='/api/stats':self.send(200,store.stats())
                elif url.path=='/api/pairs':self.send(200,store.records(args))
                elif url.path=='/api/pair':self.send(200,store.record(args.get('id','')))
                elif url.path in {'/','/index.html','/app.js','/style.css'}:
                    name='index.html' if url.path=='/' else url.path[1:]
                    mime={'index.html':'text/html','app.js':'text/javascript','style.css':'text/css'}[name]
                    self.send(200,(WEB/name).read_bytes(),mime+'; charset=utf-8')
                else:self.send(404,dict(error='Nie znaleziono strony'))
            except ValueError as exc:self.send(400,dict(error=str(exc)))
            except KeyError as exc:self.send(404,dict(error=str(exc).strip("'")))
            except sqlite3.Error:
                logging.exception('Checkpoint read failed');self.send(503,dict(error='Baza chwilowo niedostępna. Spróbuj ponownie.'))
            except (BrokenPipeError,ConnectionResetError):pass

        def log_message(self,fmt,*args):logging.info(fmt,*args)
    return ThreadingHTTPServer((host,port),Handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=ROOT/'reports/wikimedia_pipeline/v1')
    parser.add_argument('--port',type=int,default=8767)
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    server=make_server(Store(args.run_dir),port=args.port)
    directory=ROOT/'reports/wiki_training_browser';directory.mkdir(parents=True,exist_ok=True)
    import os
    (directory/'service.json').write_text(json.dumps(dict(pid=os.getpid(),url=f'http://127.0.0.1:{server.server_port}',
        run_dir=str(args.run_dir),started_at=datetime.now(timezone.utc).isoformat()),indent=2),encoding='utf-8')
    logging.info('Training gallery: http://127.0.0.1:%s',server.server_port)
    try:server.serve_forever()
    finally:server.server_close()


if __name__=='__main__':main()
