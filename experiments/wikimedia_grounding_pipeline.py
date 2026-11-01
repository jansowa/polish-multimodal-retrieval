"""Open-license Wikimedia captions: grounding, separate recycling/recheck, optional EN->PL."""
from __future__ import annotations

import argparse
import asyncio
import base64
from collections import OrderedDict
import contextvars
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
from urllib.error import URLError
from urllib.parse import urlparse
from wiki_archive_client import WikiArchiveClient,ArchiveDeferred,ArchiveTransportError

import httpx
from dotenv import load_dotenv
from PIL import Image

import visualnews_pipeline as v
from pl_mmb_collection.config import FilterConfig, HttpConfig
from pl_mmb_collection.pipeline.filters import license_is_allowed
from pl_mmb_collection.sources.http import HttpClient
from pl_mmb_collection.sources.rate_limit import RetryLater
from pl_mmb_translate.models import ImageRef, TranslationUnit

ROOT = v.ROOT
BASE = ROOT / 'reports/wikimedia_pipeline'
STOP = ROOT / 'reports/wikimedia_pipeline.stop'
ALLOWED = tuple(FilterConfig().allowed_licenses)
CLASS_ORDER = {'A':0, 'B':1, 'C':2, 'D':3}


def retryable_error(exc):
    if isinstance(exc,(PermissionError,httpx.TransportError,RetryLater,TimeoutError,URLError,ArchiveDeferred,ArchiveTransportError)):
        return True
    if isinstance(exc,httpx.HTTPStatusError):
        return exc.response.status_code in [408,429] or exc.response.status_code>=500
    # The translation tool can wrap transport errors in TranslationError.
    return any(term in str(exc) for term in
        ['InternalServerError:', 'APIConnectionError:', 'APITimeoutError:', 'RateLimitError:', '504 Gateway Time-out'])


def activity_status(counts):
    if counts.get('running',0) or counts.get('pending',0):return 'running'
    return 'blocked_errors' if counts.get('error',0) else 'waiting_for_source'
RECYCLE_PROMPT = v.RECYCLE_PROMPT.replace('English', 'Polish or English').replace('ENGLISH', 'POLISH OR ENGLISH') + '''
The input explicitly supplies caption_language (pl or en). Write recycled_caption in exactly that language.
Do not translate. Treat names, dates, entity descriptions and categories only as unverified context.
Do not discard useful descriptions of architecture, artwork, maps, documents or legible text merely because this is not a news photo.
'''


def open_license(record):
    return license_is_allowed(record.get('license'), record.get('license_url'), list(ALLOWED))


def candidates(record):
    """Per-image descriptions first; retain context only as separately judged fallback."""
    values = list(record.get('text_candidates') or [])
    for lang in ['pl', 'en']:
        text = record.get('caption_' + lang)
        if text:
            values.append(dict(language=lang, text=text,
                source=(record.get('caption_source') or {}).get(lang),
                quality=(record.get('caption_quality') or {}).get(lang)))
    ranks = {'structured_caption': 0, 'image_description': 1, 'image_title': 2,
             'entity_description': 3, 'entity_title': 4, 'category_description': 5, 'category_title': 6}
    values.sort(key=lambda t: (ranks.get(t.get('source'), 7), t.get('language') != 'pl'))
    seen = set(); result = []
    for value in values:
        lang, text = value.get('language'), (value.get('text') or '').strip()
        if lang not in {'pl', 'en'} or len(text) < 3 or (lang, text) in seen:
            continue
        seen.add((lang, text)); result.append(dict(value, text=text))
    return result


def verify_original(raw, record):
    expected = record.get('source_sha1')
    if expected and record.get('image_url') == record.get('image_original_url'):
        actual = hashlib.sha1(raw).hexdigest()
        expected = expected.lower()
        if actual != expected and int(actual, 16) != int(expected, 36):
            raise ValueError('Source image SHA1 changed')


def prepared_image(raw, record):
    verify_original(raw, record)
    with Image.open(io.BytesIO(raw)) as image:
        image = image.convert('RGB'); image.thumbnail((1280, 1280))
        stream = io.BytesIO(); image.save(stream, 'JPEG', quality=88)
    encoded = stream.getvalue()
    return dict(bytes=encoded, uri='data:image/jpeg;base64,' + base64.b64encode(encoded).decode(),
        image_hash=hashlib.sha256(raw).hexdigest(), request_hash=hashlib.sha256(encoded).hexdigest(),
        image_path=record['image_url'])


def archived_image(result,record):
    """Use an authenticated full-frame master without pretending it is the original stream."""
    master=base64.b64decode(result['master_base64'],validate=True)
    receipt=result['receipt']
    if hashlib.sha256(master).hexdigest()!=receipt['master_sha256']:raise ValueError('Archive bridge master hash mismatch')
    with Image.open(io.BytesIO(master)) as image:
        image=image.convert('RGB');image.thumbnail((1280,1280));stream=io.BytesIO();image.save(stream,'JPEG',quality=88)
    payload=stream.getvalue()
    return dict(bytes=master,uri='data:image/jpeg;base64,'+base64.b64encode(payload).decode(),
        image_hash=receipt['sha256'],request_hash=hashlib.sha256(payload).hexdigest(),image_path=record['image_original_url'],
        archive=dict(receipt_path=result['receipt_path'],source_metadata_path=result['metadata_path'],
            original_sha1=receipt['original_sha1'],original_sha256=receipt['sha256'],pixels_sha256=receipt['pixels_sha256'],
            phash=receipt['phash'],master_sha256=receipt['master_sha256'],native_sha256=receipt['native_sha256'],
            processor_config_sha256=receipt['processor_config_sha256'],model_image_source='verified_full_frame_master1536'))


class WikiPipeline(v.Pipeline):
    def __init__(self, run=BASE/'v1', source=ROOT/'outputs/polish_culture_fast', ids=None):
        self.run = Path(run); self.run.mkdir(parents=True, exist_ok=True)
        self.source = Path(source); self.selected_ids = set(ids) if ids else None
        load_dotenv(ROOT/'.env'); os.environ['SSL_CERT_FILE'] = str(ROOT/'combined-ca.pem')
        os.environ['REQUESTS_CA_BUNDLE'] = str(ROOT/'combined-ca.pem')
        self.settings = replace(v.Settings.from_env(ROOT/'.env'), temperature=0, max_output_tokens=8192,
            quality_review=False, save_drafts=False, max_retries=0, request_timeout=600,
            image_max_side=1280, image_jpeg_quality=88, max_images_per_record=1)
        self.judge_prompt = (ROOT/'reports/visualnews_judge_pilot/B_taxonomy_v3.txt').read_text(encoding='utf-8').replace(
            'ENGLISH CAPTION', 'POLISH OR ENGLISH CAPTION') + '\nThe caption language is explicit. Assess Polish and English on the same visual rubric. Source quality grades A-D describe provenance, not grounding.\n'
        config = dict(version='wikimedia-grounding-v1', model=self.settings.model,
            source=str(self.source.resolve()), judge_prompt=self.judge_prompt, recycle_prompt=RECYCLE_PROMPT,
            judge_schema=v.judge.SCHEMA, recycle_schema=v.RECYCLE_SCHEMA,
            allowed_licenses=ALLOWED, translation_system=v.SYSTEM_PROMPT,
            translation_template=v.user_payload([{'id':'ID','texts':{'text':'CAPTION'}}], 'retrieval'),
            selected_ids=sorted(self.selected_ids) if self.selected_ids else None,
            max_candidates=8, image_max_side=1280, jpeg_quality=88,
            translation_reasoning=True, source_tool_sha256=v.text_hash((ROOT/'src/pl_mmb_translate/openai_translator.py').read_text(encoding='utf-8')))
        self.fingerprint = v.digest(config)
        manifest_path = self.run/'manifest.json'
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if self.manifest['fingerprint'] != self.fingerprint:
                raise RuntimeError('Frozen pipeline changed; use another run directory')
        else:
            self.manifest = dict(created_at=v.now(), fingerprint=self.fingerprint, config=config,
                outputs_exclude_reasoning=True, image_cache='64 MiB RAM; no original images retained on disk')
            v.save(manifest_path, self.manifest)
        self.db = sqlite3.connect(self.run/'checkpoint.sqlite3'); self.db.row_factory = sqlite3.Row
        self.db.executescript('''PRAGMA journal_mode=WAL;
          CREATE TABLE IF NOT EXISTS items(id TEXT PRIMARY KEY,caption TEXT NOT NULL,image_did TEXT,image_path TEXT,
            stage TEXT NOT NULL DEFAULT 'judge',status TEXT NOT NULL DEFAULT 'pending',priority INTEGER DEFAULT 3,
            route TEXT,final_en TEXT,final_pl TEXT,next_retry REAL DEFAULT 0,failures INTEGER DEFAULT 0,
            language TEXT NOT NULL,candidate_index INTEGER DEFAULT 0,accepted_text TEXT,
            record_json TEXT NOT NULL,candidates_json TEXT NOT NULL);
          CREATE INDEX IF NOT EXISTS item_queue ON items(status,priority,next_retry);
          CREATE INDEX IF NOT EXISTS translated_text ON items(final_en) WHERE final_pl IS NOT NULL;
          CREATE TABLE IF NOT EXISTS outputs(id TEXT,stage TEXT,output_json TEXT,created_at TEXT,origin_json TEXT,PRIMARY KEY(id,stage));
          CREATE TABLE IF NOT EXISTS requests(request_id INTEGER PRIMARY KEY,id TEXT,stage TEXT,started_at TEXT,seconds REAL,
            input_json TEXT,prompt_sha256 TEXT,response_json TEXT,error TEXT,valid INTEGER DEFAULT 0,image_sha256 TEXT,request_image_sha256 TEXT);
          CREATE INDEX IF NOT EXISTS requests_by_item ON requests(id,stage);
          CREATE TABLE IF NOT EXISTS exclusions(id TEXT PRIMARY KEY,reason TEXT,record_json TEXT,updated_at TEXT);
        ''')
        columns={row['name'] for row in self.db.execute('PRAGMA table_info(items)')}
        if 'queue_class' not in columns:
            self.db.execute('ALTER TABLE items ADD COLUMN queue_class TEXT')
            self.db.execute('ALTER TABLE items ADD COLUMN class_priority INTEGER DEFAULT 4')
        self.db.execute('CREATE INDEX IF NOT EXISTS class_queue ON items(class_priority,status,priority,next_retry,id)')
        self.db.execute("UPDATE items SET status='pending' WHERE status='running'"); self.db.commit()
        self.cache = sqlite3.connect(v.CACHE_DB.resolve().as_uri()+'?mode=ro', uri=True)
        self.context = contextvars.ContextVar('wiki_translation')
        self.translation_locks = {}
        self.translator = v.ThinkingImageTranslator(self.settings)
        create = self.translator.client.chat.completions.create
        async def audited_create(*args, **kwargs):
            context = self.context.get(); start = time.monotonic(); stamp = v.now(); body = error = None
            inputs = json.loads(json.dumps(kwargs.get('messages')))
            for message in inputs:
                if isinstance(message.get('content'), list):
                    for part in message['content']:
                        if part.get('type') == 'image_url':
                            part['image_url'] = dict(reference=context['image_path'],sha256=context['request_hash'])
            try:
                response = await create(*args, **kwargs); body = response.model_dump(); return response
            except Exception as exc:
                error = type(exc).__name__ + ': ' + str(exc); raise
            finally:
                self.audit(context['id'],'translate',stamp,start,
                    dict(caption=context['caption'],messages=inputs,with_image=True,reasoning_enabled=True,
                        generation=dict(temperature=0,max_tokens=8192)),v.text_hash(v.SYSTEM_PROMPT),body,error,context)
        self.translator.client.chat.completions.create = audited_create
        if (self.source/'manifest.json').exists():
            source_manifest = json.loads((self.source/'manifest.json').read_text(encoding='utf-8'))
        else:
            with sqlite3.connect((self.source/'checkpoint.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as metadata:
                source_manifest=json.loads(metadata.execute('SELECT json FROM manifest WHERE id=1').fetchone()[0])
        http_config = HttpConfig(**source_manifest['config']['http'])
        # HttpClient shares the scraper's durable gateway pacing and cooldown file.
        self.media = HttpClient(http_config, self.run/'media_http/raw')
        self.media_lock = asyncio.Lock()
        self.images = OrderedDict(); self.image_bytes = 0
        self.archive=WikiArchiveClient()
        self.sync_source()
        self.assign_classes()

    def assign_classes(self):
        # Keep an image's initial source class while trying fallback captions.
        with self.db:
            for row in self.db.execute('SELECT id,candidates_json FROM items WHERE queue_class IS NULL').fetchall():
                grade=json.loads(row['candidates_json'])[0].get('quality') or 'unknown'
                self.db.execute('UPDATE items SET queue_class=?,class_priority=? WHERE id=?',
                    (grade,CLASS_ORDER.get(grade,4),row['id']))

    def next_item(self, now=None):
        rank=self.db.execute("SELECT MIN(class_priority) FROM items WHERE status IN ('pending','running')").fetchone()[0]
        if rank is None:return None
        # Do not start class B while class A is running or waiting for a retry.
        return self.db.execute("SELECT * FROM items WHERE status='pending' AND class_priority=? AND next_retry<=? ORDER BY priority,id LIMIT 1",
            (rank,time.time() if now is None else now)).fetchone()

    def sync_source(self):
        db = sqlite3.connect((self.source/'checkpoint.sqlite3').resolve().as_uri()+'?mode=ro', uri=True)
        with self.db:
            for (raw,) in db.execute('SELECT json FROM records'):
                record = json.loads(raw); uid = record['id']
                if self.selected_ids and uid not in self.selected_ids:
                    continue
                # Recheck licensing on each sync, including already queued or ready records.
                if not open_license(record):
                    self.db.execute('INSERT OR REPLACE INTO exclusions VALUES (?,?,?,?)',
                        (uid,'license_not_allowed',raw,v.now()))
                    self.db.execute("UPDATE items SET stage='terminal',status='completed',route='excluded_license',final_pl=NULL WHERE id=?",(uid,))
                    continue
                texts = candidates(record)[:8]
                if not texts:
                    self.db.execute('INSERT OR REPLACE INTO exclusions VALUES (?,?,?,?)',(uid,'no_pl_en_caption',raw,v.now()))
                    continue
                self.db.execute('INSERT OR IGNORE INTO items(id,caption,image_did,image_path,language,record_json,candidates_json) VALUES (?,?,?,?,?,?,?)',
                    (uid,texts[0]['text'],record.get('source_file_id'),record['image_url'],texts[0]['language'],raw,json.dumps(texts,ensure_ascii=False)))
        db.close()
        self.assign_classes()

    def adopt(self, run):
        source=sqlite3.connect((Path(run)/'checkpoint.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
        source.row_factory=sqlite3.Row
        manifest=json.loads((Path(run)/'manifest.json').read_text(encoding='utf-8'))
        expected={k:value for k,value in self.manifest['config'].items() if k!='selected_ids'}
        actual={k:value for k,value in manifest['config'].items() if k!='selected_ids'}
        if v.digest(actual)!=v.digest(expected): raise RuntimeError('Cannot adopt a different Wiki pipeline')
        count=0
        with self.db:
            for row in source.execute("SELECT * FROM items WHERE status='completed'"):
                target=self.db.execute('SELECT * FROM items WHERE id=?',(row['id'],)).fetchone()
                if not target or target['status']=='completed' or target['candidates_json']!=row['candidates_json']:
                    continue
                if not open_license(json.loads(target['record_json'])): continue
                keys=['caption','language','candidate_index','stage','status','priority','route','accepted_text','final_en','final_pl']
                self.db.execute('UPDATE items SET '+','.join(key+'=?' for key in keys)+' WHERE id=?',
                    tuple(row[key] for key in keys)+(row['id'],))
                for output in source.execute('SELECT * FROM outputs WHERE id=?',(row['id'],)):
                    self.db.execute('INSERT OR REPLACE INTO outputs VALUES (?,?,?,?,?)',tuple(output))
                for request in source.execute('SELECT * FROM requests WHERE id=? ORDER BY request_id',(row['id'],)):
                    columns=[key for key in request.keys() if key!='request_id']
                    self.db.execute('INSERT INTO requests('+','.join(columns)+') VALUES ('+','.join('?' for _ in columns)+')',
                        tuple(request[key] for key in columns))
                count+=1
        source.close()
        return count

    async def ensure_image(self, item):
        uid = item['id']
        if uid in self.images:
            self.images.move_to_end(uid); return
        record = json.loads(item['record_json'])
        if not open_license(record):
            raise ValueError('Image license is not allowed')
        url = urlparse(record['image_original_url'])
        if url.scheme != 'https' or url.hostname != 'upload.wikimedia.org':
            raise ValueError('Unexpected image URL')
        async with self.media_lock:
            if uid in self.images:
                self.images.move_to_end(uid); return
            def download():
                result=self.archive.begin(record)
                if result['action']=='ready':return archived_image(result,record)
                token=result['token']
                try:
                    def read():
                        with self.media._request(record['image_original_url']) as response:
                            raw = response.read(50_000_001)
                        if len(raw)>50_000_000:raise ValueError('Image exceeds 50 MB')
                        return raw
                    raw=self.media._retry(read,record['image_original_url'])
                    result=self.archive.commit(token,raw)
                    self.media.limiter.success();self.media.metrics['successes']+=1
                    self.media.metrics['bytes_received']+=len(raw);self.media.save_metrics()
                    return archived_image(result,record)
                finally:self.archive.abort(token)
            image = await asyncio.to_thread(download)
            self.images[uid] = image
            self.image_bytes += len(image['uri']) + len(image['bytes'])
            while self.image_bytes > 64*1024**2 and len(self.images) > 1:
                _, old = self.images.popitem(last=False)
                self.image_bytes -= len(old['uri']) + len(old['bytes'])

    def image(self, item):
        if '_prepared_image' in item:return item['_prepared_image']
        return self.images[item['id']]

    def audit(self,uid,stage,stamp,start,input_data,prompt_hash,body,error,image):
        if image.get('archive'):input_data=dict(input_data,image_archive=image['archive'])
        return super().audit(uid,stage,stamp,start,input_data,prompt_hash,body,error,image)

    async def request(self, client, item, stage, image):
        recycle = stage == 'recycle'
        prompt = RECYCLE_PROMPT if recycle else self.judge_prompt
        schema = v.RECYCLE_SCHEMA if recycle else v.judge.SCHEMA
        inputs = dict(caption_language=item['language'],candidate_index=item['candidate_index'])
        if recycle:
            inputs.update(original_caption=item['caption'],initial_assessment=self.output(item['id'],'judge'))
        else:
            inputs['caption'] = item['caption'] if stage == 'judge' else self.output(item['id'],'recycle')['recycled_caption']
        payload = dict(model=self.settings.model,temperature=0 if recycle else .2,max_tokens=1800,
            chat_template_kwargs={'enable_thinking':False},
            response_format={'type':'json_schema','json_schema':dict(name=stage+'_output',strict=True,schema=schema)},
            messages=[dict(role='system',content=prompt),dict(role='user',content=[
                dict(type='image_url',image_url={'url':image['uri']}),dict(type='text',text=json.dumps(inputs,ensure_ascii=False))])])
        start=time.monotonic(); stamp=v.now(); body=error=None
        try:
            response=await client.post(self.settings.base_url+'/chat/completions',json=payload,headers={'Authorization':'Bearer '+self.settings.api_key})
            try: body=response.json()
            except ValueError: body=dict(http_status=response.status_code,body=response.text)
            response.raise_for_status()
            if body['choices'][0].get('finish_reason') != 'stop':
                raise ValueError('Incomplete model output')
            obj=json.loads(body['choices'][0]['message']['content'])
            if not (v.recycle_valid(obj) if recycle else v.judge.validate(obj)):
                raise ValueError('Invalid model output schema')
            return obj
        except Exception as exc:
            error=type(exc).__name__+': '+str(exc); raise
        finally:
            self.audit(item['id'],stage,stamp,start,dict(inputs,with_image=True,image_path=image['image_path'],
                generation=dict(temperature=0 if recycle else .2,max_tokens=1800,thinking=False)),v.text_hash(prompt),body,error,image)

    def fallback(self, item, route):
        texts=json.loads(item['candidates_json']); index=item['candidate_index']+1
        if index < len(texts):
            candidate=texts[index]
            self.db.execute('DELETE FROM outputs WHERE id=?',(item['id'],))
            self.db.execute("UPDATE items SET caption=?,language=?,candidate_index=?,stage='judge',status='pending',priority=3,route=NULL,accepted_text=NULL,final_en=NULL,final_pl=NULL,next_retry=0,failures=0 WHERE id=?",
                (candidate['text'],candidate['language'],index,item['id']))
        else:
            self.set_stage(item['id'],'terminal',route)

    def accept(self, item, text, route):
        self.db.execute('UPDATE items SET accepted_text=? WHERE id=?',(text,item['id']))
        if item['language'] == 'pl':
            self.db.execute('UPDATE items SET final_pl=? WHERE id=?',(text,item['id']))
            self.set_stage(item['id'],'terminal','ready_native_'+route)
        else:
            self.set_stage(item['id'],'translate',route,text)

    async def process(self, client, item):
        if not open_license(json.loads(item['record_json'])):
            self.set_stage(item['id'],'terminal','excluded_license'); return
        if '_prepared_image' not in item:await self.ensure_image(item)
        stage=item['stage']; uid=item['id']
        if stage == 'translate':
            return await super().process(client,item)
        obj=await self.request(client,item,stage,self.image(item))
        self.db.execute('INSERT OR REPLACE INTO outputs VALUES (?,?,?,?,?)',(uid,stage,json.dumps(obj,ensure_ascii=False),v.now(),None))
        self.db.execute('UPDATE requests SET valid=1 WHERE request_id=(SELECT MAX(request_id) FROM requests WHERE id=? AND stage=?)',(uid,stage))
        if stage == 'judge':
            if obj['image_caption_mismatch']:
                self.fallback(item,'review_mismatch')
            elif obj['visual_grounding'] == 0:
                self.fallback(item,'rejected_grounding')
            elif obj['visual_grounding'] <= 2:
                self.set_stage(uid,'recycle','recycling')
            else:
                self.accept(item,item['caption'],'original')
        elif stage == 'recycle':
            if obj['usable']:
                self.set_stage(uid,'recheck','recycled')
            else:
                self.fallback(item,'rejected_no_visual_core')
        elif obj['visual_grounding'] >= 3 and not obj['image_caption_mismatch']:
            self.accept(item,self.output(uid,'recycle')['recycled_caption'],'recycled')
        else:
            self.fallback(item,'review_recycling_failed')

    def export(self):
        target=self.run/'final_pairs.jsonl'; tmp=target.with_suffix('.tmp')
        with tmp.open('w',encoding='utf-8') as stream:
            for row in self.db.execute("SELECT * FROM items WHERE status='completed' AND route LIKE 'ready%' AND final_pl IS NOT NULL ORDER BY id"):
                item=dict(row); record=json.loads(item['record_json'])
                if not open_license(record): continue
                result=dict(id=item['id'],text_pl=item['final_pl'],text_en=item['final_en'],
                    accepted_text=item['accepted_text'],source_caption=item['caption'],source_language=item['language'],
                    selected_candidate=json.loads(item['candidates_json'])[item['candidate_index']],
                    route=item['route'],source_record=record,pipeline_fingerprint=self.fingerprint,
                    stages={r['stage']:json.loads(r['output_json']) for r in self.db.execute('SELECT * FROM outputs WHERE id=?',(item['id'],))})
                stream.write(json.dumps(result,ensure_ascii=False)+'\n')
        tmp.replace(target)

    def state(self, status, concurrency):
        if getattr(self,'shared_slots',None):concurrency=self.shared_slots.limit('wikimedia')
        routes=dict(self.db.execute("SELECT route,COUNT(*) FROM items WHERE status='completed' GROUP BY route"))
        if status=='running':status=activity_status(dict(self.db.execute('SELECT status,COUNT(*) FROM items GROUP BY status')))
        data=dict(status=status,pid=os.getpid(),updated_at=v.now(),concurrency=concurrency,
            source=str(self.source),run_dir=str(self.run),fingerprint=self.fingerprint,
            total=self.db.execute('SELECT COUNT(*) FROM items').fetchone()[0],routes=routes,
            ready=sum(n for route,n in routes.items() if route and route.startswith('ready')),
            requests=self.db.execute('SELECT COUNT(*) FROM requests').fetchone()[0],
            latest_request_at=self.db.execute('SELECT MAX(started_at) FROM requests').fetchone()[0],
            exclusions=dict(self.db.execute('SELECT reason,COUNT(*) FROM exclusions GROUP BY reason')),
            stages=[dict(r) for r in self.db.execute('SELECT stage,status,COUNT(*) AS n FROM items GROUP BY stage,status')],
            queue_order='A -> B -> C -> D',
            classes=[dict(r) for r in self.db.execute('SELECT queue_class,status,COUNT(*) AS n FROM items GROUP BY queue_class,status')],
            disk_free_bytes=shutil.disk_usage(self.run).free)
        v.save(self.run/'state.json',data)
        if self.run == BASE/'v1': v.save(ROOT/'reports/wikimedia_pipeline_state.json',data)
        print(json.dumps(data),flush=True)

    async def work(self, concurrency=32, once=False):
        self.state('running',concurrency)
        async with httpx.AsyncClient(verify=str(ROOT/'combined-ca.pem'),timeout=httpx.Timeout(600,connect=45),
                limits=httpx.Limits(max_connections=concurrency)) as client:
            async def worker():
                while not STOP.exists():
                    if shutil.disk_usage(self.run).free < 4*1024**3:
                        raise RuntimeError('Disk reserve below 4 GiB')
                    row=self.next_item()
                    if row is None:
                        if once and not self.db.execute("SELECT COUNT(*) FROM items WHERE status IN ('pending','running')").fetchone()[0]: return
                        await asyncio.sleep(1); continue
                    item=dict(row)
                    self.db.execute("UPDATE items SET status='running' WHERE id=?",(item['id'],)); self.db.commit()
                    try:
                        if item['stage']=='translate':
                            async with self.translation_locks.setdefault(item['final_en'],asyncio.Lock()):
                                await self.process(client,item)
                        else:
                            await self.process(client,item)
                        # A source refresh can revoke a license during an in-flight request.
                        revoked=self.db.execute("SELECT 1 FROM exclusions WHERE id=? AND reason='license_not_allowed'",(item['id'],)).fetchone()
                        if revoked:
                            self.db.execute("UPDATE items SET stage='terminal',status='completed',route='excluded_license',final_pl=NULL WHERE id=?",(item['id'],))
                        self.db.commit()
                    except asyncio.CancelledError: raise
                    except Exception as exc:
                        self.db.rollback(); failures=item['failures']+1
                        transient=retryable_error(exc)
                        retry_at=getattr(exc,'retry_at',time.time()+min(300,10*2**min(failures,5)))
                        self.db.execute('UPDATE items SET status=?,next_retry=?,failures=? WHERE id=?',
                            ('pending' if transient or failures<3 else 'error',retry_at,failures,item['id']));self.db.commit()
                        with (self.run/'errors.jsonl').open('a',encoding='utf-8') as stream:
                            stream.write(json.dumps(dict(id=item['id'],stage=item['stage'],error=str(exc),error_type=type(exc).__name__,transient=transient,at=v.now()))+'\n')
            group=asyncio.gather(*(worker() for _ in range(concurrency)))
            async def monitor():
                tick=0
                while True:
                    await asyncio.sleep(30); tick+=1
                    if not self.selected_ids and tick%4==0: self.sync_source()
                    self.state('running',concurrency)
                    if tick%2==0: self.export()
            monitor_task=asyncio.create_task(monitor())
            try:
                done,_=await asyncio.wait([group,monitor_task],return_when=asyncio.FIRST_COMPLETED)
                if monitor_task in done: await monitor_task
                await group
                self.state('stopped_by_request' if STOP.exists() else 'complete',concurrency)
            except BaseException:
                self.state('interrupted',concurrency); raise
            finally:
                group.cancel();monitor_task.cancel()
                await asyncio.gather(group,monitor_task,return_exceptions=True)
                self.export()

    async def close(self):
        await self.translator.close(); self.archive.close(); self.media.close(); self.cache.close(); self.db.close()


async def main(args):
    pipeline=WikiPipeline(args.run_dir,args.source,args.ids)
    try:
        if args.adopt_run: print('ADOPTED '+str(pipeline.adopt(args.adopt_run)),flush=True)
        if args.prepare: pipeline.state('prepared',args.concurrency)
        else: await pipeline.work(args.concurrency,args.once)
    finally: await pipeline.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--concurrency',type=int,choices=range(1,33))
    parser.add_argument('--run-dir',type=Path,default=BASE/'v1')
    parser.add_argument('--source',type=Path)
    parser.add_argument('--ids',nargs='+');parser.add_argument('--once',action='store_true');parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--adopt-run',type=Path)
    args=parser.parse_args()
    runtime_path=BASE/'runtime_config.json'
    runtime=json.loads(runtime_path.read_text(encoding='utf-8')) if runtime_path.exists() else {}
    if args.source is None:args.source=Path(runtime.get('source',str(ROOT/'outputs/polish_culture_fast')))
    if args.concurrency is None:
        args.concurrency=runtime.get('concurrency',32)
        if not 1<=args.concurrency<=32:parser.error('concurrency must be between 1 and 32')
    args.run_dir.mkdir(parents=True,exist_ok=True)
    import msvcrt
    with (args.run_dir/'worker.lock').open('a+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        asyncio.run(main(args))
