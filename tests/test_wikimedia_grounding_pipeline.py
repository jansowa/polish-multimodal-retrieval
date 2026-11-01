"""License admission, source fallback and actual grounding acceptance gates."""
import asyncio
import io
import json
from pathlib import Path
import sqlite3
import sys

from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import wikimedia_grounding_pipeline as wiki
from pl_mmb_collection.config import HttpConfig
from pl_mmb_collection.sources.rate_limit import RateLimiter, RetryLater


def test_only_explicit_matching_open_licenses():
    for name,url,accepted in [
        ('CC BY 4.0','https://creativecommons.org/licenses/by/4.0/',True),
        ('CC BY-SA 3.0','https://creativecommons.org/licenses/by-sa/3.0/',True),
        ('CC0','https://creativecommons.org/publicdomain/zero/1.0/',True),
        ('CC BY-NC 4.0','https://creativecommons.org/licenses/by-nc/4.0/',False),
        ('CC BY-ND 4.0','https://creativecommons.org/licenses/by-nd/4.0/',False),
        ('CC BY 4.0','https://example.org/licenses/by/4.0/',False),
        ('CC BY 4.0','https://creativecommons.org/licenses/by-sa/4.0/',False),
        (None,None,False)]:
        assert wiki.open_license(dict(license=name,license_url=url)) is accepted


def test_caption_candidates_prefer_per_image_over_polish_entity_context():
    result=wiki.candidates(dict(text_candidates=[
        dict(language='pl',text='Polski pisarz',source='entity_description'),
        dict(language='en',text='A man sits beside a window.',source='structured_caption'),
        dict(language='zxx',text='Q1',source='depicts'),
        dict(language='und',text='IMG_1000',source='image_title')]))
    assert [r['language'] for r in result] == ['en','pl']


def test_download_source_hash_must_match_before_model():
    stream=io.BytesIO();Image.new('RGB',(20,20)).save(stream,'JPEG')
    with pytest.raises(ValueError,match='SHA1'):
        wiki.prepared_image(stream.getvalue(),dict(source_sha1='0',image_url='same',image_original_url='same'))


def instance(tmp_path,language='pl',stage='judge'):
    obj=wiki.WikiPipeline.__new__(wiki.WikiPipeline)
    obj.db=sqlite3.connect(':memory:');obj.db.row_factory=sqlite3.Row
    obj.db.executescript('''CREATE TABLE items(id TEXT PRIMARY KEY,caption TEXT,image_did TEXT,image_path TEXT,
      stage TEXT,status TEXT,priority INTEGER,route TEXT,final_en TEXT,final_pl TEXT,next_retry REAL,failures INTEGER,
      language TEXT,candidate_index INTEGER,accepted_text TEXT,record_json TEXT,candidates_json TEXT);
      CREATE TABLE outputs(id TEXT,stage TEXT,output_json TEXT,created_at TEXT,origin_json TEXT,PRIMARY KEY(id,stage));
      CREATE TABLE requests(request_id INTEGER PRIMARY KEY,id TEXT,stage TEXT,valid INTEGER);''')
    texts=[dict(language=language,text='Widok budynku o zmierzchu.',source='image_description'),
           dict(language='en',text='A building at dusk.',source='structured_caption')]
    source=dict(license='CC BY 4.0',license_url='https://creativecommons.org/licenses/by/4.0/')
    obj.db.execute('INSERT INTO items VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
        ('x',texts[0]['text'],'1','url',stage,'running',3,None,None,None,0,0,language,0,None,json.dumps(source),json.dumps(texts)))
    async def no_download(item): pass
    obj.ensure_image=no_download;obj.image=lambda item: {}
    return obj


def assessment(score,mismatch=False):
    return dict(visual_grounding=score,image_caption_mismatch=mismatch)


@pytest.mark.parametrize('score,mismatch,stage,route',[
    (4,False,'terminal','ready_native_original'),
    (2,False,'recycle','recycling'),
    (0,False,'judge',None),
    (4,True,'judge',None)])
def test_native_polish_routing_and_fallback(tmp_path,score,mismatch,stage,route):
    obj=instance(tmp_path)
    async def request(*args): return assessment(score,mismatch)
    obj.request=request
    asyncio.run(obj.process(None,dict(obj.db.execute('SELECT * FROM items').fetchone())))
    row=obj.db.execute('SELECT * FROM items').fetchone()
    assert row['stage']==stage and row['route']==route
    if route=='ready_native_original':
        assert row['final_pl']=='Widok budynku o zmierzchu.' and row['final_en'] is None
    if stage=='judge': assert row['candidate_index']==1 and row['final_pl'] is None
    obj.db.close()


@pytest.mark.parametrize('score,mismatch,ready',[(4,False,True),(2,False,False),(4,True,False)])
def test_recycled_polish_requires_independent_passing_recheck(tmp_path,score,mismatch,ready):
    obj=instance(tmp_path,stage='recheck')
    obj.db.execute('INSERT INTO outputs VALUES (?,?,?,?,?)',('x','recycle',json.dumps(dict(recycled_caption='Budynek o zmierzchu.')),'now',None))
    async def request(*args): return assessment(score,mismatch)
    obj.request=request
    asyncio.run(obj.process(None,dict(obj.db.execute('SELECT * FROM items').fetchone())))
    row=obj.db.execute('SELECT * FROM items').fetchone()
    assert bool(row['final_pl']) is ready
    if ready: assert row['route']=='ready_native_recycled'
    obj.db.close()


def test_independent_media_and_scraper_clients_share_cooldown(tmp_path):
    config=HttpConfig();path=tmp_path/'http.json'
    scraper=RateLimiter(config,path);media=RateLimiter(config,path)
    scraper.cooldown('https://commons.wikimedia.org/w/api.php',60,'429')
    with pytest.raises(RetryLater):
        media.reserve('https://www.wikidata.org/w/api.php')


def test_independent_clients_reload_reservations(tmp_path,monkeypatch):
    config=HttpConfig();path=tmp_path/'http.json'
    first=RateLimiter(config,path);second=RateLimiter(config,path)
    monkeypatch.setattr('pl_mmb_collection.sources.rate_limit.time.time',lambda:100)
    first.reserve('https://upload.wikimedia.org/a')
    second.config=HttpConfig(max_inline_wait=0,interval=0,requests_per_minute=600)
    with pytest.raises(RetryLater): second.reserve('https://commons.wikimedia.org/w/api.php')


def test_class_order_has_priority_over_stage_and_fallback(tmp_path):
    obj=instance(tmp_path)
    obj.db.execute('ALTER TABLE items ADD COLUMN queue_class TEXT')
    obj.db.execute('ALTER TABLE items ADD COLUMN class_priority INTEGER DEFAULT 4')
    obj.db.execute("UPDATE items SET status='pending',priority=3,queue_class='A',class_priority=0")
    obj.db.execute("INSERT INTO items(id,status,priority,queue_class,class_priority,next_retry) VALUES ('b','pending',0,'B',1,0)")
    assert obj.next_item(now=10)['id']=='x'
    obj.db.execute("UPDATE items SET status='running' WHERE id='x'")
    assert obj.next_item(now=10) is None
    obj.db.execute("UPDATE items SET status='pending',next_retry=20 WHERE id='x'")
    assert obj.next_item(now=10) is None
    obj.db.execute("UPDATE items SET status='completed' WHERE id='x'")
    assert obj.next_item(now=10)['id']=='b'
    obj.db.close()


def test_class_priority_orders_all_four_grades():
    obj=wiki.WikiPipeline.__new__(wiki.WikiPipeline)
    obj.db=sqlite3.connect(':memory:');obj.db.row_factory=sqlite3.Row
    obj.db.execute('CREATE TABLE items(id TEXT,status TEXT,priority INTEGER,class_priority INTEGER,next_retry REAL)')
    for grade in ['D','C','B','A']:
        obj.db.execute('INSERT INTO items VALUES (?,?,?,?,?)',(grade,'pending',0,wiki.CLASS_ORDER[grade],0))
    for grade in ['A','B','C','D']:
        assert obj.next_item(now=10)['id']==grade
        obj.db.execute("UPDATE items SET status='completed' WHERE id=?",(grade,))
    assert obj.next_item(now=10) is None
    obj.db.close()


def test_transient_io_and_wrapped_translation_timeouts_are_retried():
    assert wiki.retryable_error(PermissionError(13,'Permission denied'))
    assert wiki.retryable_error(wiki.v.TranslationError('InternalServerError: 504 Gateway Time-out'))
    assert wiki.retryable_error(wiki.v.TranslationError('APITimeoutError: server timed out'))
    assert not wiki.retryable_error(ValueError('Image exceeds 50 MB'))
    assert not wiki.retryable_error(wiki.v.TranslationError('Model returned invalid JSON'))


def test_prefetched_image_survives_lru_eviction_and_is_not_downloaded_twice(tmp_path):
    obj=instance(tmp_path)
    prepared={'bytes':b'prepared-jpeg'}
    obj.images={}
    assert wiki.WikiPipeline.image(obj,dict(id='x',_prepared_image=prepared)) is prepared
    async def forbidden_download(item):raise AssertionError('Prepared image must not be downloaded again')
    async def request(*args):return assessment(4)
    obj.ensure_image=forbidden_download;obj.request=request
    item=dict(obj.db.execute('SELECT * FROM items').fetchone());item['_prepared_image']=prepared
    asyncio.run(obj.process(None,item))
    assert obj.db.execute('SELECT final_pl FROM items').fetchone()[0]=='Widok budynku o zmierzchu.'
    obj.db.close()


def test_valid_archive_hit_bypasses_http_and_retains_original_fingerprint(tmp_path):
    import base64,hashlib
    from collections import OrderedDict
    obj=instance(tmp_path)
    stream=io.BytesIO();Image.new('RGB',(50,30)).save(stream,'JPEG');master=stream.getvalue()
    receipt=dict(master_sha256=hashlib.sha256(master).hexdigest(),sha256='a'*64,original_sha1='b'*40,
        pixels_sha256='c'*64,phash='d'*16,native_sha256='e'*64,processor_config_sha256='f'*64)
    class Archive:
        def begin(self,record):return dict(action='ready',master_base64=base64.b64encode(master).decode(),
            receipt=receipt,receipt_path='verified.json',metadata_path='source.json')
    class NoNetwork:
        def __getattr__(self,name):raise AssertionError('A valid archive hit must not use HTTP')
    obj.archive=Archive();obj.media=NoNetwork();obj.media_lock=asyncio.Lock()
    obj.images=OrderedDict();obj.image_bytes=0
    item=dict(obj.db.execute('SELECT * FROM items').fetchone())
    source=json.loads(item['record_json']);source['image_original_url']='https://upload.wikimedia.org/a.jpg'
    item['record_json']=json.dumps(source)
    # Use the real method, rather than the routing fixture's no-download stub.
    asyncio.run(wiki.WikiPipeline.ensure_image(obj,item))
    assert obj.images['x']['image_hash']=='a'*64
    assert obj.images['x']['archive']['model_image_source']=='verified_full_frame_master1536'
    obj.db.close()


def test_idle_worker_never_reports_running_with_only_errors():
    assert wiki.activity_status({'completed':4021,'error':4214})=='blocked_errors'
    assert wiki.activity_status({'completed':8235})=='waiting_for_source'
    assert wiki.activity_status({'error':3,'pending':1})=='running'
    assert wiki.activity_status({'running':16})=='running'
