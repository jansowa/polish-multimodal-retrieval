"""The gallery must expose only ready open-license pairs and their final captions."""
import json
from pathlib import Path
import sqlite3
import sys
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from wiki_training_browser import Store,make_server


@pytest.fixture
def store(tmp_path):
    db=sqlite3.connect(tmp_path/'checkpoint.sqlite3')
    db.executescript('''CREATE TABLE items(id TEXT PRIMARY KEY,caption TEXT,language TEXT,final_pl TEXT,final_en TEXT,
      accepted_text TEXT,route TEXT,candidate_index INTEGER,record_json TEXT,candidates_json TEXT,queue_class TEXT,status TEXT);
      CREATE TABLE outputs(id TEXT,stage TEXT,output_json TEXT,created_at TEXT,origin_json TEXT);''')
    for index in range(18):
        grade='A' if index%2==0 else 'B'
        record=dict(image_url='https://upload.wikimedia.org/wikipedia/commons/a/ab/Example.jpg',
            image_original_url='https://upload.wikimedia.org/wikipedia/commons/a/ab/Example.jpg',
            license='CC BY 4.0',license_url='https://creativecommons.org/licenses/by/4.0/',
            entity_label_pl='Łódź',creator='Autor',source_page_url='https://commons.wikimedia.org/wiki/File:Example.jpg')
        status='completed';text=f'Gotowy polski podpis: Łódź {index}'
        if index==15:status='pending'
        if index==16:text=None
        if index==17:record['license']='CC BY-NC 4.0'
        db.execute('INSERT INTO items VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
            (str(index),'Original source caption','en',text,'Final English caption','Final English caption',
             'ready_recycled' if index%2 else 'ready_original',0,json.dumps(record),
             json.dumps([dict(quality=grade,source='structured_caption')]),grade,status))
        db.execute('INSERT INTO outputs VALUES (?,?,?,?,?)',(str(index),'judge',json.dumps(dict(visual_grounding=4)),f'2026-10-21T12:00:{index:02d}',None))
    db.commit();db.close()
    return Store(tmp_path)


def test_only_finished_open_license_pairs_are_visible(store):
    stats=store.stats()
    assert stats['ready']==15 and stats['original']==8 and stats['recycled']==7
    rows=store.records({})
    assert rows['total']==15 and len(rows['items'])==12 and rows['pages']==2
    assert all(r['final_pl'].startswith('Gotowy polski') for r in rows['items'])
    for uid in ['15','16','17']:
        with pytest.raises(KeyError):store.record(uid)


def test_source_grade_route_unicode_and_pagination(store):
    assert store.records(dict(grade='A'))['total']==8
    assert store.records(dict(route='recycled'))['total']==7
    assert store.records(dict(q='ŁÓDŹ'))['total']==15
    first=store.records({});second=store.records(dict(page='2'))
    assert not {r['id'] for r in first['items']} & {r['id'] for r in second['items']}
    assert store.records(dict(q="' OR 1=1 --"))['total']==0
    with pytest.raises(ValueError):store.records(dict(grade='E'))


def test_details_and_random_order_are_read_only(store):
    checkpoint=store.run/'checkpoint.sqlite3';before=checkpoint.read_bytes()
    row=store.record('0')
    assert row['source_caption']=='Original source caption'
    assert row['final_en']=='Final English caption'
    assert row['source_record']['creator']=='Autor'
    assert row['stages']['judge']['output']['visual_grounding']==4
    assert '/thumb/a/ab/Example.jpg/640px-Example.jpg' in row['thumbnail_url']
    args=dict(sort='random',seed='test')
    assert store.records(args)==store.records(args)
    with store.connect() as db:
        with pytest.raises(sqlite3.OperationalError):db.execute('DELETE FROM items')
    assert checkpoint.read_bytes()==before


def test_http_gallery_api_and_unknown_paths(store):
    server=make_server(store,port=0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(base+'/') as response:
            assert 'Gotowe do treningu' in response.read().decode()
            assert "frame-ancestors 'none'" in response.headers['Content-Security-Policy']
        with urlopen(base+'/api/pairs?grade=A') as response:assert json.load(response)['total']==8
        with pytest.raises(HTTPError) as error:urlopen(base+'/api/pair?id=17')
        assert error.value.code==404
        with pytest.raises(HTTPError) as error:urlopen(base+'/../.env')
        assert error.value.code==404
    finally:server.shutdown();server.server_close();thread.join()
