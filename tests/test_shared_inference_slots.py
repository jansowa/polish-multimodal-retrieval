import asyncio
import json
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from inference_slots import SharedSlots
from migrate_wikimedia_source import migrate
import visualnews_pipeline as v


def test_equal_shares_then_full_borrowing_then_return_to_equal_shares():
    async def exercise():
        demands={'visualnews':True,'wikimedia':True}
        pool=SharedSlots(24,{name:lambda name=name:demands[name] for name in demands})
        leases=[]
        for name in demands:
            for _ in range(12):
                lease=pool.slot(name);await lease.__aenter__();leases.append((name,lease))
        assert pool.active=={'visualnews':12,'wikimedia':12}
        waiting=asyncio.create_task(pool.slot('visualnews').__aenter__())
        await asyncio.sleep(.01);assert not waiting.done();waiting.cancel()
        await asyncio.gather(waiting,return_exceptions=True)
        for name,lease in leases:
            if name=='wikimedia':await lease.__aexit__(None,None,None)
        demands['wikimedia']=False
        extra=[]
        for _ in range(12):
            lease=pool.slot('visualnews');await lease.__aenter__();extra.append(lease)
        assert pool.active=={'visualnews':24,'wikimedia':0}
        demands['wikimedia']=True
        wiki_lease=pool.slot('wikimedia');waiting=asyncio.create_task(wiki_lease.__aenter__())
        await asyncio.sleep(.01);assert not waiting.done()
        await extra.pop().__aexit__(None,None,None)
        await asyncio.wait_for(waiting,1)
        assert sum(pool.active.values())==24
        assert pool.limit('visualnews')==12
        await wiki_lease.__aexit__(None,None,None)
        for lease in extra:await lease.__aexit__(None,None,None)
        for name,lease in leases:
            if name=='visualnews':await lease.__aexit__(None,None,None)
        assert sum(pool.active.values())==0
    asyncio.run(exercise())


def test_slots_release_on_model_error():
    async def exercise():
        pool=SharedSlots(24,{'visualnews':lambda:True,'wikimedia':lambda:True})
        with pytest.raises(ValueError):
            async with pool.slot('wikimedia'):raise ValueError('model failed')
        assert sum(pool.active.values())==0
    asyncio.run(exercise())


def test_download_waits_do_not_reserve_inference_capacity():
    async def exercise():
        pool=SharedSlots(30,{'visualnews':lambda:True,'wikimedia':lambda:True},
            shares={'visualnews':6,'wikimedia':24},work_conserving=True)
        leases=[]
        for _ in range(30):
            lease=pool.slot('visualnews');await lease.__aenter__();leases.append(lease)
        assert pool.active=={'visualnews':30,'wikimedia':0}
        wiki=pool.slot('wikimedia');waiting=asyncio.create_task(wiki.__aenter__())
        await asyncio.sleep(.01)
        assert pool.limit('visualnews')==6
        await leases.pop().__aexit__(None,None,None)
        await asyncio.wait_for(waiting,1)
        assert pool.active=={'visualnews':29,'wikimedia':1}
        assert sum(pool.active.values())==30
        await wiki.__aexit__(None,None,None)
        for lease in leases:await lease.__aexit__(None,None,None)
        assert pool.waiting=={'visualnews':0,'wikimedia':0}
    asyncio.run(exercise())


def test_wiki_priority_shares_and_borrowing():
    async def exercise():
        demand={'visualnews':True,'wikimedia':True}
        pool=SharedSlots(32,{name:lambda name=name:demand[name] for name in demand},shares={'visualnews':12,'wikimedia':20})
        assert pool.limit('visualnews')==12 and pool.limit('wikimedia')==20
        leases=[]
        for name,count in [('visualnews',12),('wikimedia',20)]:
            for _ in range(count):
                lease=pool.slot(name);await lease.__aenter__();leases.append(lease)
        assert sum(pool.active.values())==32
        for lease in leases:await lease.__aexit__(None,None,None)
        demand['wikimedia']=False
        assert pool.limit('visualnews')==32
        with pytest.raises(ValueError):SharedSlots(32,pool.demands,shares={'visualnews':16,'wikimedia':20})
    asyncio.run(exercise())


def test_32_slots_split_16_each_and_borrow_32():
    async def exercise():
        demand={'visualnews':True,'wikimedia':True}
        pool=SharedSlots(32,{name:lambda name=name:demand[name] for name in demand})
        assert pool.limit('visualnews')==pool.limit('wikimedia')==16
        leases=[]
        for name in demand:
            for _ in range(16):
                lease=pool.slot(name);await lease.__aenter__();leases.append(lease)
        assert sum(pool.active.values())==32
        assert pool.active=={'visualnews':16,'wikimedia':16}
        for lease in leases:await lease.__aexit__(None,None,None)
        demand['wikimedia']=False
        assert pool.limit('visualnews')==32
    asyncio.run(exercise())


def migration_fixture(tmp_path,changed=False):
    run=tmp_path/'run';source=tmp_path/'expanded';run.mkdir();source.mkdir()
    before=dict(source_file_id='1',image_original_url='https://example.org/1.jpg',source_sha1='abc')
    db=sqlite3.connect(run/'checkpoint.sqlite3')
    db.execute('CREATE TABLE items(id TEXT,record_json TEXT,status TEXT,final_pl TEXT)')
    db.execute('INSERT INTO items VALUES (?,?,?,?)',('id',json.dumps(before),'completed','Gotowy podpis.'));db.commit();db.close()
    after=dict(before,source_sha1='changed' if changed else 'abc')
    db=sqlite3.connect(source/'checkpoint.sqlite3');db.execute('CREATE TABLE records(id TEXT PRIMARY KEY,json TEXT)')
    db.execute('INSERT INTO records VALUES (?,?)',('id',json.dumps(after)));db.commit();db.close()
    config=dict(source='old-source',judge_prompt='frozen')
    manifest=dict(config=config,fingerprint=v.digest(config))
    (run/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    return run,source


def test_source_expansion_preserves_results_and_backup(tmp_path):
    run,source=migration_fixture(tmp_path)
    result=migrate(run,source)
    assert result['verified_existing_images']==1
    assert (run/'checkpoint.before_source_expansion.sqlite3').exists()
    assert json.loads((run/'manifest.json').read_text())['config']['judge_prompt']=='frozen'
    db=sqlite3.connect(run/'checkpoint.sqlite3')
    assert db.execute('SELECT final_pl FROM items').fetchone()[0]=='Gotowy podpis.';db.close()
    assert migrate(run,source)['status']=='already_migrated'


def test_changed_image_blocks_source_migration_before_any_writes(tmp_path):
    run,source=migration_fixture(tmp_path,changed=True)
    before=(run/'manifest.json').read_bytes()
    with pytest.raises(RuntimeError,match='changed image'):migrate(run,source)
    assert (run/'manifest.json').read_bytes()==before
    assert not (run/'checkpoint.before_source_expansion.sqlite3').exists()
