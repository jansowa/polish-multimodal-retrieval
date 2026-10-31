"""Routing, exact-text reuse, recycling gates and removal of reasoning traces."""
import asyncio
from pathlib import Path
import sqlite3
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import visualnews_pipeline as pipeline

def assessment(score,mismatch=False):
    return dict(visual_grounding=score,external_context_dependency=1,named_entity_dependency='incidental',
      event_or_temporal_context=False,location_context_not_visually_verifiable=False,ocr_dependency=False,
      image_caption_mismatch=mismatch,caption_type='visual_description_plus_context',confidence=.9,
      evidence='Visible activity.',unsupported_information=['identity'])

class PipelineTests(unittest.TestCase):
    def test_routing_all_scores_and_mismatch(self):
        expected={0:'terminal',1:'recycle',2:'recycle',3:'translate',4:'translate'}
        for score,stage in expected.items():
            self.assertEqual(pipeline.judge_route(assessment(score))[0],stage)
            self.assertEqual(pipeline.judge_route(assessment(score,True))[:2],('terminal','review_mismatch'))

    def instance(self,stage='translate',final_en='A dog runs.'):
        obj=pipeline.Pipeline.__new__(pipeline.Pipeline)
        obj.run=Path('test-run')
        obj.db=sqlite3.connect(':memory:')
        obj.db.executescript('''CREATE TABLE items(id TEXT PRIMARY KEY,stage TEXT,status TEXT,priority INTEGER,
          route TEXT,final_en TEXT,final_pl TEXT,next_retry REAL,failures INTEGER);
          CREATE TABLE outputs(id TEXT,stage TEXT,output_json TEXT,created_at TEXT,origin_json TEXT,PRIMARY KEY(id,stage));
          CREATE TABLE requests(request_id INTEGER PRIMARY KEY,id TEXT,stage TEXT,valid INTEGER);''')
        obj.db.execute('INSERT INTO items VALUES (?,?,?,?,?,?,?,?,?)',('x',stage,'running',0,'original',final_en,None,0,0))
        obj.cache=sqlite3.connect(':memory:')
        obj.cache.execute('CREATE TABLE translations(en TEXT PRIMARY KEY,pl TEXT,origin_json TEXT)')
        obj.cache.execute('INSERT INTO translations VALUES (?,?,?)',('A dog runs.','Pies biegnie.','{"checkpoint":"old"}'))
        self.addCleanup(obj.db.close);self.addCleanup(obj.cache.close)
        return obj

    def test_exact_english_reuses_translation_without_image_or_api(self):
        obj=self.instance()
        obj.image=lambda item: (_ for _ in ()).throw(AssertionError('Image/API should not be used'))
        asyncio.run(obj.process(None,dict(id='x',stage='translate',final_en='A dog runs.',route='original')))
        self.assertEqual(obj.db.execute('SELECT final_pl,route,status FROM items').fetchone(),
          ('Pies biegnie.','ready_reused_original','completed'))

    def test_recycled_or_different_text_cannot_reuse_original_translation(self):
        obj=self.instance(final_en='A dog sits.')
        obj.image=lambda item: (_ for _ in ()).throw(LookupError('New translation needed'))
        with self.assertRaisesRegex(LookupError,'New translation needed'):
            asyncio.run(obj.process(None,dict(id='x',stage='translate',final_en='A dog sits.',route='recycled')))
        self.assertIsNone(obj.db.execute('SELECT final_pl FROM items').fetchone()[0])

    def test_reuses_an_identical_text_completed_during_the_current_run(self):
        obj=self.instance(final_en='A dog sits.')
        obj.db.execute('INSERT INTO items VALUES (?,?,?,?,?,?,?,?,?)',
          ('previous','terminal','completed',9,'ready_recycled','A dog sits.','Pies siedzi.',0,0))
        obj.image=lambda item: (_ for _ in ()).throw(AssertionError('Current-run result should be reused'))
        asyncio.run(obj.process(None,dict(id='x',stage='translate',final_en='A dog sits.',route='recycled')))
        self.assertEqual(obj.db.execute('SELECT final_pl,route FROM items WHERE id=?',('x',)).fetchone(),
          ('Pies siedzi.','ready_reused_recycled'))

    def test_recycled_text_must_pass_recheck_before_translation(self):
        for score,mismatch,expected in [(2,False,'terminal'),(3,True,'terminal'),(3,False,'translate')]:
            obj=self.instance(stage='recheck',final_en=None)
            obj.image=lambda item:{}
            obj.output=lambda uid,stage:{'recycled_caption':'A dog runs.'}
            async def request(*args,answer=assessment(score,mismatch)):return answer
            obj.request=request
            asyncio.run(obj.process(None,dict(id='x',stage='recheck',route='recycled')))
            stage,en=obj.db.execute('SELECT stage,final_en FROM items').fetchone()
            self.assertEqual(stage,expected)
            self.assertEqual(en,'A dog runs.' if expected=='translate' else None)

    def test_reasoning_content_removed_but_final_output_and_usage_retained(self):
        response={'choices':[{'message':{'content':'{"text":"Pies"}','reasoning':'private','reasoning_content':'private'}}],
          'usage':{'completion_tokens_details':{'reasoning_tokens':80}}}
        cleaned=pipeline.without_reasoning(response)
        self.assertEqual(cleaned['choices'][0]['message'],{'content':'{"text":"Pies"}'})
        self.assertEqual(cleaned['usage']['completion_tokens_details']['reasoning_tokens'],80)
        self.assertEqual(pipeline.without_reasoning('<think>trace</think>{"text":"Pies"}'),'{"text":"Pies"}')

    def test_unusable_recycling_cannot_hide_a_generated_caption(self):
        obj=dict(usable=False,recycled_caption='A person.',retained_visual_information=[],removed_information=[],reason='No core')
        self.assertFalse(pipeline.recycle_valid(obj));obj['recycled_caption']=None
        self.assertTrue(pipeline.recycle_valid(obj))

    def test_detects_known_semantic_mistranslation_without_blocking_correct_polish(self):
        self.assertIsNotNone(pipeline.translation_issue('Police wear riot gear.',{'text':'Policja ma sprzęt antyradziecki.'}))
        self.assertIsNone(pipeline.translation_issue('Police wear riot gear.',{'text':'Policja ma wyposażenie ochronne.'}))
        self.assertIsNone(pipeline.translation_issue('An anti-Soviet poster.',{'text':'Plakat antyradziecki.'}))

if __name__=='__main__':unittest.main()
