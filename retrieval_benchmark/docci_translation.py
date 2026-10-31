"""Resumable text-only DOCCI test translation with the recycling model/API."""
import argparse
import asyncio
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import ssl
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from retrieval_benchmark.core import local_path,read,sha,write,identity
from pl_mmb_translate.prompts import SYSTEM_PROMPT,user_payload

SOURCE=local_path('C:/Users/jsowa/benchmark-translations/data-ood-20261030/B/docci/descriptions.jsonl')
PREVIOUS=local_path('C:/Users/jsowa/benchmark-translations/retrieval-benchmark-v1/docci-qwen3.8-27b-20261031')
DEST=local_path('C:/Users/jsowa/benchmark-translations/retrieval-benchmark-v1/docci-qwen3.8-27b-20261031-nothinking-c16')
RECYCLING=ROOT/'reports/visualnews_pipeline/v1/manifest.json'


def prepare():
    config=read(RECYCLING)['config']
    if config['model']!='qwen3.8-27b':raise ValueError('Unexpected recycling model')
    rows=[json.loads(line) for line in SOURCE.read_text(encoding='utf-8').splitlines() if line.strip()]
    selected=sorted((r for r in rows if r['split']=='test'),key=lambda r:r['example_id'])
    if len(selected)!=5000 or len({r['example_id'] for r in selected})!=5000:raise ValueError('Expected the official 5000-row test split')
    value=dict(status='translation_inputs_frozen_not_a_retrieval_release',dataset='DOCCI',source_sha256=sha(SOURCE),
        source_path=str(SOURCE),selected_split='test',selected_rows=5000,selected_ids=[r['example_id'] for r in selected],
        model=config['model'],base_url=config['base_url'],recycling_manifest_sha256=sha(RECYCLING),
        temperature=config['translation_temperature'],max_tokens=config['translation_max_tokens'],
        reasoning=False,reasoning_control='chat_template_kwargs.enable_thinking=false',concurrency=16,images_sent=False,new_captions_generated=False,
        reused_receipts={p.name:sha(p) for p in sorted((PREVIOUS/'translations').glob('*.json'))},
        previous_registration_sha256=sha(PREVIOUS/'registration.json'),
        prompt_sha256=identity(dict(system=SYSTEM_PROMPT,user=user_payload([dict(id='EXAMPLE',texts=dict(description='SOURCE'))],'faithful'))),
        code_sha256=sha(__file__),qa='Structural checks and numeric flags; automated QA is not human semantic verification')
    write(DEST/'registration.json',value,immutable=True)
    return value,selected


def reuse_completed(registration,rows):
    previous_registration=read(PREVIOUS/'registration.json')
    if any(previous_registration[k]!=registration[k] for k in ('source_sha256','model','prompt_sha256','temperature')):
        raise ValueError('Previous translations have different source/model/prompt')
    by_id={row['example_id']:row for row in rows}
    for name,expected in registration['reused_receipts'].items():
        path=PREVIOUS/'translations'/name
        if sha(path)!=expected:raise ValueError('Previous translation changed: '+name)
        saved=read(path);row=by_id[saved['example_id']]
        if saved['source_sha256']!=hashlib.sha256(row['description'].encode()).hexdigest():raise ValueError('Changed reused source')
        if saved['registration_sha256']!=registration['previous_registration_sha256']:raise ValueError('Changed previous registration')
        value=dict(saved,registration_sha256=sha(DEST/'registration.json'),
            reused_from_registration_sha256=saved['registration_sha256'],reused_receipt_sha256=expected,
            reasoning=True,reused_from=str(path))
        write(DEST/'translations'/name,value,immutable=True)


def request_payload(registration,messages):
    return dict(model=registration['model'],messages=messages,temperature=registration['temperature'],
                max_tokens=registration['max_tokens'],chat_template_kwargs={'enable_thinking':False})


def parse_translation(content,uid):
    content=content.strip()
    if content.startswith('```'):
        content=re.sub(r'^```(?:json)?\s*','',content);content=re.sub(r'\s*```$','',content)
    value=json.loads(content)
    if set(value)!= {'translations'} or len(value['translations'])!=1:raise ValueError('Expected exactly one translation')
    record=value['translations'][0]
    if record.get('id')!=uid or set(record.get('texts',{}))!={'description'}:raise ValueError('Changed ID or field shape')
    text=record['texts']['description']
    if not isinstance(text,str) or not text.strip():raise ValueError('Empty translation')
    return text


def numeric_flags(source,target):
    def numbers(s):return Counter(v.replace(',','.') for v in re.findall(r'\d+(?:[.,]\d+)?',s))
    return [] if numbers(source)==numbers(target) else ['Numeric tokens differ; word/digit changes can also cause this flag']


async def run(smoke=False):
    import httpx
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env',override=False)
    registration,rows=prepare()
    reuse_completed(registration,rows)
    context=ssl.create_default_context(cafile=str(ROOT/'combined-ca.pem'))
    headers={'Authorization':'Bearer '+os.environ.get('OPENAI_API_KEY','not-needed')}
    completed={p.stem:read(p) for p in (DEST/'translations').glob('*.json')}
    done=len(completed);flagged=sum(bool(v['qa_flags']) for v in completed.values());active=0
    async with httpx.AsyncClient(base_url=registration['base_url'].rstrip('/')+'/',headers=headers,
        verify=context,timeout=httpx.Timeout(180,connect=20),limits=httpx.Limits(max_connections=16,max_keepalive_connections=16)) as client:
        async def one(row):
            nonlocal active
            target=DEST/'translations'/(row['example_id']+'.json')
            source_sha=hashlib.sha256(row['description'].encode()).hexdigest()
            if target.exists():
                saved=read(target)
                if saved['source_sha256']!=source_sha or saved['registration_sha256']!=sha(DEST/'registration.json'):
                    raise ValueError('Changed translation identity')
                return saved
            user=user_payload([dict(id=row['example_id'],texts=dict(description=row['description']))],'faithful')
            messages=[dict(role='system',content=SYSTEM_PROMPT),dict(role='user',content=user)]
            last='not attempted'
            for attempt in range(4):
                try:
                    active+=1
                    try:response=await client.post('chat/completions',json=request_payload(registration,messages))
                    finally:active-=1
                    if response.status_code in (401,403,404):raise RuntimeError('Permanent API status '+str(response.status_code))
                    response.raise_for_status();raw=response.json();choice=raw['choices'][0]
                    details=(raw.get('usage') or {}).get('completion_tokens_details') or {}
                    if details.get('reasoning_tokens',0) or (choice['message'].get('reasoning_content') or '').strip():
                        raise RuntimeError('Backend generated reasoning despite enable_thinking=false')
                    if choice.get('finish_reason') not in (None,'stop'):raise ValueError('Incomplete model response: '+str(choice.get('finish_reason')))
                    translated=parse_translation(choice['message'].get('content') or '',row['example_id'])
                    if translated.strip()==row['description'].strip():raise ValueError('Untranslated paragraph')
                    saved=dict(example_id=row['example_id'],split=row['split'],image_file=row['image_file'],
                        description_en=row['description'],description_pl=translated,source_sha256=source_sha,
                        registration_sha256=sha(DEST/'registration.json'),model=registration['model'],
                        response_model=raw.get('model'),usage=raw.get('usage'),qa_flags=numeric_flags(row['description'],translated),
                        translated_at=datetime.now(timezone.utc).isoformat(),images_sent=False,caption_origin='machine_translation',reasoning=False)
                    write(DEST/'responses'/(row['example_id']+'.json'),raw,immutable=True)
                    write(target,saved,immutable=True)
                    return saved
                except RuntimeError:
                    raise
                except Exception as exc:
                    last=type(exc).__name__
                    if attempt==3:break
                    await asyncio.sleep(2**attempt)
                    if isinstance(exc,ValueError):messages=messages[:2]+[dict(role='user',content='Return the complete faithful translation as the required JSON, with unchanged ID and keys. Do not summarize. Previous output failed validation.')]
            raise RuntimeError('Translation failed after retries: '+row['example_id']+' ('+last+')')
        selected=[row for row in rows if row['example_id'] not in completed]
        if smoke:selected=selected[:3]
        pending=iter(selected)
        async def worker():
            nonlocal done,flagged
            for row in pending:
                value=await one(row);done+=1;flagged+=bool(value['qa_flags'])
                write(DEST/'status.json',dict(status='translating',completed_rows=done,total_rows=len(rows),qa_flagged_rows=flagged,
                    updated_at=datetime.now(timezone.utc).isoformat(),model=registration['model'],local_GPU_used=False,
                    reasoning=False,concurrency=16,active_requests=active))
                print('DOCCI_PROGRESS',done,len(rows),flagged,flush=True)
        await asyncio.gather(*(worker() for _ in range(min(registration['concurrency'],len(selected)))))
    if smoke:
        write(DEST/'smoke.json',dict(status='passed',rows=len(selected),completed_total=done,model=registration['model'],images_sent=False,reasoning=False,concurrency=16),immutable=True)
    else:
        output=DEST/'docci_test_pl.jsonl'
        payload='\n'.join(json.dumps(read(DEST/'translations'/(r['example_id']+'.json')),ensure_ascii=False,sort_keys=True) for r in rows)+'\n'
        if output.exists() and output.read_text(encoding='utf-8')!=payload:raise ValueError('Completed export changed')
        if not output.exists():output.write_text(payload,encoding='utf-8')
        write(DEST/'status.json',dict(status='complete_translations_only',completed_rows=done,total_rows=len(rows),
            qa_flagged_rows=flagged,model=registration['model'],export_sha256=sha(output),local_GPU_used=False,
            retrieval_ready=False,remaining='Images, leakage audit, frozen gallery/qrels and evaluation'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','smoke','run']);args=parser.parse_args()
    if args.command=='prepare':
        reg,_=prepare();print('DOCCI_REGISTERED',reg['selected_rows'],reg['model'])
    else:
        try:asyncio.run(run(args.command=='smoke'))
        except Exception as exc:
            write(DEST/'failure.json',dict(status='failed',error_type=type(exc).__name__,
                safe_reason=str(exc) if isinstance(exc,RuntimeError) else 'See local log; credentials are not recorded'))
            raise
