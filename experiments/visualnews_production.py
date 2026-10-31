"""Resume frozen VisualNews translations in Windows, independently of WSL."""
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'src')]
import absence_news_refined as news
import httpx
from pl_mmb_translate import runner
from pl_mmb_translate.registry import DATASETS, DatasetSpec

STATE = ROOT / 'reports/visualnews_production_state.json'


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(.05 * 2 ** attempt, .5))


def counts(path):
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=30) as connection:
        return {key: connection.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0]
                for key, table in [('completed', 'completed'), ('failures', 'failures')]}


class ProductionTranslator(runner.OpenAITranslator):
    def __init__(self, settings):
        super().__init__(settings)
        self.client.timeout = httpx.Timeout(connect=75, read=settings.request_timeout, write=600, pool=75)


async def translate(settings, job, baseline, db):
    state = dict(baseline, status='translating', updated_at=utcnow())
    save(STATE, state)
    async def monitor():
        while True:
            await asyncio.sleep(30)
            current = counts(db)
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(baseline['started_at'])).total_seconds()
            state.update(current, updated_at=utcnow(), added=current['completed'] - baseline['starting_completed'],
                         texts_per_hour=(current['completed'] - baseline['starting_completed']) * 3600 / elapsed)
            save(STATE, state)
            print(json.dumps(dict(event='production_progress', **state)), flush=True)
            if shutil.disk_usage(settings.output_dir).free < 8 * 1024 ** 3:
                raise RuntimeError('Less than 8 GiB free on translation output volume')
    monitor_task = asyncio.create_task(monitor())
    try:
        # Full production passes; retry unresolved records after the rest of the source.
        for attempt in range(1, 4):
            state.update(status='translating', **{'pass': attempt})
            translation = asyncio.create_task(runner.run_dataset(job['key'], settings, with_image=False))
            done, _ = await asyncio.wait([translation, monitor_task], return_when=asyncio.FIRST_COMPLETED)
            if monitor_task in done:
                translation.cancel()
                await asyncio.gather(translation, return_exceptions=True)
                await monitor_task
            await translation
            current = counts(db)
            if current['completed'] == job['expected'] and current['failures'] == 0:
                state.update(status='complete', **current)
                break
            state.update(status='retry_wait', **current)
            save(STATE, state)
            await asyncio.sleep(60 * attempt)
        else:
            state.update(status='incomplete', **counts(db))
    except BaseException as exc:
        state.update(status='interrupted', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        monitor_task.cancel()
        await asyncio.gather(monitor_task, return_exceptions=True)
        state.update(updated_at=utcnow(), **counts(db))
        save(STATE, state)


def main():
    if (ROOT / 'reports/visualnews_image_plan.json').exists():
        raise RuntimeError('VisualNews now uses image continuation; run experiments/visualnews_images_production.py')
    if os.name != 'nt':
        raise RuntimeError('This entry point is for the existing Windows translation environment')
    import msvcrt
    os.chdir(ROOT)
    locks = []
    for filename in ['absence-full-worker.lock', 'absence-full-mbeir_visualnews.lock',
                     'visualnews-concurrency-benchmark.lock']:
        lock = (ROOT / 'reports' / filename).open('a+b')
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        locks.append(lock)
    plan = json.loads(news.PLAN.read_text(encoding='utf-8'))
    os.environ['SSL_CERT_FILE'] = str(ROOT / 'combined-ca.pem')
    os.environ['REQUESTS_CA_BUNDLE'] = str(ROOT / 'combined-ca.pem')
    output = plan['output_dir']
    if output.startswith('/mnt/'):
        output = output[5].upper() + ':/' + output[7:]
    settings = replace(news.base.suite.Settings.from_env(ROOT / '.env'), **plan['settings'],
                       output_dir=Path(output), run_name=plan['run_name'])
    job = plan['jobs'][0]
    if settings.batch_size != 16 or settings.request_timeout != 3900:
        raise RuntimeError('Expected concurrency=16 and read timeout=3900 seconds')
    if runner._translation_fingerprint(settings, with_image=False) != job['translation_config']:
        raise RuntimeError('Frozen VisualNews translation policy changed')
    if news.base.suite.digest(ROOT / job['source']) != job['source_sha256']:
        raise RuntimeError('Frozen source changed')
    if shutil.disk_usage(settings.output_dir).free < 8 * 1024 ** 3:
        raise RuntimeError('Less than 8 GiB free on translation output volume')
    DATASETS[job['key']] = DatasetSpec(job['key'], job['key'],
                                     lambda: news.base.suite.PreparedAdapter(job), job['expected'])
    runner.OpenAITranslator = ProductionTranslator
    db = settings.output_dir / settings.run_name / job['key'] / 'text_only/checkpoint.sqlite3'
    current = counts(db)
    baseline = dict(started_at=utcnow(), pid=os.getpid(), run_name=settings.run_name,
                    starting_completed=current['completed'], starting_failures=current['failures'],
                    expected=job['expected'], concurrency=16,
                    timeout_seconds=dict(connect=75, read=3900, write=600, pool=75),
                    checkpoint=str(db), source_sha256=job['source_sha256'],
                    translation_config=job['translation_config'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    save(ROOT / 'reports' / ('visualnews_resume_' + stamp + '.json'), baseline)
    save(ROOT / 'reports/visualnews_resume_baseline.json', baseline)
    with (ROOT / 'reports/visualnews_resume_history.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(baseline, ensure_ascii=False) + '\n')
    print(json.dumps(dict(event='production_resume', **baseline)), flush=True)
    try:
        asyncio.run(translate(settings, job, baseline, db))
    finally:
        for lock in locks:
            lock.close()


if __name__ == '__main__':
    main()
