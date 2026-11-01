"""Timed production-policy concurrency comparison; launched explicitly in WSL."""
import asyncio
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import time

if os.name == 'nt':
    import msvcrt
else:
    import fcntl

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'src')]
# Imports the exact frozen VisualNews profile used by the production wrapper.
import absence_news_refined as news
from pl_mmb_translate.checkpoint import CheckpointStore
from pl_mmb_translate.models import TranslationUnit
from pl_mmb_translate.openai_translator import OpenAITranslator
from pl_mmb_translate.runner import _translation_fingerprint, _write_manifest

SECONDS = 1200
OUT = ROOT / 'reports/visualnews_concurrency_comparison.json'
EVENTS = ROOT / 'reports/visualnews_concurrency_events.jsonl'


def now():
    return datetime.now(timezone.utc).isoformat()


def save(value):
    tmp = OUT.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(OUT)


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * q))] if values else None


async def stage(concurrency, items, settings, store, event_stream):
    translator = OpenAITranslator(replace(settings, batch_size=concurrency))
    request_stats = Counter()
    request_seconds = []
    original_create = translator.client.chat.completions.create

    async def instrumented_create(*args, **kwargs):
        started = time.monotonic()
        request_stats['attempts'] += 1
        try:
            response = await original_create(*args, **kwargs)
            request_stats['http_success'] += 1
            usage = getattr(response, 'usage', None)
            request_stats['completion_tokens'] += getattr(usage, 'completion_tokens', 0) or 0
            return response
        except asyncio.CancelledError:
            request_stats['cancelled_attempts'] += 1
            raise
        except Exception as exc:
            request_stats['error_' + str(getattr(exc, 'status_code', type(exc).__name__))] += 1
            raise
        finally:
            request_seconds.append(time.monotonic() - started)

    translator.client.chat.completions.create = instrumented_create
    started = time.monotonic()
    deadline = started + SECONDS
    successes = failures = window_successes = window_failures = submitted = 0
    latencies, chars = [], []
    sample_cursor = iter(items)
    pending = set()

    async def one(item):
        unit = TranslationUnit(unit_id=item['id'], texts=item['texts'], images=[],
                               metadata=item['metadata'], prompt_profile=item['profile'])
        before = time.monotonic()
        try:
            translated = await translator.translate_one(unit, with_image=False)
            return unit, translated, None, time.monotonic() - before
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return unit, None, str(exc), time.monotonic() - before

    def submit():
        nonlocal submitted
        if time.monotonic() >= deadline:
            return
        for item in sample_cursor:
            if not store.is_completed(item['id']):
                pending.add(asyncio.create_task(one(item)))
                submitted += 1
                return
        raise RuntimeError('Benchmark sample exhausted')

    for _ in range(concurrency):
        submit()
    next_report = started + 30
    print(json.dumps(dict(event='stage_start', concurrency=concurrency, at=now())), flush=True)
    try:
        while pending:
            # Allow a bounded drain, then cancel outstanding requests before the next stage.
            if time.monotonic() > deadline + 240:
                break
            done, _ = await asyncio.wait(pending, timeout=1, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                pending.remove(task)
                unit, translated, error, elapsed = await task
                in_window = time.monotonic() <= deadline
                if error is None:
                    store.put_completed(unit, translated, model=settings.model, with_image=False)
                    successes += 1
                    window_successes += int(in_window)
                    latencies.append(elapsed)
                    chars.append(len(unit.texts['text']))
                else:
                    store.put_failure(unit.unit_id, error)
                    failures += 1
                    window_failures += int(in_window)
                event_stream.write(json.dumps(dict(at=now(), concurrency=concurrency, id=unit.unit_id,
                                                   seconds=elapsed, success=error is None,
                                                   in_window=in_window, error=error), ensure_ascii=False) + '\n')
                event_stream.flush()
                submit()
            if time.monotonic() >= next_report:
                print(json.dumps(dict(event='progress', concurrency=concurrency,
                                      elapsed=round(time.monotonic() - started), successes=successes,
                                      window_successes=window_successes, failures=failures,
                                      in_flight=len(pending), requests=dict(request_stats))), flush=True)
                next_report = time.monotonic() + 30
        cancelled = len(pending)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    finally:
        await translator.close()
    elapsed = time.monotonic() - started
    result = dict(concurrency=concurrency, finished_at=now(), measurement_seconds=SECONDS,
                  wall_seconds=elapsed, submitted=submitted, succeeded=successes, failures=failures,
                  window_successes=window_successes, window_failures=window_failures,
                  texts_per_hour=window_successes * 3600 / SECONDS,
                  texts_per_hour_including_drain=successes * 3600 / elapsed,
                  cancelled_after_drain=cancelled, requests=dict(request_stats),
                  text_seconds_median=statistics.median(latencies) if latencies else None,
                  text_seconds_p95=percentile(latencies, .95),
                  request_seconds_median=statistics.median(request_seconds) if request_seconds else None,
                  source_chars_mean=statistics.mean(chars) if chars else None)
    print(json.dumps(dict(event='stage_complete', **result)), flush=True)
    return result


async def main():
    os.chdir(ROOT)
    if OUT.exists():
        raise RuntimeError('Comparison report already exists; do not overwrite or repeat implicitly')
    locks = []
    for filename in ['absence-full-worker.lock', 'absence-full-mbeir_visualnews.lock',
                     'visualnews-concurrency-benchmark.lock']:
        lock = (ROOT / 'reports' / filename).open('a+b')
        if os.name == 'nt':
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    plan = json.loads(news.PLAN.read_text(encoding='utf-8'))
    if os.name == 'nt':
        # Production .env points at WSL certificates. Use the same local CA bundle.
        os.environ['SSL_CERT_FILE'] = str(ROOT / 'combined-ca.pem')
        os.environ['REQUESTS_CA_BUNDLE'] = str(ROOT / 'combined-ca.pem')
    output = plan['output_dir']
    if os.name == 'nt' and output.startswith('/mnt/'):
        output = output[5].upper() + ':/' + output[7:]
    settings = replace(news.base.suite.Settings.from_env(ROOT / '.env'), **plan['settings'],
                       output_dir=Path(output), run_name=plan['run_name'])
    job = plan['jobs'][0]
    if _translation_fingerprint(settings, with_image=False) != job['translation_config']:
        raise RuntimeError('Frozen production translation policy differs')
    source = ROOT / job['source']
    sha = hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    if sha.hexdigest() != job['source_sha256']:
        raise RuntimeError('Prepared source changed')
    if shutil.disk_usage(settings.output_dir).free < 8 * 1024**3:
        raise RuntimeError('Less than 8 GiB free')
    directory = settings.output_dir / settings.run_name / job['key'] / 'text_only'
    _write_manifest(directory / 'manifest.json', dataset_key=job['key'], settings=settings, with_image=False)
    store = CheckpointStore(directory)
    completed = {r[0] for r in store.conn.execute('SELECT unit_id FROM completed')}
    # Sample a large pending prefix, shuffle reproducibly, then use distinct random texts per stage.
    items = []
    with source.open(encoding='utf-8') as stream:
        for line in stream:
            item = json.loads(line)
            if item['id'] not in completed:
                items.append(item)
            if len(items) == 20000:
                break
    random.Random(20261020).shuffle(items)
    result = dict(status='running', started_at=now(), pid=os.getpid(), platform=os.name,
                  production_run=settings.run_name, model=settings.model,
                  policy=job['translation_config'], source_sha256=job['source_sha256'],
                  starting_completed=len(completed), seconds_per_stage=SECONDS,
                  order=[4, 8, 16], sampling='seeded shuffle of first 20000 pending texts; distinct texts per stage',
                  stages=[])
    save(result)
    try:
        with EVENTS.open('x', encoding='utf-8') as events:
            for concurrency in [4, 8, 16]:
                result['current_concurrency'] = concurrency
                save(result)
                result['stages'].append(await stage(concurrency, items, settings, store, events))
                save(result)
        result.update(status='complete', finished_at=now(), completed=store.counts()[0])
    except BaseException as exc:
        result.update(status='interrupted', error=type(exc).__name__ + ': ' + str(exc), finished_at=now())
        raise
    finally:
        save(result)
        store.export_jsonl()
        store.close()
        for lock in locks:
            lock.close()
    print(json.dumps(dict(event='complete', report=str(OUT), completed=result['completed'])), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
