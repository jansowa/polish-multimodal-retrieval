"""Production image continuation: skip every previously translated VisualNews example."""
import asyncio
from bisect import bisect_right
from collections import OrderedDict
from contextlib import closing
import contextvars
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'src')]
import visualnews_production as base
from visualnews_image_assets import CACHE, DB, IMAGE_REPO, IMAGE_REV, sha
from pl_mmb_translate import runner
from pl_mmb_translate.models import ImageRef, TranslationUnit
from pl_mmb_translate.registry import DATASETS, DatasetSpec
import pyarrow.parquet as pq

STATE = ROOT / 'reports/visualnews_production_state.json'
STOP = ROOT / 'reports/visualnews_image_stop.request'


def text_leaves(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from text_leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from text_leaves(item)


def collect_completed(paths, canonical):
    """Canonical IDs and exact source captions identify examples across methods and aliases."""
    by_text = {text: uid for uid, text in canonical.items()}
    completed, audit = set(), []
    for path in sorted(paths):
        before = len(completed)
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=30)) as connection:
            count = 0
            for uid, source in connection.execute('SELECT unit_id,source_json FROM completed'):
                count += 1
                source = json.loads(source)
                if uid in canonical:
                    if source.get('text') != canonical[uid]:
                        raise ValueError(f'Completed source mismatch: {uid} in {path}')
                    completed.add(uid)
                for text in text_leaves(source):
                    # Covers the MMEB variant as well as M-BEIR aliases/pilots.
                    cleaned = text.removeprefix('Retrieve an image of this news caption. ').strip()
                    if cleaned in by_text:
                        completed.add(by_text[cleaned])
        audit.append(dict(checkpoint=str(path), completed_rows=count, newly_covered_ids=len(completed) - before))
    return completed, audit


def checkpoint_paths(output):
    return {p for parent in [ROOT / 'outputs', output]
            for p in parent.rglob('checkpoint.sqlite3')
            if 'visualnews' in p.parent.parent.name.casefold()}


class ImageLoader:
    """Read only needed Parquet row groups; retain at most four groups in memory."""
    def __init__(self):
        self.files, self.groups = {}, OrderedDict()

    def close(self):
        self.groups.clear()
        for file, _ in self.files.values():
            file.close()
        self.files.clear()

    def get(self, shard, row_index, did, expected_path):
        if shard not in self.files:
            file = pq.ParquetFile(shard)
            starts, offset = [], 0
            for i in range(file.num_row_groups):
                starts.append(offset)
                offset += file.metadata.row_group(i).num_rows
            self.files[shard] = file, starts
        file, starts = self.files[shard]
        group = bisect_right(starts, row_index) - 1
        key = shard, group
        if key not in self.groups:
            self.groups[key] = file.read_row_group(group, columns=['did', 'img_path', 'img'])
            while len(self.groups) > 4:
                self.groups.popitem(last=False)
        self.groups.move_to_end(key)
        row = self.groups[key].slice(row_index - starts[group], 1).to_pylist()[0]
        if row['did'] != did or row['img_path'] != expected_path or not row['img']['bytes']:
            raise ValueError('Image bytes do not match the verified caption/image join')
        return row['img']['bytes']


class ThinkingImageTranslator(base.ProductionTranslator):
    def __init__(self, settings):
        super().__init__(settings)
        self.current_unit = contextvars.ContextVar('visualnews_image_unit')
        original_create = self.client.chat.completions.create
        async def create(*args, **kwargs):
            if not any(isinstance(message.get('content'), list) and
                       any(item.get('type') == 'image_url' for item in message['content'])
                       for message in kwargs['messages']):
                raise RuntimeError('Refusing a VisualNews image request without an image')
            extra = dict(kwargs.get('extra_body') or {})
            extra['chat_template_kwargs'] = dict(extra.get('chat_template_kwargs') or {}, enable_thinking=True)
            kwargs['extra_body'] = extra
            response = await original_create(*args, **kwargs)
            details = getattr(getattr(response, 'usage', None), 'completion_tokens_details', None)
            unit = self.current_unit.get()
            unit.metadata['translation_inputs']['reasoning_tokens_last_response'] = getattr(details, 'reasoning_tokens', None)
            return response
        self.client.chat.completions.create = create

    async def translate_one(self, unit, *, with_image):
        if not with_image or len(unit.images) != 1:
            raise RuntimeError('This continuation requires exactly one verified image')
        token = self.current_unit.set(unit)
        try:
            return await super().translate_one(unit, with_image=True)
        finally:
            self.current_unit.reset(token)


class ImageAdapter:
    def __init__(self, job, settings, canonical, db):
        self.job, self.settings, self.canonical, self.db = job, settings, canonical, db

    def iter_units(self, settings, with_image):
        if not with_image:
            raise RuntimeError('Text-only fallback is forbidden')
        done, _ = collect_completed(checkpoint_paths(settings.output_dir), self.canonical)
        loader = ImageLoader()
        with closing(loader), closing(sqlite3.connect(DB.as_uri() + '?mode=ro', uri=True)) as assets:
            with (ROOT / self.job['source']).open(encoding='utf-8') as stream:
                for line in stream:
                    if STOP.exists():
                        return
                    item = json.loads(line)
                    if item['id'] in done:
                        continue
                    image = assets.execute('SELECT i.did,i.path,i.shard,i.row_index,c.matching_images '
                                           'FROM captions c JOIN images i ON c.image_did=i.did WHERE c.did=?',
                                           (item['metadata']['corpus_id'],)).fetchone()
                    if image is None:
                        continue  # Accounted for in the plan; never send without its image.
                    did, path, shard, row_index, matches = image
                    payload = loader.get(shard, row_index, did, path)
                    inputs = dict(with_image=True, reasoning_enabled=True, reasoning_control='enable_thinking=true',
                                  image_did=did, source_image=path, image_sha256=hashlib.sha256(payload).hexdigest(),
                                  image_repository=IMAGE_REPO, image_revision=IMAGE_REV,
                                  matching_original_images=matches,
                                  image_selection='lexicographically first verified original path')
                    item['metadata']['translation_inputs'] = inputs
                    yield TranslationUnit(unit_id=item['id'], texts=item['texts'],
                                          images=[ImageRef('verified_caption_image', payload)],
                                          metadata=item['metadata'], prompt_profile=item['profile'])


async def work(settings, job, baseline, db):
    state = dict(baseline, status='translating')
    def refresh():
        current = base.counts(db)
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(baseline['started_at'])).total_seconds()
        state.update(updated_at=base.utcnow(), with_image_completed=current['completed'], failures=current['failures'],
                     completed=baseline['completed_other_methods'] + current['completed'],
                     added=current['completed'] - baseline['starting_with_image_completed'],
                     texts_per_hour=(current['completed'] - baseline['starting_with_image_completed']) * 3600 / max(elapsed, 1))
        base.save(STATE, state)
    async def monitor():
        while True:
            await asyncio.sleep(30)
            refresh()
            print(json.dumps(dict(event='image_production_progress', **state)), flush=True)
            if shutil.disk_usage(settings.output_dir).free < 8 * 1024 ** 3:
                raise RuntimeError('Less than 8 GiB free; checkpoint preserved')
    refresh()
    monitoring = asyncio.create_task(monitor())
    try:
        for attempt in range(1, 4):
            state.update(status='translating', **{'pass': attempt})
            translation = asyncio.create_task(runner.run_dataset(job['key'], settings, with_image=True))
            done, _ = await asyncio.wait([translation, monitoring], return_when=asyncio.FIRST_COMPLETED)
            if monitoring in done:
                translation.cancel()
                await asyncio.gather(translation, return_exceptions=True)
                await monitoring
            await translation
            refresh()
            if STOP.exists():
                state['status'] = 'stopped_by_request'
                break
            if state['completed'] == baseline['expected']:
                state['status'] = 'complete'
                break
            state['status'] = 'retry_wait'
            refresh()
            await asyncio.sleep(60 * attempt)
        else:
            state['status'] = 'incomplete'
    except BaseException as exc:
        state.update(status='interrupted', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        monitoring.cancel()
        await asyncio.gather(monitoring, return_exceptions=True)
        refresh()


def main():
    if os.name != 'nt':
        raise RuntimeError('Use the existing Windows translation environment')
    import msvcrt
    os.chdir(ROOT)
    if STOP.exists():
        raise RuntimeError('A stop request exists; inspect before resuming')
    locks = []
    for filename in ['absence-full-worker.lock', 'absence-full-mbeir_visualnews.lock',
                     'visualnews-concurrency-benchmark.lock']:
        lock = (ROOT / 'reports' / filename).open('a+b')
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        locks.append(lock)
    old_plan = json.loads(base.news.PLAN.read_text(encoding='utf-8'))
    os.environ['SSL_CERT_FILE'] = str(ROOT / 'combined-ca.pem')
    os.environ['REQUESTS_CA_BUNDLE'] = str(ROOT / 'combined-ca.pem')
    output = old_plan['output_dir']
    if output.startswith('/mnt/'):
        output = output[5].upper() + ':/' + output[7:]
    settings = replace(base.news.base.suite.Settings.from_env(ROOT / '.env'), **old_plan['settings'],
                       output_dir=Path(output), run_name=old_plan['run_name'])
    if settings.batch_size != 16 or settings.request_timeout != 3900 or settings.quality_review:
        raise ValueError('Expected the existing 16-request, 3900-second, single-pass production settings')
    job = old_plan['jobs'][0]
    if runner._translation_fingerprint(settings, with_image=False) != job['translation_config']:
        raise ValueError('The existing frozen translation policy changed')
    if sha(ROOT / job['source']) != job['source_sha256']:
        raise ValueError('Prepared text source changed')
    assets_manifest = json.loads((CACHE / 'manifest.json').read_text(encoding='utf-8'))
    if assets_manifest['status'] != 'complete' or assets_manifest['image_revision'] != IMAGE_REV:
        raise ValueError('Image assets are not verified')
    if sha(DB) != assets_manifest['index_sha256']:
        raise ValueError('Verified caption/image index changed')
    # Verify the full cached image data before any model calls.
    for asset in assets_manifest['image_shards']:
        if sha(asset['path']) != asset['sha256']:
            raise ValueError('Cached image shard changed')
    with closing(sqlite3.connect(DB.as_uri() + '?mode=ro', uri=True)) as assets:
        canonical = {'MBeirvisualnews:' + did: text for did, text in assets.execute('SELECT did,text FROM captions')}
        mapped = {'MBeirvisualnews:' + did for (did,) in assets.execute('SELECT did FROM captions WHERE image_did IS NOT NULL')}
    directory = settings.output_dir / settings.run_name / job['key'] / 'with_image'
    runner._write_manifest(directory / 'manifest.json', dataset_key=job['key'], settings=settings, with_image=True)
    from pl_mmb_translate.checkpoint import CheckpointStore
    initial_store = CheckpointStore(directory)
    initial_store.close()
    db = directory / 'checkpoint.sqlite3'
    paths = checkpoint_paths(settings.output_dir)
    done, audit = collect_completed(paths, canonical)
    prior, _ = collect_completed(paths - {db}, canonical)
    current = base.counts(db)
    # Existing successful image outputs must never overlap any other method.
    with closing(sqlite3.connect(db.as_uri() + '?mode=ro', uri=True)) as connection:
        own_ids = {r[0] for r in connection.execute('SELECT unit_id FROM completed')}
    if prior & own_ids:
        raise ValueError('Existing image checkpoint overlaps a previously completed method')
    pending = mapped - done
    unmapped = set(canonical) - mapped - done
    baseline = dict(started_at=base.utcnow(), pid=os.getpid(), run_name=settings.run_name,
                    starting_completed=len(done), completed_other_methods=len(prior),
                    starting_with_image_completed=current['completed'], starting_failures=current['failures'],
                    expected=job['expected'], concurrency=16, with_image=True, reasoning_enabled=True,
                    timeout_seconds=dict(connect=75, read=3900, write=600, pool=75),
                    checkpoint=str(db), pending_with_verified_image=len(pending),
                    remaining_without_verified_image=len(unmapped), source_sha256=job['source_sha256'],
                    assets_manifest_sha256=sha(CACHE / 'manifest.json'),
                    translation_config=runner._translation_fingerprint(settings, with_image=True),
                    completed_checkpoints=audit)
    plan = dict(baseline, image_repository=IMAGE_REPO, image_revision=IMAGE_REV,
                no_text_only_fallback=True, deduplication='all completed VisualNews IDs and exact source captions across checkpoints',
                unmatched_ids=sorted(unmapped))
    base.save(ROOT / 'reports/visualnews_image_plan.json', plan)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    base.save(ROOT / 'reports' / ('visualnews_resume_' + stamp + '.json'), baseline)
    base.save(ROOT / 'reports/visualnews_resume_baseline.json', baseline)
    with (ROOT / 'reports/visualnews_resume_history.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(baseline, ensure_ascii=False) + '\n')
    base.save(ROOT / 'reports/visualnews_image_prior_completed.json', sorted(prior))
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest.update(reasoning_enabled=True, reasoning_control='chat_template_kwargs.enable_thinking=true',
                    continuation_plan=str(ROOT / 'reports/visualnews_image_plan.json'))
    base.save(manifest_path, manifest)
    DATASETS[job['key']] = DatasetSpec(job['key'], job['key'],
                                     lambda: ImageAdapter(job, settings, canonical, db), len(pending) + current['completed'])
    runner.OpenAITranslator = ThinkingImageTranslator
    print(json.dumps(dict(event='image_production_resume', **baseline)), flush=True)
    try:
        asyncio.run(work(settings, job, baseline, db))
    finally:
        for lock in locks:
            lock.close()


if __name__ == '__main__':
    main()
