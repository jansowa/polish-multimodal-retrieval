"""Pinned text-only EN→PL translation and automatic QA, always explicit CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import queue
import sys
import tarfile
import threading
import time
import zipfile

from prepare import ROOT, OUT, COORD, fetch, read, write, load_common, committed_bytes, sha
from acquire import ASSETS, download

MODEL = 'Helsinki-NLP/opus-mt-en-zlw'
REVISION = '92ffd5aa936924688bd715b26d6bb75393082c03'


def qa(en, pl, identifier):
    flags = []
    if not pl.strip() or len(pl.split()) < 1:
        flags.append('empty')
    language, confidence = identifier.classify(pl)
    if language != 'pl' or confidence < 0.7:
        flags.append('language_uncertain')
    if Counter(re.findall(r'\d+(?:[.,]\d+)?', en)) != Counter(re.findall(r'\d+(?:[.,]\d+)?', pl)):
        flags.append('digits_changed')
    en_negative = bool(re.search(r"\b(no|not|without|never|neither|cannot)\b|n't\b", en, re.I))
    pl_negative = bool(re.search(r'\b(nie|bez|brak|żad\w*|nigdy|ani)\b', pl, re.I))
    if en_negative != pl_negative:
        flags.append('negation_changed')
    words = pl.casefold().split()
    grams = [tuple(words[i:i+3]) for i in range(max(0, len(words)-2))]
    if grams and Counter(grams).most_common(1)[0][1] >= 3:
        flags.append('repeated_trigram')
    if re.search(r'<(?:unk|pad|s)>|translation:|tłumaczenie:|error', pl, re.I):
        flags.append('error_or_control_token')
    return dict(flags=flags, language=language, language_probability=confidence,
                accepted=not flags, language_detector='langid-1.1.6 restricted pl/en/cs/sk/de/ru; automatic proxy, not human validation')


def inputs():
    # Translation only consumes caption strings, never images, filenames, IDs or scores.
    with zipfile.ZipFile(ASSETS / 'vizwiz/annotations.zip') as archive:
        validation = json.loads(archive.read('annotations/val.json'))
    decision = read(OUT / 'vizwiz_acquisition_decision.json')
    filenames = {Path(item['name']).name for item in decision['original_members']}
    identifiers = {row['id'] for row in validation['images'] if row['file_name'] in filenames}
    captions = [row['caption'] for row in validation['annotations'] if row['image_id'] in identifiers
                and not row['is_rejected'] and not row['is_precanned']]
    return sorted(set(captions), key=lambda text: hashlib.sha256(text.encode()).hexdigest())


def main():
    common = load_common()
    metadata = read(OUT / 'deep_research_v2.json')['sources']
    api = next(r for r in metadata if r['url'] == 'https://huggingface.co/api/models/' + MODEL)
    if json.loads(api['text'])['sha'] != REVISION:
        raise ValueError('Translator metadata revision differs')
    # Fetch pinned tree, not mutable main: its LFS SHA binds the original weight.
    tree = fetch(f'https://huggingface.co/api/models/{MODEL}/tree/{REVISION}')
    package = fetch('https://pypi.org/pypi/langid/1.1.6/json')
    if tree['status'] != 'fetched' or package['status'] != 'fetched':
        raise ValueError('Cannot pin translator or language detector')
    files = json.loads(tree['text'])
    package_info = json.loads(package['text'])
    source = next(f for f in package_info['urls'] if f['packagetype']=='sdist')
    captions = inputs()
    decision = dict(status='preregistered_before_translation_or_scores', model=MODEL, revision=REVISION,
        translator_license='Apache-2.0', text_prefix='>>pol<< ', input='Only original caption text',
        parameters=dict(device='cpu', dtype='float32', torch_threads=1, interop_threads=1,
                        batch_size=4, num_beams=4, do_sample=False, max_new_tokens=256,
                        input_truncation=False, max_input_tokens=512),
        captions=len(captions), caption_set_sha256=hashlib.sha256(json.dumps(captions).encode()).hexdigest(),
        language_detector=dict(package='langid', version='1.1.6', source=source,
                               languages=['pl','en','cs','sk','de','ru'], minimum_probability=0.7),
        qa_rules=['nonempty', 'restricted language identification pl probability>=0.7',
                  'same digit-token multiset', 'negation-presence proxy agrees',
                  'no trigram repeated >=3 times', 'no control/error/translation-prefix output',
                  'no source truncation or generation token-limit exhaustion'],
        failed_QA='Mark whole image ineligible if any original accepted caption has unresolved translation QA; preserve all accepted captions for selected images.',
        automatic_QA_limitations='Surface rules and language-ID do not establish semantic equivalence. No manual labels or image-conditioned repair.',
        deterministic_retry=False, QA_thresholds_selected_without_retrieval_scores=True,
        new_manual_annotations=0, cuda_allowed=False, metadata=[tree, package])
    decision_path = OUT / 'translation_decision.json'
    if decision_path.exists():
        previous = read(decision_path)
        for key in ('model', 'revision', 'text_prefix', 'parameters', 'captions', 'caption_set_sha256', 'qa_rules'):
            if previous[key] != decision[key]:
                raise ValueError('Existing translation decision differs: ' + key)
        decision = previous  # Preserve pinned evidence timestamps and cache identity on resume.
    else:
        write(decision_path, decision)
    wanted = {'config.json', 'generation_config.json', 'pytorch_model.bin', 'source.spm', 'target.spm',
              'tokenizer_config.json', 'vocab.json', 'README.md'}
    downloaded = []
    for item in files:
        if item['path'] not in wanted:
            continue
        downloaded.append(download(f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{item["path"]}',
            'translator/' + item['path'], item['size'], item.get('lfs', {}).get('oid')))
    package_file = download(source['url'], 'language_qa/langid-1.1.6.tar.gz', source['size'], source['digests']['sha256'])
    vendor = ASSETS / 'language_qa/vendor'
    common.reserve('B', 'language-qa-code', {'C': 8*1024**2}, committed_bytes=committed_bytes())
    with tarfile.open(ASSETS / 'language_qa/langid-1.1.6.tar.gz') as archive:
        for member in archive.getmembers():
            parts = Path(member.name).parts
            if member.isfile() and len(parts) >= 3 and parts[1] == 'langid' and member.name.endswith('.py'):
                relative = Path(*parts[1:])
                if '..' in relative.parts or member.size > 5*1024**2:
                    raise ValueError('Unsafe language detector member')
                target = vendor / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                data = archive.extractfile(member).read()
                if target.exists():
                    if target.read_bytes() != data:
                        raise ValueError('Language detector code changed')
                else:
                    with target.open('xb') as stream:
                        stream.write(data)
    common.reserve('B', 'language-qa-code', {'C': 0}, committed_bytes=committed_bytes())
    sys.path.insert(0, str(vendor))
    import torch
    from transformers import MarianTokenizer, MarianMTModel
    from langid.langid import LanguageIdentifier, model as language_model
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    folder = ASSETS / 'translator'
    workers=threading.local()
    states=queue.SimpleQueue()
    # HF model initialization patches process-global tensor creation hooks.
    # Load both instances sequentially before either thread starts inference.
    for number in range(2):
        tokenizer=MarianTokenizer.from_pretrained(folder,local_files_only=True)
        translator=MarianMTModel.from_pretrained(folder,local_files_only=True,
            torch_dtype=torch.float32,device_map=None,low_cpu_mem_usage=False).to('cpu').eval()
        if any(parameter.device.type!='cpu' for parameter in translator.parameters()):
            raise ValueError('Translator has non-CPU or uninitialized parameters')
        identifier=LanguageIdentifier.from_modelstring(language_model,norm_probs=True)
        identifier.set_languages(['pl','en','cs','sk','de','ru'])
        states.put((tokenizer,translator,identifier))
    def initialize_worker():
        workers.tokenizer,workers.translator,workers.identifier=states.get()
    def translate_batch(batch):
        tokenized=workers.tokenizer(['>>pol<< '+text for text in batch],return_tensors='pt',padding=True,truncation=False)
        if tokenized['input_ids'].shape[1]>512:
            raise ValueError('Source exceeds preregistered context; no silent truncation')
        with torch.inference_mode():
            generated=workers.translator.generate(**{k:v.to('cpu') for k,v in tokenized.items()},
                num_beams=4,do_sample=False,max_new_tokens=256)
        texts=workers.tokenizer.batch_decode(generated,skip_special_tokens=True)
        output=[]
        for en,pl,ids in zip(batch,texts,generated):
            quality=qa(en,pl,workers.identifier)
            if len(ids)>=257 and int(ids[-1])!=workers.tokenizer.eos_token_id:
                quality['flags'].append('generation_limit');quality['accepted']=False
            output.append((en,pl,quality))
        return output
    cache = ASSETS / 'translations/vizwiz_cpu.jsonl'
    cache.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if cache.exists():
        for line in cache.open(encoding='utf-8'):
            row = json.loads(line)
            if row['model_revision'] != REVISION or row['translation_decision_sha256'] != sha(OUT / 'translation_decision.json'):
                raise ValueError('Translation cache belongs to another decision')
            existing[row['text_en']] = row
    remaining = [text for text in captions if text not in existing]
    budget = 16*1024**2
    committed_start = committed_bytes()
    bytes_written = 0
    def accounted():
        ledger = read(COORD/'DISK_RESERVATIONS.json')
        return max(committed_start+bytes_written,ledger.get('committed',{}).get('B',0))
    common.reserve('B', 'translation-output', {'C': budget}, committed_bytes=committed_start)
    started = time.monotonic()
    decision_sha = sha(OUT / 'translation_decision.json')
    with cache.open('a', encoding='utf-8') as stream, ThreadPoolExecutor(max_workers=2,initializer=initialize_worker) as pool:
        last_report=len(existing)//100
        for offset in range(0, len(remaining), 8):
            batches=[remaining[i:i+4] for i in range(offset,min(offset+8,len(remaining)),4)]
            translated=[row for output in pool.map(translate_batch,batches) for row in output]
            for en, pl, quality in translated:
                row = dict(text_en=en, text_pl=pl, model=MODEL, model_revision=REVISION,
                           translation_decision_sha256=decision_sha, qa=quality,
                           elapsed_run_seconds=time.monotonic()-started)
                payload = json.dumps(row, ensure_ascii=False) + '\n'
                stream.write(payload)
                budget -= len(payload.encode())
                bytes_written += len(payload.encode())
                existing[en] = row
            stream.flush()
            os.fsync(stream.fileno())
            common.reserve('B', 'translation-output', {'C': max(0,budget)}, committed_bytes=accounted())
            if offset==0 or len(existing)//100>last_report:
                last_report=len(existing)//100
                write(OUT / 'translation_progress.json', dict(status='running_CPU', done=len(existing), total=len(captions),
                    pid=os.getpid(), QA_flags=dict(Counter(flag for r in existing.values() for flag in r['qa']['flags'])),
                    elapsed_seconds=time.monotonic()-started, device='cpu', torch_threads_per_worker=1,workers=2))
                print(f'TRANSLATED_CPU {len(existing)}/{len(captions)}', flush=True)
    common.reserve('B', 'translation-output', {'C': 0}, committed_bytes=committed_bytes())
    summary = dict(status='complete_CPU', rows=len(existing), unique_source_captions=len(captions),
        artifact=common.artifact('translations',cache), model=MODEL, revision=REVISION, downloaded=downloaded,
        language_detector_package=package_file, translation_decision_sha256=decision_sha,
        accepted=sum(r['qa']['accepted'] for r in existing.values()),
        flags=dict(Counter(flag for row in existing.values() for flag in row['qa']['flags'])),
        elapsed_seconds=time.monotonic()-started, external_API_cost=0, local_compute_cost='unpriced CPU time',
        no_image_input=True, no_retrieval_scores=True)
    write(OUT / 'translation_summary.json', summary)
    print(json.dumps(summary))


if __name__=='__main__':
    main()
