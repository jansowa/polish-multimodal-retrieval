"""User-released XM3600 native-PL gallery and serial seven-model evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'experiments/data_study_20261030')]
from retrieval_benchmark.core import HERE, identity, local_path, read, sha, write

ALIASES = ['R0_seed44', 'R1_seed44', 'R3_seed44', 'embeddinggemma-2',
           'metaclip-b16-32768', 'metaclip-m16-32768', 'metaclip-s16-32768']
MODES = ['v2_no_instructions', 'author_recommended']


def destination():
    return local_path('C:/Users/jsowa/benchmark-translations/retrieval-benchmark-v1/xm3600-20261031')


def prepare_gallery():
    from audit import fingerprints
    from pl_mmb_training.holdouts import Components
    from pl_mmb_training.leakage import normalized_text
    source = ROOT/'training/runs/xm3600-pl'
    original = read(source/'manifest.json')
    if sha(source/'pairs.jsonl') != original['pairs_sha256'] or sha(source/'blacklist.json') != original['blacklist_sha256']:
        raise ValueError('Historical XM source changed')
    pairs = [json.loads(s) for s in (source/'pairs.jsonl').read_text().splitlines() if s.strip()]
    blacklist = read(source/'blacklist.json')['images']
    phashes = read(ROOT/'training/runs/tinyclip-wiki-pilot-v1/data/additional_protected_phashes.json')['hashes']
    paths = sorted({r['image'] for r in pairs})
    def verify(path):
        fp = fingerprints(path=path)
        expected = blacklist[path]
        if any(fp[k] != expected[k] for k in ('sha256', 'pixels_sha256')) or fp['phash'] != phashes[path]:
            raise ValueError('Original XM fingerprint mismatch: '+path)
        return path, fp
    with ThreadPoolExecutor(max_workers=4) as pool:
        verified = dict(pool.map(verify, paths))
    rows = [dict(r, **verified[r['image']], text_pl=r['text'], lineage_verified=True,
                 split='leaderboard_test', caption_origin='native_human') for r in pairs]
    if len(paths) != 3597 or len(rows) != 7133:
        raise ValueError('Unexpected historical clean-subset size')
    dest = destination()
    manifest = dest/'manifest.jsonl'
    payload = '\n'.join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in sorted(rows,key=lambda r:r['id']))+'\n'
    if manifest.exists():
        if manifest.read_text(encoding='utf-8') != payload:raise ValueError('Frozen XM manifest changed')
    else:
        manifest.parent.mkdir(parents=True,exist_ok=True);manifest.write_text(payload,encoding='utf-8')
    image_rows = {r['pixels_sha256']:r for r in rows}
    image_ids = sorted(image_rows);image_index = {v:i for i,v in enumerate(image_ids)}
    by_text = {}
    for r in rows:by_text.setdefault(normalized_text(r['text_pl']),[]).append(r)
    text_keys = sorted(by_text)
    # Native EN/PL captions are independently written, NOT parallel translations.
    # Canonicalize Polish only; never merge PL queries using an English caption.
    edges = sorted({(ti,image_index[r['pixels_sha256']]) for ti,key in enumerate(text_keys) for r in by_text[key]})
    components = Components();nodes = [components.add(image_rows[v]) for v in image_ids]
    query_nodes = []
    for key in text_keys:
        values = by_text[key];node = nodes[image_index[values[0]['pixels_sha256']]]
        for r in values[1:]:components.join(node,nodes[image_index[r['pixels_sha256']]])
        query_nodes.append(node)
    component_names = {}
    for uid,node in zip(image_ids,nodes):
        root = components.find(node);component_names[root] = min(uid,component_names.get(root,uid))
    protocol = dict(status='frozen_image_verified',id='xm3600_pl',manifest_sha256=sha(manifest),
        images=[image_rows[v]['image'] for v in image_ids],image_ids=image_ids,
        texts=[min(by_text[k],key=lambda r:r['id'])['text_pl'] for k in text_keys],
        query_ids=[hashlib.sha256(('pl:'+k).encode()).hexdigest() for k in text_keys],edges=edges,
        image_components=[component_names[components.find(n)] for n in nodes],
        text_components=[component_names[components.find(n)] for n in query_nodes],
        canonicalization='Exact pixels; normalized native Polish only; known caption-image positives; pHash only resampling components',
        source_pairs_sha256=sha(source/'pairs.jsonl'),original_images=3600,excluded_before_this_run=3,
        user_authorization='Use XM3600PL as an ordinary leaderboard dataset; no final-blind restriction',
        metrics='t2i/i2t recall@1/5/10, MRR, NDCG@10')
    write(dest/'retrieval_protocol.json',protocol,immutable=True)
    config = read(HERE/'local/config.json')
    config['datasets']['xm3600_pl'] = dict(protocol=str(dest/'retrieval_protocol.json'),
        protocol_sha256=sha(dest/'retrieval_protocol.json'),image_manifest=str(manifest))
    write(HERE/'local/config.json',config)
    print('XM_GALLERY_READY',len(paths),len(rows),len(text_keys),flush=True)


def prepare_runs():
    from retrieval_benchmark.completion import metadata
    config = read(HERE/'local/config.json')
    jobs = []
    for alias in ALIASES:
        spec = config['models'][alias]
        info = metadata(spec['repository'],spec['revision']) if 'repository' in spec else {}
        jobs.append(dict(alias=alias,model_id=('hf:'+spec['repository']+'@'+spec['revision']) if info else 'checkpoint:'+spec['revision'],
            **info,missing=[['xm3600_pl',mode] for mode in MODES],parity=None))
    from retrieval_benchmark import completion
    reg = dict(status='closed_before_XM_scores',jobs=jobs,config=config,
        worker_sha256=sha(completion.__file__),encoder_sha256=sha(ROOT/'src/pl_mmb_benchmark/models.py'),
        scorer_sha256=sha(ROOT/'src/pl_mmb_benchmark/full_gallery.py'),
        xm_protocol_sha256=config['datasets']['xm3600_pl']['protocol_sha256'],
        authorization='Explicit user request: XM3600 normal leaderboard; seven named models; R3_seed44 corrected from R2',
        simultaneous_models=1,cleanup='Only newly downloaded exclusive SHA-verified weights; existing primary weights protected')
    write(destination()/'registration.json',reg,immutable=True)
    print('XM_RUNS_REGISTERED',len(jobs),flush=True)


def worker(alias):
    from retrieval_benchmark import completion
    reg_path = destination()/'registration.json';reg = read(reg_path)
    spec = reg['config']['models'][alias]
    if 'repository' in spec:
        completion.destination = destination
        completion.worker(alias,reg_path)
    else:
        from retrieval_benchmark.runner import plan,run_worker
        execution = plan(reg['config'],alias,['xm3600_pl'],MODES,64)
        execution['profile'] = 'historical-native-v2'
        run_worker(reg['config'],execution)


def run():
    from pl_mmb_benchmark.models import worker_python
    from retrieval_benchmark.ingest import sync,configuration
    reg = read(destination()/'registration.json')
    if (destination()/'complete.json').exists():
        complete=read(destination()/'complete.json')
        if complete['registration_sha256']!=sha(destination()/'registration.json') or complete['models']!=ALIASES:
            raise RuntimeError('Completed XM registration changed')
        print('XM_ALREADY_COMPLETE',flush=True);return
    sync_config=configuration();sync_config.update(reg['config'])
    for job in reg['jobs']:
        alias=job['alias'];spec=reg['config']['models'][alias]
        executable=worker_python(spec['repository'],sys.executable) if 'repository' in spec else str(Path.home()/'.venvs/pl-mmb-benchmark/bin/python')
        log=destination()/(alias+'.log')
        with log.open('a',encoding='utf-8') as out:
            result=subprocess.run([executable,'-u',str(Path(__file__).resolve()),'worker','--model',alias],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('XM worker failed: '+alias+'; see '+str(log))
        print('XM_MODEL_COMPLETE',alias,flush=True);print(json.dumps(sync(sync_config)),flush=True)
    write(destination()/'complete.json',dict(status='complete',models=ALIASES,registration_sha256=sha(destination()/'registration.json')),immutable=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['gallery','register','run','worker']);parser.add_argument('--model');args=parser.parse_args()
    {'gallery':prepare_gallery,'register':prepare_runs,'run':run,'worker':lambda:worker(args.model)}[args.command]()
