"""One-model GPU workers and capacity-checked orchestration for the frozen list."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pl_mmb_training.series_20261023 import ROOT,OUT,read,freeze,file_sha,require_space

REG=OUT/'reference_eval_registration_v3_20261029.json'
DEST=Path('/mnt/c/Users/jsowa/benchmark-translations/series-2026-10-23/evaluation/references')


def model_weight_items(alias):
    budget=read(OUT/'reference_storage_preflight_20261029.json')
    aliases={alias}
    if alias in ('clip-b32-pl','clip-b32-pl-32768'):aliases.add('shared-clip-image')
    if alias=='mclip-b16plus':aliases.add('mclip-image-component')
    return [item for item in budget['models'] if item['alias'] in aliases]


def remaining_weights(alias):
    missing=0
    for item in model_weight_items(alias):
        folder=Path.home()/'.cache/huggingface/hub'/('models--'+item['repo'].replace('/','--'))/'snapshots'/item['revision']
        for entry in item['files']:
            path=folder/entry['name']
            if not path.is_file() or path.stat().st_size!=entry['bytes']:missing+=entry['bytes']
    return missing


def remove_completed_weights(alias,registration):
    """Remove only checked files belonging to the completed reference worker."""
    if not registration['cache_and_storage'].get('automatic_deletion'):
        raise ValueError('Weight removal not authorized by active registration')
    completed=read(DEST/alias/'complete.json')
    if completed['status']!='complete' or completed['registration_sha256']!=file_sha(REG):
        raise ValueError('Cannot remove weights of an incomplete or differently registered worker')
    for dataset in registration['datasets']:
        for mode in ('v2_no_instructions','author_recommended'):
            result=read(DEST/alias/dataset['name']/(mode+'.json'))
            if result['registration_sha256']!=file_sha(REG):raise ValueError('Result identity differs before cleanup')
    read(DEST/alias/'efficiency.json')
    hub=(Path.home()/'.cache/huggingface/hub').resolve()
    protected={'wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M','OPI-PIB/PolDense-32M'}
    plans=[];seen=set()
    for item in model_weight_items(alias):
        if item['repo'] in protected:raise ValueError('Primary model is protected from reference cleanup')
        repo=(hub/('models--'+item['repo'].replace('/','--'))).resolve()
        if not repo.is_relative_to(hub):raise ValueError('HF repository escapes named cache root')
        snapshot=repo/'snapshots'/item['revision']
        for entry in item['files']:
            relative=Path(entry['name'])
            if relative.is_absolute() or '..' in relative.parts or relative.suffix not in ('.safetensors','.pt','.bin'):
                raise ValueError('Not an authorized model weight filename')
            link=snapshot/relative
            if not link.is_file():continue
            target=link.resolve(strict=True)
            if not target.is_relative_to(repo) or not target.is_file():raise ValueError('Weight target escapes its exact repository')
            if target in seen:continue
            seen.add(target)
            if target.stat().st_size!=entry['bytes']:raise ValueError('Weight size differs; cleanup stopped')
            digest=file_sha(target)
            if len(target.name)==64 and all(c in '0123456789abcdef' for c in target.name) and digest!=target.name:
                raise ValueError('LFS weight hash differs; cleanup stopped')
            plans.append(dict(path=str(target),snapshot_path=str(link),bytes=entry['bytes'],sha256=digest))
    proof=DEST/alias/'weight_cleanup_plan.json'
    freeze(proof,dict(status='registered_before_weight_file_removal',files=plans,
        registration_sha256=file_sha(REG),user_authorization=registration['cache_and_storage']['user_authorization'],
        result_and_efficiency_complete=True,repository_scope_verified=True,no_recursive_directory_removal=True))
    for entry in plans:
        target=Path(entry['path'])
        if file_sha(target)!=entry['sha256']:raise ValueError('Weight changed after cleanup registration')
        target.unlink()
    freeze(DEST/alias/'weight_cleanup_complete.json',dict(status='complete',
        removed_weight_bytes=sum(p['bytes'] for p in plans),plan_sha256=file_sha(proof),
        configs_and_embeddings_retained=True,primary_TinyCLIP_and_PolDense_untouched=True,
        physical_C_free_bytes_after=shutil.disk_usage('/mnt/c').free))
    print('REFERENCE_WEIGHTS_REMOVED '+alias+' bytes='+str(sum(p['bytes'] for p in plans)),flush=True)


def one(alias):
    # Set before importing HF libraries; do not print credentials or environment.
    os.environ['REQUESTS_CA_BUNDLE']=str(ROOT/'combined-ca.pem')
    os.environ['SSL_CERT_FILE']=str(ROOT/'combined-ca.pem')
    import torch
    from pl_mmb_benchmark.models import Encoder,check_cache,model_batch_size
    from pl_mmb_benchmark.projector_adapter import AuthorPromptEncoder
    from pl_mmb_benchmark.full_gallery import evaluate_encoder
    from pl_mmb_benchmark.efficiency import measure_efficiency
    registration=read(REG)
    item=next(m for m in registration['reference_models'] if m['alias']==alias)
    check_cache()
    require_space(['/mnt/c','/mnt/d','/'])
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with (ROOT/'training/runs/tinyclip-visualnews-v1/worker.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.monotonic()
        encoder=Encoder(item['model'],model_batch_size(item['model'],64))
        if encoder.revision!=item['revision']:raise ValueError('Reference revision differs')
        native_save=encoder._save_cache
        def checked_save(path,tensor):
            require_space(['/mnt/c','/mnt/d','/'])
            require_space(['/mnt/c'],expected_bytes=tensor.numel()*tensor.element_size()+1024**2)
            native_save(path,tensor)
        encoder._save_cache=checked_save
        for dataset_item in registration['datasets']:
            protocol=ROOT/dataset_item['protocol_path']
            if file_sha(protocol)!=dataset_item['protocol_sha256']:raise ValueError('Frozen protocol changed')
            for mode in ('v2_no_instructions','author_recommended'):
                target=DEST/alias/dataset_item['name']/(mode+'.json')
                if target.exists():continue
                require_space(['/mnt/c','/mnt/d','/'])
                wrapped=AuthorPromptEncoder(encoder,alias,mode)
                result=evaluate_encoder(wrapped,dict(read(protocol),prompt_mode=mode),registration)
                result.update(model_alias=alias,model=item['model'],revision=item['revision'],
                    registration_sha256=file_sha(REG),precision=encoder.precision,
                    native_batch_size=encoder.batch_size,parameters=encoder.parameter_count,
                    components=encoder.components,torch=torch.__version__,
                    libraries={k:version(k) for k in ('transformers','sentence-transformers','Pillow')},
                    worker_sha256=file_sha(__file__),elapsed_worker_seconds=time.monotonic()-started,
                    model_role='English-only diagnostic control' if item['english_text_only_control'] else 'Polish retrieval competitor')
                freeze(target,result)
                print('REFERENCE_RESULT '+alias+' '+dataset_item['name']+' '+mode+' '+json.dumps(
                    {d:v['metrics'] for d,v in result['directions'].items()}),flush=True)
        efficiency_path=DEST/alias/'efficiency.json'
        if not efficiency_path.exists():
            probe=read(OUT/'vn_dev_protocol/retrieval_protocol.json')
            measurements=measure_efficiency(encoder,probe['images'],probe['texts'])
            measurements.update(model_alias=alias,registration_sha256=file_sha(REG),
                procedure_sha256=file_sha(ROOT/'src/pl_mmb_benchmark/efficiency.py'),
                probe_protocol_sha256=file_sha(OUT/'vn_dev_protocol/retrieval_protocol.json'))
            freeze(efficiency_path,measurements)
        freeze(DEST/alias/'complete.json',dict(status='complete',model_alias=alias,
            registration_sha256=file_sha(REG),seconds=time.monotonic()-started,weights_cleanup_pending=True))
        del encoder;gc.collect();torch.cuda.empty_cache()


def orchestrate():
    from pl_mmb_benchmark.models import worker_python
    registration=read(REG)
    if registration['status']!='closed_ready_for_execution':raise ValueError('Registration not closed')
    require_space(['/mnt/c','/mnt/d','/'])
    decision=OUT/'reference_execution_sequential_decision_20261029.json'
    freeze(decision,dict(status='registered_before_reference_scores',registration_sha256=file_sha(REG),
        worker_sha256=file_sha(__file__),efficiency_sha256=file_sha(ROOT/'src/pl_mmb_benchmark/efficiency.py'),
        models=[m['alias'] for m in registration['reference_models']],
        storage_policy='one reference model at a time; all registered datasets and timings, then remove checked weight files',
        simultaneous_reference_models=1,physical_C_free_bytes=shutil.disk_usage('/mnt/c').free,
        per_model_cache_and_temporary_budget_bytes=1024**3,minimum_reserve_bytes=4*2**30,
        model_selection_unchanged=True,weights_automatically_deleted=True,
        primary_TinyCLIP_and_PolDense_protected=True))
    for item in registration['reference_models']:
        require_space(['/mnt/c','/mnt/d','/'])
        if (DEST/item['alias']/'weight_cleanup_complete.json').exists():continue
        if (DEST/item['alias']/'complete.json').exists():
            remove_completed_weights(item['alias'],registration)
            continue
        missing=remaining_weights(item['alias'])
        require_space(['/mnt/c'],expected_bytes=missing+1024**3)
        print('REFERENCE_WORKER_START '+item['alias'],flush=True)
        command=[worker_python(item['model'],sys.executable),str(Path(__file__).resolve()),'--model',item['alias']]
        completed=subprocess.run(command,cwd=ROOT)
        if completed.returncode:raise RuntimeError('Reference worker failed; stop without substituting models: '+item['alias'])
        remove_completed_weights(item['alias'],registration)
    freeze(OUT/'reference_execution_complete_20261029.json',dict(status='complete',
        registration_sha256=file_sha(REG),models=[m['alias'] for m in registration['reference_models']],
        output=str(DEST),completed_reference_weights_removed=True,primary_weights_untouched=True))
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model');args=parser.parse_args()
    if args.model:one(args.model)
    else:sys.exit(orchestrate())
