"""Close execution registration before new-gallery results, preserving v1 bytes."""
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pl_mmb_training.series_20261023 import ROOT,OUT,read,freeze,file_sha,require_space


def main():
    require_space(['/mnt/c','/mnt/d','/'])
    previous=OUT/'reference_eval_registration.json'
    registration=copy.deepcopy(read(previous))
    adapter=read(OUT/'projector_adapter_verification_20261029.json')
    parity=read(OUT/'reference_shared_harness_parity_20261029.json')
    if adapter['status']!='passed' or parity['status']!='passed':raise ValueError('Execution validation failed')
    source_list=read(OUT/'prompt_sources/sources.json')
    sources={s['alias']:s for s in source_list}
    for model in registration['reference_models']:
        source=sources[model['alias']]
        if (source['model'],source['revision'])!=(model['model'],model['revision']):raise ValueError('Prompt revision differs')
        card=OUT/'prompt_sources'/(model['alias']+'.md')
        if file_sha(card)!=source['sha256']:raise ValueError('Author source changed')
        model['author_prompt_status']='source_audited_before_first_new_gallery_score'
        model['author_prompt']=dict(query_prefix='task: search result | query: ' if model['alias']=='embeddinggemma-2' else '',
            document_prefix='title: none | text: ' if model['alias']=='embeddinggemma-2' else '',
            image_prefix='',source_url=source['url'],source_sha256=source['sha256'],
            rationale='Gemma SearchQuery/Document asymmetric retrieval; otherwise raw caption, no universal prefix documented for the registered encoder')
        if model['alias']=='bge-vl-base':
            model['author_prompt']['rationale']='Registered base CLIP encoder; MMEB task_instruction on the MLLM model-card section does not apply to this encoder'
    datasets=[]
    for relative,role in [
        ('wiki_v2/retrieval_protocol.json','zero-shot for our registered never-Wiki-trained checkpoints'),
        ('xtd10_v2/full_retrieval_protocol.json','translation-style control; known source overlap disclosed'),
        ('xtd10_v2/clean_retrieval_protocol.json','translation-style control; no known protected overlap'),
        ('vn_dev_protocol/retrieval_protocol.json','in-domain; automatically generated/recycled/translated labels')]:
        path=OUT/relative;value=read(path)
        if value['status']!='frozen_image_verified':raise ValueError('Unverified gallery')
        datasets.append(dict(name=value['id'],protocol_path=str(path.relative_to(ROOT)),
            protocol_sha256=file_sha(path),manifest_sha256=value['manifest_sha256'],
            images=len(value['images']),texts=len(value['texts']),role=role))
    checkpoints={
        'VN_FULL_seed42':ROOT/'training/runs/tinyclip-source-ablation-v1/VN_FULL/best.pt',
        'VN_FULL_seed43':Path('/mnt/c/Users/jsowa/benchmark-translations/series-2026-10-23/runs/vn_full_seeds/seed-43/VN_FULL/best.pt'),
        'VN_FULL_seed44':Path('/mnt/c/Users/jsowa/benchmark-translations/series-2026-10-23/runs/vn_full_seeds/seed-44/VN_FULL/best.pt'),
        'VN_only':ROOT/'training/runs/tinyclip-source-ablation-v1/VN_only/best.pt',
        'historical_A1':ROOT/'training/runs/tinyclip-adaptation/A1/best.pt'}
    registration.update(schema=2,status='closed_ready_for_execution',
        supersedes=dict(path=str(previous.relative_to(ROOT)),sha256=file_sha(previous),
            reason='Complete frozen galleries, author sources, exact numerical projector parity and shared-scorer historical rank parity before first new-gallery scores'),
        datasets=datasets,
        deferred_datasets=[dict(name='human_short_queries',reason='no real human annotations yet'),
            dict(name='XTD10_paired_styles',reason='zero unambiguous existing exact-EN human/Qwen pairs; no synthetic replacement'),
            dict(name='local_CIRR_FashionIQ_RefCOCO',scope='selected finalist only; historical protocol kept separate from full galleries')],
        our_checkpoints={name:dict(path=str(path),sha256=file_sha(path),never_trained_on_Wiki=True) for name,path in checkpoints.items()},
        selected_finalist='VN_FULL_seed44',
        parity_gate=dict(status='passed',models=list(parity['checks']),proof='reference_shared_harness_parity_20261029.json',
                         proof_sha256=file_sha(OUT/'reference_shared_harness_parity_20261029.json')),
        projector_adapter_verification=dict(path='projector_adapter_verification_20261029.json',
            sha256=file_sha(OUT/'projector_adapter_verification_20261029.json')),
        remaining_execution_blockers=[],
        cache_and_storage=dict(our_new_embeddings='C:/Users/jsowa/benchmark-translations/series-2026-10-23/evaluation',
            reference_cache='native WSL ~/.cache/huggingface',reserve_GiB_each_physical_volume=4,
            automatic_deletion=False,weights_retained_until_user_manually_deletes_them=True,
            reference_worker_start_requires_fresh_capacity_check=True),
        implementation_sha256={name:file_sha(ROOT/name) for name in (
            'src/pl_mmb_benchmark/full_gallery.py','src/pl_mmb_benchmark/projector_adapter.py',
            'src/pl_mmb_benchmark/models.py','src/pl_mmb_benchmark/special_models.py')},
        reproducibility_note='Our encoders use original Torch 2.11/cu128 and Transformers 4.57.6; the is_causal compatibility warning is retained rather than changing the trained embedding space. Adapter embeddings matched historical cache exactly on the registered verification sample.')
    if [(m['model'],m['revision']) for m in registration['reference_models']] != [(m['model'],m['revision']) for m in read(previous)['reference_models']]:
        raise ValueError('Reference model list changed')
    path=OUT/'reference_eval_registration_v2_20261029.json'
    freeze(path,registration)
    print(json.dumps(dict(path=str(path),status=registration['status'],models=len(registration['reference_models']),
        datasets=datasets,our_models=list(checkpoints)),indent=2))


if __name__=='__main__':main()
