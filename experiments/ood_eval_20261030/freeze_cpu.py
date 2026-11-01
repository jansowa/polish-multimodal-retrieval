"""Freeze audited data and original qrels only after full CPU translation QA."""
import json
from pathlib import Path
from collections import Counter, defaultdict

from prepare import ROOT, OUT, COORD, read, write, load_common, verify_marker, sha, committed_bytes, local_path
from acquire import ASSETS
from protocol import graph, choose_components
from handoff import verify_A


def immutable_json(common,path,value):
    if path.exists():
        if read(path)!=value:
            raise ValueError('Different immutable dataset artifact already exists: '+str(path))
    else:
        common.atomic_json(path,value,immutable=True)


def main():
    common = load_common()
    if (COORD/'B_EVAL_FROZEN.json').exists():
        verify_marker('B_EVAL_FROZEN')
        print('B_EVAL_ALREADY_FROZEN')
        return
    verify_marker('A_UNIVERSE_FROZEN')
    finalists, done = verify_A()
    audit = read(OUT/'candidate_overlap_audit.json')
    translation = read(OUT/'translation_summary.json')
    if audit['status'] != 'CPU_originals_and_final_translations_audited':
        raise ValueError('Final PL captions must be reaudited')
    if audit['A_UNIVERSE_FROZEN_sha256'] != sha(COORD/'A_UNIVERSE_FROZEN.json'):
        raise ValueError('Audit belongs to another A universe')
    if audit['final_translation_summary_sha256'] != sha(OUT/'translation_summary.json'):
        raise ValueError('Audit belongs to another translation summary')
    path = local_path(audit['artifact']['path'])
    if sha(path) != audit['artifact']['sha256']:
        raise ValueError('Candidate audit artifact changed')
    candidates = read(path)
    translation_path = local_path(translation['artifact']['path'])
    if sha(translation_path) != translation['artifact']['sha256']:
        raise ValueError('Translation cache changed')
    translations = {row['text_en']:row for row in (json.loads(line) for line in translation_path.open())}
    flickr_qa_summary=read(OUT/'flickr_translation_QA_summary.json')
    flickr_qa_path=local_path(flickr_qa_summary['artifact']['path'])
    if sha(flickr_qa_path)!=flickr_qa_summary['artifact']['sha256']:
        raise ValueError('Ready Flickr translation QA changed')
    flickr_qa={row['id']:row for row in read(flickr_qa_path)}
    qa_rejected, groups = Counter(), defaultdict(list)
    for image in candidates:
        if image['excluded']:
            continue
        if image['dataset']=='flickr30k_pl_local_test' and not flickr_qa[image['id']]['accepted']:
            qa_rejected['Flickr_images']+=1
            continue
        if image['dataset']=='vizwiz_pl_local_validation':
            if any(c['caption'] not in translations for c in image['captions']):
                raise ValueError('Incomplete translation cache')
            failures = [translations[c['caption']]['qa'] for c in image['captions'] if not translations[c['caption']]['qa']['accepted']]
            if failures:
                qa_rejected['images'] += 1
                qa_rejected.update(flag for failure in failures for flag in failure['flags'])
                continue
        for caption in image['captions']:
            if image['dataset']=='vizwiz_pl_local_validation':
                en = caption['caption']
                pl = translations[en]['text_pl']
                identifier = 'vizwiz:caption:' + str(caption['id'])
            else:
                en, pl, identifier = caption['text_en'], caption['text_pl'], 'flickr30k:caption:'+caption['id']
            groups[image['dataset']].append(dict(id=identifier, text_en=en, text_pl=pl,
                image=image['image'], target_id=image['pixels_sha256'], sha256=image['sha256'],
                pixels_sha256=image['pixels_sha256'], phash=image['phash'],
                original_image_id=image['id'], parent_ids=image['parent_ids'],
                source_split=image.get('split','validation'),
                verified_location=image.get('verified_location',dict(path=image['image'])),
                translation=('NLLB-200-1.3B; repack supplied; unavailable original translator revision'
                             if image['dataset'].startswith('flickr') else 'Helsinki-NLP/opus-mt-en-zlw@'+translation['revision'])))
    expected_sources={'flickr30k_pl_local_test','vizwiz_pl_local_validation'}
    if set(groups) != expected_sources:
        raise ValueError('Need two independently available sources')
    common.reserve('B','freeze-data',{'C':64*1024**2,'D':2*1024**2},committed_bytes=committed_bytes())
    outputs, counts = [], {}
    for name in sorted(groups):
        records = groups[name]
        draft = graph(records,name,'selection_draft_only')
        selected = set(choose_components(draft,1000,20261023))
        if len(selected)<500:
            raise ValueError(f'{name}: {len(selected)} images below minimum after registered QA/audit; no freeze')
        records = [r for r in records if r['pixels_sha256'] in selected]
        folder = ASSETS/'frozen'/name
        folder.mkdir(parents=True,exist_ok=True)
        manifest = folder/'manifest.json'
        immutable_json(common,manifest,records)
        protocol = graph(records,name,sha(manifest))
        protocol['status']='frozen_image_verified'
        protocol_path=folder/'retrieval_protocol.json'
        immutable_json(common,protocol_path,protocol)
        qrels_path=folder/'qrels.json'
        immutable_json(common,qrels_path,dict(image_ids=protocol['image_ids'],query_ids=protocol['query_ids'],edges=protocol['edges']))
        components_path=folder/'components.json'
        immutable_json(common,components_path,dict(image_ids=protocol['image_ids'],image_components=protocol['image_components'],
            query_ids=protocol['query_ids'],text_components=protocol['text_components']))
        outputs.extend([common.artifact('manifest:'+name,manifest), common.artifact('protocol:'+name,protocol_path),
                        common.artifact('qrels:'+name,qrels_path),common.artifact('components:'+name,components_path)])
        counts[name]=dict(images=len(protocol['images']),unique_queries=len(protocol['texts']),original_captions=len(records),
            positive_edges=len(protocol['edges']),components=len(set(protocol['image_components'])),
            source_pool_images=len({r['pixels_sha256'] for r in groups[name]}),
            protocol_sha256=sha(protocol_path),manifest_sha256=sha(manifest))
    decision=dict(status='closed_before_first_retrieval_score',generation=1,
        selected_sources=sorted(groups),counts=counts,QA_rejected_whole_images=dict(qa_rejected),
        candidate_overlap_audit_sha256=sha(OUT/'candidate_overlap_audit.json'),
        source_selection_rationale='Flickr has aligned existing PL and original IDs; VizWiz has public original validation photos, accepted human captions, feasible CPU text-only MT and range acquisition inside budget. DOCCI archive exceeds B budget; AIDe original photos/rights unresolved without correspondence.',
        minimum_sources=2,no_scores_used=True,no_manual_annotations=True,
        Flickr_restricted_research_use='Original provider noncommercial research/education conditions and Flickr terms; do not transfer repack CC BY to every photograph; no external redistribution.',
        VizWiz_license='CC BY 4.0, original VizWiz-Captions website',
        no_official_Flickr30k1K_claim=True,
        source_OOD='Relative to projector training universe; pretraining exposure unestablished',
        final_translation_summary_sha256=sha(OUT/'translation_summary.json'))
    write(OUT/'dataset_decision.json',decision)
    registration=dict(read(OUT/'model_registration.json'),status='closed_ready_for_execution',datasets=counts,
        prompt_modes=['v2_no_instructions','author_recommended'],
        freeze_policy='No model/dataset additions/removals after first score',
        bootstrap=read(OUT/'evaluation_preregistration.json')['bootstrap'],
        dataset_decision_sha256=sha(OUT/'dataset_decision.json'))
    write(OUT/'execution_registration.json',registration)
    outputs += [common.artifact('evaluation_registration',OUT/'execution_registration.json'),
                common.artifact('dataset_decision',OUT/'dataset_decision.json'),
                common.artifact('overlap_audit',OUT/'candidate_overlap_audit.json'),
                common.artifact('translations',translation_path),common.artifact('translation_summary',OUT/'translation_summary.json'),
                common.artifact('translation_decision',OUT/'translation_decision.json'),
                common.artifact('Flickr_translation_QA',flickr_qa_path),
                common.artifact('Flickr_translation_QA_summary',OUT/'flickr_translation_QA_summary.json'),
                common.artifact('source_metadata',OUT/'source_metadata.json'),
                common.artifact('source_splits',OUT/'deep_research_v2.json'),
                common.artifact('originals_Flickr30k',ASSETS/'flickr30k/test.parquet'),
                common.artifact('originals_VizWiz_index',ASSETS/'vizwiz/candidate_images.json')]
    marker=dict(schema_version=1,study_id='data-ood-20261030',generation=1,owner='B',phase='B_EVAL_FROZEN',status='complete',
        contract_sha256=sha(COORD/'COORDINATION.md'),artifacts=outputs,
        A_UNIVERSE_FROZEN_sha256=sha(COORD/'A_UNIVERSE_FROZEN.json'),
        A_FINALISTS_FROZEN_sha256=sha(COORD/'A_FINALISTS_FROZEN.json'),
        dependencies={'A_UNIVERSE_FROZEN':sha(COORD/'A_UNIVERSE_FROZEN.json')})
    common.atomic_json(COORD/'B_EVAL_FROZEN.json',marker,immutable=True)
    common.reserve('B','freeze-data',{'C':0,'D':0},committed_bytes=committed_bytes())
    print('B_EVAL_FROZEN',json.dumps(counts))


if __name__=='__main__':
    main()
