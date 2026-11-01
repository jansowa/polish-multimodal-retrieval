"""Import immutable full-gallery artifacts; no model loads, scores or downloads."""
import copy
from pathlib import Path
from .core import HERE, ROOT, CATALOG, identity, local_path, parameters, public_check, read, sha, write

OWN_PARAMETERS = dict(vision=38587392, text=31883136, head=196992, shared=0)


def default_paths():
    base = 'C:/Users/jsowa/benchmark-translations'
    return {
        'series_results': base + '/series-2026-10-23/evaluation',
        'ood_results': base + '/data-ood-20261030/B/results',
        'flickr_parquet': base + '/data-ood-20261030/B/flickr30k/test.parquet',
        'ood_frozen': base + '/data-ood-20261030/B/frozen',
        'data_study_results': base + '/data-ood-20261030/A/evaluation',
        'output': base + '/retrieval-benchmark-v1/runs',
        'datasets': {}
    }


def configuration(path=None):
    result = default_paths()
    if path:
        result.update(read(path))
    return result


def sync(config, output=HERE / 'space/results.json'):
    catalog = read(CATALOG)
    doc = dict(schema_version=1, suite=catalog['version'], protocol_version='full-gallery-v2',
               datasets=copy.deepcopy(catalog['datasets']), models={}, results=[], comparisons=[],
               notes=['Scores are fractions; UI shows percentages.',
                      'UI macro-average requires complete selected tasks, equally weighted; task scores average t2i/i2t. XTD full excluded by default to avoid duplicate weighting.',
                      'Exposure concerns projector training only; pretraining exposure is unestablished.',
                      'Imported historical results; no new model evaluation performed by sync.'])
    series = local_path(config['series_results'])
    ood = local_path(config['ood_results'])
    b = read(ROOT / 'reports/ood_eval_20261030/summary.json')
    registration = read(ROOT / 'reports/series_2026-10-23/reference_eval_registration_v3_20261029.json')
    references = {r['alias']: r for r in registration['reference_models']}
    by_checkpoint = {r['sha256']: r for r in b['models']}
    efficiency = read(ROOT / 'reports/series_2026-10-23/reference_new_gallery_comparisons_20261029.json')['efficiency']
    sources, index, aliases = [], {}, {}

    def add(path, origin, equivalent_prompt_mode=None):
        result = read(path)
        if equivalent_prompt_mode:
            if not result.get('checkpoint_sha256') or result['prompt_mode'] != 'v2_no_instructions':
                raise ValueError('Prompt equivalence applies only to our intrinsic-prefix checkpoints')
            result = dict(result, prompt_mode=equivalent_prompt_mode)
        dataset = result.get('dataset_id')
        if dataset not in doc['datasets'] or doc['datasets'][dataset]['status'] != 'ready':
            return
        checksum = result.get('checkpoint_sha256')
        alias = result.get('model_alias', result.get('model'))
        if checksum:
            model_id = 'checkpoint:' + checksum
            item = by_checkpoint.get(checksum, {})
            label = item.get('id', alias)
            role = 'diagnostic' if item.get('variant') == 'R3' or label in ('VN_only', 'historical_A1') else 'project'
            p = OWN_PARAMETERS
            eff = efficiency.get('our_finalist') if checksum == '4bf977049098ca6d8bcc7abc177183aace8c03c5f08e0ecaf011d82aad5b6844' else None
            model = dict(label=label, repository=None, revision=checksum, parameters=parameters(p), role=role,
                         representative=item.get('representative', label not in ('VN_FULL_seed42', 'VN_FULL_seed43')),
                         seed=item.get('seed'), trained_on_wiki=item.get('trained_on_Wiki', False),
                         trained_on_coco=label != 'VN_only', trained_on_visualnews=label != 'historical_A1',
                         encoder_revisions=item.get('encoder_revisions', {}), efficiency=eff)
        else:
            item = references.get(alias)
            if not item:
                raise ValueError('Unregistered reference: ' + str(alias))
            repo, revision = item['model'], result['revision']
            if revision != item['revision']:
                raise ValueError('Reference revision differs: ' + alias)
            model_id = 'hf:' + repo + '@' + revision
            eff = efficiency[alias]
            model = dict(label=alias, repository=repo, revision=revision, parameters=parameters(eff['parameters']),
                         role='english_control' if item.get('english_text_only_control') or alias in ('mobileclip2-s2', 'pe-core-b16') else 'reference',
                         representative=True, efficiency=eff)
        aliases[alias] = model_id
        if model_id not in doc['models']:
            doc['models'][model_id] = model
        exposure = 'unknown'
        if checksum:
            exposure = (('ID / development' if model['trained_on_visualnews'] else 'source-OOD to projector') if dataset == 'VN_dev' else
                        ('ID / development' if model['trained_on_wiki'] else 'development / source-OOD') if dataset == 'wiki_native_dev' else
                        ('ID source COCO' if model['trained_on_coco'] else 'source-OOD to projector') if dataset.startswith('xtd10') else 'source-OOD to projector')
        fingerprint = {k: result[k] for k in ('manifest_sha256', 'gallery_sha256', 'text_gallery_sha256', 'qrels_sha256')}
        row = dict(model_id=model_id, dataset_id=dataset, prompt_mode=result['prompt_mode'],
                   profile='historical-native-v2', protocol=result['protocol'], **fingerprint,
                   metrics={d: result['directions'][d]['metrics'] for d in ('t2i', 'i2t')},
                   query_counts={d: len(result['directions'][d]['query_ids']) for d in ('t2i', 'i2t')},
                   precision=result.get('precision', 'fp32-weights-bf16-autocast-cache-fp32') if checksum else result['precision'], exposure=exposure, source_sha256=sha(path),
                   registration_sha256=result.get('registration_sha256'), source=origin)
        tok = result.get('tokenization')
        if tok:
            def compact(value):
                return {k: compact(v) if isinstance(v, dict) else v for k, v in value.items() if k != 'tokens'}
            row['tokenization'] = compact(tok)
        key = (model_id, dataset, row['prompt_mode'], row['profile'])
        if key in index:
            previous = index[key]
            if previous['metrics'] != row['metrics'] or any(previous[k] != v for k, v in fingerprint.items()):
                raise ValueError('Conflicting results; do not silently pick the best: ' + str(path))
            return
        index[key] = row
        doc['results'].append(row)
        sources.append(dict(path=str(path), sha256=sha(path), model_id=model_id, dataset_id=dataset,
                            prompt_mode=row['prompt_mode'], profile=row['profile'],
                            equivalent_prompt_mode=equivalent_prompt_mode))

    for path in sorted(series.glob('*/*/*.json')):
        if path.stem in ('v2_no_instructions', 'author_recommended'):
            add(path, 'consultation-series-20261023')
    for path in sorted((series / 'references').glob('*/*/*.json')):
        if path.stem in ('v2_no_instructions', 'author_recommended'):
            add(path, 'consultation-series-20261023')
    for path in sorted(ood.glob('*/*/*.json')):
        if path.stem in ('v2_no_instructions', 'author_recommended'):
            add(path, 'ood-study-20261030')
    study = local_path(config['data_study_results'])
    # A's R0_seed44 was verified metric-for-metric against the original scorer.
    # Only already registered R1/R3 checkpoints are added; R2 was not selected.
    baseline = next(r for r in b['models'] if r['id']=='R0_seed44')
    baseline_id = 'checkpoint:' + baseline['sha256']
    doc['import_checks'] = []
    for name in ('VN_dev','wiki_native_dev','xtd10_full_1k','xtd10_no_overlap'):
        path = study/'R0'/'seed-44'/(name+'.json')
        original = read(path)
        previous = index[(baseline_id,name,'v2_no_instructions','historical-native-v2')]
        if any(previous[k]!=original[k] for k in ('manifest_sha256','gallery_sha256','text_gallery_sha256','qrels_sha256')) or any(
            previous['metrics'][d]!=original['directions'][d]['metrics'] for d in ('t2i','i2t')):
            raise ValueError('A-to-historical baseline parity failed before importing R1/R3')
        doc['import_checks'].append(dict(check='A_R0_historical_exact_parity',dataset=name,source_sha256=sha(path)))
    for variant in ('R1', 'R3'):
        for path in sorted((study / variant).glob('seed-*/*.json')):
            if read(path).get('checkpoint_sha256') not in by_checkpoint:
                raise ValueError('Unregistered data-study checkpoint')
            add(path, 'data-study-A-20261030')
            add(path, 'data-study-A-20261030:identical-intrinsic-query-prefix', 'author_recommended')
    # Future runs coexist with imported measurements under their own numerical profile.
    for path in sorted(local_path(config['output']).glob('*/result.json')):
        run = read(path)
        if run.get('status') != 'complete':
            continue
        model = doc['models'].setdefault(run['model_id'], run['model_metadata'])
        if model['parameters'] != run['model_metadata']['parameters']:
            raise ValueError('Parameter counts changed for the same model revision')
        row = run['public_result']
        doc['results'].append(row)
        sources.append(dict(path=str(path), sha256=sha(path), model_id=run['model_id'],
                            dataset_id=row['dataset_id'], prompt_mode=row['prompt_mode'], profile=row['profile']))
    for ci in b.get('conditional_CI', []):
        if ci['ours'] in aliases and ci['reference'] in aliases:
            allowed = ('dataset', 'direction', 'metrics', 'prompt_mode', 'replicates', 'seed', 'units', 'conditional_on')
            doc['comparisons'].append(dict(model_id=aliases[ci['ours']], reference_id=aliases[ci['reference']],
                                          **{k: ci[k] for k in allowed}))
    for name, item in doc['datasets'].items():
        matching = [r for r in doc['results'] if r['dataset_id'] == name]
        if matching:
            item['images'] = matching[0]['query_counts']['i2t']
            item['texts'] = matching[0]['query_counts']['t2i']
    license_path = HERE/'local/model_licenses.json'
    if license_path.is_file():
        licenses = read(license_path)['models']
        for model_id, model in doc['models'].items():
            if model_id in licenses:
                model['license'] = copy.deepcopy(licenses[model_id])
    public_check(doc)
    write(output, doc)
    write(HERE / 'local/source_index.json', dict(results_sha256=sha(output), artifacts=sources))
    return dict(models=len(doc['models']), results=len(doc['results']), comparisons=len(doc['comparisons']),
                ready_datasets=sum(d['status'] == 'ready' for d in doc['datasets'].values()), output=str(output))
