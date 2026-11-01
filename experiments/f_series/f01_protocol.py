"""Freeze/check F01-a artifacts without scoring, training, or changing registries.

WSL: PYTHONPATH=src ~/.venvs/pl-mmb-benchmark/bin/python \
    experiments/f_series/f01_protocol.py --write  # then --check
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import subprocess

from pl_mmb_training.current_holdouts import (
    Components, DEFAULT_REGISTRY, GUARD, Registry, canonical_path, digest, local_path,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('protocols/vizwiz-v3-f')
REPORT = Path('reports/f_series/S03-F01-a')
SOURCE = Path('docs/agent_spec_f/02_PROTOKOL_EWALUACJI.md')
SEED = 20261101


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def put(path, payload, write):
    path = ROOT / path
    if write and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    if not path.exists() or path.read_bytes() != payload:
        raise RuntimeError(f'Absent or changed frozen output: {path}')
    return digest(path)


def build():
    registry = Registry()
    rows = list(registry.rows())  # validates all registered SHA and pHash availability
    source_entry = next(x for x in registry.value['inputs'] if x['key'] == 'vizwiz_pl_local_validation')
    config = json.loads((ROOT / 'retrieval_benchmark/local/config.json').read_text())
    entry = config['datasets']['vizwiz_pl_local_validation']
    source_path = local_path(entry['protocol'])
    protocol = json.loads(source_path.read_text())
    assert digest(source_path) == entry['protocol_sha256']
    assert protocol['manifest_sha256'] == source_entry['sha256']
    assert digest(local_path(entry['image_manifest'])) == source_entry['sha256']
    protected = [Path(p) for p in (
        'src/pl_mmb_training/holdouts.py', 'src/pl_mmb_training/current_holdouts.py',
        'retrieval_benchmark/catalog.json', 'retrieval_benchmark/catalog-current.json',
        'retrieval_benchmark/holdouts/registry-v2.json', str(DEFAULT_REGISTRY.relative_to(ROOT)),
        str(GUARD.relative_to(ROOT)), str(SOURCE),
    )]
    baseline = {str(p): digest(ROOT / p) for p in protected}
    # Compare protected files to Git, not merely to an in-process snapshot.
    for path, sha in baseline.items():
        blob = subprocess.check_output(['git', 'show', 'HEAD:' + path], cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest() == sha, path
    graph = Components()
    viz_nodes = defaultdict(list)
    for row, label in rows:
        node = graph.add(row, [label])
        if label == 'holdout:vizwiz_pl_local_validation':
            viz_nodes[canonical_path(row['image'])].append(node)
    image_nodes = []
    existing = {}
    for image, component in zip(protocol['images'], protocol['image_components']):
        nodes = viz_nodes[canonical_path(image)]
        assert nodes, image  # image-only gallery members must also be protected
        node = nodes[0]
        for other in nodes[1:]:
            graph.join(node, other)
        if component in existing:
            graph.join(node, existing[component])
        existing[component] = node
        image_nodes.append(node)
    # Multi-positive qrels and source text components must stay together as well.
    by_query = defaultdict(list)
    for q, i in protocol['edges']:
        by_query[q].append(i)
    for indices in by_query.values():
        for i in indices[1:]:
            graph.join(image_nodes[indices[0]], image_nodes[i])
    text_components = {}
    for q, c in enumerate(protocol['text_components']):
        assert by_query[q], 'Query with no positive'
        node = image_nodes[by_query[q][0]]
        if c in existing:
            graph.join(node, existing[c])
        if c in text_components:
            graph.join(node, text_components[c])
        text_components[c] = node
    groups = defaultdict(list)
    for i, node in enumerate(image_nodes):
        groups[graph.find(node)].append(i)
    components = {}
    for indices in groups.values():
        cid = hashlib.sha256(encoded(sorted(protocol['image_ids'][i] for i in indices))).hexdigest()
        components[cid] = sorted(indices)
    order = sorted(components)
    random.Random(SEED).shuffle(order)
    halves = {'A': [], 'B': []}
    membership = {}
    for cid in order:
        half = min(halves, key=lambda x: (len(halves[x]), x))
        halves[half].extend(components[cid])
        membership[cid] = half
    image_component = {i: cid for cid, indices in components.items() for i in indices}
    query_half = {q: membership[image_component[indices[0]]] for q, indices in by_query.items()}
    artifacts = {}
    stats = {}
    for half, indices in halves.items():
        indices.sort()
        queries = [q for q in range(len(protocol['texts'])) if query_half[q] == half]
        imap = {i: n for n, i in enumerate(indices)}
        qmap = {q: n for n, q in enumerate(queries)}
        edges = [[qmap[q], imap[i]] for q, i in protocol['edges'] if q in qmap]
        output = {k: protocol[k] for k in ('canonicalization', 'manifest_sha256', 'status')}
        output.update(id=f'vizwiz_pl_{half.lower()}', parent_protocol_sha256=digest(source_path),
                      split_seed=SEED, source_image_indices=indices, source_query_indices=queries,
                      edges=edges)
        for key in ('images', 'image_ids'):
            output[key] = [protocol[key][i] for i in indices]
        for key in ('texts', 'query_ids', 'original_caption_ids'):
            output[key] = [protocol[key][q] for q in queries]
        output['image_components'] = [image_component[i] for i in indices]
        output['text_components'] = [image_component[by_query[q][0]] for q in queries]
        artifacts[str(OUT / f'{half.lower()}.json')] = encoded(output)
        stats[half] = dict(images=len(indices), queries=len(queries), edges=len(edges),
                           components=len(set(output['image_components'])))
    assignments = dict(schema=1, seed=SEED, algorithm='sorted-component-SHA256; Python Random.shuffle; greedy image balance; ties A',
                       graph_scope='all registry-v4 rows plus source protocol image/text components and qrels',
                       registry_sha256=registry.sha256, source_manifest=source_entry,
                       source_protocol=dict(path=entry['protocol'], sha256=digest(source_path)),
                       components=[dict(id=cid, half=membership[cid], image_indices=components[cid]) for cid in sorted(components)])
    artifacts[str(OUT / 'assignments.json')] = encoded(assignments)
    catalog = json.loads((ROOT / 'retrieval_benchmark/catalog-current.json').read_text())
    catalog['version'] = 'retrieval-pl-v3-f'
    catalog['parent_catalog_sha256'] = baseline['retrieval_benchmark/catalog-current.json']
    catalog['holdout_registry'] = dict(path=str(DEFAULT_REGISTRY.relative_to(ROOT)), sha256=registry.sha256,
                                      scope='Both VizWiz halves are subsets of the already protected complete parent; v4 unchanged')
    roles = {'wiki_native_dev': 'SEL-ID', 'VN_dev': 'SEL-OOD', 'flickr30k_pl_local_test': 'GUARD',
             'xm3600_pl': 'GUARD', 'xtd10_no_overlap': 'GUARD', 'xtd10_full_1k': 'REF',
             'vizwiz_pl_local_validation': 'REF'}
    for key, role in roles.items():
        catalog['datasets'][key]['f_role'] = role
    catalog['datasets']['vizwiz_pl_local_validation']['f_notes'] = 'Historical full gallery only; use A/B for F selection/guard. No new full-gallery scores for selection.'
    for half in halves:
        path = str(OUT / f'{half.lower()}.json')
        catalog['datasets'][f'vizwiz_pl_{half.lower()}'] = dict(
            title=f'VizWiz PL {half}', role='selection' if half == 'A' else 'guard',
            f_role='SEL-OOD' if half == 'A' else 'GUARD', status='ready', caption_origin='machine_translation',
            parent='vizwiz_pl_local_validation', protocol=path,
            protocol_sha256=hashlib.sha256(artifacts[path]).hexdigest(), counts=stats[half],
            assignments=str(OUT / 'assignments.json'),
            assignments_sha256=hashlib.sha256(artifacts[str(OUT / 'assignments.json')]).hexdigest())
    catalog['datasets']['coco_val'] = dict(title='COCO val', f_role='SEL-OOD', role='selection', status='pending_f01_panel',
                                          notes='Validation IDs only; setup/evaluation belongs to later F01 stages, not a new training release.')
    catalog['datasets']['compositional_validation'] = dict(title='ARO/SugarCrepe validation', f_role='GUARD', role='control',
                                                          status='pending_f01_panel', notes='Validation IDs only; never the complete training corpora.')
    catalog['future_roles'] = {'wiki_f_dev': 'SEL-ID after F05', 'new_holdout_dev_queries': 'SEL-OOD after F10',
                               'new_holdout_dev_sentences': 'SEL-OOD after F10', 'new_holdout_test': 'TEST after F10/H4'}
    artifacts['retrieval_benchmark/catalog-f-v3.json'] = encoded(catalog)
    catalog_sha = hashlib.sha256(artifacts['retrieval_benchmark/catalog-f-v3.json']).hexdigest()
    normative = (ROOT / SOURCE).read_text(encoding='utf-8')
    md = ('# Zamrożony protokół v3-F — 2026-11-01\n\n'
          'Źródło normatywne: `' + str(SOURCE) + '`; SHA256 `' + digest(ROOT / SOURCE) + '`.\n'
          'Katalog: `retrieval_benchmark/catalog-f-v3.json`; SHA256 `' + catalog_sha + '`.\n'
          'Bieżąca ochrona treningowa: registry-v4; SHA256 `' + registry.sha256 + '`.\n'
          'Połówki i przydziały: `protocols/vizwiz-v3-f/`; SHA w katalogu i `SHA256SUMS.json`.\n'
          'SHA tego pliku i JSON: `protocols/protocol-v3-f.SHA256SUMS.json` (wiąż oba z configiem runu).\n'
          'Commit zamrożenia identyfikuje Git; dowód w raporcie S03-F01-a. '
          'Brak wyników serii F i nowych dopuszczeń treningowych.\n'
          'COCO i panel kompozycyjny: konfiguracja w dalszych etapach F01; '
          'Wiki-F i nowy holdout: przyszłe pakiety F05/F10.\n\n' + normative)
    artifacts['protocols/protocol-v3-f.md'] = md.encode()
    sections = {}
    for section in normative.split('\n## ')[1:]:
        heading, body = section.split('\n', 1)
        sections[heading] = body.strip()
    specification = dict(
        schema=1, version='v3-f', frozen_date='2026-11-01', series='F',
        normative_source=dict(path=str(SOURCE), sha256=digest(ROOT / SOURCE)),
        normative_markdown=dict(path='protocols/protocol-v3-f.md', sha256=hashlib.sha256(artifacts['protocols/protocol-v3-f.md']).hexdigest()),
        normative_sections=sections,
        sha256_binding='Detached full-file SHA256 in protocols/protocol-v3-f.SHA256SUMS.json; pin MD and JSON plus catalog in every run config',
        catalog=dict(path='retrieval_benchmark/catalog-f-v3.json', sha256=catalog_sha),
        holdout_registry=catalog['holdout_registry'],
        licensing=dict(default_tier='WEIGHTS', H1_status='pending', STRICT='requires H1',
                       new_training_release=False, publish_requires='H6', new_NC_REF_evaluation_requires='H1'),
        roles={role: [k for k, v in catalog['datasets'].items() if v.get('f_role') == role]
               for role in ('SEL-ID', 'SEL-OOD', 'GUARD', 'REF')},
        future_roles=catalog['future_roles'],
        selection=dict(direction='t2i', metric='R@1', formula='0.5 * mean(SEL-ID) + 0.5 * mean(SEL-OOD)',
                       parallel_metric='SEL_i2t', units='percentage points', tie_abs_delta_lt_pp=0.3,
                       tie_break_order=['SEL_i2t', 'simpler_variant', 'earlier_checkpoint'],
                       SEL_ID_tie_until_Wiki_F_dev_abs_delta_lt_pp=2.0,
                       expansions='Between experiments only, decision record, controls recomputed; holdout-dev queries/sentences are separate OOD columns',
                       choices_on_SEL_only=['checkpoint', 'alpha', 'head', 'bridge', 'filter', 'threshold', 'parameter_allocation']),
        scoring=dict(implementation='src/pl_mmb_benchmark/full_gallery.py', precision='FP32',
                     candidates='full gallery of each named subset', multi_positive_qrels=True,
                     directions=['t2i', 'i2t'], metrics=['R@1', 'R@5', 'R@10', 'MRR', 'nDCG@10'],
                     prompt_mode='v2_no_instructions', historical_protocols_separate=True,
                     compositional=dict(validation_pairs_only=True, ties=0.5, blind_baseline_percent=50,
                                        columns=['relation', 'attribution', 'SugarCrepe'], independent_test=False)),
        statistics=dict(bootstrap=dict(samples=10000, seed=20261023, unit='whole_component', confidence=0.95,
                                        conditional_on_recipe=True), screening_seed=42, confirmation_seeds=[42, 43, 44],
                        report='mean +/- SD across seeds and bootstrap per seed; no seed selection',
                        hubness=['N10 skewness', 'fraction of text hubs'], truncation_tokens=128,
                        truncation_report='every dataset'),
        identity=dict(phase='F', model='OPI-PIB/PolDense-32M', revision='ca70494a25dad90d980db7c735658dc7bf88ed25',
                      probe_path='protocols/poldense_identity_probe_v1.jsonl', probe_texts=1000,
                      reference_path='protocols/poldense_identity_reference_v1.pt', precision='FP32', batches=[1, 32],
                      same_precision_and_batch_max_abs_diff=0, other_precision_or_batch_tolerance=1e-5,
                      tensor_sha_equal_required=True, failure_invalidates_run=True,
                      status='probe/reference implementation in F01-b; not created by F01-a'),
        success=dict(export_parameters_lt=100000000, identity_pass_required=True, licensing_H1_tier_required=True,
                     reference='F-base R1, all three seeds, same protocol',
                     test=dict(types=['short_queries', 'sentences'], both_t2i_delta_ge_pp=0,
                               at_least_one_t2i_delta_ge_pp=3.0, delta_CI_lower_gt=0,
                               i2t_delta_ge_pp=-1.0),
                     guardrails_pp={'xm3600_pl': {'t2i': -1.0, 'i2t': -1.0},
                                    'flickr30k_pl_local_test': -1.5, 'vizwiz_pl_b': -1.5,
                                    'xtd10_no_overlap': -2.0, 'compositional_validation': -2.0},
                     phase_M_text_retention=dict(mean_ndcg10_delta_ge_pp=-0.5, every_task_delta_ge_pp=-1.0),
                     external_REF_levels=dict(required_to_pass=False,
                         A='MetaCLIP-2 S16 PL 32768 on XM t2i and >=3/5 Flickr/VizWiz/Wiki/VN/XTD897; historical full-gallery comparisons separate',
                         B='MetaCLIP-2 M16 PL 32768 on XM t2i'),
                     latency=dict(owner_budget_before='FE3', default_CPU_image_ms_le=154, default_CPU_text_ms_le=29)),
        efficiency=dict(implementation='src/pl_mmb_benchmark/efficiency.py', same_machine=True,
                        CPU=dict(precision='FP32', batch=1, threads=2), GPU=dict(precision='BF16', batch=64),
                        warmups=10, measurements=100, statistic='median',
                        parameters='unique tensors in exported state_dict; all inference components included'),
        read_limits=dict(GUARD_new_per_F_series=3, historical_GUARD_0=dict(date='2026-10-31', dataset='xm3600_pl', new_read=False),
                         GUARD_requires='three confirmed seeds plus F-base', TEST_total=2, TEST_requires='H4 for each read',
                         TEST_models_predeclared=True, log='reports/f_series/eval_reads_log.jsonl',
                         test_informs_recipe='TEST becomes dev; new independent annotated TEST required',
                         text_only_diagnostics='unlimited on SEL; forbidden on GUARD/TEST'),
        reporting=dict(directions_separate=True, datasets_separate=True, facts_hypotheses_interpretations_separate=True,
                       required_columns=['vision/text/head parameters', 'latency', 'license tier', 'manifest SHA256',
                                         'COMMERCIAL_CANDIDATE/COMMERCIAL_BASELINE/RESEARCH_ONLY/REF', 'PolDense identity'],
                       test_size=dict(min_recommended_queries=1500, preferred_queries=2400)),
    )
    artifacts['protocols/protocol-v3-f.json'] = encoded(specification)
    artifacts['protocols/protocol-v3-f.SHA256SUMS.json'] = encoded({
        path: hashlib.sha256(artifacts[path]).hexdigest()
        for path in ('protocols/protocol-v3-f.md', 'protocols/protocol-v3-f.json')})
    return artifacts, baseline, stats, registry, protocol


def validate(artifacts, source):
    a, b = [json.loads(artifacts[str(OUT / f'{h}.json')]) for h in ('a', 'b')]
    for field in ('source_image_indices', 'source_query_indices', 'image_components', 'text_components', 'image_ids', 'query_ids'):
        assert not set(a[field]) & set(b[field]), field
    assert sorted(a['source_image_indices'] + b['source_image_indices']) == list(range(len(source['images'])))
    assert sorted(a['source_query_indices'] + b['source_query_indices']) == list(range(len(source['texts'])))
    reconstructed = []
    for half in (a, b):
        for field, index in (('images', 'source_image_indices'), ('image_ids', 'source_image_indices'),
                             ('texts', 'source_query_indices'), ('query_ids', 'source_query_indices'),
                             ('original_caption_ids', 'source_query_indices')):
            assert half[field] == [source[field][i] for i in half[index]]
        reconstructed.extend((half['source_query_indices'][q], half['source_image_indices'][i]) for q, i in half['edges'])
    assert Counter(reconstructed) == Counter(map(tuple, source['edges']))
    assert abs(len(a['images']) - len(b['images'])) <= max(Counter(a['image_components'] + b['image_components']).values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    assert args.write != args.check
    artifacts, baseline, stats, registry, source = build()
    validate(artifacts, source)
    hashes = {p: put(p, payload, args.write) for p, payload in artifacts.items()}
    summary = dict(status='passed', registry_status='registry_verified', registry_sha256=registry.sha256,
                   protected_sha256=baseline, artifacts_sha256=hashes, halves=stats,
                   checks=['all registry inputs SHA and original-image pHash verified',
                           'full-registry transitive components; original protocol components retained',
                           'image/query pool and qrels multiset conserved exactly', 'component/image/query A-B intersections empty',
                           'deterministic regeneration byte-identical', 'protected inputs byte-identical to Git HEAD'],
                   source_counts=dict(images=len(source['images']), queries=len(source['texts']), edges=len(source['edges'])),
                   new_reads={'GUARD': 0, 'TEST': 0}, gpu_jobs=[])
    for path, sha in baseline.items():
        assert digest(ROOT / path) == sha
    split_summary = dict(summary, artifacts_sha256={p: sha for p, sha in hashes.items() if not p.startswith('protocols/protocol-v3-f')})
    put(REPORT / 'split-summary.json', encoded(split_summary), args.write)
    put(REPORT / 'summary.json', encoded(summary), args.write)
    print(json.dumps({'status': summary['status'], 'halves': stats, 'summary_sha256': digest(ROOT / REPORT / 'summary.json')}))


if __name__ == '__main__':
    main()
