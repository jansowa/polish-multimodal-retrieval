"""Refresh report metadata from frozen counts, without model/scoring reruns."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pl_mmb_benchmark.report_parameters import HEADERS, encoder_counts, parameter_cells
from pl_mmb_training.series_20261023 import ROOT, OUT, read, freeze, file_sha, local_path

FILES = [OUT / 'REFERENCE_COMPARISON_20261029.md', OUT / 'OUR_NEW_GALLERIES_20261029.md',
         OUT / 'REPORT.md', ROOT / 'reports/benchmark_twenty_models.md']
FOCUS = {'Model', 'Referencja', 'Seed 44 minus kontrola', 'Worker referencji'}
NOTE = ('Parametry podano w milionach: **Enkodery V+T = wizyjny + tekstowy**. '
        'Pozostałe enkodery oraz osobny projektor nie wchodzą do tej sumy. '
        'Nasz model: **70,47 mln = 38,59 mln wizja + 31,88 mln tekst**, '
        'projektor osobno **0,197 mln**. EmbeddingGemma2: **438,76 mln = '
        '167,76 mln wizja + 271,00 mln tekst**; enkoder audio był wyłączony.')


def cells(line):
    return [value.strip() for value in line.strip().strip('|').split('|')]


def line(values):
    return '| ' + ' | '.join(values) + ' |'


def update_tables(content, counts, resolve):
    lines, result = content.splitlines(), []
    i = tables = rows = 0
    while i < len(lines):
        header = cells(lines[i]) if lines[i].startswith('|') else []
        if not header or header[0] not in FOCUS:
            result.append(lines[i]); i += 1; continue
        old_parameter = {j for j, value in enumerate(header)
                         if value == 'Parametry [M]' or value.startswith('Vision / text / head')}
        kept = [j for j in range(len(header)) if j not in old_parameter]
        clean = [header[j] for j in kept]
        alignment = cells(lines[i + 1])
        if len(alignment) != len(header) or not all(set(v) <= {':', '-'} for v in alignment):
            raise ValueError('Expected a valid Markdown table alignment')
        result.append(line([clean[0], *HEADERS, *clean[1:]]))
        clean_align = [alignment[j] for j in kept]
        result.append(line([clean_align[0], '---:', '---:', '---:', *clean_align[1:]]))
        tables += 1; i += 2
        while i < len(lines) and lines[i].startswith('|'):
            before = cells(lines[i])
            if len(before) != len(header):
                raise ValueError('Table row width differs')
            alias = resolve(before[0])
            values = [before[j] for j in kept]
            parameters = parameter_cells(counts[alias]).split(' | ')
            after = [values[0], *parameters, *values[1:]]
            if [after[0], *after[4:]] != values:
                raise AssertionError('A result cell changed')
            result.append(line(after)); rows += 1; i += 1
    result[1:1] = ['', NOTE, '']
    return '\n'.join(result) + '\n', dict(tables=tables, rows=rows, nonparameter_cells_unchanged=True)


def preserve(path):
    target = path.with_name(path.stem + '_before_encoder_columns_20261030' + path.suffix)
    payload = path.read_bytes()
    if target.exists():
        if target.read_bytes() != payload:
            raise ValueError('Previous report backup differs')
    else:
        with target.open('xb') as stream:
            stream.write(payload)
    return dict(path=str(target.relative_to(ROOT)), sha256=file_sha(target))


def main():
    comparison_path = OUT / 'reference_new_gallery_comparisons_20261029.json'
    comparison_sha = file_sha(comparison_path)
    comparison = read(comparison_path)
    reg = read(OUT / 'reference_eval_registration_v3_20261029.json')
    counts = {alias: entry['parameters'] for alias, entry in comparison['efficiency'].items()}
    aliases = {m['model']: m['alias'] for m in reg['reference_models']}
    for alias in reg['our_checkpoints']:
        counts[alias] = counts['our_finalist']

    def resolve(value):
        value = value.strip('`* ')
        if value in counts:
            return value
        if value in aliases:
            return aliases[value]
        for model, alias in aliases.items():
            if value.startswith(model + ' + '):
                return alias
        raise ValueError('Unknown model in table: ' + value)

    gemma = read(local_path('/mnt/c/Users/jsowa/benchmark-translations/series-2026-10-23/evaluation/references/embeddinggemma-2/wiki_native_dev/v2_no_instructions.json'))
    if gemma['components']['audio_encoder'] is not False:
        raise ValueError('Gemma audio must be excluded')
    if encoder_counts(counts['embeddinggemma-2'])['total'] != gemma['parameters']:
        raise ValueError('Gemma active parameter total differs')
    rendered = {path: update_tables(path.read_text(encoding='utf-8'), counts, resolve) for path in FILES}
    freeze(OUT / 'parameter_table_update_decision_20261030.json', dict(
        status='registered_before_report_metadata_update', scoring_and_CI_unchanged=True,
        source_comparison_sha256=comparison_sha, script_sha256=file_sha(__file__),
        policy='sum of vision and text only; exclude other encoders, separate projector and scalar scoring parameters',
        gemma_audio_disabled=True, no_model_downloads=True, no_model_reruns=True,
        original_reports={str(p.relative_to(ROOT)): file_sha(p) for p in FILES}))
    backups = {str(path.relative_to(ROOT)): preserve(path) for path in [*FILES, OUT / 'summary.json']}
    for path, (content, _) in rendered.items():
        path.write_text(content, encoding='utf-8')
    policy = dict(status='complete', unit='parameters', display_unit='millions, two decimals',
        total_definition='distinct vision plus text encoder parameters; other modalities and standalone projector excluded',
        models={alias: encoder_counts(value) for alias, value in counts.items()},
        our_standalone_projector=counts['our_finalist']['head'], gemma_audio_disabled=True,
        source_comparison_sha256=comparison_sha,
        validation={str(path.relative_to(ROOT)): metadata for path, (_, metadata) in rendered.items()})
    freeze(OUT / 'encoder_parameter_reporting_20261030.json', policy)
    summary = read(OUT / 'summary.json')
    summary['encoder_parameter_reporting'] = policy
    summary['latest_report_update'] = 'report_completion_encoder_columns_20261030.json'
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    if file_sha(comparison_path) != comparison_sha:
        raise AssertionError('Frozen scientific comparison changed')
    updated = {str(path.relative_to(ROOT)): file_sha(path) for path in [*FILES, OUT / 'summary.json']}
    freeze(OUT / 'report_completion_encoder_columns_20261030.json', dict(status='complete',
        updated_reports_sha256=updated, previous_reports=backups,
        parameter_metadata_sha256=file_sha(OUT / 'encoder_parameter_reporting_20261030.json'),
        scientific_comparison_unchanged_sha256=comparison_sha, quality_and_CI_cells_unchanged=True))
    for name, value in policy['validation'].items():
        print(name, value)


if __name__ == '__main__':
    main()
