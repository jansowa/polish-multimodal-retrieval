"""Read-only project facts for the standalone model-review brief; no model evaluation."""
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pl_mmb_training.common import ROOT, file_sha, read_json, write_json
from pl_mmb_training.leakage import normalized_text


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def verify_brief():
    path = ROOT / 'reports/project_consultation_brief.md'
    document = path.read_text(encoding='utf-8')
    snapshot = read_json(ROOT / 'reports/project_consultation_snapshot.json')
    inputs = read_json(ROOT / 'training/runs/tinyclip-adaptation/inputs.json')['inputs']
    for source, sha in inputs.items():
        assert file_sha(source) == sha
        if Path(source).name in ('FULL_CLEAN.jsonl', 'validation.jsonl', 'data.jsonl', 'blacklist.json'):
            assert sha in document, source
    assert read_json(ROOT / 'training/runs/tinyclip-adaptation/validation_graph.json')['sha256'] in document
    tables = [[s.strip().replace('**', '') for s in line.split('|')[1:-1]]
              for line in document.splitlines() if line.startswith('| ')]
    def percentage(value):
        return f'{100*value:.2f}'.replace('.', ',')
    aliases = {'D_LoRA': 'D LoRA', 'E_KD_0.1': 'E KD λ=0,1', 'E_KD_0.3': 'E KD λ=0,3',
               'F_TinyCLIP61': 'F TinyCLIP-61M', 'B1': 'B1 = A1', 'C0': 'C0 = A1'}
    for result in snapshot['adaptation']['rows']:
        label = aliases.get(result['experiment'], result['experiment'])
        cells = next(r for r in tables if r[0] == label)
        assert cells[4].replace(' ', '') == str(result['trainable_params']), label
        assert cells[5:8] == [percentage(result['val_t2i_r1']), percentage(result['val_i2t_r1']), str(result['best_epoch'])], label
    checked = []
    for result in snapshot['optimization_comparison']:
        if result['experiment'][0] not in ('S', 'T'):
            continue
        label = result['experiment'].split('_')[0]
        cells = next(r for r in tables if r[0].startswith(label + ' '))
        assert cells[7:10] == [percentage(result['val_t2i_r1']), percentage(result['val_i2t_r1']), str(result['best_epoch'])], label
        checked.append(result['experiment'])
    reference = (ROOT / 'reports/benchmark_twenty_models.md').read_text(encoding='utf-8')
    count = 0
    for line in reference.splitlines():
        if not line.startswith('| ') or line.startswith('| Model '):
            continue
        original = [s.strip() for s in line.split('|')[1:-1]]
        if len(original) != 9:
            continue
        cells = next(r for r in tables if len(r) == 6 and r[0].startswith(original[0]))
        assert cells[1] == original[1].replace('.', ','), original[0]
        assert cells[2:6] == [v.replace('.', ',') for v in original[4:8]], original[0]
        count += 1
    assert count == 20 and len(checked) == 10
    assert '\ufffd' not in document
    value = dict(status='passed', brief_sha256=file_sha(path),
                 snapshot_sha256=file_sha(ROOT / 'reports/project_consultation_snapshot.json'),
                 immutable_inputs_verified=True, adaptation_rows_verified=11,
                 optimization_rows_verified=len(checked), pretrained_benchmark_rows_verified=count,
                 no_training_or_evaluation_performed=True)
    write_json(ROOT / 'reports/project_consultation_verification.json', value)
    print(json.dumps(value, indent=2))


def main():
    if '--verify-brief' in sys.argv:
        verify_brief()
        return
    train = rows(ROOT / 'training/runs/clean-v1/FULL_CLEAN.jsonl')
    val = rows(ROOT / 'training/runs/clean-v1/validation.jsonl')
    audit = read_json(ROOT / 'training/runs/clean-v1/audit.json')
    assert len(train) == 70289 and len(val) == 7695
    for path, sha in read_json(ROOT / 'training/runs/tinyclip-adaptation/inputs.json')['inputs'].items():
        assert file_sha(path) == sha
    text_overlap = {}
    retrieval_text_overlap = {}
    for field in ('text_en', 'text_pl'):
        tr = {normalized_text(r[field]) for r in train}
        va = {normalized_text(r[field]) for r in val}
        common = tr & va
        text_overlap[field] = dict(unique_normalized_strings_shared=len(common),
            train_rows_with_shared_string=sum(normalized_text(r[field]) in common for r in train),
            validation_rows_with_shared_string=sum(normalized_text(r[field]) in common for r in val))
        retrieval_rows = [r for r in val if r['dataset'] == 'mbeir_mscoco']
        retrieval_common = tr & {normalized_text(r[field]) for r in retrieval_rows}
        retrieval_text_overlap[field] = dict(unique_normalized_strings_shared=len(retrieval_common),
            validation_retrieval_rows_with_shared_string=sum(normalized_text(r[field]) in retrieval_common for r in retrieval_rows))
    parents_t = {p for r in train for p in r['parent_ids']}
    parents_v = {p for r in val for p in r['parent_ids']}
    pixels_t = {r['target_id'] for r in train}
    pixels_v = {r['target_id'] for r in val}
    assert not parents_t & parents_v and not pixels_t & pixels_v
    compact = read_json(ROOT / 'training/runs/vision-four/common/evaluation_compact.json')
    evaluation = {}
    for task, examples in compact['datasets'].items():
        sizes = [len(r['images']) for r in examples]
        evaluation[task] = dict(queries=len(examples), candidate_pool_min=min(sizes), candidate_pool_max=max(sizes),
            candidate_pool_sizes=dict(Counter(sizes)), image_plus_text_queries=sum(r['query_image'] is not None for r in examples),
            candidate_image_paths=len({i for r in examples for i in r['images']}))
    image_manifest = read_json(ROOT / 'training/runs/e1-full/cache.json')
    locks = read_json(ROOT / 'training/runs/vision-commercial-five/model_lock.json')
    config_path = next(Path(p) for p in locks['tinyclip-39m']['files'] if p.endswith('/config.json'))
    clip = read_json(config_path)
    value = dict(observed_at=datetime.now(timezone.utc).isoformat(),
        document='project_consultation_brief.md',
        train=dict(records=len(train), unique_pairs=len({(r['target_id'], r['text_pl']) for r in train}),
                   unique_images=len(pixels_t), sources=dict(Counter(r['dataset'] for r in train))),
        validation=dict(records=len(val), unique_images=len(pixels_v), sources=dict(Counter(r['dataset'] for r in val))),
        train_validation_overlap=dict(parent_ids=len(parents_t & parents_v), canonical_pixels=len(pixels_t & pixels_v),
                                      normalized_caption_strings=text_overlap,
                                      primary_retrieval_normalized_caption_strings=retrieval_text_overlap),
        legacy_evaluation=evaluation,
        feature_manifest=dict(images=len(image_manifest['images']), texts=len(image_manifest['texts'])),
        tinyclip_vision_architecture=clip['vision_config'], tinyclip_projection_dim=clip['projection_dim'],
        cleanroom=audit['variants'],
        adaptation=read_json(ROOT / 'reports/tinyclip_adaptation_summary.json'),
        optimization=read_json(ROOT / 'reports/tinyclip_optimization_summary.json'),
        optimization_comparison=read_json(ROOT / 'reports/tinyclip_optimization.json')['comparison'],
        visualnews_pipeline_local_status=read_json(ROOT / 'reports/visualnews_pipeline_state.json'),
        xm3600_preparation=read_json(ROOT / 'reports/xm3600_pl_preparation.json'),
        no_training_or_evaluation_performed=True)
    write_json(ROOT / 'reports/project_consultation_snapshot.json', value)
    print(json.dumps({k: value[k] for k in ('observed_at', 'train_validation_overlap', 'legacy_evaluation',
                                         'feature_manifest', 'tinyclip_vision_architecture')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
