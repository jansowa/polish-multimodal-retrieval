"""Export compact comparison tables and complete epoch diagnostics, without evaluation."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pl_mmb_training.common import ROOT, read_json, write_json
from pl_mmb_training.tinyclip_adaptation import OUT


def export():
    results = {p.parent.name: read_json(p) for p in sorted(OUT.glob('*/result.json')) if p.parent.name != 'finalist_evaluation'}
    finalist_path = OUT / 'finalist_evaluation/evaluation.json'
    legacy_path = OUT / 'finalist_evaluation/result.json'
    if legacy_path.exists() and not finalist_path.exists():
        legacy_path.rename(finalist_path)
    finalist = read_json(finalist_path) if finalist_path.exists() else None
    rows, epochs = [], []
    names = ['A1', 'A2', 'B1', 'B2', 'C0', 'C1', 'C2', 'D_LoRA', 'E_KD_0.1', 'E_KD_0.3', 'F_TinyCLIP61']
    for name in names:
        source = name
        if name in ('B1', 'C0'):
            p = OUT / f'{name}_reuse.json'
            if not p.exists():
                continue
            source = read_json(p)['source']
        if source not in results:
            continue
        r = results[source]
        v = r['selected_validation']
        row = dict(experiment=name, reused_from=source if source != name else '',
                   feature_source=r['feature'], projector='linear' if not r['hidden'] else f'MLP {r["hidden"]} GELU',
                   vision_trainable_blocks=r['vision_blocks'], lora=r.get('lora', False),
                   trainable_params=r['trainable_params'], val_t2i_r1=v['t2i']['recall@1'],
                   val_t2i_r5=v['t2i']['recall@5'], val_t2i_r10=v['t2i']['recall@10'],
                   val_i2t_r1=v['i2t']['recall@1'], best_epoch=r['best_epoch'],
                   peak_vram_gib=r['peak_vram_bytes'] / 2**30, test_mscoco_r1='')
        rows.append(row)
        if finalist and source == finalist['selection']['winner']:
            row['test_mscoco_r1'] = finalist['local_mscoco']['recall@1']
        if source != name:
            continue
        for e in r['history']:
            epochs.append(dict(experiment=name, epoch=e['epoch'], train_loss=e['train_loss'],
                               val_loss=e['validation_loss'], train_gold_loss=e.get('train_gold_loss'),
                               train_kd_loss=e.get('train_kd_loss'), val_kd_loss=e.get('validation_kd_loss'),
                               val_t2i_r1=e['validation']['t2i']['recall@1'],
                               val_t2i_r5=e['validation']['t2i']['recall@5'],
                               val_t2i_r10=e['validation']['t2i']['recall@10'],
                               val_i2t_r1=e['validation']['i2t']['recall@1'],
                               gradient_norm_mean=e['gradient_norm_mean'], gradient_norm_max=e.get('gradient_norm_max'),
                               peak_vram_gib=e['peak_vram_bytes'] / 2**30, trainable_params=e['trainable_params']))
    for filename, values in [('tinyclip_adaptation.csv', rows), ('tinyclip_adaptation_loss.csv', epochs)]:
        if values:
            with (ROOT / 'reports' / filename).open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    lines = ['# TinyCLIP adaptation — FULL_CLEAN, seed 42', '',
             'Fixed 70,289 train rows / 7,695 validation rows; unchanged snapshot and validation retrieval gallery.',
             'Selection: validation t2i R@1, then mean t2i/i2t R@1, then earliest epoch. XM3600 remains blind.', '',
             '| experiment | feature source | projector | vision trainable blocks | trainable params | val t2i R@1 | val i2t R@1 | best epoch | test MSCOCO |',
             '|---|---|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        blocks = f'{r["vision_trainable_blocks"]} (LoRA)' if r['lora'] else str(r['vision_trainable_blocks'])
        test = '—' if r['test_mscoco_r1'] == '' else f'{100*r["test_mscoco_r1"]:.2f}%'
        lines.append(f'| {r["experiment"]} | {r["feature_source"]} | {r["projector"]} | {blocks} | '
                     f'{r["trainable_params"]:,} | {100*r["val_t2i_r1"]:.2f}% | {100*r["val_i2t_r1"]:.2f}% | '
                     f'{r["best_epoch"]} | {test} |')
    lines += ['', 'B1 and C0 reuse the corresponding completed run exactly. Trainable params include sigmoid scale/bias.',
              'A2 is native pooled CLS after final native LayerNorm, before CLIP visual_projection.',
              'Projector LR 0.001; C vision LR 1e-5. D uses rank-8, alpha-16 q/v LoRA in the final two blocks, LR 1e-4.',
              'E uses frozen SigLIP2 Base/16 256 image/EN-caption similarity distributions, symmetric KL at temperature 0.07;',
              'student uses image/PL-caption similarities. No raw embedding regression. Lambda 0.1 and 0.3 only.',
              'F uses the same best frozen projector recipe, with native projected TinyCLIP-61M image features.',
              'Peak VRAM is peak PyTorch allocated CUDA memory; full losses, retrieval and gradients are in the CSV/JSON logs.', '']
    if finalist:
        lines += [f'Finalist selected before test: **{finalist["selection"]["winner"]}**.',
                  f'Local MSCOCO t2i R@1: **{100*finalist["local_mscoco"]["recall@1"]:.2f}%**. '
                  f'Historical FULL_CLEAN frozen reference: {100*finalist["historical_frozen_reference"]["recall@1"]:.2f}%.',
                  'This is the existing local MSCOCO protocol, not independent cross-dataset validation.',
                  'All configurations have one seed. The small KD gain needs replication before a strong conclusion.', '']
    write_json(ROOT / 'reports/tinyclip_adaptation_summary.json', dict(rows=rows, selection_uses_test=False, xm3600_used=False))
    (ROOT / 'reports/tinyclip_adaptation.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    export()
