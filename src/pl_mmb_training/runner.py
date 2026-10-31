import csv
import io
import json
import random
import statistics
from datetime import datetime, timezone

from .common import ROOT, TASKS, digest, file_sha, read_json, write_json
from .data import load_evaluation, prepare_aro, published_datasets
from .leakage import make_blacklist, LeakageGuard
from .baselines import reproduce, values_for
from .evaluation import evaluate_features, scores_table, delta


def bind_config(config, dest):
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / 'config.json'
    if path.exists() and read_json(path) != config:
        raise ValueError('Output directory belongs to another config; choose a new --output')
    write_json(path, config)


def source_snapshot():
    """Code changes must not silently reuse completed run identities."""
    files = [ROOT / 'train.py', *(ROOT / 'src/pl_mmb_training').glob('*.py'),
             *(ROOT / 'src/pl_mmb_benchmark').glob('*.py')]
    return {str(p.relative_to(ROOT)): file_sha(p) for p in sorted(files)}


def write_records(path, records):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in records), encoding='utf-8')


def records_at(dest):
    return [json.loads(line) for line in (dest / 'data.jsonl').read_text(encoding='utf-8').splitlines()]


def check_e0(config, dest):
    e0 = read_json(dest / 'e0.json')
    if e0['config_sha256'] != digest(config) or e0['status'] != 'complete':
        raise ValueError('E0 is incomplete or belongs to another config')
    for filename, sha in e0['artifacts'].items():
        if file_sha(dest / filename) != sha:
            raise ValueError(f'E0 artifact changed: {filename}')
    if published_datasets() != set(e0['published_datasets']):
        raise ValueError('Leaderboard dataset scope changed; build a new E0 snapshot before training')
    records = records_at(dest)
    guard = LeakageGuard(read_json(dest / 'blacklist.json'))
    if any(guard.reasons(r) for r in records):
        raise ValueError('Training manifest overlaps evaluation')
    return e0, records


def run_e0(config, dest, cache_root):
    bind_config(config, dest)
    datasets, provenance, english = load_evaluation()
    baselines = reproduce(config['baseline_models'], datasets, provenance)
    lineage = read_json(cache_root / 'coco_annotation_parents.json')
    blacklist = make_blacklist(datasets, english, lineage)
    blacklist['lineage_metadata'] = dict(coco=file_sha(cache_root / 'coco_annotation_parents.json'),
                                      visual_genome=file_sha(cache_root / 'visual_genome_image_data.json'))
    blacklist['sha256'] = digest({k: v for k, v in blacklist.items() if k != 'sha256'})
    write_json(dest / 'blacklist.json', blacklist)
    print(f'E0 blacklist: {len(blacklist["ids"])} queries, {len(blacklist["images"])} image paths', flush=True)
    if any('checkpoint' in source for source in config['data']['sources']):
        from .full_data import prepare_full
        records, audit = prepare_full(config, blacklist, cache_root, dest)
    else:
        records, audit = prepare_aro(config, blacklist, cache_root)
    write_records(dest / 'data.jsonl', records)
    write_json(dest / 'data_audit.json', audit)
    write_json(dest / 'eval_english.json', english)
    result = dict(stage='E0', status='complete', config_sha256=digest(config), evaluation=provenance,
                  benchmark_protocol=__import__('pl_mmb_benchmark.protocol', fromlist=['PROTOCOL']).PROTOCOL,
                  published_datasets=sorted(published_datasets()), baselines=baselines,
                  blacklist_sha256=blacklist['sha256'], cross_dataset_parent_groups=len(blacklist['cross_dataset_parents']),
                  cross_dataset_pixel_groups=len(blacklist['cross_dataset_pixels']),
                  train=audit['train'], validation=audit['validation'], removed=audit['rejected'],
                  artifacts={name: file_sha(dest / name) for name in ('blacklist.json', 'data.jsonl', 'data_audit.json', 'eval_english.json')})
    write_json(dest / 'e0.json', result)
    print(f'E0 complete: train={audit["train"]}, validation={audit["validation"]}, excluded={audit["rejected"]}', flush=True)
    return result


def run_cache(config, dest, cache_root):
    from .features import encode_cached
    e0, records = check_e0(config, dest)
    lock = read_json(dest / 'model_lock.json')
    if lock['config_sha256'] != digest(config):
        raise ValueError('Wrong model lock')
    datasets, provenance, english = load_evaluation()
    if provenance != e0['evaluation']:
        raise ValueError('Evaluation changed since E0')
    blacklist = read_json(dest / 'blacklist.json')
    for path, identity in blacklist['images'].items():
        if file_sha(path) != identity['sha256']:
            raise ValueError(f'Eval image changed since E0: {path}')
    for r in records:
        if file_sha(r['image']) != r['sha256']:
            raise ValueError('Training image changed since E0')
    images = list(dict.fromkeys([p for examples in datasets.values() for p in values_for(examples)[0]] + [r['image'] for r in records]))
    # Keep EN beside PL in both the manifest and offline text feature cache.
    texts = list(dict.fromkeys([t for examples in datasets.values() for t in values_for(examples)[1]]
                 + [r['text_pl'] for r in records] + [r['text_en'] for r in records] + list(english.values())))
    _, image_meta = encode_cached(images, 'images', config, lock, cache_root)
    _, text_meta = encode_cached(texts, 'texts', config, lock, cache_root)
    manifest = dict(config_sha256=digest(config), e0_sha256=file_sha(dest / 'e0.json'), model_lock=lock,
                    images=images, texts=texts, image_features=image_meta, text_features=text_meta)
    write_json(dest / 'cache.json', manifest)
    print('All frozen features are cached; encoder inference is finished.', flush=True)


def regression_loss(predicted, target):
    import torch.nn.functional as F
    return (1 - F.cosine_similarity(predicted, target, dim=-1)).mean()


def multi_positive_infonce(queries, targets, target_ids, temperature=0.07):
    """E2 building block: all occurrences of the target are positives, never negatives."""
    import torch
    import torch.nn.functional as F
    if temperature <= 0 or len(target_ids) != len(targets) or len(queries) != len(targets):
        raise ValueError('Invalid contrastive batch or temperature')
    logits = F.normalize(queries, dim=-1) @ F.normalize(targets, dim=-1).T / temperature
    # IDs may be arbitrary strings; canonical equivalence defines positives.
    groups = {value: i for i, value in enumerate(dict.fromkeys(target_ids))}
    ids = torch.tensor([groups[v] for v in target_ids], device=logits.device)
    mask = ids[:, None] == ids[None, :]
    return (torch.logsumexp(logits, dim=1) - torch.logsumexp(logits.masked_fill(~mask, -torch.inf), dim=1)).mean()


def run_e1(config, dest):
    import torch
    from .features import validate_features
    torch.set_num_threads(4)
    e0, records = check_e0(config, dest)
    manifest = read_json(dest / 'cache.json')
    if manifest['config_sha256'] != digest(config) or manifest['e0_sha256'] != file_sha(dest / 'e0.json'):
        raise ValueError('Feature cache belongs to another E0/config')
    tensors = []
    for kind, values in (('image', manifest['images']), ('text', manifest['texts'])):
        meta = manifest[kind + '_features']
        if file_sha(meta['path']) != meta['sha256']:
            raise ValueError('Feature cache checksum mismatch')
        tensors.append(validate_features(values, torch.load(meta['path'], weights_only=True, map_location='cpu')))
    image_features, text_features = tensors
    ii = {v: i for i, v in enumerate(manifest['images'])}
    ti = {v: i for i, v in enumerate(manifest['texts'])}
    datasets, provenance, _ = load_evaluation()
    if provenance != e0['evaluation']:
        raise ValueError('Evaluation changed since feature extraction')
    splits = {}
    for split in ('train', 'validation'):
        rows = [r for r in records if r['split'] == split]
        splits[split] = (image_features[[ii[r['image']] for r in rows]], text_features[[ti[r['text_pl']] for r in rows]])
    opt_config = config['optimization']
    all_results = []
    for seed in config['seeds']:
        run_dir = dest / f'E1-seed-{seed}'
        run_dir.mkdir(exist_ok=True)
        identity = dict(config=config, seed=seed, model_lock=manifest['model_lock'], source=source_snapshot(),
                        e0_sha256=file_sha(dest / 'e0.json'), cache_manifest_sha256=file_sha(dest / 'cache.json'))
        run_id = digest(identity)
        result_path = run_dir / 'result.json'
        if result_path.exists():
            result = read_json(result_path)
            if result['run_id'] != run_id:
                raise ValueError('Seed output belongs to another immutable run')
            all_results.append(result)
            continue
        write_json(run_dir / 'config.json', identity)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        projector = torch.nn.Linear(image_features.shape[1], text_features.shape[1])
        optimizer = torch.optim.AdamW(projector.parameters(), lr=opt_config['learning_rate'], weight_decay=opt_config['weight_decay'])
        generator = torch.Generator().manual_seed(seed)
        # The E0 control is this exact untrained initialization; its delta isolates training.
        def evaluate_projector():
            with torch.inference_mode():
                projected = projector(image_features)
            results, predictions = {}, {}
            for key in TASKS:
                results[key], predictions[key] = evaluate_features(datasets[key], manifest['images'], projected,
                                                                  manifest['texts'], text_features)
            return results, predictions
        initial_metrics, _ = evaluate_projector()
        initial_scores = scores_table(initial_metrics)
        best_loss, best_epoch, history = float('inf'), None, []
        train_x, train_t = splits['train']
        val_x, val_t = splits['validation']
        with torch.inference_mode():
            initial_train_loss = sum(
                float(regression_loss(projector(train_x[start:start + opt_config['batch_size']]),
                                      train_t[start:start + opt_config['batch_size']]))
                * len(train_x[start:start + opt_config['batch_size']])
                for start in range(0, len(train_x), opt_config['batch_size'])
            ) / len(train_x)
            initial_val_loss = float(regression_loss(projector(val_x), val_t))
        initial_loss = dict(epoch=0, train_loss=initial_train_loss, validation_loss=initial_val_loss)
        write_json(run_dir / 'e0_untrained.json', dict(metrics=initial_metrics, scores=initial_scores,
                                                     loss=initial_loss))
        (run_dir / 'epochs.jsonl').write_text(json.dumps(dict(seed=seed, **initial_loss)) + '\n', encoding='utf-8')
        print(f'E1 seed={seed} epoch=0/{opt_config["epochs"]} train={initial_train_loss:.5f} val={initial_val_loss:.5f}', flush=True)
        for epoch in range(1, opt_config['epochs'] + 1):
            projector.train()
            permutation = torch.randperm(len(train_x), generator=generator)
            total = 0.
            for offset in range(0, len(permutation), opt_config['batch_size']):
                indices = permutation[offset:offset + opt_config['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                loss = regression_loss(projector(train_x[indices]), train_t[indices])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(indices)
            projector.eval()
            with torch.inference_mode():
                val_loss = float(regression_loss(projector(val_x), val_t))
            history.append(dict(epoch=epoch, train_loss=total / len(train_x), validation_loss=val_loss))
            with (run_dir / 'epochs.jsonl').open('a', encoding='utf-8') as progress:
                progress.write(json.dumps(dict(seed=seed, **history[-1]), allow_nan=False) + '\n')
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                torch.save(dict(state_dict=projector.state_dict(), input_dim=image_features.shape[1],
                                output_dim=text_features.shape[1], run_id=run_id), run_dir / 'projector.pt')
            print(f'E1 seed={seed} epoch={epoch}/{opt_config["epochs"]} train={total/len(train_x):.5f} val={val_loss:.5f}', flush=True)
        checkpoint = torch.load(run_dir / 'projector.pt', weights_only=True)
        projector.load_state_dict(checkpoint['state_dict'])
        benchmark_metrics, predictions = evaluate_projector()
        scores = scores_table(benchmark_metrics)
        previous = initial_scores
        previous_label = 'E0 untrained projector (same seed)'
        if config.get('previous_run'):
            previous = read_json(ROOT / config['previous_run'])['scores']
            previous_label = config['previous_run']
        result = dict(stage='E1', seed=seed, run_id=run_id, config=config, metrics=benchmark_metrics,
                      benchmark_protocol=e0['benchmark_protocol'], evaluation=e0['evaluation'],
                      blacklist_sha256=e0['blacklist_sha256'],
                      scores=scores, delta=delta(scores, previous), previous=previous_label,
                      timestamp=datetime.now(timezone.utc).isoformat(), history=history, initial_loss=initial_loss,
                      checkpoint_epoch=best_epoch, validation_loss=best_loss,
                      train_pairs=len(train_x), validation_pairs=len(val_x),
                      trainable_parameters=sum(p.numel() for p in projector.parameters()),
                      frozen_text=True, frozen_vision=True, optimizer_inputs='offline_features_only',
                      fusion='raw-addition-before-final-l2 (diagnostic until E4)', full_dataset=True,
                      environment=dict(torch=torch.__version__, device='cpu', extraction_device='cuda',
                                       model_lock=manifest['model_lock']))
        for key in TASKS:
            write_records(run_dir / f'{key}.predictions.jsonl', predictions[key])
        write_json(result_path, result)
        all_results.append(result)
    aggregate = {k: statistics.mean(r['scores'][k] for r in all_results) for k in (*TASKS, 'AVG')}
    std = {k: statistics.stdev(r['scores'][k] for r in all_results) if len(all_results) > 1 else None for k in (*TASKS, 'AVG')}
    summary = dict(stage='E1', seeds=config['seeds'], scores=aggregate, std=std,
                   delta={k: statistics.mean(r['delta'][k] for r in all_results) for k in (*TASKS, 'AVG')},
                   runs=[str(dest / f'E1-seed-{r["seed"]}' / 'result.json') for r in all_results])
    write_json(dest / 'summary.json', summary)
    report(config, dest, e0, all_results, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def report(config, dest, e0, runs, summary):
    columns = (*TASKS, 'AVG')
    header = '| Experiment | CIRR | FashionIQ | MSCOCO_t2i | RefCOCO | AVG |\n|---|---:|---:|---:|---:|---:|\n'
    def row(label, scores, signed=False):
        return '| ' + label + ' | ' + ' | '.join(f'{scores[k]*100:{"+" if signed else ""}.2f}' for k in columns) + ' |\n'
    text = '# E0 + E1\n\nScores: Recall@1 (%); deltas: percentage points. AVG requires all four tasks.\n\n'
    text += 'Data selection and hyperparameters were fixed before evaluation. Checkpoints use only grouped training-validation loss.\n\n'
    text += header
    for b in e0['baselines']:
        text += row('E0 replay: ' + b['model'], b['scores'])
    for r in runs:
        initial = read_json(dest / f'E1-seed-{r["seed"]}' / 'e0_untrained.json')
        text += row(f'E0 untrained seed {r["seed"]}', initial['scores'])
        text += row(f'E1 seed {r["seed"]}', r['scores'])
        text += row('Δ vs ' + r['previous'], r['delta'], True)
    text += row('E1 mean', summary['scores'])
    if all(v is not None for v in summary['std'].values()):
        text += row('E1 std across seeds', summary['std'])
    text += row('E1 mean Δ vs previous', summary['delta'], True)
    text += '\nE1 trains only the image projector with positive-pair cosine regression. Composed queries use the existing raw sum as a diagnostic until E4.\n'
    text += f'\nTrain pairs: {e0["train"]}; validation: {e0["validation"]}; excluded during data preparation: {e0["removed"]}. See the per-source audit for leakage, missing images and missing positive mappings.\n'
    text += '\nARO is now training data for these checkpoints and must not be reported as an independent ARO evaluation. '
    text += 'The eval blacklist covers IDs, bytes, RGB pixels, English/Polish captions and VG→COCO parent IDs; '
    text += 'arbitrary transformed duplicates without recoverable source lineage are outside what exact hashes can certify.\n'
    (dest / 'comparison.md').write_text(text, encoding='utf-8')
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(['stage', 'seed', *columns, *['delta_' + k for k in columns]])
    for r in runs:
        writer.writerow(['E1', r['seed'], *[r['scores'][k] for k in columns], *[r['delta'][k] for k in columns]])
    (dest / 'comparison.csv').write_text(out.getvalue(), encoding='utf-8')
