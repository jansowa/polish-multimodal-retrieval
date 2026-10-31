"""Close source study, reproduce VN selection, then score COCO without feedback."""
import csv
import json
import math
import sys
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import numpy as np
import torch
from pl_mmb_training import source_ablation as study
from pl_mmb_training.common import ROOT,digest,file_sha,read_json,write_json
from pl_mmb_training.losses import PairLoss
from pl_mmb_training.evaluation import evaluate_features
from pl_mmb_training.retrieval import evaluate_graph,graph_from_examples
from verify_tinyclip_generalization import components,bootstrap


def load_head(result):
    saved=torch.load(result['checkpoint_path'],weights_only=True,map_location='cpu')
    model=torch.nn.Linear(512,384).eval();model.load_state_dict(saved['projector'])
    return model,saved


def benchmark(data,results,registry):
    baseline=ROOT/'training/runs/e1-full'
    manifest=read_json(baseline/'cache.json')
    compact=read_json(ROOT/'training/runs/vision-four/common/evaluation_compact.json')
    assert compact['sha256']==digest(compact['datasets'])
    assert compact['feature_manifest_sha256']==file_sha(baseline/'cache.json')
    rows=compact['datasets']['mscoco_t2i']
    examples=[SimpleNamespace(id=r['id'],images=[manifest['images'][i] for i in r['images']],
        texts=[manifest['texts'][i] for i in r['texts']],
        query_image=manifest['images'][r['query_image']] if r['query_image'] is not None else None) for r in rows]
    assert len(examples)==1000 and all(e.query_image is None for e in examples)
    meta=read_json(ROOT/'training/runs/vision-commercial-five/models/tinyclip-39m/feature_cache.json')
    assert file_sha(meta['path'])==meta['sha256']
    features=torch.load(meta['path'],weights_only=True,map_location='cpu')
    assert features.shape==(len(manifest['images']),512)
    old_indices={p:i for i,p in enumerate(manifest['images'])}
    reference=features[[old_indices[p] for p in data.ctx.base.images]]
    minimum_cosine=float(torch.nn.functional.cosine_similarity(reference,data.native[:data.ctx.offset]).min())
    assert minimum_cosine>.999
    blacklist=read_json(baseline/'blacklist.json')
    graph=graph_from_examples(examples,blacklist,manifest['images'],manifest['texts'])
    values={}
    by_name={r['experiment']:r for r in results}
    for name in registry['finalists']:
        result=by_name[name];model,_=load_head(result)
        with torch.inference_mode():projected=model(features)
        metrics,predictions=evaluate_features(examples,manifest['images'],projected,manifest['texts'],data.ctx.base.texts)
        global_metrics=evaluate_graph(graph,projected,data.ctx.base.texts,include_ranks=True)
        values[name]=dict(local=metrics,fixed_gallery=global_metrics,predictions=predictions,
            no_known_COCO_training=result['no_known_COCO_training'],checkpoint_sha256=result['checkpoint_sha256'])
        (study.OUT/name/'mscoco_test_predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in predictions),encoding='utf-8')
        print(f'TEST_FINALIST {name} local_t2i_R1={metrics["recall@1"]:.5f}',flush=True)
    protocol=dict(name='Existing local MMEB MSCOCO_t2i and separately fixed whole gallery; not official COCO 5k',
        examples=len(examples),candidate_count_distribution=dict(Counter(len(e.images) for e in examples)),
        fixed_gallery_images=len(graph['images']),fixed_gallery_texts=len(graph['texts']),
        compact_sha256=file_sha(ROOT/'training/runs/vision-four/common/evaluation_compact.json'),
        native_feature_sha256=meta['sha256'],minimum_train_cache_feature_cosine=minimum_cosine,
        caveat='Existing legacy benchmark, inspected in earlier studies. No COCO score fed back into this new study.')
    return values,protocol


def compare(results,validation,dev,data):
    coco_groups=components(data.ctx.base)
    lookup={r['text_pl']:r['component'] for r in data.ctx.dev}
    keys=sorted({r['component'] for r in data.ctx.dev});index={v:i for i,v in enumerate(keys)}
    vn_groups=np.array([index[lookup[t]] for t in data.ctx.dev_graph['texts']])
    pairs=[('VN_'+code,'VN_only') for code in ('A','R','S','C','nonCOCO_A','nonCOCO_R','nonCOCO_AR')]
    pairs+=[('VN_FULL','VN_FULL_drop_'+code) for code in study.SOURCES]
    pairs+=[('VN_FULL','VN_only'),('FULL_only','VN_only')]
    return {a+'_minus_'+b:dict(
        VN_dev=bootstrap(dev[a]['ranks']['t2i'],dev[b]['ranks']['t2i'],vn_groups),
        COCO_val_posthoc=bootstrap(validation[a]['ranks']['t2i'],validation[b]['ranks']['t2i'],coco_groups)) for a,b in pairs}


def plots(results,validation,comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,6),sharey=True)
    names=[r['experiment'] for r in results];ys=np.arange(len(names))
    for ax,which,title in zip(axes,['VN_dev','COCO_val'],['Selection: VisualNews dev','COCO validation: post hoc only']):
        values=[100*(r['selected_VN_dev'] if which=='VN_dev' else validation[r['experiment']])['t2i']['recall@1'] for r in results]
        colors=['#009E73' if r['no_known_COCO_training'] else '#777777' for r in results]
        ax.barh(ys,values,color=colors)
        for y,value in zip(ys,values):ax.text(value+.5,y,f'{value:.2f}',va='center',fontsize=8)
        ax.set_xlim(0,100);ax.set_xlabel('t2i R@1 (%)');ax.set_title(title);ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(ys,names,fontsize=8);axes[0].invert_yaxis()
    fig.suptitle('Frozen encoders; random linear head; one seed; green = no known COCO-derived training')
    fig.tight_layout();fig.savefig(ROOT/'reports/tinyclip_source_ablation.png',dpi=170);plt.close(fig)
    chosen=['VN_'+c+'_minus_VN_only' for c in ('A','R','S','C','nonCOCO_A','nonCOCO_R','nonCOCO_AR')]
    chosen+=['VN_FULL_minus_VN_FULL_drop_'+c for c in study.SOURCES]
    labels=['Add '+c for c in ('A','R','S','C','nonCOCO_A','nonCOCO_R','nonCOCO_AR')]+['Keep '+c+' vs replace by VN' for c in study.SOURCES]
    fig,axes=plt.subplots(1,2,figsize=(13,6),sharey=True)
    for ax,split,title in zip(axes,['VN_dev','COCO_val_posthoc'],['VN dev delta','COCO val post hoc delta']):
        for y,name in enumerate(chosen):
            c=comparisons[name][split];v=c['delta_percentage_points'];lo,hi=c['interval_95_percentage_points']
            ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',color='#0072B2',capsize=3)
        ax.axvline(0,color='black',linewidth=.8);ax.set_title(title);ax.set_xlabel('t2i R@1 difference (percentage points)');ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(range(len(chosen)),labels,fontsize=8);axes[0].invert_yaxis()
    fig.suptitle('Conditional paired component-bootstrap 95% intervals; not training-seed uncertainty')
    fig.tight_layout();fig.savefig(ROOT/'reports/tinyclip_source_ablation_effects.png',dpi=170);plt.close(fig)


def export(data,results,validation,tests,protocol,comparisons,registry):
    value=dict(protocol='source-ablation-v1',status='complete',results=results,COCO_validation_posthoc=validation,
        COCO_finalists=tests,COCO_protocol=protocol,comparisons=comparisons,data_audit=data.audit,
        selection_registry=registry,COCO_used_for_selection=False,wiki_used=False,xm3600_used=False,
        caveats=['One training seed, conditional query-component intervals only.',
            'Checkpoint selection uses VN dev and may favor the VN domain. COCO val is reported only after all training/selection closed.',
            'Source additions and replacements jointly change image diversity, captions and batch-negative composition.',
            data.audit['caveat']])
    write_json(ROOT/'reports/tinyclip_source_ablation.json',value)
    lines=['# Ablacja źródeł i COCO bez danych COCO w naszym treningu','',
        'TinyCLIP-39M i PolDense frozen. Projector Linear 512→384 od losowych wag, świeży sigmoid loss; 196 994 trenowalne parametry. Seed 42.',
        'Każdy wariant: 5500 kroków i 1 405 780 prezentacji. LR 0.001, AdamW, native features bez augmentacji.',
        '**Checkpointy i finaliści wybrani wyłącznie po VN dev, przed oceną COCO.** Brak wiki i retrieval XM3600.','',
        'A = ARO Attribution, R = ARO Relation, S = SugarCrepe, C = M-BEIR MSCOCO.','',
        '| Wariant | Bez znanego COCO train | Train pary | Train obrazy | VN dev t2i R@1 | COCO val t2i R@1 — post hoc | COCO val i2t R@1 — post hoc | Best krok |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    csv_rows=[]
    for r in results:
        name=r['experiment'];v=validation[name];dev=r['selected_VN_dev']
        lines.append(f'| {name} | {"tak" if r["no_known_COCO_training"] else "nie"} | {r["train"]["rows"]:,} | {r["train"]["images"]:,} | {100*dev["t2i"]["recall@1"]:.2f}% | {100*v["t2i"]["recall@1"]:.2f}% | {100*v["i2t"]["recall@1"]:.2f}% | {r["best_updates"]} |')
        csv_rows.append(dict(experiment=name,feature_source='TinyCLIP original projected 512D',projector='Linear512to384',
            vision_trainable_blocks=0,trainable_params=196994,no_known_COCO_training=r['no_known_COCO_training'],
            train_rows=r['train']['rows'],train_images=r['train']['images'],steps=r['steps'],best_updates=r['best_updates'],
            VN_dev_t2i_R1=dev['t2i']['recall@1'],COCO_val_t2i_R1=v['t2i']['recall@1'],COCO_val_t2i_R5=v['t2i']['recall@5'],
            COCO_val_t2i_R10=v['t2i']['recall@10'],COCO_val_i2t_R1=v['i2t']['recall@1'],
            COCO_test_local_R1=tests.get(name,{}).get('local',{}).get('recall@1'),checkpoint_sha256=r['checkpoint_sha256']))
    with (ROOT/'reports/tinyclip_source_ablation.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    lines+=['','## Wybrani finaliści: benchmark COCO','',protocol['name'],
        f'Local: {protocol["examples"]} zapytań; liczby kandydatów: {protocol["candidate_count_distribution"]}. '
        f'Osobna pełna galeria: {protocol["fixed_gallery_images"]} obrazów / {protocol["fixed_gallery_texts"]} tekstów.','',
        '| Finalista | Bez znanego COCO train | Local R@1 | Local R@5 | Local R@10 | Fixed gallery t2i R@1 |',
        '|---|---|---:|---:|---:|---:|']
    for name,t in tests.items():
        lines.append(f'| {name} | {"tak" if t["no_known_COCO_training"] else "nie — kontrola"} | '+
            ' | '.join(f'{100*t["local"][f"recall@{k}"]:.2f}%' for k in (1,5,10))+f' | {100*t["fixed_gallery"]["t2i"]["recall@1"]:.2f}% |')
    lines+=['','## Wpływ źródeł','',
        'Add: 50% VN / 50% dodane źródło kontra samo VN. Keep: pełna mieszanka kontra zastąpienie slotów danego źródła przez VN; pozostałe oryginalne sloty identyczne.',
        'Przedziały są warunkowe: bootstrap grup zapytań (obrazy/artykuły/powiązania), bez niepewności seeda i selekcji checkpointu.','',
        '| Porównanie | VN dev Δ p.p. [95% CI] | COCO val post hoc Δ p.p. [95% CI] |','|---|---:|---:|']
    for name,comp in comparisons.items():
        vals=[]
        for split in ('VN_dev','COCO_val_posthoc'):
            c=comp[split];lo,hi=c['interval_95_percentage_points']
            vals.append(f'{c["delta_percentage_points"]:+.2f} [{lo:+.2f}, {hi:+.2f}]')
        lines.append(f'| {name} | '+ ' | '.join(vals)+' |')
    lines+=['','![Wyniki](tinyclip_source_ablation.png)','', '![Różnice](tinyclip_source_ablation_effects.png)','',
        '## Ograniczenia i pochodzenie','',
        '- Bez COCO dotyczy danych i wag projectora w naszym treningu. Oficjalna metadana VG, znane Flickr/COCO lineage oraz aliasy pliku/pikseli zostały sprawdzone. Nie jest to dowód braku nieznanych kopii ani braku COCO w pretreningu gotowych encoderów.',
        '- Wszystkie 24 473 wiersze / 3037 obrazów strict ARO mają jawny wpis VG bez COCO ID; recheck transytywnych powiązań nie zmniejszył tej puli.',
        '- VN dev to ten sam pipeline i dwa źródła newsowe co VN train, z modelowymi captionami. Wybór po VN dev może preferować VN kosztem COCO. Dane COCO nie służyły jednak żadnemu wyborowi w tej serii.',
        '- Efekt źródła obejmuje również zmianę liczby różnych obrazów, stylu i liczby captionów oraz negatywów w batchu. Spadek nie dowodzi, że źródło szkodzi przy każdej proporcji lub przy większym budżecie.',
        '- Jeden seed wystarcza do wstępnego screeningu dużych różnic; małe efekty wymagają powtórzeń. Przedziały nie korygują wielu porównań ani selekcji.',
        '- Wyniki tej serii nie są bezpośrednią ablają wcześniejszego warm-startu A1: zmieniono selekcję na VN dev, losowy start, LR i augmentację. Wszystkie warunki wewnątrz tej serii są wspólne.',
        '- Użyty benchmark COCO to dotychczasowy lokalny benchmark MMEB, oglądany w poprzednich badaniach. XM3600 pozostaje blind; wiki nie było ładowane.']
    (ROOT/'reports/tinyclip_source_ablation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    registry=read_json(study.OUT/'selection_registry.json')
    assert registry['status']=='all_training_closed_before_COCO_evaluation'
    report=read_json(ROOT/'reports/tinyclip_source_ablation.json');results=report['results']
    data=study.Data();assert len(results)==len(data.plans)==14
    initial_path,initial=study.initialize()
    checks={};dev={}
    for r in results:
        name=r['experiment'];config=read_json(study.OUT/name/'config.json')
        assert digest(config)==r['run_id'] and config==r['config']
        assert config['initial_sha256']==file_sha(initial_path) and config['initialization']==initial['initialized_from']
        for source,sha in config['source'].items():assert file_sha(ROOT/source)==sha
        assert registry['checkpoints'][name]['sha256']==r['checkpoint_sha256']==file_sha(r['checkpoint_path'])
        assert r['example_presentations']==1405780 and r['steps']==5500
        assert not r['COCO_used_for_selection'] and not r['wiki_used'] and not r['xm3600_used']
        indices=data.eligible(data.plans[name])
        assert digest(indices)==config['train_indices_sha256']
        assert all(i>=data.nbase or i in data.strict for i in indices)==r['no_known_COCO_training']
        chosen=max(r['history'],key=study.key);assert chosen['updates']==r['best_updates']
        assert chosen['VN_dev']==r['selected_VN_dev'] and r['history'][-1]['updates']==5500
        model,saved=load_head(r);model=model.to('cuda')
        assert saved['run_id']==r['run_id'] and saved['updates']==r['best_updates']
        assert saved['initialization_sha256']==config['initial_sha256']
        actual=data.retrieve(model,data.ctx.dev_graph)
        check=dict(actual);check.pop('ranks');assert check==r['selected_VN_dev']
        criterion=PairLoss('siglip_sigmoid').to('cuda');criterion.load_state_dict(saved['loss'])
        with torch.inference_mode():loss=sum(float(criterion(model(data.dev_x[p]),data.dev_t[p],mask))*len(p)
            for p,mask in data.dev_batches)/len(data.ctx.dev)
        assert abs(loss-r['best_validation_loss'])<2e-5
        assert all(math.isfinite(h['validation_loss']) for h in r['history'])
        dev[name]=actual;checks[name]=dict(source_verified=True,fresh_initialization_verified=True,
            VN_selection_and_loss_reproduced=True,train_ancestry_verified=True,full_budget_verified=True)
        print('VERIFIED_VN',name,flush=True)
    marker=study.OUT/'evaluation_started.json'
    marker_identity=dict(registry_sha256=file_sha(study.OUT/'selection_registry.json'),
        evaluator_sha256=file_sha(__file__),data=data.identity)
    if marker.exists():assert read_json(marker)['identity']==marker_identity
    else:write_json(marker,dict(identity=marker_identity,created_at=datetime.now(timezone.utc).isoformat(),
        all_training_closed=True,COCO_used_for_selection=False))
    validation={}
    for r in results:
        model,_=load_head(r);validation[r['experiment']]=data.retrieve(model.to('cuda'),data.ctx.base.graph)
        print('COCO_POSTHOC',r['experiment'],validation[r['experiment']]['t2i']['recall@1'],flush=True)
    tests,protocol=benchmark(data,results,registry)
    comparisons=compare(results,validation,dev,data)
    data.verify();export(data,results,validation,tests,protocol,comparisons,registry);plots(results,validation,comparisons)
    write_json(ROOT/'reports/tinyclip_source_ablation_verification.json',dict(status='passed',checks=checks,
        immutable_inputs_unchanged=True,COCO_used_for_selection=False,wiki_used=False,xm3600_scored=False,
        registry_sha256=file_sha(study.OUT/'selection_registry.json'),evaluation_marker=read_json(marker)))
    write_json(study.OUT/'status.json',dict(status='complete',experiments=14,COCO_scored_after_selection=True,
        COCO_used_for_selection=False,finalists=registry['finalists']))
    print('SOURCE_ABLATION_COMPLETE',flush=True)


if __name__=='__main__':
    import fcntl
    with (study.vn.OUT/'worker.lock').open('a') as lock:
        print('Waiting for completed source study and exclusive GPU lock',flush=True)
        fcntl.flock(lock,fcntl.LOCK_EX)
        main()
