"""Post-hoc COCO transfer of the closed wiki pilot; no training/selection."""
import csv
import json
import sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from pl_mmb_training import wiki_pilot as pilot
from pl_mmb_training import tinyclip_visualnews as vn
from pl_mmb_training.source_ablation import UnionFind
from pl_mmb_training.common import ROOT,digest,file_sha,read_json,write_json
from pl_mmb_training.evaluation import evaluate_features
from pl_mmb_training.leakage import normalized_text
from pl_mmb_training.retrieval import evaluate_graph,graph_from_examples
from verify_tinyclip_generalization import bootstrap

OUT=pilot.OUT/'coco_transfer'
NAMES=[f'WIKI_{p:02d}pct' for p in (0,3,10)]


def query_groups(graph,blacklist):
    uf=UnionFind()
    for path,pixel in zip(graph['images'],graph['image_ids']):
        uf.join(['pixel:'+pixel]+['parent:'+p for p in blacklist['images'][path]['parent_ids']])
    for t,i in graph['edges']:
        uf.join(['text:'+normalized_text(graph['texts'][t]),'pixel:'+graph['image_ids'][i]])
    def labels(nodes):
        roots=[uf.find(n) for n in nodes];index={v:i for i,v in enumerate(sorted(set(roots)))}
        return np.array([index[r] for r in roots])
    return dict(t2i=labels(['text:'+normalized_text(t) for t in graph['texts']]),
        i2t=labels(['pixel:'+graph['image_ids'][i] for i in sorted({i for _,i in graph['edges']})]))


def contrasts(records,groups):
    output={}
    for a,b in [(NAMES[1],NAMES[0]),(NAMES[2],NAMES[0]),(NAMES[2],NAMES[1])]:
        output[a+'_minus_'+b]={}
        for protocol,labels in groups.items():
            directions=('t2i',) if protocol=='COCO_local_test' else ('t2i','i2t')
            for direction in directions:
                def ranks(name):
                    value=records[name][protocol]
                    return [p['rank'] for p in value['predictions']] if protocol=='COCO_local_test' else value['ranks'][direction]
                for k in (1,5,10):
                    x=np.array(ranks(a));y=np.array(ranks(b))
                    c=bootstrap(np.where(x<=k,1,2),np.where(y<=k,1,2),labels[direction])
                    c['caveat']='Conditional paired query-component interval, not seed/checkpoint-selection uncertainty; COCO is historically observed.'
                    output[a+'_minus_'+b][protocol+'_'+direction+f'_R{k}']=c
    return output


def load_head(name,file,expected):
    path=pilot.OUT/name/file;assert file_sha(path)==expected['sha256']
    saved=torch.load(path,weights_only=True,map_location='cpu')
    assert saved['run_id']==expected['run_id'] and saved['updates']==expected['updates']
    model=torch.nn.Linear(512,384).eval();model.load_state_dict(saved['projector'])
    return model


def evaluate():
    OUT.mkdir(exist_ok=True)
    verified_path=ROOT/'reports/tinyclip_wiki_pilot_verification.json'
    verified=read_json(verified_path);assert verified['status']=='passed'
    ctx=vn.Context()
    native_meta=read_json(vn.OUT/'native_features.json')
    assert native_meta['identity']['data']==ctx.identity
    assert file_sha(vn.OUT/'native_features.pt')==native_meta['sha256']
    native=torch.load(vn.OUT/'native_features.pt',weights_only=True,map_location='cpu')
    manifest_path=ROOT/'training/runs/e1-full/cache.json';manifest=read_json(manifest_path)
    compact_path=ROOT/'training/runs/vision-four/common/evaluation_compact.json';compact=read_json(compact_path)
    assert compact['sha256']==digest(compact['datasets'])
    assert compact['feature_manifest_sha256']==file_sha(manifest_path)
    compact_rows=compact['datasets']['mscoco_t2i']
    examples=[SimpleNamespace(id=r['id'],images=[manifest['images'][i] for i in r['images']],
        texts=[manifest['texts'][i] for i in r['texts']],query_image=None) for r in compact_rows]
    assert len(examples)==1000 and all(r['query_image'] is None for r in compact_rows)
    feature_meta=read_json(ROOT/'training/runs/vision-commercial-five/models/tinyclip-39m/feature_cache.json')
    assert file_sha(feature_meta['path'])==feature_meta['sha256']
    test_x=torch.load(feature_meta['path'],weights_only=True,map_location='cpu')
    im={p:i for i,p in enumerate(manifest['images'])}
    minimum=float(torch.nn.functional.cosine_similarity(test_x[[im[p] for p in ctx.base.images]],native[:ctx.offset]).min())
    assert minimum>.999
    blacklist_path=ROOT/'training/runs/e1-full/blacklist.json';blacklist=read_json(blacklist_path)
    test_graph=graph_from_examples(examples,blacklist,manifest['images'],manifest['texts'])
    # The legacy benchmark blacklist contains test images, while validation
    # identities live in the separately frozen validation manifest.
    group_metadata=dict(blacklist,images={**blacklist['images'],**{r['image']:r for r in ctx.base.val}})
    groups=dict(COCO_val=query_groups(ctx.base.graph,group_metadata),COCO_fixed_test=query_groups(test_graph,group_metadata))
    text_labels=dict(zip(test_graph['texts'],groups['COCO_fixed_test']['t2i']))
    groups['COCO_local_test']=dict(t2i=np.array([text_labels[e.texts[0]] for e in examples]))
    registry=dict(status='all_models_frozen_before_COCO_scoring',checkpoint_selection='VN_dev_only',
        selected_variants=NAMES,diagnostic_checkpoints='fixed_predeclared_step5500',
        training_verification_sha256=file_sha(verified_path),
        inputs={str(p):file_sha(p) for p in (manifest_path,compact_path,blacklist_path)},
        native_feature_sha256=native_meta['sha256'],test_feature_sha256=feature_meta['sha256'],
        checkpoints={},COCO_used_for_training=False,COCO_used_for_selection=False,xm3600_scored=False)
    results={}
    for name in NAMES:
        folder=pilot.OUT/name;result=read_json(folder/'result.json');config=read_json(folder/'config.json')
        assert result['config']==config and digest(config)==result['run_id']
        for source,sha in config['source'].items():assert file_sha(ROOT/source)==sha
        assert max(result['history'],key=pilot.selection_key)['updates']==result['best_updates']
        assert file_sha(folder/'best.pt')==result['checkpoint_sha256']
        assert not result['COCO_scored'] and not result['COCO_used_for_selection']
        registry['checkpoints'][name]={file:dict(sha256=file_sha(folder/file),updates=updates,run_id=result['run_id'])
            for file,updates in [('best.pt',result['best_updates']),('last.pt',5500)]}
        registry['checkpoints'][name]['result_sha256']=file_sha(folder/'result.json')
        results[name]=result
    if (OUT/'registry.json').exists():assert read_json(OUT/'registry.json')['identity']==registry
    else:write_json(OUT/'registry.json',dict(created_at_UTC=datetime.now(timezone.utc).isoformat(),identity=registry))
    # Registry written before the first COCO model score; nothing below selects a model.
    output={kind:{} for kind in ('selected','fixed5500')}
    with torch.inference_mode():
        for name in NAMES:
            for kind,file in [('selected','best.pt'),('fixed5500','last.pt')]:
                model=load_head(name,file,registry['checkpoints'][name][file])
                if kind=='selected':
                    gx=ctx.dev_graph;pred=model(native[gx['image_indices']])
                    local=dict(gx,image_indices=list(range(len(pred))),text_indices=list(range(len(gx['texts']))))
                    actual=evaluate_graph(local,pred,ctx.texts[gx['text_indices']],include_ranks=True)
                    old=results[name]['selected']['VN_dev']['retrieval']
                    assert actual['t2i']==old['t2i'] and actual['i2t']==old['i2t']
                val=evaluate_graph(ctx.base.graph,model(native[:ctx.offset]),ctx.base.texts,include_ranks=True)
                projected=model(test_x)
                fixed=evaluate_graph(test_graph,projected,ctx.base.texts,include_ranks=True)
                metrics,predictions=evaluate_features(examples,manifest['images'],projected,manifest['texts'],ctx.base.texts)
                assert [p['id'] for p in predictions]==[e.id for e in examples]
                record=dict(COCO_val=val,COCO_fixed_test=fixed,COCO_local_test=dict(metrics=metrics,predictions=predictions),
                    checkpoint=registry['checkpoints'][name][file],VN_dev_t2i_R1=results[name]['selected']['VN_dev']['retrieval']['t2i']['recall@1'])
                output[kind][name]=record
                write_json(OUT/f'{name}_{kind}.json',record)
                print(f'COCO_TRANSFER {name} {kind}: local_R1={metrics["recall@1"]:.5f} val_R1={val["t2i"]["recall@1"]:.5f}',flush=True)
    prior=read_json(ROOT/'reports/tinyclip_source_ablation.json')
    current=output['selected'][NAMES[0]]
    assert current['COCO_local_test']['metrics']==prior['COCO_finalists']['VN_only']['local']
    assert current['COCO_val']['t2i']==prior['COCO_validation_posthoc']['VN_only']['t2i']
    comparisons={kind:contrasts(records,groups) for kind,records in output.items()}
    ctx.verify()
    for path,sha in registry['inputs'].items():assert file_sha(path)==sha
    for name in NAMES:
        for file in ('best.pt','last.pt'):assert file_sha(pilot.OUT/name/file)==registry['checkpoints'][name][file]['sha256']
    report=dict(status='complete',registry=read_json(OUT/'registry.json'),results=output,comparisons=comparisons,
        protocols=dict(local_test=dict(queries=len(examples),candidate_counts=dict(Counter(len(e.images) for e in examples)),
            independent_components=len(set(groups['COCO_local_test']['t2i'])),name='Existing local MMEB MSCOCO_t2i; not official COCO5k'),
            fixed_test=dict(images=len(test_graph['images']),texts=len(test_graph['texts']),graph_sha256=test_graph['sha256']),
            validation=dict(images=len(ctx.base.graph['images']),texts=len(ctx.base.graph['texts']),graph_sha256=ctx.base.graph['sha256'])),
        verification=dict(status='passed',previous_VN_only_COCO_reproduced=True,VN_selection_reproduced=True,
            inputs_and_checkpoints_unchanged=True,minimum_native_feature_cosine=minimum),
        evaluator_sha256=file_sha(__file__),COCO_used_for_selection=False,new_training=False,xm3600_scored=False,
        caveats=['One training seed; conditional intervals exclude seed and checkpoint-selection uncertainty.',
            'COCO is a historically inspected benchmark, not a newly blind evaluation.',
            'The VN/wiki-only pilot recommendation is restricted to those domains and does not establish cross-dataset utility.',
            'No known COCO in supervised projector training; encoder pretraining and unidentified copies are outside this claim.'])
    write_json(ROOT/'reports/tinyclip_wiki_coco_transfer.json',report)
    export(report)
    write_json(OUT/'status.json',dict(status='complete',checkpoints=6,training=False,COCO_scored=True,xm3600_scored=False))


def export(report):
    lines=['# Wiki → COCO: transfer poza domenę treningową','',
        'Wszystkie trzy warianty i checkpointy były zamrożone przed oceną COCO. Wybór checkpointów: wyłącznie VN dev. Brak nowych treningów; XM3600 pozostaje nieocenione.',
        'TinyCLIP-39M i PolDense frozen; Linear 512→384. Seed42. Train: samo VN lub VN z 3%/10% prezentacji wiki, bez znanego COCO w naszym nadzorowanym treningu.','',
        'Główny punkt porównania: dotychczasowy lokalny MMEB MSCOCO_t2i, 1000 zapytań i po1000 kandydatów; nie jest to oficjalny COCO5k. Zależności zapytań: '
        f'{report["protocols"]["local_test"]["independent_components"]} komponentów obrazów/podpisów/powiązań źródłowych.','',
        '| Udział wiki | COCO test R@1 | R@5 | R@10 | COCO val t2i R@1, post hoc | COCO val i2t R@1, post hoc | Best krok po VN |',
        '|---|---:|---:|---:|---:|---:|---:|']
    csv_rows=[]
    for name in NAMES:
        r=report['results']['selected'][name];m=r['COCO_local_test']['metrics'];v=r['COCO_val']
        lines.append(f'| {name} | '+' | '.join(f'{100*m[f"recall@{k}"]:.2f}%' for k in (1,5,10))+
            f' | {100*v["t2i"]["recall@1"]:.2f}% | {100*v["i2t"]["recall@1"]:.2f}% | {r["checkpoint"]["updates"]} |')
        csv_rows.append(dict(experiment=name,checkpoint_sha256=r['checkpoint']['sha256'],best_updates=r['checkpoint']['updates'],
            COCO_test_R1=m['recall@1'],COCO_test_R5=m['recall@5'],COCO_test_R10=m['recall@10'],
            COCO_val_t2i_R1=v['t2i']['recall@1'],COCO_val_i2t_R1=v['i2t']['recall@1']))
    lines+=['','Parowane 95% przedziały warunkowe; obejmują komponenty zapytań, nie seedy lub wybór checkpointu. Różnice w punktach procentowych.','',
        '| Wariant − 0% | COCO test t2i Δ [95% CI] | COCO val t2i Δ [95% CI] |','|---|---:|---:|']
    for name in NAMES[1:]:
        c=report['comparisons']['selected'][name+'_minus_'+NAMES[0]];values=[]
        for key in ('COCO_local_test_t2i_R1','COCO_val_t2i_R1'):
            x=c[key];lo,hi=x['interval_95_percentage_points'];values.append(f'{x["delta_percentage_points"]:+.2f} [{lo:+.2f}, {hi:+.2f}]')
        lines.append(f'| {name} | '+' | '.join(values)+' |')
    lines+=['','## Stały krok5500 — pomocnicza kontrola','',
        'Ten checkpoint był wcześniej zadeklarowany w pilotażu jako diagnoza stałego budżetu; nie wybierano go po COCO.','',
        '| Wariant | COCO test R@1 | COCO val t2i R@1 |','|---|---:|---:|']
    for name in NAMES:
        r=report['results']['fixed5500'][name]
        lines.append(f'| {name} | {100*r["COCO_local_test"]["metrics"]["recall@1"]:.2f}% | {100*r["COCO_val"]["t2i"]["recall@1"]:.2f}% |')
    lines+=['','## Interpretacja','',
        'Poprzednia rekomendacja z pilotażu VN/wiki dotyczyła ochrony wyniku VN i zachowania na wiki dev. Nie rozstrzygała transferu na inne dane; właściwe porównanie dla COCO znajduje się powyżej.',
        'Mała różnica na COCO przy jednym seedzie pozostaje wynikiem wstępnym. Nie użyto ocen COCO do przestawiania checkpointów, proporcji danych ani uruchamiania nowych treningów. COCO było już oglądane w poprzednich eksperymentach.',
        'Osobne wyniki dla pełnej galerii testowej oraz R@1/5/10 w obu kierunkach są w JSON; nie należy mieszać ich z metrykami lokalnych pul kandydatów.',
        '', 'Weryfikacja: passed; odtworzono wcześniejsze wyniki COCO VN-only oraz VN dev wszystkich trzech modeli. Dane i checkpointy zachowały SHA256.']
    (ROOT/'reports/tinyclip_wiki_coco_transfer.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (ROOT/'reports/tinyclip_wiki_coco_transfer.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    old=ROOT/'reports/tinyclip_wiki_pilot.md'
    note='[Ocena transferu na COCO](tinyclip_wiki_coco_transfer.md)'
    if note not in old.read_text(encoding='utf-8'):
        with old.open('a',encoding='utf-8') as stream:
            stream.write('\n## Późniejsza ocena transferu\n\n'+note+
                '. Rekomendacja powyżej powstała przed tą oceną i dotyczyła VN/wiki; nie stanowiła rozstrzygnięcia o użyteczności wiki na innych domenach.\n')
    print('WIKI_COCO_TRANSFER_COMPLETE',flush=True)


if __name__=='__main__':
    import fcntl
    with (vn.OUT/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        evaluate()
