"""Reproduce selected COCO/VN dev metrics; never score MSCOCO test or XM3600."""
import csv
import json
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import numpy as np
import torch
from pl_mmb_training import tinyclip_visualnews as study
from pl_mmb_training import tinyclip_adaptation as base
from pl_mmb_training.common import ROOT,digest,file_sha,read_json,write_json
from pl_mmb_training.losses import PairLoss
from verify_tinyclip_generalization import components,bootstrap


def plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    colors={'0':'#777777','10k':'#0072B2','all':'#D55E00'}
    for r in results:
        if r['spec'].get('replication_of'):
            continue
        volume='0' if r.get('reused_reference') else r['spec']['volume']
        blocks=r['spec']['blocks']
        style='-' if blocks else '--'
        label=f'VN {volume}, '+('last2 + anchor' if blocks else 'frozen')
        rows=r['history'];epochs=[x['epoch'] for x in rows]
        axes[0,0].plot(epochs,[100*x['validation']['t2i']['recall@1'] for x in rows],style,color=colors[volume],label=label)
        axes[0,1].plot(epochs[1:],[x['train_loss'] for x in rows[1:]],style,color=colors[volume])
        axes[1,0].plot(epochs,[x['validation_loss'] for x in rows],style,color=colors[volume])
        axes[1,1].plot(epochs,[x['validation_feature_cosine_drift'] for x in rows],style,color=colors[volume])
    for ax,title in zip(axes.flat,['COCO validation t2i R@1 (%)','Train gold loss','COCO validation loss','COCO vision feature drift (1−cos)']):
        ax.set_title(title);ax.set_xlabel('Budget epoch (275 updates)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('VisualNews data-volume scout; seed 42; checkpoint selection on COCO validation')
    fig.tight_layout();fig.savefig(ROOT/'reports/tinyclip_visualnews_curves.png',dpi=160);plt.close(fig)


def stats(values):
    values=np.asarray(values,dtype=float)
    return dict(mean=float(values.mean()),sample_std=float(values.std(ddof=1)) if len(values)>1 else None,
        minimum=float(values.min()),maximum=float(values.max()),values=values.tolist())


def fixed_seed_cluster_interval(delta,groups):
    """Bootstrap queries by component after averaging the paired effect over fixed seeds."""
    ng=int(groups.max())+1
    sums=np.bincount(groups,weights=delta,minlength=ng)
    counts=np.bincount(groups,minlength=ng)
    rng=np.random.default_rng(20261021)
    samples=[]
    for _ in range(10):
        indices=rng.integers(ng,size=(1000,ng))
        samples.append(sums[indices].sum(1)/counts[indices].sum(1))
    return dict(delta_percentage_points=float(np.mean(delta)*100),
        interval_95_percentage_points=(np.quantile(np.concatenate(samples),[.025,.975])*100).tolist(),
        bootstrap_components=ng,queries=len(delta),repetitions=10000,
        caveat='Conditional on these three fitted adaptation seeds; no seed or checkpoint/configuration-selection uncertainty.')


def repetitions(results,ranks,groups):
    by_name={r['experiment']:r for r in results}
    names={kind:[f'VN_all_{kind}'+(f'_seed{seed}' if seed!=42 else '') for seed in (42,43,44)]
        for kind in ('frozen','last2_anchor')}
    if not all(name in by_name for family in names.values() for name in family):
        return None
    summary=dict(seeds=[42,43,44],initialization='Same A1 epoch-9 head and native TinyCLIP for all fits',
        caveat='Adaptation seeds only; not independent repetitions of the original linear-probe pretraining.',
        metrics={},paired_effect={})
    for kind,family in names.items():
        rows=[by_name[name] for name in family]
        for seed,row in zip(summary['seeds'],rows):
            assert row['spec']['seed']==seed and row['spec']['volume']=='all'
        metrics={}
        for split,key in [('coco','selected_validation'),('vn','selected_vn_dev')]:
            metrics[split]={direction:{f'recall@{k}':stats([100*r[key][direction][f'recall@{k}'] for r in rows])
                for k in (1,5,10)} for direction in ('t2i','i2t')}
        metrics['best_budget_epoch']=stats([r['best_epoch'] for r in rows])
        metrics['peak_vram_bytes']=stats([max(h['peak_vram_bytes'] for h in r['history']) for r in rows])
        summary['metrics'][kind]=metrics
    for split,key in [('coco','selected_validation'),('vn','selected_vn_dev')]:
        delta=np.mean([(np.asarray(ranks[a][split])==1).astype(float)-(np.asarray(ranks[b][split])==1)
            for a,b in zip(names['last2_anchor'],names['frozen'])],axis=0)
        summary['paired_effect'][split]=dict(
            per_seed_delta_percentage_points=stats([100*(by_name[a][key]['t2i']['recall@1']-by_name[b][key]['t2i']['recall@1'])
                for a,b in zip(names['last2_anchor'],names['frozen'])]),
            mean_fixed_seed_component_bootstrap=fixed_seed_cluster_interval(delta,groups[split]))
    return summary


def plot_repetitions(summary):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,split,title in zip(axes,['coco','vn'],['Original COCO validation','Separate VisualNews dev']):
        for i,(seed,color) in enumerate(zip(summary['seeds'],['#0072B2','#D55E00','#009E73'])):
            values=[summary['metrics'][kind][split]['t2i']['recall@1']['values'][i] for kind in ('frozen','last2_anchor')]
            ax.plot([0,1],values,'o-',color=color,label=f'Seed {seed}')
        ax.set_xticks([0,1],['Frozen vision','Last 2 blocks + anchor'])
        ax.set_ylabel('t2i R@1 (%)');ax.set_title(title);ax.grid(axis='y',alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('33,851 added VisualNews train images; matched adaptation seeds; same LP initialization')
    fig.tight_layout();fig.savefig(ROOT/'reports/tinyclip_visualnews_seeds.png',dpi=160);plt.close(fig)


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ctx=study.Context();engine=study.Engine(ctx)
    report=read_json(ROOT/'reports/tinyclip_visualnews.json')
    results=report['results']
    assert len(results)>=6
    ranks,checks={},{}
    with torch.inference_mode():
        for result in results:
            name=result['experiment'];path=Path(result['checkpoint_path'])
            assert file_sha(path)==result['checkpoint_sha256']
            saved=torch.load(path,weights_only=True,map_location='cpu')
            config=read_json(path.parent/'config.json')
            assert digest(config)==result['run_id']==saved['run_id']
            for source,sha in config['source'].items():
                assert file_sha(ROOT/source)==sha
            if result['spec'].get('replication_of'):
                assert file_sha(ROOT/'training/repeat_tinyclip_visualnews.py')==result['spec']['launcher_sha256']
            if not result.get('reused_reference'):
                assert config['data']==ctx.identity
            else:
                assert config['inputs']==ctx.base.identity
            chosen=max(result['history'],key=base.checkpoint_key)
            assert chosen['epoch']==result['best_epoch']==saved['epoch']
            assert chosen['validation']==result['selected_validation']
            assert result['history'][-1]['updates']==3300
            assert not result['selection_uses_test'] and not result['xm3600_used']
            assert not result['vn_dev_used_for_selection']
            assert all(math.isfinite(x['validation_loss']) for x in result['history'])
            model=base.projector(512,0).to('cuda').eval();model.load_state_dict(saved['projector'])
            suffix=study.copy_suffix(engine.vision,result['spec']['blocks'],engine.hook).to('cuda').eval() if result['spec']['blocks'] else None
            if suffix is not None:suffix.load_state_dict(saved['vision_suffix'])
            current=engine.coco_features(suffix)
            coco=engine.retrieval(ctx.graph,current,model)
            dev=engine.selected_dev(suffix,model)
            for actual,expected in [(coco,result['selected_validation']),(dev,result['selected_vn_dev'])]:
                for direction in ('t2i','i2t'):
                    for k in (1,5,10):
                        assert actual[direction][f'recall@{k}']==expected[direction][f'recall@{k}']
            criterion=PairLoss('siglip_sigmoid').to('cuda');criterion.load_state_dict(saved['loss'])
            loss=0.
            for part in torch.arange(len(ctx.vi)).split(256):
                loss+=float(criterion(model(current[ctx.vi[part].to('cuda')]),ctx.vt[part].to('cuda'),ctx.vp.mask(part.tolist()).to('cuda')))*len(part)
            assert abs(loss/len(ctx.vi)-chosen['validation_loss'])<2e-5
            ranks[name]=dict(coco=coco['ranks']['t2i'],vn=dev['ranks']['t2i'])
            checks[name]=dict(checkpoint_verified=True,primary_and_dev_reproduced=True,val_loss_reproduced=True,
                full_budget_verified=True,source_verified=True)
            print('VERIFIED',name,flush=True)
    # Resample whole image/article/caption components, rather than treating captions as independent.
    coco_groups=components(ctx.base)
    dev_lookup={r['text_pl']:r['component'] for r in ctx.dev}
    dev_keys=sorted({r['component'] for r in ctx.dev});dev_map={k:i for i,k in enumerate(dev_keys)}
    vn_groups=np.array([dev_map[dev_lookup[t]] for t in ctx.dev_graph['texts']])
    pairs=[('VN_0_last2_anchor','VN_0_frozen'),('VN_10k_last2_anchor','VN_10k_frozen'),
        ('VN_all_last2_anchor','VN_all_frozen'),('VN_10k_frozen','VN_0_frozen'),
        ('VN_all_frozen','VN_10k_frozen'),('VN_all_last2_anchor','VN_0_last2_anchor')]
    comparisons={a+'_minus_'+b:{split:bootstrap(ranks[a][split],ranks[b][split],groups)
        for split,groups in [('coco',coco_groups),('vn',vn_groups)]} for a,b in pairs}
    for seed in (43,44):
        a,b=f'VN_all_last2_anchor_seed{seed}',f'VN_all_frozen_seed{seed}'
        if a in ranks and b in ranks:
            comparisons[a+'_minus_'+b]={split:bootstrap(ranks[a][split],ranks[b][split],groups)
                for split,groups in [('coco',coco_groups),('vn',vn_groups)]}
    repeated=repetitions(results,ranks,dict(coco=coco_groups,vn=vn_groups))
    ctx.verify()
    for path,sha in ctx.audit['blacklists'].items():assert file_sha(path)==sha
    for path,sha in ctx.audit['base_inputs'].items():assert file_sha(path)==sha
    write_json(ROOT/'reports/tinyclip_visualnews_verification.json',dict(status='passed',checks=checks,
        comparisons=comparisons,replication=repeated,inputs_unchanged=True,test_used=False,xm3600_scores_computed=False,
        caveat='Conditional paired bootstrap; excludes configuration/checkpoint-selection uncertainty and seed variance.'))
    study.export(ctx,results);plot(results)
    if repeated:
        plot_repetitions(repeated)
        report_path=ROOT/'reports/tinyclip_visualnews.md'
        text=report_path.read_text(encoding='utf-8')
        text=text.replace('Seed 42.','Scout: seed 42; pełny dodatek powtórzony na seedach 43 i 44.')
        text=text.replace('Seed scout. Test MSCOCO','Scout i trzy seedy adaptacji dla pełnego dodatku. Test MSCOCO')
        report_path.write_text(text,encoding='utf-8')
    with (ROOT/'reports/tinyclip_visualnews_metrics.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['experiment','seed','vn_train_images','feature_source','projector',
            'vision_trainable_blocks','trainable_params','best_budget_epoch','val_t2i_r1','val_t2i_r5','val_t2i_r10',
            'val_i2t_r1','vn_dev_t2i_r1','vn_dev_i2t_r1','peak_vram_bytes','checkpoint_sha256','test_mscoco'])
        writer.writeheader()
        for r in results:
            v=r['selected_validation'];dev=r['selected_vn_dev']
            writer.writerow(dict(experiment=r['experiment'],seed=r['spec']['seed'],vn_train_images=r['vn_train_images'],
                feature_source='TinyCLIP original projected 512D',projector='Linear 512->384',
                vision_trainable_blocks=r['spec']['blocks'],trainable_params=r['trainable_params'],best_budget_epoch=r['best_epoch'],
                val_t2i_r1=v['t2i']['recall@1'],val_t2i_r5=v['t2i']['recall@5'],val_t2i_r10=v['t2i']['recall@10'],
                val_i2t_r1=v['i2t']['recall@1'],vn_dev_t2i_r1=dev['t2i']['recall@1'],vn_dev_i2t_r1=dev['i2t']['recall@1'],
                peak_vram_bytes=max(h['peak_vram_bytes'] for h in r['history']),checkpoint_sha256=r['checkpoint_sha256'],test_mscoco='not run'))
    with (ROOT/'reports/tinyclip_visualnews.md').open('a',encoding='utf-8') as stream:
        stream.write('\nWyniki z wybranych checkpointów odtworzono po ponownym wczytaniu z dysku.\n\n')
        stream.write('| Porównanie | COCO Δ R@1 p.p. [95% CI] | VN dev Δ R@1 p.p. [95% CI] |\n|---|---|---|\n')
        for name,c in comparisons.items():
            values=[]
            for split in ('coco','vn'):
                lo,hi=c[split]['interval_95_percentage_points']
                values.append(f'{c[split]["delta_percentage_points"]:+.2f} [{lo:+.2f}, {hi:+.2f}]')
            stream.write(f'| {name} | {values[0]} | {values[1]} |\n')
        stream.write('\nBootstrap po komponentach; bez korekty selekcji i zmienności seedów.\n\n![Krzywe](tinyclip_visualnews_curves.png)\n')
        if repeated:
            stream.write('\n## Pełny dodatek VN: trzy seedy adaptacji\n\n')
            stream.write('Seedy 42/43/44 mają wspólny checkpoint startowy A1 ep. 9. Zmieniono wyłącznie seed adaptacji. '
                'Wartości poniżej to średnia ± odchylenie standardowe między seedami, w procentach.\n\n')
            stream.write('| Vision | COCO val t2i R@1 | COCO val i2t R@1 | VN dev t2i R@1 |\n|---|---:|---:|---:|\n')
            for kind in ('frozen','last2_anchor'):
                values=[]
                for split,direction in [('coco','t2i'),('coco','i2t'),('vn','t2i')]:
                    s=repeated['metrics'][kind][split][direction]['recall@1']
                    values.append(f'{s["mean"]:.2f} ± {s["sample_std"]:.2f}')
                stream.write(f'| {kind} | '+ ' | '.join(values)+' |\n')
            stream.write('\n| Last2 − frozen, t2i R@1 | Średnia ± SD różnicy (p.p.) | Warunkowy przedział 95% (p.p.) |\n|---|---:|---:|\n')
            for split in ('coco','vn'):
                effect=repeated['paired_effect'][split];s=effect['per_seed_delta_percentage_points']
                lo,hi=effect['mean_fixed_seed_component_bootstrap']['interval_95_percentage_points']
                stream.write(f'| {split} | {s["mean"]:+.2f} ± {s["sample_std"]:.2f} | [{lo:+.2f}, {hi:+.2f}] |\n')
            stream.write('\nPrzedział resampluje komponenty zapytań po uśrednieniu trzech dopasowanych seedów. '
                'Nie obejmuje niepewności seedów, wcześniejszego pretreningu projectora ani selekcji konfiguracji/checkpointów. '
                'Trzy seedy nie są wystarczające do mocnego wniosku o istotności populacyjnej.\n\n'
                '![Powtórzenia](tinyclip_visualnews_seeds.png)\n')
            effect=repeated['paired_effect']['coco']['per_seed_delta_percentage_points']
            dev_effect=repeated['paired_effect']['vn']['per_seed_delta_percentage_points']
            stream.write('\n## Interpretacja i ograniczenia\n\n')
            stream.write(f'Na pełnym dodatku VN adaptacja dwóch bloków daje średnio {effect["mean"]:+.2f} p.p. '
                f'COCO val t2i R@1 wobec dopasowanego frozen (SD różnic {effect["sample_std"]:.2f} p.p.). '
                f'Na osobnym VN dev średnia różnica wynosi {dev_effect["mean"]:+.2f} p.p. '
                'To wynik konkretnej recepty: dwa bloki + anchor, warm start, warmup i augmentacja. '
                'Nie rozstrzyga osobno wpływu każdego z tych składników.\n\n')
            stream.write('Porównanie 0 VN z dodatkiem zmienia jednocześnie domenę danych, liczbę unikalnych obrazów '
                'i proporcję ekspozycji na FULL_CLEAN. Porównanie 10k/all trzyma proporcję źródeł i koszt treningu, '
                'ale obejmuje tylko jeden seed. Nie można przypisać całego zysku samej liczbie obrazów.\n\n')
            stream.write('COCO val ma 419 obrazów i było używane we wcześniejszych eksperymentach. '
                'Warunkowe przedziały nie korygują wielokrotnego wyboru recepty i checkpointów. '
                'VN dev ma te same dwa źródła i pipeline generujący captiony co VN train; '
                'nie jest niezależnym testem ludzkich zapytań. Grupowanie artykułów i pHash ogranicza przeciek, '
                'ale nie gwarantuje rozdzielenia wszystkich wydarzeń ani wszystkich cropów.\n\n')
            stream.write('Peak VRAM w nowych przebiegach obejmuje pełny cache RGB umieszczony na GPU. '
                'Nie należy porównywać go bezpośrednio z pamięcią historycznych kontroli 0 VN. '
                'Liczby parametrów i pełne logi lossów/gradientów są zapisane w JSON; tabela CSV zawiera metryki i checkpoint SHA.\n\n')
            stream.write('Kolejny logiczny test metodologiczny to dopasowany LP→FT z projectorem najpierw '
                'wytrenowanym na całej mieszance FULL_CLEAN+VN, z kontrolą frozen o identycznym dodatkowym budżecie. '
                'W tej serii wszystkie adaptacje startują ze starego A1, wytrenowanego przed dodaniem VN. '
                'Ten wariant nie został tutaj uruchomiony. Przed końcową oceną MSCOCO należy zamrozić receptę i '
                'regułę wyboru seeda; XM3600 pozostaje blind i wymaga osobnej decyzji.\n')
    print(json.dumps(dict(status='passed',comparisons=comparisons),ensure_ascii=False),flush=True)

if __name__=='__main__':
    import fcntl
    with (study.OUT/'worker.lock').open('a') as lock:
        print('Waiting for exclusive GPU worker lock',flush=True)
        fcntl.flock(lock,fcntl.LOCK_EX)
        main()
