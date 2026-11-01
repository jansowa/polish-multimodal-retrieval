"""Reproduce the three frozen wiki-pilot checkpoints; never score COCO/XM3600."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from pl_mmb_training import wiki_pilot as study
from pl_mmb_training.common import ROOT,digest,file_sha,read_json,write_json
from pl_mmb_training.losses import PairLoss


def groups(data,split,direction='t2i'):
    rows=data.ctx.dev if split=='VN_dev' else data.wiki_dev
    lookup={}
    for row in rows:
        text=row['text_pl'] if direction=='t2i' else row['target_id'];component=row['component']
        if text in lookup:assert lookup[text]==component
        lookup[text]=component
    labels=[lookup[t] for t in data.graphs[split]['texts' if direction=='t2i' else 'image_ids']]
    keys=sorted(set(labels));index={s:i for i,s in enumerate(keys)}
    return np.array([index[s] for s in labels])


def bootstrap(a,b,labels):
    delta=(np.array(a)<=1).astype(float)-(np.array(b)<=1).astype(float)
    count=np.bincount(labels);sums=np.bincount(labels,weights=delta);n=len(count)
    rng=np.random.default_rng(3819);draws=[]
    for _ in range(20):
        weights=rng.multinomial(n,np.full(n,1/n),size=500)
        draws.extend((weights@sums)/(weights@count))
    return dict(delta_percentage_points=float(100*delta.mean()),
        interval_95_percentage_points=(100*np.quantile(draws,[.025,.975])).tolist(),
        components=n,queries=len(delta),resamples=10000,
        caveat='Conditional paired component bootstrap; excludes training-seed and checkpoint-selection uncertainty.')


def verify(data,results):
    initial,path=study.initialize();checks={}
    for result in results:
        folder=study.OUT/result['experiment'];config=read_json(folder/'config.json')
        assert result['config']==config and digest(config)==result['run_id']
        assert file_sha(path)==config['initial_sha256']
        for source,sha in config['source'].items():assert file_sha(ROOT/source)==sha
        assert file_sha(folder/'best.pt')==result['checkpoint_sha256']
        assert max(result['history'],key=study.selection_key)['updates']==result['best_updates']
        assert result['example_presentations']==1405780 and result['steps']==5500
        assert result['presentations'].get('wikimedia',0)==1405780*result['wiki_percent']//100
        assert not result['COCO_scored'] and not result['xm3600_scored'] and not result['wiki_used_for_selection']
        model=study.base.projector(512,0).cuda().eval();criterion=PairLoss('siglip_sigmoid',.07,-10.).cuda()
        for file,key in [('best.pt','selected'),('last.pt','final_fixed_budget')]:
            saved=torch.load(folder/file,weights_only=True,map_location='cpu')
            assert saved['run_id']==result['run_id'] and saved['initial_sha256']==file_sha(path)
            assert saved['updates']==(result['best_updates'] if file=='best.pt' else 5500)
            model.load_state_dict(saved['projector']);criterion.load_state_dict(saved['loss'])
            actual=study.selected(model,criterion,data)
            for split in actual:
                assert actual[split]['retrieval']==result[key][split]['retrieval'],(result['experiment'],file,split)
                assert abs(actual[split]['loss']-result[key][split]['loss'])<2e-5
        checks[result['experiment']]=dict(selected_and_final_retrieval_reproduced=True,
            selected_and_final_losses_reproduced=True,source_and_initialization_verified=True,
            exact_presentation_budget_verified=True,checkpoint_selection_VN_only=True)
        print('VERIFIED_WIKI',result['experiment'],flush=True)
    prior=read_json(study.source.OUT/'VN_only/result.json')
    current=results[0]
    old=prior['selected_VN_dev'];new=current['selected']['VN_dev']['retrieval']
    baseline=dict(previous_VN_R1=old['t2i']['recall@1'],current_VN_R1=new['t2i']['recall@1'],
        previous_best_updates=prior['best_updates'],current_best_updates=current['best_updates'],
        same_R1_and_selected_step=(old['t2i']['recall@1']==new['t2i']['recall@1'] and prior['best_updates']==current['best_updates']))
    assert baseline['same_R1_and_selected_step'],baseline
    # Recompute a small known-text sample to check newly encoded wiki vectors
    # against the frozen VN cache, including the current tokenizer/model runtime.
    from pl_mmb_training.features import encode_cached
    config=read_json(ROOT/'training/runs/e1-full/config.json')
    lock=read_json(ROOT/'training/runs/e1-full/cache.json')['model_lock']
    values=sorted({r['text_pl'] for r in data.ctx.vn},key=lambda t:digest(['text-cache-parity',t]))[:32]
    fresh,meta=encode_cached(values,'texts',config,lock,study.OUT/'parity_text_cache')
    old=data.ctx.texts[[data.ctx.ti[t] for t in values]]
    cosine=torch.nn.functional.cosine_similarity(fresh,old)
    minimum_cosine=float(cosine.min());assert minimum_cosine>.9999,minimum_cosine
    data.verify()
    return dict(status='passed',checks=checks,previous_VN_only_reproduced=baseline,immutable_inputs_unchanged=True,
        text_cache_parity=dict(texts=len(values),minimum_cosine=minimum_cosine,fresh_meta=meta),
        COCO_scored=False,xm3600_scored=False)


def export(data,results,verification):
    comparisons={}
    for a,b in [(1,0),(2,0),(2,1)]:
        name=results[a]['experiment']+'_minus_'+results[b]['experiment'];comparisons[name]={}
        for key in ('selected','final_fixed_budget'):
            comparisons[name][key]={split:bootstrap(
                results[a][key][split]['retrieval']['ranks']['t2i'],
                results[b][key][split]['retrieval']['ranks']['t2i'],groups(data,split)) for split in ('VN_dev','wiki_dev')}
            comparisons[name][key].update({split+'_i2t':bootstrap(
                results[a][key][split]['retrieval']['ranks']['i2t'],
                results[b][key][split]['retrieval']['ranks']['i2t'],groups(data,split,'i2t')) for split in ('VN_dev','wiki_dev')})
    baseline=results[0];candidates=[]
    for r in results[1:]:
        vn=r['selected']['VN_dev']['retrieval']['t2i']['recall@1']-baseline['selected']['VN_dev']['retrieval']['t2i']['recall@1']
        wiki=r['selected']['wiki_dev']['retrieval']['t2i']['recall@1']-baseline['selected']['wiki_dev']['retrieval']['t2i']['recall@1']
        if vn>=-.005 and wiki>0:candidates.append(r['experiment'])
    recommendation=candidates[0] if candidates else 'WIKI_00pct'
    report=read_json(ROOT/'reports/tinyclip_wiki_pilot.json')
    report.update(status='complete',verification=verification,comparisons=comparisons,
        pilot_recommendation=recommendation,recommendation_scope='Exploratory one-seed pilot; wiki metrics now examined, so this dev is no longer blind.')
    write_json(ROOT/'reports/tinyclip_wiki_pilot.json',report)
    write_json(ROOT/'reports/tinyclip_wiki_pilot_verification.json',verification)
    lines=['# Pilotaż wiki + VisualNews','',
        'Frozen TinyCLIP-39M (oryginalny embedding 512D) i PolDense-32M; losowy Linear 512→384, świeże skale sigmoid. 196 994 trenowalne parametry; seed 42.',
        'Trzy treningi od wspólnych wag losowych: po 5500 kroków i 1 405 780 prezentacji, LR 0.001, AdamW, bez augmentacji. Wiki zastępuje 0%, 3% albo 10% slotów VN.',
        '**Checkpoint wybierany wyłącznie po VN dev.** Wiki dev, wiki train probe i wynik w stałym kroku 5500 służą diagnozie. Nie oceniano COCO ani XM3600.','',
        f'Zamrożony eksport: {data.audit["snapshot"]["rows"]} par, {data.audit["snapshot"]["created_at"]}; poprawnie pobrane {data.audit["download_ready"]}, niepobrane {data.audit["download_failed"]}.',
        f'Po kontroli: train {data.audit["train"]["rows"]} par / {data.audit["train"]["images"]} obrazów / {data.audit["groups"]["train_components"]} komponentów; dev {data.audit["dev"]["rows"]} par / {data.audit["dev"]["images"]} obrazów / {data.audit["groups"]["dev_components"]} komponentów.',
        'Podział łączy rzeczywiste encje Wikidata, źródłowy SHA1, piksele, podpisy i pHash≤6. Komponent z rozpoznanym powiązaniem z chronionymi danymi jest wykluczany w całości. Oryginalny FULL_CLEAN i split VN pozostają bez zmian.',
        'Część obrazów pobrano jako oryginały z kontrolą SHA1, pozostałe jako oficjalne rendery Wikimedia 1280 px po potwierdzeniu SHA1 oryginału przez API. Tożsamość pobranych bajtów i przetwarzanie do wejścia CLIP są zapisane w audycie.','',
        '| Wariant | VN dev t2i R@1 / R@5 / R@10 | VN i2t R@1 | Wiki dev t2i R@1 / R@5 / R@10 | Wiki i2t R@1 | Wiki train probe R@1 | Best krok |',
        '|---|---:|---:|---:|---:|---:|---:|']
    csv_rows=[]
    for r in results:
        selected=r['selected'];v=selected['VN_dev']['retrieval'];w=selected['wiki_dev']['retrieval'];p=selected['wiki_train_probe']['retrieval']
        triple=lambda m:' / '.join(f'{100*m["t2i"][f"recall@{k}"]:.2f}' for k in (1,5,10))
        lines.append(f'| {r["experiment"]} | {triple(v)} | {100*v["i2t"]["recall@1"]:.2f} | {triple(w)} | {100*w["i2t"]["recall@1"]:.2f} | {100*p["t2i"]["recall@1"]:.2f} | {r["best_updates"]} |')
        best=next(h for h in r['history'] if h['updates']==r['best_updates'])
        csv_rows.append(dict(experiment=r['experiment'],feature_source='TinyCLIP original projected 512D',projector='Linear512to384',
            vision_trainable_blocks=0,trainable_params=196994,wiki_percent=r['wiki_percent'],best_updates=r['best_updates'],
            best_budget_epoch=r['best_updates']/275,VN_t2i_R1=v['t2i']['recall@1'],VN_t2i_R5=v['t2i']['recall@5'],VN_t2i_R10=v['t2i']['recall@10'],
            VN_i2t_R1=v['i2t']['recall@1'],wiki_t2i_R1=w['t2i']['recall@1'],wiki_i2t_R1=w['i2t']['recall@1'],
            train_loss=best['train_loss'],VN_loss=selected['VN_dev']['loss'],wiki_loss=selected['wiki_dev']['loss'],
            projector_grad_mean=best['gradient_norm']['projector']['mean'] if best['gradient_norm'] else None,
            peak_vram_bytes=max(h['peak_vram_bytes'] for h in r['history']),wiki_presentations_selected=best['presentations'].get('wikimedia',0),
            wiki_presentations_total=r['presentations'].get('wikimedia',0),checkpoint_sha256=r['checkpoint_sha256']))
    lines+=['','## Różnice względem VN bez wiki','',
        'Warunkowe 95% przedziały z parowanego bootstrapu komponentów zapytań; nie obejmują zmienności seeda ani wyboru checkpointu. Liczby w punktach procentowych.','',
        '| Wariant − 0% | VN dev Δ [95% CI] | Wiki dev Δ [95% CI] |','|---|---:|---:|']
    for r in results[1:]:
        c=comparisons[r['experiment']+'_minus_'+baseline['experiment']]['selected'];values=[]
        for split in ('VN_dev','wiki_dev'):
            x=c[split];lo,hi=x['interval_95_percentage_points'];values.append(f'{x["delta_percentage_points"]:+.2f} [{lo:+.2f}, {hi:+.2f}]')
        lines.append(f'| {r["experiment"]} | '+' | '.join(values)+' |')
    lines+=['','## Stały pełny budżet — diagnoza','',
        '| Wariant | VN dev R@1, krok 5500 | Wiki dev R@1, krok 5500 | Wiki train probe R@1, krok 5500 |','|---|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["experiment"]} | '+' | '.join(f'{100*r["final_fixed_budget"][s]["retrieval"]["t2i"]["recall@1"]:.2f}' for s in ('VN_dev','wiki_dev','wiki_train_probe'))+' |')
    lines+=['','## Kierunek i2t — diagnoza','',
        '| Wariant − 0% | VN dev i2t Δ [95% CI] | Wiki dev i2t Δ [95% CI] |','|---|---:|---:|']
    for r in results[1:]:
        c=comparisons[r['experiment']+'_minus_'+baseline['experiment']]['selected'];values=[]
        for split in ('VN_dev_i2t','wiki_dev_i2t'):
            x=c[split];lo,hi=x['interval_95_percentage_points'];values.append(f'{x["delta_percentage_points"]:+.2f} [{lo:+.2f}, {hi:+.2f}]')
        lines.append(f'| {r["experiment"]} | '+' | '.join(values)+' |')
    tokens=data.features['text_tokens']
    lines+=['',f'Długość wejścia tekstowego: stały limit {tokens["max_length"]} tokenów. '
        f'Przekracza go {tokens["exceeding_max"]} z {len(data.features["text_values"])} unikalnych podpisów wiki; '
        f'mediana {tokens["median"]:.0f}, maksimum {tokens["maximum"]}. W tej serii pozostawiono wspólny limit bez zmian.']
    lines+=['',f'Wariant wskazany przez zapisaną przed treningami regułę pilotażu: **{recommendation}** (najniższy udział wiki z dodatnią różnicą wiki i spadkiem VN nie większym niż 0.5 p.p.).',
        'Wynik opisuje reakcję na tę niewielką pulę wiki; nie przesądza o wartości większego korpusu. Wiki dev jest małe, podpisy przeszły automatyczną ocenę, a model może zapamiętywać treści powtarzane przez oversampling. Nie porównujemy surowych wyników wiki i VN między sobą: mają inne galerie i trudność.',
        'Rozdzielność encji dotyczy wiki train względem wiki dev; VN nie ma kompletnego mapowania osób i obiektów do Wikidata, więc wspólne encje między VN a wiki nie są wykluczone.',
        'Wiki dev zostało teraz obejrzane diagnostycznie; kolejne iteracje wymagają uwzględnienia tego w interpretacji. Brak znanych powiązań COCO dotyczy naszego treningu, bez gwarancji dla pretreningu gotowych encoderów lub nierozpoznanych kopii.',
        '', '![Krzywe treningu](tinyclip_wiki_pilot.png)','',
        f'Kontrola: {verification["status"]}; odtworzono metryki i loss wszystkich wybranych i końcowych checkpointów, budżety oraz wcześniejszy wariant VN-only.']
    (ROOT/'reports/tinyclip_wiki_pilot.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (ROOT/'reports/tinyclip_wiki_pilot.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    return recommendation


def plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,split,title in zip(axes,('VN_dev','wiki_dev','wiki_train_probe'),('VN dev: checkpoint selection','Wiki dev: diagnostic','Wiki train probe: diagnostic')):
        for r in results:
            x=[h['updates'] for h in r['history']];y=[100*h[split]['t2i']['recall@1'] for h in r['history']]
            ax.plot(x,y,label=f'{r["wiki_percent"]}% wiki')
            best=next(h for h in r['history'] if h['updates']==r['best_updates'])
            ax.scatter([best['updates']],[100*best[split]['t2i']['recall@1']],s=35)
        ax.set_title(title);ax.set_xlabel('Updates');ax.set_ylabel('t2i R@1 (%)');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Frozen encoders; one seed; dots = checkpoints selected only by VN dev')
    fig.tight_layout();fig.savefig(ROOT/'reports/tinyclip_wiki_pilot.png',dpi=170);plt.close(fig)


def main():
    import fcntl
    with (study.vn.OUT/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        data=study.Data();results=[read_json(study.OUT/f'WIKI_{p:02d}pct/result.json') for p in (0,3,10)]
        verification=verify(data,results);recommendation=export(data,results,verification);plot(results)
        write_json(study.OUT/'status.json',dict(status='complete',experiments=3,pilot_recommendation=recommendation,
            verification='passed',COCO_scored=False,xm3600_scored=False))
        print('WIKI_PILOT_COMPLETE',recommendation,flush=True)


if __name__=='__main__':main()
