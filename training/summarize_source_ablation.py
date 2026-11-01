"""Descriptive findings for the closed study; no training or model selection."""
import sys
import json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from pl_mmb_training import source_ablation as study
from pl_mmb_training.common import ROOT,read_json,write_json,file_sha
from verify_tinyclip_generalization import bootstrap


def interval(value):
    lo,hi=value['interval_95_percentage_points']
    return f'{value["delta_percentage_points"]:+.2f} [{lo:+.2f}, {hi:+.2f}]'


def main():
    source=ROOT/'reports/tinyclip_source_ablation.json'
    report=read_json(source)
    verified=read_json(ROOT/'reports/tinyclip_source_ablation_verification.json')
    assert report['status']=='complete' and verified['status']=='passed'
    assert not report['COCO_used_for_selection'] and not report['wiki_used']
    runs={r['experiment']:r for r in report['results']}
    tests=report['COCO_finalists']
    compact=read_json(ROOT/'training/runs/vision-four/common/evaluation_compact.json')
    manifest=read_json(ROOT/'training/runs/e1-full/cache.json')
    blacklist=read_json(ROOT/'training/runs/e1-full/blacklist.json')
    coco=compact['datasets']['mscoco_t2i'];uf=study.UnionFind()
    for r in coco:
        positive=manifest['images'][r['images'][0]]
        pixel=blacklist['images'][positive]['pixels_sha256']
        text=manifest['texts'][r['texts'][0]]
        uf.join(['pixel:'+pixel,'text:'+text])
    roots=[uf.find('pixel:'+blacklist['images'][manifest['images'][r['images'][0]]]['pixels_sha256']) for r in coco]
    labels={key:i for i,key in enumerate(sorted(set(roots)))}
    groups=np.array([labels[key] for key in roots])
    for name,record in tests.items():
        assert [p['id'] for p in record['predictions']]==[r['id'] for r in coco]
        assert record['checkpoint_sha256']==runs[name]['checkpoint_sha256']
    contrasts={a+'_minus_VN_only':bootstrap([p['rank'] for p in tests[a]['predictions']],
        [p['rank'] for p in tests['VN_only']['predictions']],groups) for a in ('VN_nonCOCO_AR','VN_FULL')}
    for c in contrasts.values():
        c['caveat']='Conditional paired interval on fixed benchmark queries; no training-seed or checkpoint/model-selection uncertainty.'
    additions={};vn=runs['VN_only']['train']
    for code,source_name in study.SOURCES.items():
        r=runs['VN_'+code]
        additions[source_name]=dict(rows=r['train']['rows']-vn['rows'],images=r['train']['images']-vn['images'],
            add_vs_VN=report['comparisons']['VN_'+code+'_minus_VN_only'],
            keep_vs_replace_by_VN=report['comparisons']['VN_FULL_minus_VN_FULL_drop_'+code])
    write_json(ROOT/'reports/tinyclip_source_ablation_findings.json',dict(status='complete',
        report_sha256=file_sha(source),verification_sha256=file_sha(ROOT/'reports/tinyclip_source_ablation_verification.json'),
        test_component_contrasts=contrasts,source_additions=additions,training_after_test=False,
        COCO_used_for_selection=False,wiki_used=False,xm3600_used=False))
    lines=['# Wnioski: COCO bez nadzorowanego treningu na COCO','',
        '**Samo VisualNews: COCO R@1 = 45,0%, R@5 = 75,0%, R@10 = 84,2%.**',
        'To dotychczasowy lokalny MMEB MSCOCO_t2i: 1000 zapytań i po 1000 kandydatów. Nie jest to oficjalny COCO 5k.',
        'Zamrożone TinyCLIP-39M i PolDense; Linear 512→384 od losowych wag, świeży loss. Żadnych wag projectora wcześniej trenowanych na COCO. Checkpoint i konfiguracje wyłącznie po VN dev.','',
        '| Dane | COCO w naszym train | COCO test R@1 | R@5 | R@10 | COCO val R@1 — post hoc | VN dev R@1 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name in ('VN_only','VN_nonCOCO_AR','VN_FULL'):
        result=runs[name];t=tests[name]['local'];v=report['COCO_validation_posthoc'][name]
        lines.append(f'| {name} | {"nie" if result["no_known_COCO_training"] else "tak — kontrola"} | '+
            ' | '.join(f'{100*t[f"recall@{k}"]:.2f}%' for k in (1,5,10))+
            f' | {100*v["t2i"]["recall@1"]:.2f}% | {100*result["selected_VN_dev"]["t2i"]["recall@1"]:.2f}% |')
    lines+=['',f'Różnice na teście (warunkowy paired bootstrap po {len(labels)} grupach pozytywnych obrazów/tekstów; 1000 zapytań nie jest niezależnymi obserwacjami):','']
    for name,c in contrasts.items():lines.append(f'- {name}: {interval(c)} p.p.; 95% przedział nie obejmuje zmienności seedów ani selekcji checkpointu.')
    lines+=['','## Wkład każdego źródła','',
        'Poniższe różnice dotyczą **COCO validation ocenionej po zamknięciu wszystkich treningów**, nie dodatkowych testów kolejnych konfiguracji.',
        'Dodanie źródła: mieszanka 50/50 z VN względem samego VN. Wkład w pełnej mieszance: pełna mieszanka względem zastąpienia źródła przez VN, przy zachowaniu ekspozycji pozostałych źródeł.','',
        '| Źródło | Pary źródła | Różne obrazy źródła | Dodanie do VN: Δ p.p. [95% CI] | Wkład w pełnej mieszance: Δ p.p. [95% CI] |',
        '|---|---:|---:|---:|---:|']
    for code,source_name in study.SOURCES.items():
        a=additions[source_name]
        lines.append(f'| {source_name} | {a["rows"]:,} | {a["images"]:,} | {interval(a["add_vs_VN"]["COCO_val_posthoc"])} | {interval(a["keep_vs_replace_by_VN"]["COCO_val_posthoc"])} |')
    lines+=['','## Decyzja po wstępnym screeningu','',
        '- Do recepty bez COCO wybieramy obecnie samo VN. Wyboru dokonano po VN dev, przed otwarciem ocen COCO.',
        '- Samo 45,0% kontra 42,4% nie dowodzi przewagi na populacji testowej: przedział różnicy dla strict ARO względem VN obejmuje zero. To ważne ograniczenie jednego seeda i małej liczby niezależnych grup benchmarkowych.',
        '- ARO Attribution i Relation przy wadze 50% obniżają wyniki względem samego VN. Usunięcie ich z pełnej mieszanki poprawia VN dev, a małe różnice COCO val nie dowodzą korzyści z ich obecności. Nie ma podstaw, aby domyślnie nadawać im tak dużą wagę w tej recepcie.',
        '- SugarCrepe ma mały, niepewny dodatni efekt na COCO val; jest oparte na COCO, więc nie kwalifikuje się do gałęzi bez COCO.',
        '- M-BEIR MSCOCO daje duży dodatni efekt na COCO val; pełni rolę kontroli ekspozycji na domenę COCO i nie kwalifikuje się do gałęzi bez COCO.',
        '- Dane ARO/SugarCrepe użyto jako dodatnich par. Ich jawne błędne captiony/hard negatives nie uczestniczyły w lossie. Wynik nie rozstrzyga potencjału innego celu kompozycyjnego ani mniejszej proporcji ARO.','',
        '## Granice wniosku','',
        '- To jeden seed, zgodnie z założeniem screeningu. Różnice i przedziały opisują tę receptę i dostępne pule, nie uniwersalną jakość źródeł.',
        '- Budżet maksymalny był równy; wybrane checkpointy mogą wykorzystywać różne jego części. Reguła wyboru po VN dev może sprzyjać VN kosztem innych domen.',
        '- Liczba wierszy ARO jest duża, ale liczba różnych obrazów znacznie mniejsza niż w VN. Przy stałym koszcie zamiana połowy batcha zmienia również różnorodność obrazów i strukturę negatywów.',
        '- Bez COCO oznacza brak znanego pochodzenia COCO w naszym nadzorowanym treningu i brak wyboru po COCO. Nie dowodzi braku COCO w oryginalnym pretreningu gotowych encoderów ani braku niezidentyfikowanych kopii.',
        '- VN train/dev pochodzi z tego samego pipeline’u i dwóch źródeł newsowych, z modelowymi captionami. Benchmark COCO to istniejący, historycznie oglądany zestaw.',
        '- Nie uruchamiano danych wiki, retrieval XM3600 ani nowych treningów po poznaniu wyników COCO.','',
        '[Pełne 14 wariantów, logi i porównania](tinyclip_source_ablation.md).',
        '[Weryfikacja checkpointów i źródeł](tinyclip_source_ablation_verification.json).']
    (ROOT/'reports/tinyclip_source_ablation_findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(test_contrasts=contrasts),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
