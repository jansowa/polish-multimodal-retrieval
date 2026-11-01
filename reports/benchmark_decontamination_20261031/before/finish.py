"""CPU-only closed-generation bootstrap, final accounting and readable report."""
from config import *
import confidence_intervals as ci
from common import verify_marker


def main():
    ci.REPORT=REPORT;ci.STORAGE=STORE;ci.COORD=COORD;ci.main()
    finalists=verify_marker(COORD/'A_FINALISTS_FROZEN.json');done=verify_marker(COORD/'A_GPU_DONE.json')
    assert done['A_FINALISTS_FROZEN_sha256']==sha256(COORD/'A_FINALISTS_FROZEN.json')
    for m in finalists['models']:
        assert sha256(m['path'])==m['sha256'] and sha256(m['training_manifest_path'])==m['training_manifest_sha256']
    for p,h in read(SERIES/'preservation_before.json').items():assert sha256(ROOT/p)==h
    report=read(REPORT/'training_report.json');audit=read(REPORT/'image_audit.json');downloads=read(REPORT/'download_summary.json')
    selected=report['selection']['chosen_Wiki'];scores=report['selection']['variant_mean_sd']
    drop=100*(scores['R1']['mean']-scores[selected]['mean']);guardrail=drop<=2
    lines=['# Wiki recovery — generacja 2, 31.10.2026','',
        f'Pobieranie: {downloads["counts"]}. Archiwum: {downloads["stored_asset_bytes"]/1e9:.2f} GB; fingerprinty oryginałów policzone przed zmniejszeniem.',
        'Zapis: cały kadr do1536px JPEG92 oraz osobne, bezstratne wejście TinyCLIP224 z bitową zgodnością preprocessingu. Pierwsze20 oryginałów testowych zachowano.',
        f'Train Wiki: {audit["selected_Wiki_rows"]} rekordów, {audit["selected_Wiki_images"]} obrazów, {audit["selected_Wiki_components"]} pełnych komponentów. Wszystkie galerie chronione; pHash nie tworzy qrels.',
        'R0/R1: reuse wszystkich3 seedów po sprawdzeniu niezmienności danych, cech i reload metryk/lossu. R2/R3: świeże seedy42, następnie wybrany Wiki43/44. Recepta5500 kroków/1405780 prezentacji niezmieniona.',
        '',f'Wybrano {selected} według Wiki dev seed42. Spadek VN względem R1: {drop:.3f} p.p.; guardrail: {"zaliczony" if guardrail else "niezaliczony — brak rekomendacji zastąpienia R1"}.','',
        '| Wariant | Galeria | t2i R@1 mean±SD | i2t R@1 mean±SD |','|---|---|---:|---:|']
    for v,datasets in report['variants'].items():
        for name,directions in datasets.items():
            a=directions['t2i']['recall@1'];b=directions['i2t']['recall@1']
            lines.append(f'| {v} | {name} | {100*a["mean"]:.3f}±{100*a["sd"]:.3f}% | {100*b["mean"]:.3f}±{100*b["sd"]:.3f}% |')
    lines+=['','Wiki dev jest ID dla modeli uczonych na Wiki, źródłowo OOD dla R0/R1. VNdev jest ID; XTD to kontrola COCO/tłumaczeń. Nie łączymy galerii w jeden ranking.',
        'Wszystkie4 warianty seed42: training_report.json i osobne katalogi evaluation; R@1/5/10, MRR, NDCG@10. CI: conditional_CI.json, paired bootstrap komponentów10000, seed20261023, oddzielnie od SD seedów.',
        'V+T70470528; V38587392; T31883136; head196992 osobno. Native/recycled/translated zachowane; brak nowych ręcznych etykiet, brak decyzji o publikacji. Licencje, autorzy i strony źródłowe w rekordach.',
        'Generacja1 pozostała niezmieniona. Nowe modele wymagają nowego audytu OOD B względem uniwersum generacji2. Nie odczytywano wyników B. GPU zwolnione.']
    (REPORT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    used=reserve(remaining=0)
    value=dict(schema_version=1,study_id='data-ood-20261031-wiki-recovery',generation=2,owner='A',phase='complete',status='complete',
        updated_at=utc(),report=artifact('report',REPORT/'REPORT.md'),selected_Wiki_images=audit['selected_Wiki_images'],
        VN_drop_vs_R1_pp=drop,VN_guardrail_pass=guardrail,new_task_committed_bytes=used,physical_disk=disk_snapshot(),GPU_released=True)
    atomic_json(REPORT/'STATUS.json',value);atomic_json(COORD/'A_STATUS.json',value);atomic_json(COORD.parent/'A_STATUS.json',value)
    print('OVERNIGHT_COMPLETE',__import__('json').dumps(value),flush=True)


if __name__=='__main__':main()
