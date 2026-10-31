"""CPU-only accounting and report; never modifies frozen readiness artifacts."""
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from preflight import ROOT,REPORT,SERIES,STORAGE,read,rows
from common import COORD,atomic_json,disk_snapshot,sha256,verify_marker,reserve


def main():
    finalists=verify_marker(COORD/'A_FINALISTS_FROZEN.json');done=verify_marker(COORD/'A_GPU_DONE.json')
    assert done['A_FINALISTS_FROZEN_sha256']==sha256(COORD/'A_FINALISTS_FROZEN.json')
    assert done['further_A_cuda_planned'] is False
    assert len(finalists['models'])==9 and len(finalists['representative_model_ids'])==3
    assert all(m['text_prefix']=='[query]: ' for m in finalists['models'])
    assert read(REPORT/'conditional_CI.json')['status']=='complete'
    summary=read(REPORT/'training_report.json');selection=summary['selection'];wiki=selection['chosen_Wiki']
    preservation=read(SERIES/'preservation_before.json')
    for p,h in preservation.items():assert sha256(ROOT/p)==h
    frozen=verify_marker(COORD/'A_INPUTS_FROZEN.json')
    for item in frozen['artifacts']:
        assert sha256(item['path'])==item['sha256']
    for model in finalists['models']:assert sha256(model['path'])==model['sha256']
    for model in finalists['models']:assert sha256(model['training_manifest_path'])==model['training_manifest_sha256']
    vn1=selection['variant_mean_sd']['R1']['mean'];vnw=selection['variant_mean_sd'][wiki]['mean']
    guardrail=vnw>=vn1-.02
    files={volume:sum(p.stat().st_size for root in roots for p in root.rglob('*') if p.is_file())
        for volume,roots in {'C':[STORAGE],'D':[REPORT,ROOT/'experiments/data_study_20261030',ROOT/'experiments/data_ood_common_20261030']}.items()}
    new_runs=[p for p in (STORAGE/'runs').rglob('result.json')]
    elapsed=sum(read(p)['elapsed_seconds'] for p in new_runs)
    counts=read(REPORT/'image_audit.json');available=read(REPORT/'original_availability.json')
    wiki_rows=list(rows(STORAGE/'Wiki_train.jsonl'));wiki_components=len({r['component'] for r in wiki_rows})
    vn_rows=list(rows(STORAGE/'new_VN_train.jsonl'))
    data_counts={name:dict(rows=len(rs),unique_pixel_images=len({r['pixels_sha256'] for r in rs}),
        transitive_components=len({r['component'] for r in rs}),routes=dict(Counter(r['route'] for r in rs)))
        for name,rs in [('new_VN',vn_rows),('Wiki',wiki_rows)]}
    accounting=dict(status='complete',preservation_all_SHA_passed=True,files_bytes=files,
        new_training_runs=len(new_runs),new_training_seconds=elapsed,
        wall_times=dict(image_audit_seconds=counts['elapsed_seconds'],
            study_elapsed_seconds_since_registration=time.time()-(REPORT/'registration.json').stat().st_mtime,
            final_cached_feature_preparation_seconds=read(REPORT/'feature_preparation_timing.json')['seconds']),
        disk=disk_snapshot(),data_counts=data_counts,Wiki_components=wiki_components,VN_drop_vs_R1_pp=100*(vn1-vnw),
        final_VN_guardrail_pass=guardrail,
        recommendation='pilot only; no production replacement inference' if guardrail else 'Do not recommend replacing R1: VN loss exceeds 2 pp',
        final_OOD_model_ids=finalists['representative_model_ids'])
    atomic_json(REPORT/'final_accounting.json',accounting)
    lines=['# Audyt i pilot nowych danych — sesja A, 30.10.2026','',
        'Wykonano wyłącznie zarejestrowane R0/R1/R2/R3. Nie uruchamiano sesji B, nowego legacy COCO, XM3600, FT/LoRA/KD ani nowych backbone’ów.',
        '',f'Zamrożony eksport: 132 875 VN i 24 125 Wiki. Pilot: {len(vn_rows):,} nowych rekordów VN ({data_counts["new_VN"]["unique_pixel_images"]:,} obrazów według pixel SHA; {data_counts["new_VN"]["transitive_components"]:,} komponentów) i {len(wiki_rows)} Wiki ({data_counts["Wiki"]["unique_pixel_images"]} obrazów; {wiki_components} komponentów).',
        f'Wiki: {available["counts"]["missing_local_original"]:,} rekordów bez lokalnego oryginału; 401 dostępnych miało już bezpośrednią chronioną tożsamość. Spośród pozostałych 162 pełne komponenty dopuściły 28.',
        f'VN: oficjalne caption/DID/path i artykuły sprawdzone bez SQLite; z 20 000 deterministycznie wybranych oryginałów odrzucono {counts["new_VN_decoded"]-counts["new_VN_accepted"]}. Powody wielokrotne: `{json.dumps(counts["new_VN_rejection_labels"],ensure_ascii=False)}`.',
        '', 'Stare FULL/VN zachowują oryginalne podpisy. Nowe kopie starych komponentów wykluczono. pHash≤6 służy wykluczeniom, nie tworzy nowych pozytywnych qrels.',
        '', 'Każdy nowy run: 5500 kroków, 1 405 780 prezentacji, w tym 702 880 FULL; batch256 i ostatni145 co275. R0/R1 używają dokładnie historycznego samplera. Wiki zastępuje tylko sloty VN: 42 173 prezentacje dla3% lub140 578 dla10%.',
        'Enkodery frozen, świeży Linear512→384 i loss SigLIP; FP32 head/loss/cache, BF16 autocast przy ekstrakcji enkoderów. Wybór checkpointu: VNdev t2iR1, potem meanR1, potem wcześniejszy krok, co275 i krok0.',
        '', '| Wariant | Seed42 VNdev t2i R@1 | Seed42 Wiki dev t2i R@1 | Wybrany krok |',
        '|---|---:|---:|---:|']
    for v,r in summary['seed42_all_four_variants'].items():
        p=STORAGE/'evaluation'/v/'seed-42/wiki_native_dev.json'
        if not p.exists():p=STORAGE/'runs'/v/'seed-42/Wiki_dev.json'
        metric=read(p)['directions']['t2i']['metrics']['recall@1'] if p.exists() else None
        lines.append(f'| {v} | {100*r["selected_VN_dev"]["t2i"]["recall@1"]:.3f}% | {100*metric:.3f}% | {r["best_updates"]} |' if metric is not None else
            f'| {v} | {100*r["selected_VN_dev"]["t2i"]["recall@1"]:.3f}% | brak | {r["best_updates"]} |')
    lines+=['',f'Wybrano {wiki} według Wiki dev seed42; reguła i niekorzystne wyniki zachowane.',
        f'Spadek VN względem R1 po trzech seedach: {100*(vn1-vnw):.3f} p.p.; guardrail≤2p.p.: {"zaliczony" if guardrail else "niezaliczony — brak rekomendacji zastąpienia R1"}.','',
        '| Wariant | Galeria | Status | t2i R@1 mean±SD | i2t R@1 mean±SD |','|---|---|---|---:|---:|']
    for v,ds in summary['variants'].items():
        for name,directions in ds.items():
            a=directions['t2i']['recall@1'];b=directions['i2t']['recall@1']
            role='ID' if name=='VN_dev' or (name=='wiki_native_dev' and v==wiki) else ('źródłowo OOD' if name=='wiki_native_dev' else 'kontrola COCO/tłumaczenia')
            lines.append(f'| {v} | {name} | {role} | {100*a["mean"]:.3f}±{100*a["sd"]:.3f}% | {100*b["mean"]:.3f}±{100*b["sd"]:.3f}% |')
    lines+=['','Wszystkie warianty, seed42 (po zamknięciu selekcji, galerie posthoc):','',
        '| Wariant | Galeria | t2i R@1 | i2t R@1 |','|---|---|---:|---:|']
    for v in ('R0','R1','R2','R3'):
        for name in summary['variants']['R0']:
            score=read(STORAGE/'evaluation'/v/'seed-42'/(name+'.json'))
            a=score['directions']['t2i']['metrics']['recall@1'];b=score['directions']['i2t']['metrics']['recall@1']
            lines.append(f'| {v} | {name} | {100*a:.3f}% | {100*b:.3f}% |')
    lines+=['','Pozostałe R@5/10, MRR i NDCG@10 oraz wartości seedów: `training_report.json`. Warunkowe CI różnic reprezentantów: `conditional_CI.json`, paired bootstrap komponentów 10 000, seed20261023; osobno od mean±SD seedów. OOD B otrzyma dokładnie trzech reprezentantów; dziewięć checkpointów pozostaje analizą seedów. R0 pozostaje seed44; pozostali to mediana VNdev, remis→niższy seed.',
        '', 'V+T:70 470 528; V:38 587 392; T:31 883 136. Head196 992 liczony osobno.',
        f'Nowe runy: {len(new_runs)}; suma zmierzonego czasu pętli treningowych: {elapsed:.2f} s. Nowe pliki: C: {files["C"]/1024**3:.3f} GiB, D: {files["D"]/1024**2:.2f} MiB. Czas audytu: {counts["elapsed_seconds"]:.2f} s. Bieżące wolne miejsce i szczegóły: `final_accounting.json`. Czas od prerejestracji obejmuje przerwę na wyłączenie komputera, nie jest czasem aktywnych obliczeń.',
        '', 'Ograniczenia: tylko28 dostępnych i rozłącznych Wiki; wielokrotne prezentacje tej małej puli ograniczają wnioskowanie o całym eksporcie. Nie zbierano nowych ręcznych adnotacji. Wiki zachowuje native/recycled/translated i metadane źródeł; nie jest w całości ludzkim goldem.',
        'Licencje/atrybucja: zachowane source_page_url, creator/attribution, license i license_url Wiki oraz rewizje i oryginalne źródła VN. Metadane Wiki deklarują CC BY/CC BY-SA/CC0; podpisy recycled/translated pochodzą z pipeline’u. Nie podjęto decyzji o publikacji ani nie uznano licencji zbioru za automatyczne zezwolenie na publikację każdego obrazu VN.',
        '', 'Historyczny baseline sprzed Wiki i wszystkie osiem preservation SHA pozostały identyczne. Nowy release znajduje się w wersjonowanym pliku `Wiki_training_release*.json`; historyczny raise pozostaje bez zmian. Kod w src nie był zmieniany. Test cache wymagał odtworzenia batchów BF16; obrazy i teksty są wtedy bitowo zgodne z adapterem (`parity_resolution.md`, `adapter_cache_parity.json`).',
        '', 'SHA wszystkich dziewięciu checkpointów, manifestów treningowych i rewizje enkoderów: `../data_ood_coordination_20261030/A_FINALISTS_FROZEN.json`. Reprezentanci do OOD: R0_seed44, R1_seed44, R3_seed44. Dodatkowy R2_seed42 służy wyłącznie zamkniętej analizie pilota i nie rozszerza listy OOD.',
        '', 'Handoff: `HANDOFF.md`, markery w `reports/data_ood_coordination_20261030/`. GPU A zakończone; lock zwolniony. Nie odczytywano wyników B.']
    (REPORT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    reserve('A','small-code-and-audit',{'C':0,'D':0},committed_bytes=sum(files.values()))
    atomic_json(COORD/'A_STATUS.json',dict(schema_version=1,study_id='data-ood-20261030',generation=1,owner='A',
        phase='complete',status='complete',report=str(REPORT/'REPORT.md'),GPU_released=True,all_preserved=True))
    print(json.dumps(accounting,ensure_ascii=False))


if __name__=='__main__':main()
