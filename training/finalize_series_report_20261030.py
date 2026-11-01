"""Update the day report after the entire registered B0 comparison closes."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pl_mmb_training.series_20261023 import ROOT, OUT, read, freeze, file_sha, local_path
from pl_mmb_benchmark.report_parameters import parameter_cells

BASE = local_path('/mnt/c/Users/jsowa/benchmark-translations/series-2026-10-23/evaluation')


def preserve(path, target):
    payload = path.read_bytes()
    if target.exists():
        if target.read_bytes() != payload:
            raise ValueError('Prior report snapshot differs: ' + str(target))
    else:
        with target.open('xb') as stream:
            stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def main():
    completion = read(OUT / 'reference_execution_complete_20261029.json')
    comparison = read(OUT / 'reference_new_gallery_comparisons_20261029.json')
    reg_path = OUT / 'reference_eval_registration_v3_20261029.json'
    reg = read(reg_path)
    if completion['status'] != comparison['status'] or comparison['status'] != 'complete':
        raise ValueError('Finish all workers and conditional confidence intervals first')
    if completion['registration_sha256'] != file_sha(reg_path) or comparison['registration_sha256'] != file_sha(reg_path):
        raise ValueError('Final comparison registration differs')
    original = read(OUT / 'preservation_before.json')
    for path, digest in original.items():
        if file_sha(ROOT / path) != digest:
            raise ValueError('Protected original changed: ' + path)
    freeze(OUT / 'preservation_20261030.json', original)
    previous = {name: preserve(OUT / name, OUT / (Path(name).stem + '_before_20261030' + Path(name).suffix))
                for name in ('REPORT.md', 'summary.json')}
    statuses = {
        'P0.1': dict(status='zrobione', note='Wiki397 / 331 tekstów, guard i immutable baseline seed44; brak nowego treningu Wiki.'),
        'P0.2': dict(status='częściowo', note='XTD10 1000 i clean897 gotowe i ocenione; brak istniejących dokładnych par EN do Qwen-vs-human style.'),
        'P0.3': dict(status='częściowo', note='300 zadań i narzędzie gotowe; rzeczywiste ludzkie query pozostają do zebrania.'),
        'P0.4': dict(status='częściowo', note='200 zadań audytu i narzędzie gotowe; rzeczywiste ludzkie oceny pozostają do zebrania.'),
        'P0.5': dict(status='zrobione', note='Rejestracja, prompt sources, adapter parity, cache ranks, full-gallery i efficiency gotowe. Użytkownik wyłączył zadania obraz+tekst i grounding; zakres: tekst↔obraz (evaluation_scope_amendment_20261030.json).'),
        'P0.5.2': dict(status='zrobione', note='wiki_dev_zero_shot_pre_wiki_training.json zachowany bez nadpisania.'),
        'P0.5.3': dict(status='zrobione', note='Draft XM istnieje; execution_allowed=false; zero wyników XM3600.'),
        'P1.A': dict(status='zrobione', note='Seedy42 reuse,43/44 fresh; mediana44; legacy i fixed-gallery test już wykonane.'),
        'P1.B0': dict(status='zrobione', note='20 modeli, 4 galerie, 2 tryby, 2 kierunki; paired CI i efficiency; usunięto ukończone wagi.'),
        'P1.B': dict(status='nie zaczęte', note='Po zamknięciu B0: ekstrakcja i 1-seed scout według pierwotnego planu.'),
        'P1.C': dict(status='nie zaczęte', note='Po B: sampler i ARO0/10/20/50, 3 seedy.'),
        'P1.D': dict(status='nie zaczęte', note='Po C: Linear / LNLinear / residual MLP, 3 seedy.'),
        'P2.E': dict(status='częściowo', note='Dossier z linkami i cytatami; USA Today niedostępne, kwalifikacje prawne nierozstrzygnięte.'),
        'P2.F': dict(status='nie zaczęte', note='Text-only modality gap, licencjonowany OPUS i alignment pilot pozostają.'),
        'P2.G': dict(status='nie zaczęte', note='Plan i image audit po B; brak treningu destylacyjnego bez akceptacji.'),
        'expanded_corpus': dict(status='częściowo', note='157000 zamrożonych rekordów, metadata audit; training_ready_rows=0, image/rights audit pending.')}
    runs = []
    for model in reg['reference_models']:
        folder = BASE / 'references' / model['alias']
        completed, cleanup = read(folder / 'complete.json'), read(folder / 'weight_cleanup_complete.json')
        if completed['registration_sha256'] != file_sha(reg_path) or cleanup['status'] != 'complete':
            raise ValueError('Incomplete model: ' + model['alias'])
        runs.append(dict(model=model['alias'], worker_wall_seconds=completed['seconds'],
            removed_weight_bytes=cleanup['removed_weight_bytes'], complete_sha256=file_sha(folder / 'complete.json'),
            cleanup_sha256=file_sha(folder / 'weight_cleanup_complete.json')))
    wiki_entries = comparison['datasets']['wiki_native_dev']['v2_no_instructions']
    primary = max((entry for entry in wiki_entries if entry['model']['same_size_class_le_100M']
                   and not entry['model']['english_text_only_control']), key=lambda entry: entry['metrics']['t2i']['recall@1'])
    checkpoint_sha = read(OUT / 'vn_full_seeds_posthoc.json')['checkpoint_sha256']
    freeze(OUT / 'xm3600_gate_registration_v2_20261030.json', dict(status='draft_requires_user_approval',
        execution_allowed=False, XM3600_scored=False, previous_draft_sha256=file_sha(OUT / 'xm3600_gate_registration.json'),
        our_model=dict(alias='VN_FULL_seed44', checkpoint_sha256=checkpoint_sha),
        reference_models=reg['reference_models'], primary_reference=primary['model'],
        comparator_rule='Highest t2i R@1 on Wiki native dev v2 among registered <=100M Polish-capable references; selected only from non-XM results',
        comparator_selection_result_sha256=file_sha(OUT / 'reference_new_gallery_comparisons_20261029.json'),
        proposed_gate=dict(primary='t2i R@1 paired delta versus primary reference', minimum_point_gain_pp=2.0,
            lower_95CI_above_pp=0.0, replicates=10000, seed=20261023, unit='transitive component'),
        protocol='same frozen full gallery and qrels for all participants, v2 primary; author mode secondary',
        remaining_before_execution=['user approval', 'freeze final dataset/protocol identity without scoring',
                                    'joint approved runner; current scorer deliberately rejects XM3600']))
    disk = {str(path): dict(zip(('total', 'used', 'free'), shutil.disk_usage(path)))
            for path in (local_path('/mnt/c/'), local_path('/mnt/d/'))}
    expanded = read(OUT / 'expanded_corpus_metadata_audit_20261029.json')
    result_hashes = {str(p.relative_to(BASE)): file_sha(p) for p in sorted(BASE.rglob('*.json'))}
    result_index_sha = freeze(OUT / 'evaluation_result_index_20261030.json', result_hashes)
    artifacts = {str(p.relative_to(OUT)): file_sha(p) for p in sorted(OUT.rglob('*'))
                 if p.is_file() and p.suffix in ('.json', '.jsonl', '.md')
                 and p.name not in ('REPORT.md', 'summary.json', 'artifact_index_20261030.json', 'report_completion_20261030.json')}
    index_sha = freeze(OUT / 'artifact_index_20261030.json', artifacts)
    decisions = [name for name in artifacts if 'decision' in name or 'registration' in name]
    summary = dict(status='P1.B0_complete_further_series_pending', date='2026-10-30',
        tasks=statuses, reference_registration_sha256=file_sha(reg_path), selected_finalist='VN_FULL_seed44',
        checkpoint_sha256=checkpoint_sha,
        reference_models=20, reference_result_files=160, directions_per_result=2,
        conditional_paired_bootstrap=dict(replicates=10000, seed=20261023, unit='transitive component',
            caveat='fixed gallery/qrels; not training uncertainty; no multiplicity correction'),
        reference_runs=runs, reference_worker_wall_seconds=sum(r['worker_wall_seconds'] for r in runs),
        GPU_active_seconds=None, time_caveat='worker wall time includes CPU efficiency, preprocessing and loading; not GPU kernel time',
        P1A_reported_GPU_seconds_new_runs=read(OUT / 'vn_full_seeds_training_summary.json')['GPU_seconds_new_runs'],
        evaluation_result_index_sha256=result_index_sha,
        disk_before_reference=read(OUT / 'reference_execution_sequential_decision_20261029.json'), disk_after=disk,
        expanded_corpus=expanded, XM3600_scored=False, original_data_preserved=True,
        artifacts_sha256_index='artifact_index_20261030.json', artifact_index_sha256=index_sha,
        decisions=decisions, previous_report_sha256=previous,
        source_reports=['OUR_NEW_GALLERIES_20261029.md', 'REFERENCE_COMPARISON_20261029.md', 'LICENSING_DOSSIER_20261030.md'])
    lines = ['# Stan serii — 30.10.2026', '',
        '**P1.B0 ukończone: 20 zarejestrowanych modeli porównawczych, 4 galerie i 2 tryby promptów.** Finalista pozostaje VN_FULL seed44, wybrany wcześniej medianą VN dev.', '',
        'Nasza konfiguracja: vision `wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M`, text `OPI-PIB/PolDense-32M`, Linear 512→384; oba encodery frozen. Około 70,67 mln parametrów inference.', '',
        'Enkodery finalisty (suma V+T | wizja | tekst): '+parameter_cells(comparison['efficiency']['our_finalist']['parameters'])+' mln. Projektor osobno: 0,197 mln. Pozostałe enkodery są poza sumą; audio EmbeddingGemma2 było wyłączone.', '',
        '## Porównanie z konkurencją', '',
        '[Pełne tabele paired Δ i 95% CI, osobno ≤100M, PL-adaptowane i tryby promptów, oraz efektywność](REFERENCE_COMPARISON_20261029.md).',
        '[Wszystkie nasze checkpointy, średnia/SD trzech seedów i porównania kontroli](OUR_NEW_GALLERIES_20261029.md).', '',
        'Poniżej wyróżnione odczyty trybu v2. Δ = nasz seed44 minus referencja. Jednostka CI: komponent; 10 000 replik, seed 20261023. Przedziały są warunkowe; nie obejmują niepewności treningu/pretreningu ani korekty wielokrotnych porównań.', '']
    for dataset in reg['datasets']:
        name = dataset['name']
        entries = comparison['datasets'][name]['v2_no_instructions']
        ours = read(BASE / reg['selected_finalist'] / name / 'v2_no_instructions.json')
        eligible = [e for e in entries if not e['model']['english_text_only_control']]
        best = max(eligible, key=lambda e: e['metrics']['t2i']['recall@1'])
        selected = [e for e in eligible if e['model']['same_size_class_le_100M']]
        if best not in selected:
            selected.append(best)
        lines += ['### ' + name, '',
            f"Galeria: {dataset['images']} obrazów / {dataset['texts']} tekstów. Nasz seed44: t2i R@1 **{100*ours['directions']['t2i']['metrics']['recall@1']:.2f}%**, i2t **{100*ours['directions']['i2t']['metrics']['recall@1']:.2f}%**.", '',
            '| Referencja | Enkodery V+T [M] | Wizyjny [M] | Tekstowy [M] | t2i R@1 | Seed44 minus referencja [95% CI], p.p. |', '|---|---:|---:|---:|---:|---:|']
        for entry in selected:
            metric = entry['paired']['t2i']['metrics']['recall@1']
            lo, hi = metric['ci95_pp']
            counts = comparison['efficiency'][entry['model']['alias']]['parameters']
            lines.append(f"| {entry['model']['alias']} | {parameter_cells(counts)} | {100*entry['metrics']['t2i']['recall@1']:.2f}% | {metric['delta_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
        lines += ['']
    training = read(OUT / 'vn_full_seeds_training_summary.json')
    posthoc = read(OUT / 'vn_full_seeds_posthoc.json')
    vn = training['VN_dev']['t2i']['recall@1']
    coco = posthoc['COCO_val_posthoc']['t2i']['recall@1']
    lines += ['Wiki: odczyt przed nowym treningiem Wiki. VN: in-domain dla nas, podpisy modelowe. XTD10: kontrola stylu, nie niezależny holdout. Nie ustalono rozłączności z pretreningiem konkurencji.', '',
        '## P1.A — trzy seedy i zamknięty test', '',
        f"Seedy 42/43/44: VN dev t2i R@1 **{100*vn['mean']:.2f}% ± {100*vn['sample_sd']:.2f} p.p. SD**; COCO val post hoc **{100*coco['mean']:.2f}% ± {100*coco['sample_sd']:.2f} p.p. SD**.",
        'Seed44 wybrano wcześniej regułą mediany VN dev. Lokalny legacy MSCOCO: t2i R@1 50,7%, R@5 79,0%, R@10 88,3%. To lokalny protokół kandydatów, nie oficjalny COCO5k. Fixed-gallery zapisano osobno w `vn_full_seeds_posthoc.json`; testów nie powtarzano.', '',
        '## Stan zadań', '', '| Punkt | Status | Uwagi |', '|---|---|---|']
    lines.extend(f"| {name} | {entry['status']} | {entry['note']} |" for name, entry in statuses.items())
    lines += ['', '## Większy korpus', '',
        'Zamrożono 132 875 rekordów VN i 24 125 Wiki, razem 157 000. Wstępnie nowe, niepowiązane z chronionymi komponentami po metadanych: 96 930 VN i 22 580 Wiki. To nie są gotowe niezależne pary treningowe: brak pełnego audytu obrazów, pixel SHA, pHash≤6 i praw. `training_ready_rows=0`.',
        'Oryginalne FULL_CLEAN, validation, snapshot VN 35 918 i stary Wiki 1322 zachowały SHA. Nie podmieniono ich nowymi eksportami. Nie otwierano aktywnych SQLite/WAL ani nie zmieniano działających pipeline’ów.', '',
        '## Decyzje użytkownika', '',
        '- XM3600 pozostaje FINAL_BLIND. [Draft bramki v2](xm3600_gate_registration_v2_20261030.json) proponuje +2 p.p. i dolną granicę paired 95% CI > 0 wobec najlepszego małego polskojęzycznego modelu na Wiki dev; wymaga domknięcia protokołu i akceptacji przed wspólnym uruchomieniem. Zero wyników XM w tej serii.',
        '- Publikacja: [dossier](LICENSING_DOSSIER_20261030.md), w tym Gemma/PolDense, źródła newsowe i luka USA Today. Brak zatwierdzenia publikacji.',
        '- Nowy trening Wiki: baseline zapisany, ale wymagany rozłączny train loader, pełny audyt i decyzja. Historyczny pilot 975/201 nie jest nowym Wiki 397.',
        '- Destylacja: najpierw plan i audyt; brak treningu bez akceptacji. FT/LoRA/KD-λ na starej mieszance nie uruchamiano.', '',
        '## Dysk, czas i dowody', '',
        'Wagi referencji ładowano pojedynczo i usuwano po wszystkich galeriach oraz timingach. Zachowano konfiguracje, embeddingi, wyniki i modele bazowe TinyCLIP/PolDense. Wyjątek formatu wspólnych blobów HF obsłużono z kontrolą SHA i brakiem odwołań z innych repozytoriów. Nie wykonywano ogólnego czyszczenia cache.',
        'Natywny cache Xet miał podczas przeglądu około 3 GiB; nie usuwano jego chunków. Aktywne pipeline’y również zmieniają zajętość dysku, więc różnica wolnego miejsca nie jest wyłącznie kosztem benchmarku.', '',
        '| Worker referencji | Enkodery V+T [M] | Wizyjny [M] | Tekstowy [M] | Czas ścienny, s | Usunięte pliki wag, GiB |', '|---|---:|---:|---:|---:|---:|']
    lines.extend(f"| {r['model']} | {parameter_cells(comparison['efficiency'][r['model']]['parameters'])} | {r['worker_wall_seconds']:.1f} | {r['removed_weight_bytes']/2**30:.3f} |" for r in runs)
    lines += ['', 'Czas workera obejmuje CPU/GPU, dekodowanie, ładowanie i pomiar CPU. Nie jest pomiarem aktywnego czasu kerneli GPU; nie rekonstruujemy go z median timingów.', '']
    lines.append('Raport treningu P1.A podaje `GPU_seconds_new_runs`: ' + ', '.join(
        f"seed {seed}: {seconds:.2f} s" for seed, seconds in training['GPU_seconds_new_runs'].items()) + '. Seed42: reuse.')
    lines.extend(f"- Wolne po wykonaniu: `{path}` **{value['free']/2**30:.2f}GiB**." for path, value in disk.items())
    lines += ['', 'SHA nowych manifestów i artefaktów: [artifact_index_20261030.json](artifact_index_20261030.json). Lista rejestracji/decision JSON, czasy i dowody: [summary.json](summary.json). Poprzedni raport i summary zachowane w kopiach `*_before_20261030`.',
        'Weryfikacja: parity 3000/3000 historycznych rang wspólnym scorerem; adapter dokładnie odtworzył 16 historycznych próbek; 15 testów serii oraz 4 testy ochrony współdzielonych wag przeszły. Przedziały sprawdzają tożsamości galerii, qrels, query i komponentów.', '',
        'Najbliższy etap GPU: P1.B — frozen features i scout backbone’ów; potem P1.C/P1.D. Kończymy obecny podpunkt w całości, zgodnie z zasadą przerwania po pełnym etapie.','']
    (OUT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    freeze(OUT / 'report_completion_20261030.json', dict(status='complete', REPORT_sha256=file_sha(OUT / 'REPORT.md'),
        summary_sha256=file_sha(OUT / 'summary.json'), artifact_index_sha256=index_sha,
        reference_completion_sha256=file_sha(OUT / 'reference_execution_complete_20261029.json')))
    print('DAY_REPORT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
