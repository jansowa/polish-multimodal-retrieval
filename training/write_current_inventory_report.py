"""Human-readable inventory of current production data only."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / 'reports/current_data_inventory.json').read_text(encoding='utf-8'))
by_name = {e['dataset']: e for e in data['checkpoints']}
sources = [('aro_visual_relation','ARO Visual Relation',23937,23226),
           ('aro_visual_attribution','ARO Visual Attribution',28748,28049),
           ('sugarcrepe_continuation','SugarCrepe',7511,6747),
           ('mbeir_mscoco','M-BEIR MSCOCO',24809,19962),
           ('mbeir_visualnews','M-BEIR VisualNews',537568,0),
           ('cirr','MMEB CIRR',1000,None), ('fashioniq','MMEB FashionIQ',1000,None),
           ('mscoco_t2i','MMEB MSCOCO_t2i',1000,None), ('refcoco','MMEB RefCOCO',1000,None)]
text = ('# Aktualnie dostępne dane\n\n'
        f'Odczyt tłumaczeń: {data["translations_observed_at"]}. Wyłącznie aktualne wersje produkcyjne. '
        'Liczby to rekordy, nie liczba wszystkich zdań ani unikalnych obrazów. '
        'VisualNews jest w trakcie tłumaczenia.\n\n'
        '| Zbiór | Źródło EN: rekordy | Gotowe PL: rekordy | Pary w obecnym train+val |\n'
        '|---|---:|---:|---:|\n')
for key, name, count, retained in sources:
    text += f'| {name} | {count:,} | {by_name[key]["completed"]:,} | {retained if retained is not None else "eval — wykluczone z train"} |\n'
text += (f'\nRazem {sum(e["completed"] for e in by_name.values()):,} przetłumaczonych rekordów, '
         'w tym 4000 rekordów ewaluacyjnych. ARO i SugarCrepe mają dodatnie oraz kontrastowe ujemne podpisy; '
         'nie liczymy ich jako oddzielnych rekordów. VisualNews: opisy EN/PL i metadane, brak lokalnych '
         'obrazów potrzebnych do gotowej puli treningowej; nie policzono na nowo audytu rozszerzonego zbioru.\n\n'
         'Obecny train: 70 289 rekordów, 67 445 unikalnych par, 10 645 różnych target images. '
         'Validation: 7695 rekordów, 1216 różnych target images. Razem 77 984 pary i 11 861 obrazów '
         'według pikseli; audyt nie wykazuje overlapu z chronionym eval.\n\n'
         '## Dodatkowe lokalne dane tekstowe, nieużywane w treningach multimodalnych\n\n'
         'mFAQ PL: 63 157 train + 2025 validation = 65 182 pary pytanie–odpowiedź, bez obrazów.\n\n'
         'MAUPQA: 3 660 140 rekordów pytanie–passage z etykietą trafności, bez obrazów. '
         '454 234 pozytywne i 3 205 906 negatywnych wierszy; nie są to liczby unikalnych pytań.\n\n'
         '| MAUPQA subset | Rekordy |\n|---|---:|\n')
for subset in data['additional_text_sources']['maupqa']['subsets']:
    text += f'| {subset["subset"]} | {subset["rows"]:,} |\n'
text += ('\n## Obrazy i cache\n\n'
         'Źródłowe obrazy ARO i SugarCrepe są lokalnie w Parquet; gotowe obrazy treningowe są wyodrębnione. '
         'Cache plików MMEB zawiera CIRR 4530, FashionIQ 17 148, MSCOCO_t2i 2004, RefCOCO 16 832 plików. '
         'To liczby plików na dysku, nie unikalnych semantycznie obrazów ani rozmiary train.\n\n'
         'Wspólny manifest encoderów obejmuje 30 893 ścieżki obrazów i 137 904 teksty EN/PL, z train, val '
         'i eval. Nie są to dodatkowe niezależne przykłady treningowe. Offline image embeddings są policzone '
         'dla dziesięciu przetestowanych backbone’ów; tekstowy cache wykorzystuje frozen PolDense-32M.\n\n'
         'Mamy też oryginalne qrels, zapytania, caption pools i odwołania do obrazów dla użytych wariantów '
         'M-BEIR MSCOCO/VisualNews, oraz blacklisty eval, identyfikatory źródłowe, hashe i split train/val. '
         'Winoground ma w cache tylko README; nie zaliczamy go do gotowych danych. '
         'ImageCoDe i inne opcje rejestru nie są gotowymi przetłumaczonymi zbiorami.\n\n'
         'Lokalizacje aktualnych danych: `outputs/` w projekcie; '
         '`C:/Users/jsowa/benchmark-translations/` dla SugarCrepe, M-BEIR MSCOCO i VisualNews; '
         '`.cache/absence-v3/` dla źródeł EN; `training/runs/e1-full/` dla aktualnego splitu i audytu; '
         'cache Hugging Face dla źródłowych Parquet/CSV i embeddingów.\n')
target = ROOT / 'reports/current_data_inventory.md'
target.write_text(text, encoding='utf-8')
print(target)
