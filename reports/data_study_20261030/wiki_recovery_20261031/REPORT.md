# Wiki recovery — generacja 2, 31.10.2026

Pobieranie: {'ready': 10214}. Archiwum: 7.00 GB; fingerprinty oryginałów policzone przed zmniejszeniem.
Zapis: cały kadr do1536px JPEG92 oraz osobne, bezstratne wejście TinyCLIP224 z bitową zgodnością preprocessingu. Pierwsze20 oryginałów testowych zachowano.
Train Wiki: 3539 rekordów, 3539 obrazów, 935 pełnych komponentów. Wszystkie galerie chronione; pHash nie tworzy qrels.
R0/R1: reuse wszystkich3 seedów po sprawdzeniu niezmienności danych, cech i reload metryk/lossu. R2/R3: świeże seedy42, następnie wybrany Wiki43/44. Recepta5500 kroków/1405780 prezentacji niezmieniona.

Wybrano R3 według Wiki dev seed42. Spadek VN względem R1: 1.514 p.p.; guardrail: zaliczony.

| Wariant | Galeria | t2i R@1 mean±SD | i2t R@1 mean±SD |
|---|---|---:|---:|
| R0 | VN_dev | 73.922±0.175% | 69.967±0.708% |
| R0 | wiki_native_dev | 15.710±0.799% | 14.274±1.018% |
| R0 | xtd10_full_1k | 49.633±0.306% | 44.667±0.929% |
| R0 | xtd10_no_overlap | 51.394±0.000% | 46.302±0.948% |
| R1 | VN_dev | 74.472±0.485% | 69.800±0.218% |
| R1 | wiki_native_dev | 15.106±0.604% | 13.686±0.524% |
| R1 | xtd10_full_1k | 49.633±0.643% | 45.333±0.252% |
| R1 | xtd10_no_overlap | 50.910±0.451% | 47.083±0.341% |
| R3 | VN_dev | 72.957±0.325% | 68.433±0.651% |
| R3 | wiki_native_dev | 17.120±0.923% | 15.449±0.524% |
| R3 | xtd10_full_1k | 48.767±0.681% | 44.533±1.021% |
| R3 | xtd10_no_overlap | 50.130±1.036% | 46.117±1.287% |

Wiki dev jest ID dla modeli uczonych na Wiki, źródłowo OOD dla R0/R1. VNdev jest ID; XTD to kontrola COCO/tłumaczeń. Nie łączymy galerii w jeden ranking.
Wszystkie4 warianty seed42: training_report.json i osobne katalogi evaluation; R@1/5/10, MRR, NDCG@10. CI: conditional_CI.json, paired bootstrap komponentów10000, seed20261023, oddzielnie od SD seedów.
V+T70470528; V38587392; T31883136; head196992 osobno. Native/recycled/translated zachowane; brak nowych ręcznych etykiet, brak decyzji o publikacji. Licencje, autorzy i strony źródłowe w rekordach.
Generacja1 pozostała niezmieniona. Nowe modele wymagają nowego audytu OOD B względem uniwersum generacji2. Nie odczytywano wyników B. GPU zwolnione.
