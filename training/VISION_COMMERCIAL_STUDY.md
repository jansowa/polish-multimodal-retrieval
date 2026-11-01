# Frozen vision: pięć kandydatów z deklarowaną licencją komercyjną

Uruchamiamy tylko: TinyCLIP 39M/16 YFCC, TinyCLIP 40M/32 LAION,
Chinese-CLIP RN50, RN50 CC12M, RN50 YFCC. Kolejność jest zablokowana w kodzie.
RN101 YFCC i TinyCLIP 61M są planem; wymagają analizy pierwszej piątki.

Recepta identyczna jak E2b i poprzednie cztery testy: frozen PolDense-32M,
frozen vision, standardowy końcowy image embedding z oryginalną projekcją,
nowy liniowy projector do 384 wymiarów. Sigmoid multi-positive, uczona skala
i bias (dwa parametry funkcji celu, nie encoderów); batch 256, AdamW lr=.001,
wd=.01 dla projectora, 0 dla parametrów lossu, 20 epok, bez schedulera.
Checkpoint wybiera minimum validation loss. Dane i kolejność batchy dla
danego seeda identyczne: 70 289 train / 7695 val, 67 445 unikalnych par train.
Nie zmieniamy manifestów, blacklist ani tekstowego cache EN/PL. Natywny
preprocessing i rozdzielczość modelu; wszystkie vision features offline.

Najpierw pięć scoutów z seedem 42. Potem seedy 43/44 tylko jeśli scout:

- osiąga R@1 ≥47%; lub
- jest najlepszym kandydatem z deklaracją MIT/Apache-2.0 w swojej klasie
  parametrów: ≤40M, (40M,50M], (50M,60M]; porównanie uwzględnia wcześniejsze
  DINOv2, ConvNeXt i iBOT; remisy kwalifikują oba; lub
- traci ≤2 p.p. do MobileCLIP2-S2 49.1% przy ≥20% mniej parametrów vision.

Klasy i interpretacja „istotnie mniej” zapisane przed scoutingiem w configu.
MobileCLIP nie jest zaliczany do MIT/Apache: ma własną licencję Apple.
To jawna selekcja replikacji na podstawie testu zgodnie ze zleceniem,
nie niezależne potwierdzenie ani strojenie hiperparametrów na teście.
Oceny modelu komercyjnego odnoszą się do deklaracji licencji wag w karcie;
nie oznaczają pełnego audytu praw do zbiorów pretrainingowych lub produktu.

Każdy checkpoint i karta licencyjna mają dokładną rewizję i SHA256.
Chinese-CLIP używa trzech niezmienionych klas obrazu z oficjalnego kodu;
bez inicjalizacji chińskiego encodera tekstu. OpenCLIP zachowuje attention
pooling i jego projekcję. TinyCLIP zachowuje visual_projection. Parametry
liczymy z rzeczywiście używanego image modelu, bez oryginalnej text tower.

TinyCLIP 39M vs 40M zmienia jednocześnie patch i dane — nie izoluje osobno
tych przyczyn. RN50 CC12M vs YFCC ma identyczną architekturę i preprocessing;
pozostają to gotowe checkpointy, więc również inne ustawienia ich oryginalnego
pretrainingu mogą się różnić. Nasz trening pozostaje identyczny.

Uruchomienie w istniejącym środowisku:

```sh
python training/prepare_commercial_vision.py
python training/prepare_chinese_source.py
python training/commercial_vision_study.py smoke
python training/commercial_vision_study.py all
python training/export_commercial_vision.py
```

Na obecnym WSL brak RAM/commit hosta może przerywać CUDA mimo wolnej VRAM.
`training/observe_commercial_study.py` wznawia te same run_id i checkpointy,
sprawdza hashe zapisanych feature tensors i nie inicjalizuje ponownie ich
encoderów. Nowe ekstrakcje izoluje przez `cache_commercial_backbone.py`.
Przy silnej presji RAM worker cache można uruchomić samodzielnie, przed
wznowieniem procesu treningowego; przykładowe wejście/wyjście są w
`training/runs/vision-commercial-five/recovery/`. Nie zmienia to lossu,
budżetu ani checkpoint selection. Po zakończeniu:

```sh
python training/verify_commercial_study.py
```

Raport zawiera MSCOCO t2i R@1/5/10, osobny fixed-gallery t2i/i2t,
CIRR/FashionIQ/MSCOCO_t2i/RefCOCO/AVG, różnice do SigLIP2 i poprzedniego
testu, dokładne parametry vision/PolDense/projectora i historię strat.
To dotychczasowy lokalny benchmark MMEB, nie oficjalny COCO 5k.
