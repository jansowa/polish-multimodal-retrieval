# Frozen vision: pierwsze cztery modele

Zlecony zakres: DINOv2-Small → MobileCLIP2-S2 → ConvNeXt-Tiny → iBOT ViT-S/16,
potem **STOP i analiza**. ConvNeXt-Small, DINO ViT-S/16 i DeiT-Small pozostają
niewykonane. Ten eksperyment używa sigmoid loss wybranego w E2b i zachowuje
liniowy projector; porównanie MLP nie zostało do niego dołączone.

**Pierwsza czwórka wykonana.** Lokalne MSCOCO t2i R@1, seed 42: DINOv2 25,80%,
MobileCLIP2-S2 49,10%, ConvNeXt-Tiny 20,20%, iBOT 23,30%. DINOv2 nie osiągnął
progu 38%, więc każdy nowy model ma jeden seed. Referencyjny SigLIP2/E2b:
49,70% ±0,70 p.p. (3 seedy; sam seed 42: 48,90%).
[Porównanie jakości i parametrów](../reports/training_vision_four.md).

Stałe: frozen PolDense-32M, oryginalny cache tekstów, 70 289 rekordów train
(67 445 unikalnych par), 7695 val, identyczne splity oraz batch permutations
dla odpowiadających seedów, 20 epok, batch 256, AdamW lr=0.001, wd=0.01 dla
projectora, bez schedulera. Learnable temperature start=0.07, bias start=-10.
Checkpoint wybiera minimum validation loss, jak w E2b. Vision frozen.

| Backbone | Oficjalne źródło | Native feature | Wejście |
|---|---|---|---|
| DINOv2-Small | facebook/dinov2-small, pinned HF SHA | Final normalized CLS, 384 dims | Resize 256, center crop 224, oficjalny processor |
| MobileCLIP2-S2 | apple/MobileCLIP2-S2, pinned HF SHA | Standard `encode_image(normalize=False)`, 512 dims; oryginalna image projection zachowana | Oficjalny OpenCLIP dfndr2b preprocessing, 256 px, mean=0/std=1 |
| ConvNeXt-Tiny | TorchVision IMAGENET1K_V1, SHA checkpointu | Avgpool + final LayerNorm + flatten, 768 dims, przed końcowym Linear classifier | Oficjalne `weights.transforms()`, crop 224 |
| iBOT ViT-S/16 | Oficjalny teacher backbone ByteDance, SHA weights i kodu | Oficjalny forward: final normalized CLS, 384 dims | Resize 256 bicubic, crop 224, ImageNet normalization |

iBOT korzysta z oryginalnego kodu forward. Adaptacja dotyczy wyłącznie importu
funkcji inicjalizacji `trunc_normal_` i usunięcia nieużywanego importu registry.
Checkpoint ładowany `strict=True`. Wszystkie cztery modele przechodzą test
poprawnych, niezerowych embeddingów; MobileCLIP ma dodatkowo sprawdzenie
identyczności wyjścia `encode_image` i używanego modułu visual.

DINOv2: najpierw seed 42. Jeśli **lokalne MSCOCO t2i R@1 ≥38%**, uruchamiane są
seedy 43 i 44 (łącznie trzy). To jawny próg użytkownika oparty na teście do
replikacji, bez strojenia parametrów. 40% = mocny kandydat; 41,7% odnosi się do
E1. Aktualny sigmoid/SigLIP2 ma 49,70% ±0,70 p.p. Pozostałe nowe modele: seed 42.

Parametry liczone z faktycznie użytych modułów: vision (z zachowaną oryginalną
projekcją, jeśli występuje), nowy projector oraz cały embedder = vision +
PolDense + projector. Oryginalna text tower MobileCLIP jest zastąpiona PolDense
i wyłączona z tej sumy. Classifier ConvNeXt usunięty. Dwa skalary lossu są
w trainable params, a nie w liczbie parametrów potrzebnych do inference.

## Uruchomienie

```bash
PY=~/.venvs/pl-mmb-benchmark/bin/python
$PY training/freeze_vision_reference.py
$PY training/prepare_vision_models.py
$PY training/vision_study.py smoke
$PY training/check_compact_eval.py
$PY -u training/vision_study.py all
$PY training/export_vision_study.py
```

Pobieranie jest jawne. Reszta wykonuje się offline. Cache obrazów ma hashe
oryginalnych plików, checkpointów, preprocessing, precyzję i wersje bibliotek.
Chunki i ukończone seedy są wznawialne. Kod i trzy checkpointy referencyjnego
SigLIP2/E2b pozostają w archiwum `training/runs/vision-four/reference/`.

Snapshot benchmarków jest odczytywany strumieniowo i reprezentowany przez
indeksy współdzielonych ścieżek. `check_compact_eval.py` sprawdza każdą rangę
i metrykę czterech istniejących benchmarków względem referencyjnego E2b.
Kandydaci, kolejność, qrels i tie policy pozostają takie same.

Artefakty: `training/runs/vision-four/` (config, model lock, param counts,
smoke, decyzja DINO, per-model cache metadata, checkpointy, epochs.jsonl,
wyniki i predictions), raport `reports/training_vision_four.md/json/csv`
oraz pełna historia `reports/training_vision_four_loss.csv`.

Główna metryka: dotychczasowe lokalne MSCOCO t2i R@1/5/10. Osobno rozszerzony
fixed-gallery protocol: 1000 tekstowych / 201 obrazowych zapytań, galeria
1002 obrazów / 1000 captionów. CIRR/FashionIQ/RefCOCO i AVG diagnostyczne.
To porównanie gotowych reprezentacji; różne dane pretrainingu i preprocessing
nie pozwalają przypisać różnic wyłącznie paradygmatowi vision-only/image-text.
