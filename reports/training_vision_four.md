# Porównanie czterech frozen vision encoders

Wykonano wyłącznie pierwsze cztery testy w zleconej kolejności; dalszych modeli nie uruchomiono. Punkt odniesienia: trzy zachowane seedy SigLIP2 + sigmoid loss (E2b).

Stałe: frozen PolDense-32M, sigmoid loss z uczoną temperaturą i biasem, liniowy projector, 70 289 train / 7 695 validation (67 445 unikalnych par w train), identyczne splity i kolejność batchy dla seeda 42, 20 epok, batch 256, AdamW lr=0.001, wd=0.01, scheduler=none. Checkpoint wybiera minimum własnej validation loss. Wszystkie frozen features cache offline.

Projector jest liniowy dla każdego modelu; liczba jego wejść wynika z wymiaru backbone’u. Każdy encoder zachowuje swój oficjalny preprocessing i rozdzielczość. Wartości jakości: R@1 w %, średnia ± std tylko gdy jest więcej niż jeden seed.

## Jakość i liczba parametrów

| Model | Vision M | Nowy projector M | Cały embedder M | Seedy | MSCOCO t2i R@1 | Δ vs SigLIP2 E2b (p.p.) |
|---|---:|---:|---:|---:|---:|---:|
| SigLIP2 Base/16 256 (E2b) | 92.930 | 0.295 | 125.109 | 3 | 49.70 ± 0.70 | +0.00 |
| DINOv2-Small | 22.057 | 0.148 | 54.088 | 1 | 25.80 | -23.90 |
| MobileCLIP2-S2 | 35.815 | 0.197 | 67.895 | 1 | 49.10 | -0.60 |
| ConvNeXt-Tiny | 27.820 | 0.295 | 59.999 | 1 | 20.20 | -29.50 |
| iBOT ViT-S/16 | 21.666 | 0.148 | 53.697 | 1 | 23.30 | -26.40 |

PolDense ma **31,883,136 parametrów**. Cały embedder = używana część vision + PolDense + nowy liniowy projector. Uczone skala/bias lossu nie są potrzebne w inference. Oryginalna text tower MobileCLIP2 nie jest używana ani liczona. Jego oryginalna image projection jest zachowana i wliczona w vision. ConvNeXt pomija końcowy classifier Linear (pełny klasyfikator ImageNet ma 28 589 128 parametrów).

## Reprezentacje i pretraining

| Model | Pretraining | Input px | Feature dim | Reprezentacja |
|---|---|---:|---:|---|
| SigLIP2 Base/16 256 (E2b) | image-text pretrained SigLIP2 | 256 | 768 | original SigLIP2 vision pooler_output |
| DINOv2-Small | vision-only self-supervised DINOv2 | 224 | 384 | final normalized CLS token |
| MobileCLIP2-S2 | image-text pretrained MobileCLIP2 | 256 | 512 | standard encode_image output in CLIP space, including original image projection |
| ConvNeXt-Tiny | supervised ImageNet-1K | 224 | 768 | global average pool + final LayerNorm, before classification Linear |
| iBOT ViT-S/16 | vision-only iBOT self-supervised ImageNet-1K teacher | 224 | 384 | official backbone forward: final normalized CLS token |

DINOv2 seed 42: **25.80% R@1**. Próg ≥38% nieosiągnięty; użyto seedów [42]. Jest to jawna decyzja o replikacji na podstawie testu; nie strojono na nim LR, temperatury początkowej, architektury ani wyboru epoki. Progi 40 i 41,7% odnoszą się do wcześniejszego E1; aktualny sigmoid/SigLIP2 ma 49,70%.

Porównanie tego samego seeda 42: SigLIP2 **50.00%**, MobileCLIP2 **49.10%**. Referencyjne 49,70% jest średnią trzech seedów; jeden seed MobileCLIP2 nie wystarcza do rozstrzygnięcia niewielkiej różnicy między tymi modelami.

## MSCOCO text→image — dotychczasowe lokalne pule MMEB

| Model | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| SigLIP2 Base/16 256 (E2b) | 49.70 ± 0.70 | 81.17 ± 0.57 | 89.53 ± 1.10 |
| DINOv2-Small | 25.80 | 62.00 | 77.30 |
| MobileCLIP2-S2 | 49.10 | 78.20 | 88.80 |
| ConvNeXt-Tiny | 20.20 | 48.40 | 63.00 |
| iBOT ViT-S/16 | 23.30 | 54.30 | 70.10 |

## Retrieval dwukierunkowy — osobny fixed-gallery protocol

Ta sama galeria 1002 obrazów / 1000 captionów, 1000 zapytań t2i i 201 zapytań i2t, multi-positive qrels. To rozszerzenie MSCOCO MMEB, nie oficjalny test COCO 5k.

| Model | t2i R@1 | t2i R@5 | t2i R@10 | i2t R@1 | i2t R@5 | i2t R@10 |
|---|---:|---:|---:|---:|---:|---:|
| SigLIP2 Base/16 256 (E2b) | 49.63 ± 0.74 | 81.13 ± 0.51 | 89.50 ± 1.08 | 81.92 ± 1.44 | 98.18 ± 0.76 | 99.00 ± 0.00 |
| DINOv2-Small | 25.80 | 61.90 | 77.20 | 62.69 | 92.54 | 96.52 |
| MobileCLIP2-S2 | 49.10 | 78.20 | 88.80 | 80.60 | 97.51 | 99.00 |
| ConvNeXt-Tiny | 20.20 | 48.30 | 63.00 | 57.21 | 84.58 | 90.05 |
| iBOT ViT-S/16 | 23.30 | 54.30 | 70.10 | 60.20 | 89.05 | 94.03 |

## Cztery benchmarki — R@1

| Model | CIRR | FashionIQ | MSCOCO_t2i | RefCOCO | AVG |
|---|---:|---:|---:|---:|---:|
| SigLIP2 Base/16 256 (E2b) | 9.40 ± 0.52 | 6.20 ± 0.69 | 49.70 ± 0.70 | 61.47 ± 1.04 | 31.69 ± 0.24 |
| DINOv2-Small | 10.30 | 3.00 | 25.80 | 54.10 | 23.30 |
| Δ vs SigLIP2 E2b (p.p.) | +0.90 | -3.20 | -23.90 | -7.37 | -8.39 |
| MobileCLIP2-S2 | 21.30 | 7.90 | 49.10 | 36.90 | 28.80 |
| Δ vs SigLIP2 E2b (p.p.) | +11.90 | +1.70 | -0.60 | -24.57 | -2.89 |
| ConvNeXt-Tiny | 9.30 | 2.60 | 20.20 | 41.50 | 18.40 |
| Δ vs SigLIP2 E2b (p.p.) | -0.10 | -3.60 | -29.50 | -19.97 | -13.29 |
| iBOT ViT-S/16 | 10.10 | 2.20 | 23.30 | 46.60 | 20.55 |
| Δ vs SigLIP2 E2b (p.p.) | +0.70 | -4.00 | -26.40 | -14.87 | -11.14 |

CIRR/FashionIQ/RefCOCO i AVG pozostają diagnostyczne; nie wybierano pod nie architektury.

## Historia treningu

| Model | Seed | Initial val loss | Best val loss | Epoka | Train loss final |
|---|---:|---:|---:|---:|---:|
| DINOv2-Small | 42 | 15.09437 | 5.21798 | 4 | 2.09137 |
| MobileCLIP2-S2 | 42 | 15.27135 | 4.33743 | 8 | 1.84443 |
| ConvNeXt-Tiny | 42 | 14.85234 | 5.53651 | 2 | 1.61908 |
| iBOT ViT-S/16 | 42 | 14.67498 | 5.27499 | 4 | 2.19904 |

Dokładne liczby parametrów, rewizje i SHA checkpointów, preprocessing, feature cache, run_id i pełne metryki są w JSON. Modele różnią się też danymi pretrainingu i native preprocessingiem: porównanie bada gotowe reprezentacje, nie izoluje samej metody pretrainingu.

Źródła oficjalne: [DINOv2](https://huggingface.co/facebook/dinov2-small), [MobileCLIP2](https://huggingface.co/apple/MobileCLIP2-S2), [ConvNeXt-Tiny](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html), [iBOT](https://github.com/bytedance/ibot).
