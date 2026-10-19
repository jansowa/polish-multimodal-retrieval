# Porównanie 15 modeli na czterech datasetach

Dodano pięć modeli, po cztery pełne przebiegi każdy (1000 zapytań na dataset). Łącznie leaderboard zawiera 60 przebiegów w protokole v2. Wcześniejsze 150 plików wyników i predykcji pozostało niezmienionych.

| Model | Parametry [M] | Średnia P@1 | cirr | fashioniq | mscoco_t2i | refcoco | Średnia NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| google/embeddinggemma-2 | 438.76 | 39.10 | 19.40 | 15.40 | 57.00 | 64.60 | 57.07 |
| timm/ViT-B-16-SigLIP-i18n-256 | 370.63 | 38.40 | 15.90 | 16.10 | 54.80 | 66.80 | 56.09 |
| timm/ViT-B-32-SigLIP2-256 | 376.86 | 38.15 | 17.40 | 16.70 | 53.10 | 65.40 | 56.92 |
| alphaedge-ai/siglip2-base-patch16-512-pol-32768 | 204.38 | 37.80 | 17.40 | 18.40 | 45.20 | 70.20 | 55.59 |
| google/siglip2-base-patch16-512 | 375.82 | 37.77 | 17.30 | 18.70 | 45.00 | 70.10 | 55.62 |
| alphaedge-ai/siglip2-base-patch16-512-pol-16384 | 191.80 | 37.70 | 17.30 | 18.10 | 45.10 | 70.30 | 55.29 |
| google/siglip2-base-patch16-384 | 375.48 | 36.52 | 15.70 | 17.80 | 44.20 | 68.40 | 55.00 |
| google/siglip2-base-patch16-256 | 375.23 | 36.23 | 16.30 | 16.90 | 43.70 | 68.00 | 53.98 |
| M-CLIP/XLM-Roberta-Large-Vit-B-16Plus + timm/vit_base_patch16_plus_clip_240.laion400m_e31 | 677.76 | 35.80 | 16.00 | 12.70 | 60.00 | 54.50 | 53.91 |
| google/siglip2-base-patch16-naflex | 375.23 | 35.30 | 15.90 | 18.90 | 42.00 | 64.40 | 53.97 |
| alphaedge-ai/metaclip-2-worldwide-m16-384-pol-16384 | 85.30 | 28.73 | 10.90 | 10.70 | 34.60 | 58.70 | 45.33 |
| alphaedge-ai/metaclip-2-worldwide-s16-384-pol-16384 | 49.72 | 27.10 | 11.50 | 9.70 | 30.00 | 57.20 | 43.51 |
| alphaedge-ai/clip-ViT-B-32-multilingual-v1-pl-32768 | 156.33 | 21.85 | 8.50 | 4.20 | 24.90 | 49.80 | 36.25 |
| alphaedge-ai/clip-ViT-B-32-multilingual-v1-pl-16384 | 143.75 | 21.32 | 8.40 | 4.00 | 23.30 | 49.60 | 35.67 |
| apple/MobileCLIP2-S2 | 99.24 | 19.28 | 19.10 | 8.80 | 27.20 | 22.00 | 33.71 |

Wyniki w procentach, średnia nieważona po datasetach. Precision@1 jest główną metryką MMEB; NDCG@10 pozostaje dodatkowym widokiem.

## Konfiguracja

- Ten sam [protokół v2](../benchmark/protocol_v2.md), przypięte źródła i SHA256 tłumaczeń: brak instrukcji MMEB, natywne embeddingi, suma tekst+obraz przed końcowym cosine.
- M-CLIP: XLM-R Large, maskowany mean pooling, projekcja 640D; użyto dokładnie wskazanej wizji `timm/vit_base_patch16_plus_clip_240.laion400m_e31`, mimo że przykład karty tekstowej wskazuje `e32`. Łącznie 677760256 parametrów, bez nieużywanej angielskiej wieży tekstowej OpenCLIP.
- Google SigLIP: oficjalny preprocessing 384 lub 256, padding do 64 tokenów, BF16 autocast. Nowe modele OpenCLIP/M-CLIP: wagi BF16; OpenCLIP zachowuje oficjalną canonicalization, padding i preprocessing checkpointu.
- Wszystkie repozytoria i tokenizery mają przypięte rewizje, zapisane w `models.py`; konfiguracja bazowa XLM-R również jest przypięta. Checkpointy pozostają w `/home/jsowa/.cache/huggingface`.
- Polecenie uruchomienia i aliasy: [README](../benchmark/README.md). SigLIP używa PyTorch 2.14.1+cu130; M-CLIP używa 2.11.0+cu128 i transferów wag do 8 MB. Transformers 4.57.6, OpenCLIP 3.3.0, timm 1.0.30. Wersje bibliotek i rzeczywisty batch są w metadanych poszczególnych przebiegów.
- Karta wag tekstowych M-CLIP nie deklaruje licencji. Leaderboard jawnie pokazuje brak deklaracji dla tekstu i MIT dla wskazanej wizji; pozostałe cztery modele mają Apache-2.0.

## Walidacja

Zweryfikowano integralność wszystkich wcześniejszych 150 plików, komplet 60 unikalnych par model/dataset, po 1000 predykcji, skończone wyniki i zgodność wszystkich metryk z zapisanymi rangami. Testy benchmarku i regresyjny test historycznego checkpointu M-CLIP przeszły (10 testów). Test Chrome obejmuje pełne nazwy, jedną tabelę, sortowanie wszystkich kolumn i przełączanie metryk.

Podczas prób batcha 32 wystąpiły błędy sterownika i automatyczne restarty WSL. Po każdym przywrócono tłumaczenia i oba serwery z zachowanych punktów. Kontynuowano z batchem 8 w stałej sesji WSL; gotowe wyniki były pomijane.

Źródła adapterów: [M-CLIP](https://github.com/FreddeFrallan/Multilingual-CLIP/blob/main/multilingual_clip/pt_multilingual_clip.py), [SigLIP i18n](https://huggingface.co/timm/ViT-B-16-SigLIP-i18n-256), [SigLIP2 B32](https://huggingface.co/timm/ViT-B-32-SigLIP2-256).
