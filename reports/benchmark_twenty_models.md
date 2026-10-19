# Porównanie 20 modeli na czterech datasetach

Dodano TIPSv2 B/14 i L/14, BGE-VL-base, PE-Core-B16-224 oraz Marqo FashionSigLIP. Ukończono 20 nowych przebiegów; łącznie 80 pełnych przebiegów, po 1000 zapytań każdy. Zachowano bez zmian wszystkie 190 wcześniejszych plików wyników i predykcji.

| Model | Parametry [M] | Licencja | Średnia P@1 | cirr | fashioniq | mscoco_t2i | refcoco | Średnia NDCG@10 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| google/embeddinggemma-2 | 438.76 | Apache-2.0 | 39.10 | 19.40 | 15.40 | 57.00 | 64.60 | 57.07 |
| timm/ViT-B-16-SigLIP-i18n-256 | 370.63 | Apache-2.0 | 38.40 | 15.90 | 16.10 | 54.80 | 66.80 | 56.09 |
| timm/ViT-B-32-SigLIP2-256 | 376.86 | Apache-2.0 | 38.15 | 17.40 | 16.70 | 53.10 | 65.40 | 56.92 |
| alphaedge-ai/siglip2-base-patch16-512-pol-32768 | 204.38 | Apache-2.0 | 37.80 | 17.40 | 18.40 | 45.20 | 70.20 | 55.59 |
| google/siglip2-base-patch16-512 | 375.82 | Apache-2.0 | 37.77 | 17.30 | 18.70 | 45.00 | 70.10 | 55.62 |
| alphaedge-ai/siglip2-base-patch16-512-pol-16384 | 191.80 | Apache-2.0 | 37.70 | 17.30 | 18.10 | 45.10 | 70.30 | 55.29 |
| google/siglip2-base-patch16-384 | 375.48 | Apache-2.0 | 36.52 | 15.70 | 17.80 | 44.20 | 68.40 | 55.00 |
| google/siglip2-base-patch16-256 | 375.23 | Apache-2.0 | 36.23 | 16.30 | 16.90 | 43.70 | 68.00 | 53.98 |
| M-CLIP/XLM-Roberta-Large-Vit-B-16Plus + timm/vit_base_patch16_plus_clip_240.laion400m_e31 | 677.76 | Nie podano (tekst); MIT (wizja) | 35.80 | 16.00 | 12.70 | 60.00 | 54.50 | 53.91 |
| google/siglip2-base-patch16-naflex | 375.23 | Apache-2.0 | 35.30 | 15.90 | 18.90 | 42.00 | 64.40 | 53.97 |
| alphaedge-ai/metaclip-2-worldwide-m16-384-pol-16384 | 85.30 | CC-BY-NC-4.0 | 28.73 | 10.90 | 10.70 | 34.60 | 58.70 | 45.33 |
| Marqo/marqo-fashionSigLIP | 203.16 | Apache-2.0 | 28.23 | 12.10 | 12.20 | 24.30 | 64.30 | 44.89 |
| alphaedge-ai/metaclip-2-worldwide-s16-384-pol-16384 | 49.72 | CC-BY-NC-4.0 | 27.10 | 11.50 | 9.70 | 30.00 | 57.20 | 43.51 |
| google/tipsv2-l14 | 487.94 | Apache-2.0 | 21.93 | 8.00 | 7.90 | 3.20 | 68.60 | 33.85 |
| alphaedge-ai/clip-ViT-B-32-multilingual-v1-pl-32768 | 156.33 | MIT | 21.85 | 8.50 | 4.20 | 24.90 | 49.80 | 36.25 |
| google/tipsv2-b14 | 195.95 | Apache-2.0 | 21.48 | 9.30 | 6.60 | 3.40 | 66.60 | 32.94 |
| alphaedge-ai/clip-ViT-B-32-multilingual-v1-pl-16384 | 143.75 | MIT | 21.32 | 8.40 | 4.00 | 23.30 | 49.60 | 35.67 |
| facebook/PE-Core-B16-224 | 447.66 | Apache-2.0 | 20.88 | 9.50 | 8.80 | 6.90 | 58.30 | 34.01 |
| apple/MobileCLIP2-S2 | 99.24 | Apple AMLR | 19.28 | 19.10 | 8.80 | 27.20 | 22.00 | 33.71 |
| BAAI/BGE-VL-base | 149.62 | MIT | 6.62 | 7.40 | 0.70 | 1.40 | 17.00 | 15.71 |

Wyniki w procentach, średnia nieważona po datasetach. Główna metryka MMEB: Precision@1. NDCG@10 jest dodatkowym widokiem.

## Adaptery i protokół

Wszystkie modele mają przypięte rewizje i używają tych samych tłumaczeń, lokalnych pul kandydatów oraz [protokołu v2](../benchmark/protocol_v2.md): bez instrukcji MMEB, surowe embeddingi, suma tekst+obraz przed końcowym cosine. Nie zmieniono wcześniejszych wyników.

- [TIPSv2](https://huggingface.co/google/tipsv2-b14): oficjalny kod z przypiętej rewizji HF, 448×448, bilinear resize, piksele [0,1], pierwszy CLS jako reprezentacja globalna. Natywny SentencePiece, lowercase i 64 tokeny; FP32, batch do 2, Torch attention bez xFormers. B/14 ma 195948288 parametrów, L/14 487941120.
- [BGE-VL-base](https://huggingface.co/BAAI/BGE-VL-base): oficjalny kod, 149620737 parametrów, CLIPProcessor 224 i maksymalnie 77 tokenów. Oficjalne encode_multimodal dodaje surowe cechy tekstu i obrazu, po czym normalizuje sumę; evaluator realizuje tę samą geometrię bez wstępnej normalizacji obu części.
- [PE-Core](https://huggingface.co/facebook/PE-Core-B16-224): dokładnie checkpoint Facebooka, 447659521 parametrów; oba enkodery ładowane ze strict=True. Oficjalny kod perception_models, commit 3e352cca660658d4b5c90f42a7808b11469e4c66. Natywne 224, bilinear squash, zakres [-1,1], kontekst 32 tokenów.
- [FashionSigLIP](https://huggingface.co/Marqo/marqo-fashionSigLIP): OpenCLIP ładowany z lokalnego snapshotu HF, 203155970 parametrów, natywne 224 i 64 tokeny, canonicalization, normalize=False.
- BGE używa FP32 wag z BF16 autocast; PE i Marqo mają wagi BF16. Pierwszy próbny wynik PE w FP32 zachowano w archiwum; tabela obejmuje cztery zgodne pomiary PE w BF16. Loader PE odtwarza niemodyfikowane kodowanie RoPE i maskę przyczynową poza kontekstem meta. Stos: PyTorch 2.11.0+cu128, Transformers 4.57.6, OpenCLIP 3.3.0, timm 1.0.30, SentencePiece 0.2.1, einops 0.8.1. Wszystkie komponenty i rzeczywisty batch zapisano w result.json.

## Walidacja i dysk

Zweryfikowano 80 unikalnych par model/dataset, zgodność SHA tłumaczeń, komplet 1000 skończonych predykcji na przebieg oraz wszystkie metryki względem zapisanych rang. Dwanaście testów benchmarku przeszło, w tym numeryczna zgodność transferu wag i odzyskiwanie po przejściowych błędach CUDA. Kontrole GPU EN/PL używają wejść bez instrukcji. Test Chrome potwierdza jedną tabelę, pełne nazwy, sortowanie wszystkich kolumn i przełączanie metryk.

Po nieudanej początkowej kontroli TIPS zrestartował się WSL. Tłumaczenia i oba serwery przywrócono z zapisanych punktów; właściwe obliczenia wykonano z małym batchem TIPS. Dotychczasowe wyniki zachowano.

Wagi pobierano wyłącznie do natywnego cache HF WSL. Po walidacji tej serii wagi zostają usunięte zgodnie z preferencją oszczędzania miejsca; konfiguracje, tokenizery, wyniki i cache embeddingów pozostają. Szczegółowy protokół usunięcia znajduje się w benchmark/results/model_weight_cleanup_twenty.json.

Usunięto pięć plików wag i odzyskano 5.25 GiB w systemie plików WSL. Sumy SHA-256 235 plików wyników i leaderboardu są identyczne przed i po sprzątaniu.
