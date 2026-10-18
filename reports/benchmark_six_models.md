# Porównanie sześciu modeli — 2026-10-18

Zakończono 24 pełne przebiegi: 6 modeli × 4 te same datasety, po 1000 przykładów.
Leaderboard: http://localhost:8766. Wcześniejsze 46 plików wyników i predykcji pozostało bez zmian (sprawdzono SHA256).

| Model | Parametry [M] | Licencja | Średnia P@1 [%] | Średnia NDCG@10 [%] |
|---|---:|---|---:|---:|
| google/siglip2-base-patch16-512 | 375.82 | Apache-2.0 | 29.475 | 45.162 |
| alphaedge-ai/siglip2-base-patch16-512-pol-16384 | 191.80 | Apache-2.0 | 29.275 | 44.776 |
| alphaedge-ai/metaclip-2-worldwide-m16-384-pol-16384 | 85.30 | CC-BY-NC-4.0 | 26.375 | 42.077 |
| alphaedge-ai/metaclip-2-worldwide-s16-384-pol-16384 | 49.72 | CC-BY-NC-4.0 | 22.575 | 37.836 |
| apple/MobileCLIP2-S2 | 99.24 | Apple AMLR | 20.575 | 33.541 |
| alphaedge-ai/clip-ViT-B-32-multilingual-v1-pl-16384 | 143.75 | MIT | 19.275 | 32.433 |

Nowe modele: clip-b32-pl (multilingual CLIP), siglip2-pl, siglip2-base.
Multilingual CLIP: tekst Sentence Transformers (mean pooling + Dense), obraz OpenAI CLIP ViT-B/32. Parametry obejmują oba enkodery.
SigLIP2: preprocessing 512×512, limit 64 tokenów, padding max_length, BF16; multilingual CLIP: limit 128 tokenów, FP16. Batch początkowy 32.
Modele i pomocniczy enkoder obrazu zapisano w /home/jsowa/.cache/huggingface/hub.
Nowe rewizje oraz komponent enkodera obrazu są przypięte w src/pl_mmb_benchmark/models.py.
Szczegóły protokołu i licencji: [benchmark/README.md](../benchmark/README.md); żadne dodatkowe datasety nie były liczone.

Walidacja: kontrola 10 par EN/PL dla każdego nowego modelu, 50 testów regresyjnych, test sortowania i przełączania metryki w przeglądarce.
Proces tłumaczenia oraz przeglądarka danych pozostały uruchomione.
