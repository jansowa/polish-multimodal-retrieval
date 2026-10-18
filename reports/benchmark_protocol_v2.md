# Benchmark po korekcie protokołu (v2)

Ukończono 40 pełnych przebiegów: 10 modeli × 4 datasety × 1000 zapytań.
GPU: NVIDIA GeForce RTX 5070 Ti. Wszystkie modele i cache embeddingów są w natywnym cache WSL.

## Wyniki

Wyniki w procentach. Średnia jest nieważona. Domyślna metryka MMEB: Precision@1.

| Model | Parametry [M] | Licencja | Średnia P@1 | cirr | fashioniq | mscoco_t2i | refcoco | Średnia NDCG@10 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| google/embeddinggemma-2 | 438.76 | Apache-2.0 | 39.10 | 19.40 | 15.40 | 57.00 | 64.60 | 57.07 |
| alphaedge-ai/siglip2-base-patch16-512-pol-32768 | 204.38 | Apache-2.0 | 37.80 | 17.40 | 18.40 | 45.20 | 70.20 | 55.59 |
| google/siglip2-base-patch16-512 | 375.82 | Apache-2.0 | 37.77 | 17.30 | 18.70 | 45.00 | 70.10 | 55.62 |
| alphaedge-ai/siglip2-base-patch16-512-pol-16384 | 191.80 | Apache-2.0 | 37.70 | 17.30 | 18.10 | 45.10 | 70.30 | 55.29 |
| google/siglip2-base-patch16-naflex | 375.23 | Apache-2.0 | 35.30 | 15.90 | 18.90 | 42.00 | 64.40 | 53.97 |
| alphaedge-ai/metaclip-2-worldwide-m16-384-pol-16384 | 85.30 | CC-BY-NC-4.0 | 28.73 | 10.90 | 10.70 | 34.60 | 58.70 | 45.33 |
| alphaedge-ai/metaclip-2-worldwide-s16-384-pol-16384 | 49.72 | CC-BY-NC-4.0 | 27.10 | 11.50 | 9.70 | 30.00 | 57.20 | 43.51 |
| alphaedge-ai/clip-ViT-B-32-multilingual-v1-pl-32768 | 156.33 | MIT | 21.85 | 8.50 | 4.20 | 24.90 | 49.80 | 36.25 |
| alphaedge-ai/clip-ViT-B-32-multilingual-v1-pl-16384 | 143.75 | MIT | 21.32 | 8.40 | 4.00 | 23.30 | 49.60 | 35.67 |
| apple/MobileCLIP2-S2 | 99.24 | Apple AMLR | 19.28 | 19.10 | 8.80 | 27.20 | 22.00 | 33.71 |

## Zasady i odtwarzanie

Usunięto instrukcje MMEB przed tokenizacją. Dla CLIP/SigLIP zapytania mieszane to suma surowych embeddingów; końcowe porównanie używa cosine. Gemma koduje tekst i obraz wspólnie, bez instrukcji i prefiksów wyszukiwania. Załadowano 438760448 parametrów tekstu i wizji, bez enkodera audio.

Cosine odpowiada `get_pred(normalization=True)` w oryginalnym kodzie MMEB v1. Oryginał udostępnia też iloczyn skalarny, domyślnie `normalization=False`; brak opublikowanych flag tabeli baseline nie pozwala potwierdzić jej konfiguracji. Nie deklarujemy identyczności wszystkich ustawień z tabelą angielską. Szczegóły: [protokół v2](../benchmark/protocol_v2.md).

[Referencyjny evaluator](https://github.com/TIGER-AI-Lab/VLM2Vec/blob/28fdbdd68a66374ec1b53207ad7a24b9979f1ab6/evaluation/eval_utils.py), [karta Gemma i selektywne ładowanie](https://huggingface.co/google/embeddinggemma-2#2-selective-encoder-loading).

## Walidacja i zachowanie poprzedniej pracy

- Zweryfikowano SHA256 wszystkich 70 historycznych plików wyników i predykcji: bez zmian.
- Wszystkie 40 przebiegów ma pełne 1000 predykcji, przypięte rewizje modeli i identyczny SHA tłumaczeń w ramach datasetu.
- Wszystkie metryki ponownie sprawdzono względem zapisanych rang.
- Dziewięć testów benchmarku przeszło; pełna regresja projektu: 52 testy przeszły, jeden pominięty w oddzielnym środowisku.
- Test leaderboardu w Chrome potwierdził sortowanie, metryki i pełne nazwy modeli.
- Wyniki JSON/JSONL pozostają w `benchmark/results/<run_id>/`; poprzednia tabela w `index.protocol-v1.html`. Manifesty integralności i walidacja pozostają w katalogu wyników.
- Podczas pracy po restartach WSL wznowiono kolejki tłumaczeń oraz przeglądarkę danych. Nie usuwano wcześniejszych wyników.
- Wersje bibliotek utrwalono w plikach `benchmark/environment-*.lock.txt`. Środowisko tłumaczeń jest oddzielne.
