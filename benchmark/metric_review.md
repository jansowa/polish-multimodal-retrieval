# Wybór metryk

Weryfikacja w publikacjach i oficjalnym kodzie, 2026-10-18:

| Benchmark | Metryka | Źródło |
|---|---|---|
| MMEB (pierwsza wersja, używane lokalne pule kandydatów) | Precision@1 | [Publikacja, §2.1 i §4](https://arxiv.org/html/2410.05160v3#S2.SS1) |
| MIEB, retrieval | Głównie nDCG@10, dla wybranych zadań Recall@1 lub MAP@5 | [Publikacja, §2.1](https://openaccess.thecvf.com/content/ICCV2025/papers/Xiao_MIEB_Massive_Image_Embedding_Benchmark_ICCV_2025_paper.pdf) |
| ARO, VG Relation / Attribution | Trafność wyboru podpisu; kod raportuje również grupy relacji/atrybutów | [Oficjalny evaluator](https://github.com/mertyg/vision-language-models-are-bows/blob/main/dataset_zoo/aro_datasets.py) |

Obecne cztery policzone datasety pochodzą z MMEB, a nie z protokołu MIEB.
Każdy wiersz ma jedną oznaczoną poprawną odpowiedź i 999 negatywnych kandydatów.
Z tego powodu Precision@1, Recall@1 oraz Hit@1 są identyczne liczbowo.
Dotychczasowe wyniki opisane jako Recall@1 nie wymagają ponownego liczenia na GPU.

NDCG@10 jest użytecznym dodatkowym widokiem, ponieważ uwzględnia pozycje 2–10,
których Precision@1 nie rozróżnia. Dla jednego poprawnego kandydata na pozycji `r`:

```text
NDCG@10 = 1 / log2(r + 1), jeśli r <= 10; w przeciwnym razie 0.
```

Nie mamy stopniowanych ocen trafności ani dodatkowych pozytywów, więc nie należy
interpretować tego wyniku jako ewaluacji wielopozytywnego rankingu. Zmiana metryki
nie odtwarza również pełnej puli retrieval ani innych splitów oryginalnego datasetu.

Decyzja: domyślnie metryka właściwa dla używanego benchmarku (MMEB: Precision@1),
z przełącznikiem NDCG@10 i pozostałych metryk. Dla przyszłych adapterów MIEB należy
ustalać główną metrykę na poziomie konkretnego zadania, a nie globalnie wymuszać P@1.
Średnia w lokalnym leaderboardzie jest nieważoną średnią datasetów, liczoną tylko
dla kompletnego wiersza modelu i jednej wybranej metryki.

Obecna implementacja CLIP jest lokalnym wariantem: zachowuje instrukcje tekstowe
i sumuje embeddingi obrazu oraz tekstu. Publikacja MMEB dla baseline'ów CLIP pomija
instrukcje. Dlatego zgodność głównej metryki nie oznacza zgodności całego protokołu.
ARO w obecnym adapterze ma accuracy po przykładach; oficjalny kod zawiera dodatkowo
rozbicie na grupy. Tych różnych sposobów agregacji nie należy utożsamiać.
