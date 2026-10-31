# Aktualny plan: polska wspólna przestrzeń text–image

Ten dokument zastępuje wcześniejszą kolejność E0–E11 z `EXPERIMENTS.md`.
Zakres obecnie zlecony: **P0 oraz E2a/E2b**. Wszystkie dalsze punkty są planem.

**P0 i P1 wykonane.** Oba lossy przekroczyły próg replikacji na val; po trzy seedy.
E2b wygrał na val i jako jedyny nowy loss otrzymał test: lokalne MSCOCO t2i R@1
49,70% ± 0,70 p.p., +8,00 p.p. wobec pełnego E1. E2a pozostaje oceniony na val.
[Pełny raport P0/E2](../reports/training_p0_e2.md).

Aktualnie zlecone rozszerzenie: [pierwsze cztery frozen vision backbone'y](VISION_STUDY.md)
na sigmoid loss i liniowym projectorze, przed dalszą analizą. Po tej czwórce
zatrzymać się; nie uruchamiać automatycznie dalszych modeli ani MLP.

Cel: zachować jakość PolDense text↔text, maksymalizować text↔image i image↔text,
a composed retrieval i grounding rozbudowywać oddzielnie. PolDense pozostaje
frozen aż do warunkowej, bardzo późnej fazy P11.

## Punkt odniesienia

Pełny E1 jest stałym baseline'em dla wszystkich kolejnych eksperymentów:

- frozen `OPI-PIB/PolDense-32M` i `google/siglip2-base-patch16-256`;
- liniowy projector, cosine positive-pair regression;
- **70 289 train / 7 695 validation**, 77 984 par po audycie;
- seedy 42/43/44; MSCOCO lokalne MMEB t2i R@1 **41,70% ± 0,20 p.p.**;
- config, checkpointy, wyniki, hashe danych i oryginalny kod zabezpieczone przez
  `training/freeze_e1.py`, lock w `training/runs/p0-e2/baseline/`;
- pilot 6k jest historycznym wynikiem, nie baseline'em kolejnych ablacjów.

## P0 — infra i ewaluacja

1. Weryfikować lock E1 i feature cache przed każdym runem. Nie nadpisywać E1.
2. Zachować lokalny protokół MMEB oraz jego R@1/5/10. Dla dwukierunkowego
   retrieval raportować **oddzielny** `fixed-gallery-multi-positive-v1`:
   t2i R@1/5/10 i i2t R@1/5/10. Testowe galerie powstają wyłącznie z istniejących
   przykładów MSCOCO MMEB. T2i używa wszystkich ich kandydatów; i2t pyta tylko
   o obrazy z przypisanym captionem. Galeria tekstów zawiera unikalne podpisy PL.
   Nie jest to oficjalny globalny test COCO 5k ani odwrócenie lokalnych pul.
3. Te same piksele pod różnymi nazwami to jeden obraz; wszystkie udokumentowane
   captiony i obrazy stanowią multi-positive qrels. Zapis galerii, qrels i ich hash.
4. Do decyzji używać już wydzielonej walidacji, bez zmiany splitu E1. Główny
   validation retrieval używa **2093 par M-BEIR MSCOCO** z istniejących 7695 val.
   Brak wspólnych parent IDs i pikseli z train; blacklisty testu nadal obowiązują.
5. Logować run_id, seed, rozmiary, liczbę unikalnych par i rekordów źródłowych
   rzeczywiście odwiedzonych, ekspozycje przykładów, modele i rewizje, projector,
   loss, batch, LR, weight decay, scheduler, temperature/logit_scale/bias,
   trainable params, best i selected validation score, historię i benchmarki.
   Historyczny E1 nie miał retrieval co epokę: oznaczyć ten brak, nie dopisywać
   domniemanych wyników. Można zmierzyć retrieval jego zachowanych checkpointów.
6. Text↔text pozostaje identyczne na poziomie modelu i wejść: oryginalne frozen
   PolDense, ten sam preprocessing, zero trenowanego mapowania tekstu.

## P1 — E2a/E2b, wyłącznie zmiana lossu

Stałe: **dokładnie te same dane, kolejność batchy dla danego seeda, inicjalizacja
projectora, model, 20 epok, batch 256, AdamW lr=0.001, wd=0.01, brak schedulera**.
Tak jak w E1, checkpoint wybiera minimum własnej validation loss, nie benchmark.
Skale/bias lossu mają weight decay 0; sam projector zachowuje ustawienia E1.

- **E2a:** symmetric multi-positive InfoNCE, 0.5·L(t2i)+0.5·L(i2t), L2 normalize,
  learnable logit_scale, początkowa temperature=0.07. L w każdym kierunku
  maksymalizuje sumaryczne prawdopodobieństwo znanych pozytywów. Dla jednej
  pozytywnej pary jest to zwykłe symmetric cross entropy.
- **E2b:** SigLIP-style sigmoid, L2 normalize, learnable logit_scale i bias;
  ten sam początkowy temperature=0.07, bias=-10. Suma strat par / batch size.
  Każda znana pozytywna para ma etykietę +1, pozostałe -1.
- Najpierw seed 42 dla obu. **Gain ≥1 p.p. na validation MSCOCO t2i R@1**
  względem E1 seed 42 uruchamia łącznie seedy 42/43/44 dla danej konfiguracji.
  Próg dotyczy walidacji: użycie testu do tej decyzji łamałoby zasadę P0.
- Najlepszy nowy loss wybiera średnia validation t2i/i2t R@1 z jego wybranych
  checkpointów. Decyzję zapisać **przed** benchmarkiem. Test odpalić tylko dla
  tego lossu; drugi wariant zachowuje wyniki val i jawny status „not selected”.
- Raportować Δ każdego wybranego wariantu względem pełnego E1. Wartości strat
  z różnych funkcji celu nie są bezpośrednio porównywalne.

Źródło sigmoid loss: [kod autorów SigLIP](https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py),
[publikacja](https://arxiv.org/abs/2303.15343). Multi-positive qrels są rozszerzeniem
na powtarzające się targety w naszych danych.

## Aktualna seria vision — pięć checkpointów z deklaracją MIT/Apache

Po pierwszej czwórce uruchamiamy wyłącznie TinyCLIP 39M/16 YFCC,
TinyCLIP 40M/32 LAION, Chinese-CLIP RN50, RN50 CC12M i RN50 YFCC,
w tej kolejności, na niezmienionej recepcie E2b. Zasady i jawne progi
replikacji: [VISION_COMMERCIAL_STUDY.md](VISION_COMMERCIAL_STUDY.md).
Po piątce raport i stop. RN101 YFCC oraz TinyCLIP 61M pozostają planem;
nie uruchamiać przed analizą pierwszych pięciu. Licencję wag oceniać
osobno od praw do danych i warunków PolDense/Gemma.

## Dalsze etapy — zaplanowane, obecnie nie uruchamiać

| Faza | Eksperyment | Zmiana i warunki |
|---|---|---|
| P2 | E3a linear / E3b mały 2-layer MLP | Na najlepszym lossie; Linear(hidden)→GELU→Linear(PolDense dim)→L2. Nie zwiększać nadmiernie hidden. Residual i LayerNorm tylko jako osobne próby. |
| P3 | E4: 2–4 frozen vision backbone'y | Priorytet SigLIP/SigLIP2 i mocne modele istniejącego harnessu; ten sam PolDense, split, loss i podobny compute budget. Raportować params, dim, trainable params, oba kierunki MSCOCO R@1/5/10, trzy diagnostyczne benchmarki i latency jeśli dostępna. Wybór według walidacji dwukierunkowej, bez strojenia pod CIRR. |
| P4 | E5a frozen / E5b ostatni blok / E5c ostatnie dwa / opcjonalne E5d LoRA | PolDense frozen. LR vision znacznie mniejszy od LR projectora; overfitting i porównanie z frozen baseline. Cech zmienianej vision tower nie wolno traktować jako stałego cache. Cache możliwy dla danych, teacherów i niezmiennych części modelu. |
| P5 | E6 ranking/similarity KD | Dopiero po mocnym zwykłym baseline. Teacher EN, student PL; cache teacher offline. L=contrastive+λ·KL(softmax(S_T/τ_T)‖softmax(S_S/τ_S)), λ∈{0,0.1,0.3,1}. Główny sygnał to geometria rankingów, nie dopasowanie raw embeddings. |
| P6 | E7 hard negatives | Offline top-K=16/32 przez teachera albo najlepszego studenta, wyłącznie z train. Usuwać te same image IDs, alternatywne captiony, dokładne duplikaty i w miarę możliwości near-duplicates. Positive+hard+in-batch; porównanie z czystym contrastive. |
| P7 | PolDense scale | Po ustaleniu loss+projector+vision przenieść receptę 1:1 na 17M/32M/68M. Zmieniać wyłącznie text encoder. Raportować params, jakość multimodalną, trainable params i inference cost. |
| P8 | Data scaling | 6k/20k/40k/70k/pełne ~100k, jeśli faktycznie dostępne. Zagnieżdżone train subsets, stałe val/test; oś learning curve: liczba unikalnych par image–text. Osobno raportować liczbę rekordów i powtórzeń; nie deklarować 100k bez gotowych obrazów i audytu. |
| P9 | CIRR-A/B/C + hard negatives | Dopiero po mocnym core. Osobny composition head: normalize(r+αt), normalize(Linear([r,t])), albo normalize(r+MLP([r,t,r*t,r-t])). Core embeddings frozen; head nie może zmieniać text↔text ani bazowego retrieval. Negatywy z tej samej/podobnej grupy, bez leakage. |
| P10 | RefCOCO / grounding | Osobna capability: najpierw zero-shot core, później ewentualny grounding head. Słaby grounding nie jest powodem zmiany całej architektury. |
| P11 | Text tower, tylko jeśli konieczne | Warunki: plateau vision finetuningu, KD i hard negatives oraz brak pomocy większego backbone'u. Wymagane replay oryginalnych danych text retrieval, distillation do frozen original PolDense, text↔text benchmark przed/po i brak istotnej regresji. |

Wspólne zasady: jedna zmiana naraz, EN i PL zachowane równolegle, cache offline
gdzie wagi są stałe, zero leakage także między datasetami, multi-positive
wszędzie tam, gdzie targety się powtarzają. Wybór architektury/loss/LR na val.
Po eksperymentach tabela CIRR/FashionIQ/MSCOCO_t2i/RefCOCO/AVG i Δ vs pełny E1;
AVG ma znaczenie diagnostyczne. P9 composition nie zastępuje core modelu.
