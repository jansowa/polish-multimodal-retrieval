# Wiki → COCO: transfer poza domenę treningową

Wszystkie trzy warianty i checkpointy były zamrożone przed oceną COCO. Wybór checkpointów: wyłącznie VN dev. Brak nowych treningów; XM3600 pozostaje nieocenione.
TinyCLIP-39M i PolDense frozen; Linear 512→384. Seed42. Train: samo VN lub VN z 3%/10% prezentacji wiki, bez znanego COCO w naszym nadzorowanym treningu.

Główny punkt porównania: dotychczasowy lokalny MMEB MSCOCO_t2i, 1000 zapytań i po1000 kandydatów; nie jest to oficjalny COCO5k. Zależności zapytań: 201 komponentów obrazów/podpisów/powiązań źródłowych.

| Udział wiki | COCO test R@1 | R@5 | R@10 | COCO val t2i R@1, post hoc | COCO val i2t R@1, post hoc | Best krok po VN |
|---|---:|---:|---:|---:|---:|---:|
| WIKI_00pct | 45.00% | 75.00% | 84.20% | 58.47% | 68.02% | 1650 |
| WIKI_03pct | 45.10% | 75.10% | 83.90% | 58.47% | 66.11% | 1650 |
| WIKI_10pct | 45.00% | 74.50% | 83.80% | 58.13% | 66.11% | 2475 |

Parowane 95% przedziały warunkowe; obejmują komponenty zapytań, nie seedy lub wybór checkpointu. Różnice w punktach procentowych.

| Wariant − 0% | COCO test t2i Δ [95% CI] | COCO val t2i Δ [95% CI] |
|---|---:|---:|
| WIKI_03pct | +0.10 [-1.11, +1.39] | +0.00 [-0.91, +0.90] |
| WIKI_10pct | +0.00 [-2.60, +2.50] | -0.33 [-2.20, +1.48] |

## Stały krok5500 — pomocnicza kontrola

Ten checkpoint był wcześniej zadeklarowany w pilotażu jako diagnoza stałego budżetu; nie wybierano go po COCO.

| Wariant | COCO test R@1 | COCO val t2i R@1 |
|---|---:|---:|
| WIKI_00pct | 43.50% | 55.89% |
| WIKI_03pct | 43.70% | 56.84% |
| WIKI_10pct | 44.10% | 56.27% |

## Interpretacja

W tej zamkniętej serii mały dodatek wiki nie wykazał poprawy transferu na COCO: R@1 wynosi 45.0%, 45.1%, 45.0% dla 0%, 3%, 10%. Zysk 3% to jedno trafienie netto na 1000 zapytań, a jego warunkowy przedział [-1.11, +1.39] p.p. obejmuje zero. R@5/10 również nie pokazują spójnej poprawy. Na COCO val wynik t2i dla 3% jest identyczny z bazowym, a i2t obu dodatków wiki jest niższe o 1.91 p.p.

Pomocnicze porównanie tego samego kroku5500 daje +0.2/+0.6 p.p. test R@1 przy 3%/10% wiki. Nie uzasadnia to zmiany checkpointów wybranych wcześniej po VN dev; nie widać dużego efektu ukrytego wyłącznie przez różny czas ich wyboru. Wniosek dotyczy tej puli 975 wiki train i jednej inicjalizacji, bez rozstrzygnięcia o większym, bardziej różnorodnym korpusie wiki.

Poprzednia rekomendacja z pilotażu VN/wiki dotyczyła ochrony wyniku VN i zachowania na wiki dev. Nie rozstrzygała transferu na inne dane; właściwe porównanie dla COCO znajduje się powyżej.
Mała różnica na COCO przy jednym seedzie pozostaje wynikiem wstępnym. Nie użyto ocen COCO do przestawiania checkpointów, proporcji danych ani uruchamiania nowych treningów. COCO było już oglądane w poprzednich eksperymentach.
Osobne wyniki dla pełnej galerii testowej oraz R@1/5/10 w obu kierunkach są w JSON; nie należy mieszać ich z metrykami lokalnych pul kandydatów.

Weryfikacja: passed; odtworzono wcześniejsze wyniki COCO VN-only oraz VN dev wszystkich trzech modeli. Dane i checkpointy zachowały SHA256.
