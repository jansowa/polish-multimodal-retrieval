# Wnioski: COCO bez nadzorowanego treningu na COCO

**Samo VisualNews: COCO R@1 = 45,0%, R@5 = 75,0%, R@10 = 84,2%.**
To dotychczasowy lokalny MMEB MSCOCO_t2i: 1000 zapytań i po 1000 kandydatów. Nie jest to oficjalny COCO 5k.
Zamrożone TinyCLIP-39M i PolDense; Linear 512→384 od losowych wag, świeży loss. Żadnych wag projectora wcześniej trenowanych na COCO. Checkpoint i konfiguracje wyłącznie po VN dev.

| Dane | COCO w naszym train | COCO test R@1 | R@5 | R@10 | COCO val R@1 — post hoc | VN dev R@1 |
|---|---|---:|---:|---:|---:|---:|
| VN_only | nie | 45.00% | 75.00% | 84.20% | 58.47% | 78.08% |
| VN_nonCOCO_AR | nie | 42.40% | 70.80% | 81.40% | 52.68% | 69.40% |
| VN_FULL | tak — kontrola | 52.30% | 80.40% | 89.00% | 62.30% | 74.09% |

Różnice na teście (warunkowy paired bootstrap po 201 grupach pozytywnych obrazów/tekstów; 1000 zapytań nie jest niezależnymi obserwacjami):

- VN_nonCOCO_AR_minus_VN_only: -2.60 [-6.20, +1.01] p.p.; 95% przedział nie obejmuje zmienności seedów ani selekcji checkpointu.
- VN_FULL_minus_VN_only: +7.30 [+3.80, +11.00] p.p.; 95% przedział nie obejmuje zmienności seedów ani selekcji checkpointu.

## Wkład każdego źródła

Poniższe różnice dotyczą **COCO validation ocenionej po zamknięciu wszystkich treningów**, nie dodatkowych testów kolejnych konfiguracji.
Dodanie źródła: mieszanka 50/50 z VN względem samego VN. Wkład w pełnej mieszance: pełna mieszanka względem zastąpienia źródła przez VN, przy zachowaniu ekspozycji pozostałych źródeł.

| Źródło | Pary źródła | Różne obrazy źródła | Dodanie do VN: Δ p.p. [95% CI] | Wkład w pełnej mieszance: Δ p.p. [95% CI] |
|---|---:|---:|---:|---:|
| aro_visual_attribution | 25,434 | 3,966 | -6.94 [-9.59, -4.30] | -0.53 [-2.63, +1.54] |
| aro_visual_relation | 20,883 | 4,658 | -5.41 [-8.25, -2.68] | -0.72 [-2.45, +1.05] |
| sugarcrepe | 6,103 | 1,310 | +1.72 [-0.67, +4.10] | +0.38 [-1.58, +2.35] |
| mbeir_mscoco | 17,869 | 3,579 | +7.61 [+5.17, +10.10] | +3.78 [+1.64, +6.00] |

## Decyzja po wstępnym screeningu

- Do recepty bez COCO wybieramy obecnie samo VN. Wyboru dokonano po VN dev, przed otwarciem ocen COCO.
- Samo 45,0% kontra 42,4% nie dowodzi przewagi na populacji testowej: przedział różnicy dla strict ARO względem VN obejmuje zero. To ważne ograniczenie jednego seeda i małej liczby niezależnych grup benchmarkowych.
- ARO Attribution i Relation przy wadze 50% obniżają wyniki względem samego VN. Usunięcie ich z pełnej mieszanki poprawia VN dev, a małe różnice COCO val nie dowodzą korzyści z ich obecności. Nie ma podstaw, aby domyślnie nadawać im tak dużą wagę w tej recepcie.
- SugarCrepe ma mały, niepewny dodatni efekt na COCO val; jest oparte na COCO, więc nie kwalifikuje się do gałęzi bez COCO.
- M-BEIR MSCOCO daje duży dodatni efekt na COCO val; pełni rolę kontroli ekspozycji na domenę COCO i nie kwalifikuje się do gałęzi bez COCO.
- Dane ARO/SugarCrepe użyto jako dodatnich par. Ich jawne błędne captiony/hard negatives nie uczestniczyły w lossie. Wynik nie rozstrzyga potencjału innego celu kompozycyjnego ani mniejszej proporcji ARO.

## Granice wniosku

- To jeden seed, zgodnie z założeniem screeningu. Różnice i przedziały opisują tę receptę i dostępne pule, nie uniwersalną jakość źródeł.
- Budżet maksymalny był równy; wybrane checkpointy mogą wykorzystywać różne jego części. Reguła wyboru po VN dev może sprzyjać VN kosztem innych domen.
- Liczba wierszy ARO jest duża, ale liczba różnych obrazów znacznie mniejsza niż w VN. Przy stałym koszcie zamiana połowy batcha zmienia również różnorodność obrazów i strukturę negatywów.
- Bez COCO oznacza brak znanego pochodzenia COCO w naszym nadzorowanym treningu i brak wyboru po COCO. Nie dowodzi braku COCO w oryginalnym pretreningu gotowych encoderów ani braku niezidentyfikowanych kopii.
- VN train/dev pochodzi z tego samego pipeline’u i dwóch źródeł newsowych, z modelowymi captionami. Benchmark COCO to istniejący, historycznie oglądany zestaw.
- Nie uruchamiano danych wiki, retrieval XM3600 ani nowych treningów po poznaniu wyników COCO.

[Pełne 14 wariantów, logi i porównania](tinyclip_source_ablation.md).
[Weryfikacja checkpointów i źródeł](tinyclip_source_ablation_verification.json).
