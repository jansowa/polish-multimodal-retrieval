# Uzupełnienie benchmarku ≤500M — 31.10.2026

Dobór modeli i rewizji zamknięto przed pierwszym nowym wynikiem. To mocni kandydaci w przyjętych klasach, nie dowód znalezienia globalnie najlepszego modelu.
R0/R1/R3 to wcześniej ustaleni reprezentanci seed44; R3 pozostaje diagnostyczny po niezaliczeniu guardraila VN.
Średni wynik: R@1, najpierw średnia T2I/I2T, następnie równe wagi pięciu zadań. Pełne XTD10 ma osobne wyniki, ale nie dubluje wariantu oczyszczonego w średniej.
Parametry V+T obejmują wyłącznie używane enkodery; V i T podano osobno. Head TinyCLIP/MobileCLIP: 196 992; head SigLIP2: 295 296 dodatkowych parametrów. Logit scale/bias i inne modalności nie są wliczane.

| Klasa | Model | V+T [M] | V [M] | T [M] | Średni R@1 [%] | Flickr | VizWiz | Wiki | VN | XTD clean | Wyniki /12 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| <50M | metaclip-s16 | 49.722 | 21.959 | 27.763 | 40.628 | 48.060 | 26.325 | 30.144 | 58.033 | 40.580 | 12 |
| 50–100M | metaclip-s16-32768 | 56.014 | 21.959 | 34.054 | 49.124 | 55.487 | 33.963 | 38.303 | 68.649 | 49.220 | 12 |
| 50–100M | ours-mobileclip2-poldense32-seed42 | 67.698 | 35.815 | 31.883 | 32.025 | 43.229 | 17.928 | 9.594 | 39.595 | 49.777 | 12 |
| 50–100M | R1_seed44 | 70.471 | 38.587 | 31.883 | 41.398 | 50.240 | 21.596 | 14.480 | 72.069 | 48.606 | 12 |
| 50–100M | R0_seed44 | 70.471 | 38.587 | 31.883 | 40.963 | 48.956 | 20.797 | 15.135 | 71.545 | 48.384 | 12 |
| 50–100M | R3_seed44 | 70.471 | 38.587 | 31.883 | 40.797 | 48.546 | 20.901 | 15.991 | 69.996 | 48.551 | 12 |
| 50–100M | metaclip-m16 | 85.302 | 38.782 | 46.520 | 45.867 | 53.427 | 32.488 | 33.291 | 65.702 | 44.426 | 12 |
| 50–100M | metaclip-m16-32768 | 93.690 | 38.782 | 54.908 | 54.716 | 61.568 | 39.244 | 43.717 | 74.869 | 54.181 | 12 |
| 100–200M | ours-siglip2-poldense32-seed42 | 124.813 | 92.930 | 31.883 | 31.666 | 43.754 | 19.746 | 8.033 | 37.075 | 49.721 | 12 |
| 100–200M | metaclip-b16-32768 | 141.393 | 86.484 | 54.908 | 57.697 | 63.861 | 43.204 | 49.635 | 74.593 | 57.191 | 12 |
| 100–200M | siglip2-pl | 191.799 | 93.520 | 98.279 | 43.948 | 62.240 | 41.982 | 14.530 | 38.223 | 62.765 | 12 |
| 200–300M | siglip2-pl-32768 | 204.382 | 93.520 | 110.862 | 45.221 | 63.523 | 42.714 | 15.311 | 40.121 | 64.437 | 12 |
| 300–400M | siglip-i18n-256 | 370.626 | 92.930 | 277.696 | 58.791 | 69.710 | 49.777 | 37.170 | 75.370 | 61.929 | 12 |
| 300–400M | siglip2-base | 375.824 | 93.520 | 282.304 | 45.601 | 64.133 | 42.183 | 15.588 | 40.995 | 65.106 | 12 |
| 300–400M | siglip2-b32-256 | 376.856 | 94.552 | 282.304 | 50.712 | 64.588 | 46.585 | 25.459 | 55.833 | 61.093 | 12 |
| 400–500M | metaclip-l14-32768 | 414.837 | 303.966 | 110.871 | 62.689 | 70.754 | 50.324 | 53.614 | 77.941 | 60.814 | 12 |
| 400–500M | embeddinggemma-2 | 438.760 | 167.758 | 271.003 | 73.805 | 82.748 | 65.782 | 57.089 | 92.280 | 71.126 | 12 |

Wiki/VN są wcześniej używanymi dev, COCO jest źródłem ID dla naszych modeli FULL, a Flickr/VizWiz to lokalne galerie z MT. Średnia nie jest czystym wynikiem zero-shot.
Każdy istniejący model przed uzupełnieniem musi dokładnie odtworzyć wszystkie per-query metryki historycznej kontroli Wiki. Nowe modele przechodzą przez ten sam scorer i natywny adapter.
Nie uruchomiono XM3600 ani DOCCI. Nie oceniano nowych modeli >500M. M-CLIP 677.76M oraz SigLIP2 large/MetaCLIP huge/giant są poza limitem.
Wagi referencji są ładowane pojedynczo; usuwane są wyłącznie nowo pobrane pliki wag po pełnym ukończeniu modelu, z kontrolą SHA, rozmiaru i właścicieli snapshotów. Dane, checkpointy projektu i stare wyniki pozostają.

Źródła nowych modeli (dokładne rewizje w rejestracji):
- [alphaedge-ai/metaclip-2-worldwide-s16-384-pol-32768](https://huggingface.co/alphaedge-ai/metaclip-2-worldwide-s16-384-pol-32768/tree/9565b93ba0049a67775c24f332e46fe12e0e1796)
- [alphaedge-ai/metaclip-2-worldwide-m16-384-pol-32768](https://huggingface.co/alphaedge-ai/metaclip-2-worldwide-m16-384-pol-32768/tree/c3410d687df164929aafdd02f6d323b5b4f168dd)
- [alphaedge-ai/metaclip-2-worldwide-b16-384-pol-32768](https://huggingface.co/alphaedge-ai/metaclip-2-worldwide-b16-384-pol-32768/tree/10800b373c918e68abe60e38cc90bea2aa3831f7)
- [alphaedge-ai/metaclip-2-worldwide-l14-pol-32768](https://huggingface.co/alphaedge-ai/metaclip-2-worldwide-l14-pol-32768/tree/ca70ffb896e9e5c616d22827ab8659f31d25bc50)
