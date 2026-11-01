# Bramka zgodności cache — rozstrzygnięcie przed nowym treningiem

Pierwszy test porównywał 8 obrazów przeliczonych jako nowy batch8 z cechami
zapisanymi podczas ekstrakcji batch32. Minimum cos wyniosło0.9999522567.
Teksty odtworzyły się bitowo (max_abs=0), więc ostrzeżenie Transformers nie
było przyczyną tej różnicy.

Odtworzenie dokładnych dwóch batchów ekstrakcji po32 obrazy daje max_abs=0
dla wszystkich64 obrazów, w tym wszystkich8 pierwotnych sond. Dowód:
parity_diagnosis.json. Źródłem różnicy jest numeryka BF16 zależna od kształtu
batcha. SHA modeli, prefix, preprocessing i cache nie zmieniły się.

Runner sprawdza odtąd identyczne wejścia i batche; prógcos>0.99999 pozostaje,
a dodatkowo wymaga max_abs=0 dla obrazów. Nie zmieniono uniwersum, precyzji,
konfiguracji treningu, selekcji ani baseline’u. Wszystkie warianty korzystają
z tych samych zamrożonych cech. Nie było jeszcze nowych wyników treningu.
