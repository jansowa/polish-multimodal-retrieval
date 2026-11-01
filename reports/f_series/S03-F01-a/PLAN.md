# S03-F01-a — plan

Zakres: wyłącznie F01 kroki 1–2 (04 §F01, 09 §4 i wejściowy NEXT).
Pytanie: czy można zamrozić protokół v3-F i rozłączne komponentowo połówki
istniejącego VizWiz bez zmiany chronionych wejść?

1. Zweryfikować rejestr v4 i SHA wejść; użyć pełnych metadanych rejestru do
   przechodnich komponentów, z zachowaniem komponentów źródłowego protokołu.
2. Uporządkować komponenty po identyfikatorze, przetasować seedem określonym
   w 02 §2 i przydzielać całe komponenty do mniej licznej galerii (remis: A).
3. Zapisać nowy katalog, manifest przydziałów i protokoły połówek, zachowując
   wszystkie obrazy, zapytania i krawędzie qrels; zamrozić protokół MD/JSON.
4. Niezależnie sprawdzić zachowanie puli i qrels, brak przecięcia komponentów,
   powtarzalność oraz niezmienność chronionych SHA. Dowody w summary.json.

Brak modeli, treningu, GPU, pobierania, odczytów wyników GUARD/TEST i nowych
dopuszczeń treningowych. Budżet: lokalny CPU, bez nowych zależności.
Sukces: kryterium z wejściowego NEXT; F01 nadal in_progress. Stop: naruszenie
integralności rejestru, brak fingerprintów lub konieczność zmiany zamrożonych
plików → eskalacja według 07 §8. Pozostałe kroki F01 poza zakresem.
