# Kontrola próbki VisualNews

Obejrzano 24 natywne kadry z cache'u i odpowiadające im polskie captiony, po 6 przypadków z każdego połączenia train/dev × Washington Post/USA Today. Próbkę wylosowano deterministycznie, bez wyników modelu. Plansza i lista identyfikatorów: `tinyclip_visualnews_quality_sample.png` / `.json`.

To kontrola wykonana przez model, nie niezależna adnotacja ludzka ani potwierdzenie jakości całego zbioru. Nie zmieniła zamrożonych manifestów ani kryterium checkpointu.

Większość przykładów odpowiada głównym obiektom i czynnościom widocznym w obrazie. Zastrzeżenia do powtórnej kontroli w pipeline:

- Przykład 19: caption o pomniku wojennym i Grobie Nieznanego Żołnierza zawiera specyficzną lokalizację/cel czynności, których sam pokazany kadr z idącymi osobami nie potwierdza. To kandydat do ponownego groundingu/recycle.
- Przykład 3: opis czynności w wodzie jest trudny do niezależnego potwierdzenia ze względu na bardzo małą rozdzielczość dwóch klatek. Oznaczyć jako niepewny, nie jako pewny błąd.
- Przykłady 8/11/23: native center crop obcina część osoby lub tekstu/otoczenia. Caption może odnosić się do szczegółu pełnego zdjęcia niewidocznego w kadrze treningowym. To ograniczenie preprocessingu, niekoniecznie błędne tłumaczenie.

Również poprawnie zweryfikowany automatycznie VN dev pozostaje zbiorem z modelowymi rewrite'ami i tłumaczeniami. Nie zastępuje osobnego sprawdzenia na ludzkich krótkich zapytaniach.
