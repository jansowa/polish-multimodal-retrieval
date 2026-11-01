**Rekomendacja: B_taxonomy_v3; text-only: T1.**

Przeprowadziłem 7 wariantów multimodalnych (4 różne strategie v1, 2 rewizje v2, 1 rewizję v3) oraz 2 text-only. Każdy dostał dokładnie te same 360 naturalnych par. Dla każdego multimodalnego: 64 podpisy kontrolne na 8 obrazach oraz 3 losowania na identycznych 100 parach (pierwsze losowanie w przebiegu głównym; dwa dodatkowe). Łącznie 5088 wywołań HTTP, plus 4 wcześniejsze techniczne próby w bootstrap_v0.

**Materiały do akceptacji**

- [Rekomendowany prompt multimodalny](recommended_multimodal.txt), [dokładna konfiguracja wykonania](B_taxonomy_v3_config.json); [drugi finalista](B_taxonomy_v2.txt).
- [Rekomendowany prompt text-only](recommended_text_only.txt).
- [Finalny JSON schema](schema.json), [osobny schema text-only](text_only_schema.json).
- [Galeria: 50 rozbieżności / błędów i 30 losowych przykładów](review.html). Każdy przykład zawiera obraz, EN caption, JSON i confidence; starsze warianty można rozwinąć.
- [Wyniki liczbowe](summary.json), [porównanie CSV](comparison.csv), [surowe odpowiedzi](raw/), [kontrole](controls.json), [weryfikacja kompletności](verification.json).

Pierwszy finalista ma mniej fałszywych alarmów i większą stabilność. Drugi, B v2, ma mniej wyjść poza ręczne zakresy VG (6/60 wobec 8/60), więc pozostaje wariantem porównawczym. Nie zakładam, że różnice w tym pilocie dowodzą przewagi na całym korpusie.

**Próbka i zakres wniosków**

Źródło to VisualNews_t2i z TIGER-Lab/MMEB-eval, revision `eb3a8af230a66be5c9308cd2be5779b9d6de4cb3`. Pula zawiera 1000 rekordów; wybrano 360 różnych captionów i obrazów. 180 wybrano losowo, 180 przez rotację heurystycznych kategorii tekstowych, seed 20261020. Usunięto wyłącznie stałą instrukcję MMEB „Retrieve an image of this news caption.”; oryginalny query jest w sample.json. Obrazy powiązano z oficjalnym kandydatem pozytywnym 0, bez zgadywania ID.

Jest to diagnostyczny podzbiór benchmarku, a nie losowa próba 537 tys. captionów M-BEIR VisualNews. Nie estymuję na tej podstawie odsetka dobrych przykładów całego korpusu. Te obrazy należą do podzbioru ewaluacyjnego; pilot nie dodaje ich do treningu. Pierwsze 100 rekordów zamrożono jako grupę powtórzeń; pozostałe 260 brały udział w eksploracyjnej analizie błędów, więc nie stanowią ślepego holdoutu.

Kategorie heurystyczne zachodzą na siebie: date=90, location=168, entity=294, temporal=90, short_visual_candidate=141, cause_effect=24, event=27, role=41, ambiguous_candidate=32. Etykiety są wskazówkami doboru, nie prawdą referencyjną. Przypadki czarne i uszkodzone pozostały jako trudne przykłady; samo poprawne zdekodowanie pliku nie zapewnia użytecznej treści.

**Porównanie**

| Wariant | Surowy JSON | Schema po usunięciu samych fence | Kontrole /48 | Fałszywe alarmy /10 | Poza ręcznym zakresem /60 | Ciężkie /60 | Zgodność VG | Wszystkie etykiety |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_direct | 77.8% | 100.0% | 48 | 6 | 10 | 1 | 71.5% | 54.5% |
| B_taxonomy | 67.2% | 100.0% | 41 | 7 | 11 | 3 | 69.3% | 49.2% |
| C_claims | 96.1% | 100.0% | 48 | 6 | 12 | 2 | 75.0% | 56.5% |
| D_skeptical | 92.2% | 100.0% | 48 | 7 | 12 | 1 | 69.5% | 49.0% |
| B_taxonomy_v2 | 100.0% | 100.0% | 48 | 7 | 6 | 0 | 74.0% | 52.0% |
| C_claims_v2 | 100.0% | 100.0% | 48 | 9 | 7 | 1 | 75.5% | 53.0% |
| B_taxonomy_v3 | 100.0% | 100.0% | 48 | 6 | 8 | 0 | 82.0% | 60.0% |

Surowy JSON oznacza `json.loads` na niezmienionej odpowiedzi. Odzyskiwanie usuwa tylko kompletny zewnętrzny blok Markdown i ponownie sprawdza schema; nie naprawia treści. V1 nie używał constrained decoding; v2/v3 wysyłają `response_format=json_schema`. Poprawy formatu nie przypisuję wyłącznie promptowi. B v1 ma jeden niepoprawny wynik kontrolny (pięć zależnych porównań niewykonalnych) i jeden niepoprawny wynik powtórzenia; w tabeli kontrole niewykonalne nie dostają zaliczenia.

Zgodność powtórzeń porównuje losowania 1 i 2 z losowaniem bazowym: 200 par porównań na 100 rekordach; B v1 ma 199 poprawnych par. „Wszystkie etykiety” obejmuje 8 pól klasyfikacyjnych, bez confidence i opisów dowodów. Samo agreement nie jest miarą trafności. Ważona kappa, MAE, dokładne mianowniki oraz podział na grupy są w summary.json.

Ręczny przegląd obejmuje 60 obrazów z captionami i przedziały akceptowalnego VG zapisane po diagnozie v1, przed pełnym porównaniem przypadków v2. Są to diagnostyczne oceny Codex, bez ekspertów i bez ustalonego gold standard; przypadki OCR, landmarków i źródeł obrazu bywają niejednoznaczne. „Ciężki” oznacza odległość co najmniej 2 od tego przedziału. Osobno obejrzałem wszystkie 14 naturalnych mismatchów v2: 9 oceniłem jako fałszywe alarmy, 3 jako niejednoznaczne, 2 jako nieużyteczne pliki. Te przypadki służyły projektowaniu v3, więc poprawa na nich jest wynikiem rozwojowym, nie niezależnym precision. Końcowy przegląd v3 wykrył jeszcze jeden nowy fałszywy alarm (413), dodany po zakończeniu strojenia. Stąd końcowy mianownik 10, z czego 9 stanowiło przypadki rozwojowe.

**Iteracje i stwierdzony efekt**

Bootstrap: cztery wywołania ujawniły echo definicji schema i bloki Markdown. Zastąpiłem definicję w prompcie szablonem instancji i usunąłem instrukcję retrieval z wejścia; techniczne wyniki zachowano osobno i wyłączono z porównania.

V1: A — bezpośrednie osie; B — typy informacji; C — atomowe claims; D — audyt sceptyczny. Wspólne skale i schema. Problemami były rozpoznawanie osób jako dowodu, nadmierne karanie nazw przy widocznej akcji oraz mylenie źródła obrazu z opisem treści.

V2: jawne kredytowanie widocznego rdzenia, reguły OCR / landmarków / source credit oraz spójność uzasadnienia ze score. B: liczba wyjść poza ręczny zakres 11 → 6 /60; C: 12 → 7 /60. Format wykonania zmieniono osobno na wymuszony JSON schema.

V3: doprecyzowanie rzeczywistej sprzeczności, odbić, cutoutów, szkiców, zasłonięcia, tłumaczeń tekstu i błędów wnioskowania geograficznego. Naturalne fałszywe alarmy B v2 → B v3 w końcowym zestawie 10: 7 → 6. W pierwotnych 9 przypadkach rozwojowych spadek wyniósł 7 → 5, ale pojawił się nowy alarm 413. Jednocześnie poprawnie wykryte błędne kolory: 8/8. Poprawa jest częściowa; mismatch=true powinno kierować do ręcznego przeglądu, a nie automatycznie usuwać parę. Pozostały m.in. błędny odczyt greckiego napisu, wnioskowanie o kraju z ubioru i wyobrażenie sali sądowej. Wszystkie skutki i regresje można sprawdzić w galerii oraz raw outputs.

**Rozkład labeli finalisty**

| visual_grounding | n | % |
|---|---:|---:|
| 0 | 56 | 15.6% |
| 1 | 149 | 41.4% |
| 2 | 134 | 37.2% |
| 3 | 13 | 3.6% |
| 4 | 8 | 2.2% |

| caption_type | n |
|---|---:|
| visual_description | 8 |
| visual_description_plus_context | 95 |
| entity_caption | 84 |
| event_caption | 125 |
| mostly_contextual | 46 |
| noisy_or_mismatched | 2 |

External context 0–3: {"3": 246, "2": 86, "1": 14, "0": 14}. Flaga mismatch: 8/360. Rozkład jest silnie przesunięty ku niskiej ocenie, co w news captions jest możliwe; sam rozkład nie potwierdza jakości ani nie uzasadnia jego sztucznego wyrównania.
Grupa random: n=180, średni VG=1.339, external=2.556, mismatch=5.
Grupa targeted: n=180, średni VG=1.372, external=2.578, mismatch=3.

**Kontrole**

8 obrazów, każdy z czystym opisem, dopisanym nazwiskiem fotografa, usuniętym nazwiskiem, dokładną datą, nazwą wydarzenia, dokładną lokalizacją, późniejszym skutkiem i błędnym kolorem. Daje to 64 captiony i 48 porównań oczekiwanego kierunku. Dla daty, wydarzenia, miejsca i późniejszego skutku wymagano ścisłego wzrostu external context; przy usunięciu nazwiska VG nie może spaść; błędny widoczny kolor musi zmienić mismatch z false na true. Dodatkowo zapisano, czy dopisanie kontekstu nie zwiększa VG.

| Kontrola | Zaliczone |
|---|---:|
| remove_name | 8/8 |
| date | 8/8 |
| event | 8/8 |
| precise_location | 8/8 |
| later_outcome | 8/8 |
| wrong_color | 8/8 |

**Text-only**

| Wariant | Poprawny schema /360 | Named entity i nadal potencjalnie dobry query | External MAE wobec finalisty | Kandydaci nadmiernego odrzucania |
|---|---:|---:|---:|---:|
| T1 | 360 | 147 | 0.611 | 11 |
| T2 | 359 | 119 | 0.652 | 10 |

T1 wybrano za kompletny poprawny format i zachowanie query wizualnych pomimo nazw własnych. „Kandydat nadmiernego odrzucania” oznacza text-only external ≥2 przy multimodal VG ≥3; jest to rozbieżność dwóch ocen, a nie potwierdzony false-positive rate. Nie należy odrzucać przykładów wyłącznie przez nazwisko ani przez wysoki wynik text-only. External MAE porównuje potencjalną zależność tekstową z zależnością ocenianą na obrazie; nie są to identyczne zadania. [Przypadki do kontroli false positives, z captionami i wynikami obu trybów](text_only_false_positive_candidates.json).

**Schema i wykonanie**

Zachowano wszystkie 9 wymaganych pól. `named_entity_dependency` ma enum none/incidental/essential. Dodano `evidence` i `unsupported_information`, aby audytować, co jest widoczne i co wymaga kontekstu. Confidence oznacza pewność oceny modelu, nie jakość pary i nie skalibrowane prawdopodobieństwo. Text-only ma osobny schema, bo bez obrazu nie może stwierdzać rzeczywistego grounding, OCR ani mismatch.

Testowana konfiguracja: model `qwen3.8-27b`, temperatura 0.2, thinking wyłączone, max_tokens=1800, obraz proporcjonalnie do maksymalnie 1280 px, JPEG quality 88. Testy nie obejmują automatycznie temperatury 0 ani innych ustawień; porównanie finalistów dotyczy zapisanej konfiguracji.

Łączne raportowane użycie tokenów: {"prompt_tokens": 4274373, "completion_tokens": 812489, "total_tokens": 5086862}. Nie obliczam kosztu pieniężnego bez cennika endpointu. Wszystkie raw HTTP bodies, finish_reason, usage, czasy, hashe promptów / obrazów / captionów i numery prób zapisano w raw/.

**50 rozbieżności / błędów oraz 30 losowych przykładów**

[Galeria z obrazami i pełnymi wynikami](review.html), [50 przypadków JSON](disagreements.json), [30 losowych przypadków JSON](random30.json), [60 ocen ręcznych](manual_reviews.json), [15 przeglądów mismatch](manual_mismatch_reviews.json). Pierwszeństwo na liście 50 mają ręcznie sprawdzone mismatchy, dalej największa rozbieżność ocen / typów; przypadki bez ręcznej adjudykacji są oznaczone jako kandydaci do przeglądu.

Pilot zakończony. Prompt i schema są przygotowane do ręcznej akceptacji; pełny VisualNews judge nie został uruchomiony.
