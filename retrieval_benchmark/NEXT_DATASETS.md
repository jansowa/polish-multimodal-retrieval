# Kolejne wersje danych

Infrastruktura działa na siedmiu istniejących galeriach. Nie zamieniamy
zaplanowanych zbiorów w gotowe na podstawie samych adnotacji lub loadera.

## Priorytety po audycie 31.10.2026

Wszystkie poniższe propozycje korzystają z istniejących adnotacji. Nie wymagają
nowego ręcznego etykietowania. Adaptacja captioningu do retrievalu jest naszym
wersjonowanym protokołem, nie oficjalnym wynikiem captioningu.

| Priorytet | Zbiór | Wartość dla obecnego benchmarku | Praca do wykonania |
|---|---|---|---|
| 1 | [DOCCI](https://google.github.io/docci/) → PL | Długie ludzkie opisy, liczenie, relacje i podobne zdjęcia. Najlepsze uzupełnienie obecnych krótkich podpisów. | Obrazy, tłumaczenie testowego podzbioru, metadane klastrów i frozen retrieval. EN opisy już lokalnie. |
| 2 | [TextCaps](https://textvqa.org/textcaps/) → PL | Treść tekstu widocznego na obrazie; najbardziej bezpośredni kolejny test pod wcześniejsze zainteresowanie OCR. | Publiczny split z opisami, obrazy i tłumaczenie; zachować nazwy, liczby i cytowane napisy. |
| 3 | [RSICD](https://github.com/201528014227051/RSICD_optimal) → PL | Nowy typ obrazów: zdjęcia lotnicze/satelitarne, zamiast kolejnej galerii codziennych fotografii. | Obrazy i istniejące podpisy EN, PL, audyt duplikatów i zasad udostępniania, protokół wielopozytywny. |
| Warunkowo | [AIDe](https://aclanthology.org/L18-1337.pdf) | Natywne polskie ludzkie opisy: 2000 podpisów dla 1000 obrazów z Flickr8k. | Najpierw potwierdzić aktualny dostęp do powiązanych obrazów/adnotacji i warunki ich użycia. Nie blokować tym DOCCI/TextCaps. |
| Później | [DCI](https://github.com/facebookresearch/DCI) → PL | Gęste opisy, drugi test długiego kontekstu. | Oryginalne ludzkie opisy, obrazy SA-1B i ich warunki dostępu. Wersja CLIP-ready zawiera streszczenia/negatywy LLaMA2 i stanowi inny typ danych. |

DOCCI ma 15000 zdjęć i opisy średnio 136 słów. TextCaps obejmuje około 28000
obrazów z opisami wymagającymi odczytywania tekstu. RSICD zawiera 10921 obrazów
224x224 i po pięć opisów, lecz powtarzalne podpisy wymagają ostrożności z qrels.
DCI obejmuje 7805 obrazów. Są to rozmiary oryginalnych zbiorów, nie gotowych PL galerii.

Praktycznie: najpierw DOCCI PL, następnie TextCaps PL. RSICD ma sens, jeśli chcemy
świadomie mierzyć ogólne OOD również poza domenami typowymi dla wyszukiwarki.
Nie proponuję natychmiastowego pobierania wszystkich pięciu korpusów.

## Wspólne wymagania dla nowych galerii

* Przed pierwszym scoringiem zamknąć split, liczebność, wybór przykładów, wersję
  tłumaczenia, qrels i listę modeli. Około 1000 obrazów jest rozsądnym pilotażem
  DOCCI/TextCaps; nie wybierać ich według wyników któregokolwiek modelu.
* Tłumaczyć istniejące ludzkie opisy tekst→tekst. Nie zastępować ich podpisami
  wygenerowanymi z obrazów ani nie streszczać DOCCI pod limit konkretnego modelu.
* Zachować wszystkie znane pozytywy i scalanie identycznych podpisów; nie tworzyć
  dodatkowych pozytywów na podstawie wyników własnego encodera. Raportować
  niepełność adnotacji i automatycznego QA tłumaczeń.
* Rejestrować pochodzenie: DOCCI cluster_id, TextCaps/OpenImages ID, RSICD source ID.
  Przy dodawaniu kolejnych zbiorów sprawdzać także nakładania między testami.
* Włączyć całą galerię oraz komponenty do nowej wersji
  [rejestru holdoutów](holdouts/README.md) przed następnym treningiem. Nie zmieniać
  aktywnego katalogu/rejestru pod działającą zamrożoną generacją bez nowego audytu.
* Publikacja leaderboardu nie wymaga ponownego publikowania obrazów. Sposób
  dystrybucji obrazów/adnotacji sprawdzić oddzielnie dla każdego źródła.

WIT/Wiki PL warto rozwijać jako test nowych artykułów/komponentów. Po treningu
na Wikipedii pozostaje jednak kontrolą uogólnienia w tym źródle, nie nową domeną.
NAC wymaga dopiero zbudowania i oczyszczenia zasobu. Kolejne tłumaczenia COCO
(COCO-35L/XTD200) nie zastąpią nowych domen. ImageNet zero-shot to osobne zadanie,
którego nie dodawałbym do średniej retrievalu.

## DOCCI PL — następna galeria

Lokalne oryginalne EN:
`C:/Users/jsowa/benchmark-translations/data-ood-20261030/B/docci/descriptions.jsonl`.
Publiczne źródło: https://google.github.io/docci/ .

Do ukończenia przed pierwszym wynikiem:

1. Ustalić wersję testowego podzbioru (np. około 1000 obrazów) i metodę deterministycznego
   wyboru przed scoringiem; nie wybierać przykładów według wyników naszych modeli.
2. Pozyskać oryginalne obrazy z oficjalnego wydania/HF; zamrozić SHA i metadane
   `cluster_id`, jeśli dostępne. Nie pobierać archiwum 7.1 GB do katalogu na D.
3. Wykluczyć ujawnione nakładania z uniwersum treningowym A i chronionymi holdoutami;
   podobne, celowo kontrastujące zdjęcia DOCCI pozostają różnymi kandydatami.
4. Przetłumaczyć dosłownie istniejące ludzkie EN, tekst→tekst, z przypiętym modelem,
   parametrami i cache'em. Nie streszczać, nie generować nowych opisów ze zdjęć.
   Automatyczne QA nie daje ludzkiej gwarancji semantycznej poprawności PL.
5. Zamrozić manifest, galerię, qrels, komponenty i listę modeli, następnie dodać
   nową wersję katalogu. Długie teksty i odsetek truncation raportować per model.
6. Przeprowadzić wspólną ocenę z tym samym scorerem; niczego nie oznaczać ręcznie.

## XM3600 — zwykły test leaderboardu

Aktualna lokalna wersja ma 3597 obrazów i 7133 par po dawnych wykluczeniach;
oryginał z 3600 zdjęciami jest oddzielnym zakresem. Nie deklarować równoważności.
Użytkownik 31.10.2026 jawnie zniósł ograniczenie FINAL BLIND. Nowy przebieg
`retrieval_benchmark/xm3600.py` ocenia siedem wskazanych modeli w obu trybach promptów.
Manifest i protokół są przypięte przed scoringiem; natywnych opisów PL nie scalamy
według angielskich opisów, ponieważ XM3600 nie jest korpusem równoległych tłumaczeń.
XM3600 pozostaje chroniony przed treningiem w bieżącym rejestrze v4 i zamrożonym v2 generacji Wiki2.

## Wersjonowanie

Zmiana opisów, qrels, obrazów lub polityki wykluczeń wymaga nowego identyfikatora
galerii/protokołu i nowej wersji katalogu. Nie nadpisujemy wyników v1, nie uzupełniamy
brakujących komórek wynikiem z innej galerii ani nie wybieramy najlepszej wersji
tłumaczenia na podstawie rankingu modeli.
