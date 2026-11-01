# P2.E — źródła licencyjne, 30.10.2026

Przegląd źródeł z 29–30.10.2026, nie opinia prawna. Oddzielamy licencję wag, prawa do danych, warunki nauczyciela i status outputów. Poniższe ustalenia nie kwalifikują prawnie wytrenowanego projektora ani nie zatwierdzają publikacji.

## Wagi i nauczyciele

| Składnik | Potwierdzone źródło i krótki cytat | Wagi / nauczyciel | Dane i outputy | Status przed publikacją |
|---|---|---|---|---|
| TinyCLIP-39M | [Karta modelu](https://huggingface.co/wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M), [licencja autora](https://raw.githubusercontent.com/microsoft/Cream/main/TinyCLIP/LICENSE): „The above copyright notice and this permission notice shall be included”. | Karta deklaruje MIT; zachować tekst licencji i informacje copyright przy dystrybucji objętego nią materiału. | Deklaracja MIT modelu nie dokumentuje praw do poszczególnych obrazów YFCC ani naszego treningu. | Warunki wag są wskazane; osobno sprawdzić prawa do danych i sposób pakowania. |
| PolDense-32M | [Karta OPI-PIB](https://huggingface.co/OPI-PIB/PolDense-32M): „License: gemma”. Również lokalny README przypiętej rewizji `ca70494a25dad90d980db7c735658dc7bf88ed25` ma `license: gemma`. | Model deklaruje warunki Gemma. Nie traktować go jako MIT/Apache. Karta opisuje destylację z BGE-Multilingual-Gemma2. | Uczenie w przestrzeni jego embeddingów wymaga osobnej oceny zakresu warunków nauczyciela. | **Niejasne dla naszego projektora / destylacji; decyzja użytkownika przed publikacją.** |
| Gemma — warunki powiązane z PolDense | [Gemma Terms](https://ai.google.dev/gemma/terms), aktualizacja 1.04.2026: „including distillation methods that use intermediate data representations”; „Google claims no rights in Outputs you generate using Gemma.” | §1.1(e) szeroko definiuje Model Derivatives. §3.1 wymienia umowę, ograniczenia użycia, oznaczanie zmian i NOTICE przy właściwej dystrybucji. | Same outputy nie są Model Derivatives w tej definicji; to odrębne od statusu modelu uczonego z ich pomocą. Brak roszczenia Google do outputów nie rozstrzyga praw innych podmiotów. | Ustalić zastosowanie do osobnego heada, paczki z PolDense i przyszłego studenta. Nie rozstrzygamy tego tutaj. |
| SigLIP2 | [Oficjalna karta Google](https://huggingface.co/google/siglip2-base-patch16-256): „License: apache-2.0”. | Deklarowana Apache-2.0; zachować właściwe licencje, notices i oznaczenia zmian przy redystrybucji. | Licencja wag nie stanowi wykazu praw do wszystkich obrazów treningowych ani danych użytych w naszym projekcie. | Warunki wag są wskazane. Dla roli nauczyciela uwzględnić odrębnie dane i finalny pakiet. |
| Qwen używany przez pipeline | [Oficjalny LICENSE Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE): „Apache License”, „Version 2.0”. | Aktualny oficjalny checkpoint deklaruje Apache-2.0. Odczytane manifesty VN i Wiki wskazują alias `qwen3.8-27b`; nie dokumentują dokładnej rewizji wag serwowanego endpointu. | Licencja wag nie jest samodzielnym zezwoleniem na redystrybucję tekstów wejściowych, tłumaczeń i przerobionych captionów. Nie dowodzi też przeniesienia praw autorskich do outputów na użytkownika. | Uzupełnić identyfikację wdrożonego checkpointu i warunki usługi. Prawa do captionów zależą także od źródła. |

## Dane treningowe i ewaluacyjne

| Źródło | Potwierdzenie | Dane / outputy | Status przed publikacją |
|---|---|---|---|
| VisualNews — zbiór | [Repozytorium autorów](https://github.com/FuxiaoLiu/VisualNews-Repository), [README](https://raw.githubusercontent.com/FuxiaoLiu/VisualNews-Repository/main/README.md), [hosting Rice](https://www.cs.rice.edu/~vo9/visualnews/). W sprawdzonych materiałach nie znaleziono wyraźnej licencji całego zbioru. | Dostępność archiwów nie dokumentuje zgody wydawców na obrazy i podpisy. Recykling i tłumaczenie wymagają rozpatrzenia praw do źródła. | **Niejasne**. Nie przypisujemy automatycznie MIT ani licencji non-commercial na podstawie samej dostępności datasetu. |
| Washington Post | [Warunki wydawcy](https://www.washingtonpost.com/terms-of-service/), §5: „personal and non-commercial use only”. | Warunki ograniczają użycie treści; nie rozstrzygamy zastosowania do historycznego zbioru. | **Wymaga oceny użytkownika.** |
| USA Today | [Właściwy adres warunków](https://cm.usatoday.com/terms/) oraz [Gannett](https://www.gannett.com/terms-of-use/). Próby odczytu nie dostarczyły treści dokumentów. | Nie ma zweryfikowanego cytatu ani potwierdzonego zakresu dozwolonego użycia. Warunki osobnego portalu Gannett Media Education nie zastępują warunków USA Today. | **Luka w źródłach**: uzupełnić ręcznie właściwe warunki i ich wersję. |
| COCO | [Warunki w oficjalnym repozytorium](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm): „The COCO Consortium does not own the copyright of the images.” | Adnotacje/strona: CC BY 4.0 według tego dokumentu. Obrazy mają odrębnych właścicieli; wymagane sprawdzenie właściwych warunków zdjęć/Flickr. | Atrybucja dla objętych nią adnotacji; **odrębny audyt obrazów**. Nie oznaczamy wszystkich zdjęć jako CC BY 4.0. |
| ARO / Visual Genome | [Licencja kodu ARO](https://github.com/mertyg/vision-language-models-are-bows/blob/main/LICENSE): „MIT License”. [Strona danych Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html) odsyła do [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). | MIT kodu ARO i deklaracja na stronie VG są różnymi warstwami. Sprawdzić konkretne adnotacje, obrazy i ich pochodzenie; licencja kodu nie dokumentuje praw do zdjęć. | Zachować atrybucje i powiązania z rekordami. Zakres praw do materiałów źródłowych pozostaje do oceny. |
| SugarCrepe | [Licencja repozytorium autorów](https://github.com/RAIVNLab/sugar-crepe/blob/main/LICENSE): „MIT License”. | Kod i zmodyfikowane teksty należy rozpatrzyć oddzielnie od wykorzystanych obrazów i podpisów COCO. Nie znaleziono odrębnego potwierdzenia pełni praw do zdjęć w opakowaniu `mteb/SUGARCREPE_fmt`. | Zachować notices kodu; przeprowadzić audyt źródeł COCO. MIT repozytorium nie wystarcza jako licencja wszystkich zdjęć. |
| Wikimedia Commons | [Instrukcja ponownego użycia](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia): „Each media file has its licensing specified on its file description page.” | Licencję, autora, źródło, attribution i ewentualne ShareAlike oceniać per plik. Autorzy zdjęć i tekstów mogą być różni. Status wag nie wynika automatycznie z obecności CC BY-SA. | Zachować manifest praw per rekord. Obecność pól nie jest weryfikacją prawidłowości deklaracji. |

## Sprawdzenie metadanych zamrożonego Wiki dev

W `wiki_v2/wiki_dev.jsonl` wszystkie **397** rekordów mają niepuste `author`, `license_url` i `source_page_url`. Deklarowane etykiety:

| Etykieta | Rekordy |
|---|---:|
| CC BY-SA 4.0 | 227 |
| CC BY-SA 3.0 | 52 |
| CC BY 4.0 | 44 |
| CC0 | 43 |
| CC BY 3.0 | 19 |
| CC BY-SA 2.0 | 5 |
| CC BY 2.0 | 5 |
| CC BY-SA 2.5 | 2 |

To odczyt zachowanych deklaracji, bez ponownej kwalifikacji każdego dzieła. Nie przenosimy tego wyniku na nową pulę 24 125 rekordów Wiki; jej pełny audyt obrazów i praw pozostaje otwarty.

## Decyzje do przygotowania publikacji

1. Zakres paczki: sam head, head + encodery czy usługa; sposób spełnienia warunków Gemma i ocena statusu projektora/studentów.
2. Podstawa użycia źródeł newsowych i dopuszczalny zakres publikacji; uzupełnienie warunków USA Today. Dossier nie zatwierdza publikacji wag wyuczonych na VN.
3. Manifest atrybucji oraz źródła i licencje obrazów/tekstów z COCO, VG, SugarCrepe i Wiki.
4. Dokładny checkpoint Qwen użyty przez endpoint i ewentualne dodatkowe warunki dostawcy usługi.

Nie podejmowano publikacji ani treningu destylacyjnego. Status P2.E: **częściowo — źródła i rozdzielenie warstw zebrane; warunki USA Today oraz wskazane kwalifikacje nierozstrzygnięte**.
