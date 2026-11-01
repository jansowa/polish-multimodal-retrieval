# Kandydaci na polski eval tekst↔obraz — źródła sprawdzone 30.10.2026

To katalog do weryfikacji i przygotowania, nie gotowe zatwierdzone manifesty. Nie pobierano obrazów ani modeli. Żaden nowy benchmark nie został oceniony. OOD odnosimy do treningu projektora i potwierdzamy audytem uniwersumA; nie obiecujemy braku ekspozycji w pretreningu encodera.

| Kandydat | Język i istniejące etykiety | Dostęp / ograniczenie | Rola |
|---|---|---|---|
| [Flickr30k-multilingual](https://huggingface.co/datasets/aplominski/flickr30k-multilingual) | Dostępne opisyPL, tłumaczenia NLLB-200-1.3B oryginalnych EN, nie natywne polskie adnotacje | Karta deklaruje1000 test i 1014 validation; razem obrazy i podpisy. Sprawdzić oryginalne ID, splity i prawa do fotografii; deklaracja CC-BY repacka nie zastępuje audytu źródła | Pierwszy praktyczny kandydat bez nowego MT; nowa kolekcja zdjęć, podobna ogólna fotografia |
| [AIDe, IPI PAN](https://zil.ipipan.waw.pl/Scwad/AIDe), [publikacja](https://aclanthology.org/L18-1337/) | 1000 obrazów Flickr8k i 2000 ludzkich opisów napisanych po polsku | Adnotacje dostępne, strona odsyła po obrazy do kontaktu z autorką; sprawdzić publiczne mapowanie i dostępne oryginały bez korespondencji. Deklaracja zasobu CC-BY-SA4 | Wartościowy natywnyPL, ale nie gwarantować automatycznej dostępności obrazów |
| [VizWiz-Captions](https://vizwiz.org/tasks-and-datasets/image-captioning/) | Istniejące ludzkie opisyEN; tłumaczeniePL do wykonania | Publiczny validation 7750 zdjęć. Dostarczone flagi do wyłączenia template/spam captionów. Strona deklaruje CC-BY4. Używać jawnych adnotacji i sprawdzić faktyczny dostęp, nie wymagać serwera konkursowego | Inny profil zdjęć użytkowych od osób niewidomych, dobry kandydat na silniejszy shift |
| [DOCCI](https://google.github.io/docci/), [oficjalna karta](https://huggingface.co/datasets/google/docci) | Ludzkie, szczegółowe opisyEN; tłumaczeniePL do wykonania | Własne fotografie autorów, karta CC-BY4; test 5000. Długie captiony wymagają raportu truncation dla każdego modelu; pobierać wykonalny podzbiór oryginałów | Nowa kolekcja i diagnostyka długich, precyzyjnych opisów |

Źródła rezerwowe i zbiory poza bieżącym celem:

- [DCI](https://github.com/facebookresearch/DCI):7805 obrazów SA-1B i opisy. Dostęp do zdjęć wymaga warunkówSA-1B i archiwum; wersja „CLIP-ready” zawiera wygenerowane streszczenia. To mniej wygodny start niż powyższe źródła.
- [WIT](https://github.com/google-research-datasets/wit): teksty wielu języków zWikipedii; po dodaniu Wiki do train nie przedstawiać jako nowej domeny.
- [XM3600](https://github.com/google/crossmodal-3600): gotowe ludzkie captiony m.in.PL, ale w tym projekcie zarezerwowanyFINAL_BLIND. Nie wykorzystywać do bieżącego wyboru, nowego dev ani eksperymentów tej pary sesji.
- XTD10 bazuje na COCO, więc brak overlapu nie czyni go nową domeną. CIRR/FashionIQ/RefCOCO wyłączone zgodnie z preferencją użytkownika.

Wybór realizacyjny: co najmniej dwa wykonalne źródła, preferencyjnie Flickr30k-PL plus VizWiz-PL lub DOCCI-PL; AIDe dodatkowo jeżeli obrazy dostępne. Nie czekać na korespondencję ani ręczne oznaczanie. Nie udawać, że tłumaczenia są natywnym goldemPL. Nie traktować parsera ani modelowej kontroli języka jako pełnej ludzkiej walidacji.
