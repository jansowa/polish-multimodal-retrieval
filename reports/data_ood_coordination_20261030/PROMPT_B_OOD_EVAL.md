# Prompt B — gotowy eval tekst↔obraz bez nowych ręcznych adnotacji

Pracujesz w repozytorium:
`D:/work/translate-vision-embedder-benchmark/pl-multimodal-benchmark-translator/pl-multimodal-benchmark-translator`.

Przeczytaj i realizuj `reports/data_ood_coordination_20261030/COORDINATION.md` oraz `DATASET_CANDIDATES.md`. To kontrakt z sesją treningową A. Wykonaj przygotowanie i ocenę, nie kończ na liście pomysłów. Nie uruchamiaj A ani subagentów. Użytkownik nie ma czasu na oznaczanie, przeglądanie próbek ani generowanie ręcznych query — wykorzystaj **istniejące** powiązania obraz–opis, ewentualnie tłumacząc opisy.

## Zakres i zasoby

Tylko tekst→obraz i obraz→tekst. Bez CIRR/FashionIQ/RefCOCO, query obraz+tekst, VQA ani generowania opisów z obrazów. Nie trenuj modeli retrieval. Nie zmieniaj A, kanonicznych raportów starej serii ani `src/`; pisz we własnych katalogach. Obowiązują budżety i brak usuwania danych z kontraktu.

Przygotowanie na CPU i małe metadane można od razu. Po `COMMON_READY.json` korzystaj ze wspólnego helpera, nie implementuj konkurencyjnego locka. Nowe tłumaczenia lokalnie na CPU albo na GPU dopiero po `A_GPU_DONE`; nigdy kosztem alokacji VN/Wiki. **Żadnych operacji CUDA do chwili sprawdzenia `A_FINALISTS_FROZEN.json` i `A_GPU_DONE.json` oraz uzyskania istniejącego wspólnego locka GPU.** To obejmuje translator i sprawdzanie modeli. Przy tłumaczeniu równoległym ustaw urządzenie jawnie na CPU, bez `device_map=auto`.

## Kandydaci i wybór przed wynikami

1. Flickr30k-PL: sprawdź `https://huggingface.co/datasets/aplominski/flickr30k-multilingual`. Gotowe PL po NLLB-200-1.3B, nie natywne polskie gold. Użyj tylko validation/test i sprawdź rzeczywiste splitID, licencje fotografii i captionów oraz mapowanie na źródłowe FlickrID. Nie pobieraj całego train. Nie nazywaj lokalnego odfiltrowanego podzbioru oficjalnym Flickr30k1K.
2. VizWiz-Captions-PL: `https://vizwiz.org/tasks-and-datasets/image-captioning/`. Użyj jawnie dostępnych opisów, preferuj validation, nie zakładaj dostępności challenge-testu. Odfiltruj odrzucone/templatecaptions zgodnie z dostarczonymi `is_rejected` i `is_precanned`, nie oceną naszych encoderów. Tłumacz istniejące opisy, zachowaj ich powiązania z obrazami. To kandydat na silniejszą zmianę profilu zdjęć.
3. DOCCI-PL: `https://google.github.io/docci/`, `https://huggingface.co/datasets/google/docci`. Oryginalne ludzkie opisyEN do tłumaczenia. Zachowaj długie teksty; raportuj tokeny i odsetek truncation dla każdego modelu. Bez tworzenia krótszych „goldów” LLM-em. Oryginalne fotografie o kontrolowanym pochodzeniu, bez używania wersji syntetycznie przepisanej.
4. AIDe: `https://zil.ipipan.waw.pl/Scwad/AIDe`, publikacja `https://aclanthology.org/L18-1337/`. Natywne polskie opisy. Zweryfikuj możliwość uzyskania oryginalnych obrazów legalnie i automatycznie po Flickr8kID; jeśli wymaga korespondencji lub brak aktywów, oznacz blokadę i kontynuuj z innymi zbiorami. **Nie wysyłaj maili ani wiadomości do autorów.** Nie jest potrzebne tworzenie nowych podpisów.

Przed oceną wybierz w decisionJSON zestawy wykonalne pod względem licencji, obrazów, tłumaczeń i dysku, niezależnie od scores. Celminimum: **dwa nowe, rozłączne źródła** — Flickr30k-PL i co najmniej jedno z VizWiz/DOCCI; AIDe dodatkowo jeśli dostępne. Wszystkie4 nie są obowiązkowe. Jeżeli nic poza jednym nie jest dostępne, udokumentuj ograniczenie zamiast wymyślać dane. DCI/SA-1B ze zgodą i dużym archiwum to późniejsza rezerwa, nie blokada bieżącej pracy. WIT/Wiki po uczeniu Wiki nie spełniają celu nowej domeny; XTD/COCO też nie. XM3600 ma już PL, lecz **nadal FINAL_BLIND: żadnych wyników, dzielenia na nowy dev, probe ani wyników konkurencji**.

## Rozłączność i zamrożenie

- Wstępne pobranie katalogu nie jest zgodą na benchmark. Poczekaj na zweryfikowany `A_UNIVERSE_FROZEN.json`. Wyklucz powiązania z całym uniwersum A, starymi trainami i wszystkimi chronionymi dev/val/eval. Uwzględnij źródła/rodziców/URL, file i pixelSHA, pHash≤6, identyczne znormalizowane EN/PL. Używaj wyłącznie blacklisty XM do audytu, bez oceniania XM.
- Nie deduplikuj na podstawie embeddingów ocenianych modeli. Nie usuwaj trudnych przykładów po score. Komponenty bliskich zdjęć trzymaj razem między splitami i używaj do bootstrapu; wewnątrz DOCCI nie kasuj celowych kontrastujących zdjęć tylko dlatego, że pHash jest bliski. Qrels nie wynikają z pHash.
- Dla każdego źródła celuj początkowo w 1000 obrazów na zamrożoną galerię, minimum 500 po audycie; zachowaj wszystkie istniejące zaakceptowane captiony do wybranych obrazów. Dobór deterministyczny, hashkolejność ze stałym seedem i z uwzględnieniem komponentów, nie wyników modeli. Gdy rozmiar/split nie pozwala, raportuj faktycznyN. Nie modyfikuj oryginalnego splitu bez nowej lokalnej nazwy.
- Oryginalne pary są qrels, nie nowymi ocenami LLM. Dla tych samych captionów EN/PL po normalizacji unifikuj query i łącz znane pozytywne obrazy; exactduplikaty obrazów kanonikalizuj. Nie nadawaj pHashkomponentom wspólnej etykiety pozytywnej. Nie dorabiaj hardnegativegoldów.
- Tłumaczenie wyłącznie tekst→tekst, bez obrazu, nazw pliku, identyfikatora celu ani wyników retrievalu. Zachowaj EN, PL, oryginalnyID, model/revision, prompt/parametry, cacheSHA i koszty. Automatyczne QA: kompletność, język, powtórzenia, cyfry, negacje, puste/błędne odpowiedzi. Reguły przed oceną; nie oceniaj tłumaczeń za pomocą modeli startujących w benchmarku. Nie poprawiaj captionów, by pasowały do obrazu. Jeśli dane są z gotowych tłumaczeń, wyraźnie to oznacz.
- Nie zastępuj oryginałów miniaturami ani nie pobieraj całych archiwów bez oszacowania, tylko dlatego że są najłatwiej dostępne. Czytelna licencja i pochodzenie per źródło; nie przenoś automatycznie licencji repacka na wszystkie zdjęcia.
- Zamroź źródła/revisions/SHA, manifesty, tłumaczenia, obrazy, qrels, galerie i komponenty; publikuj `B_EVAL_FROZEN.json`. Nowe benchmarki w tej serii nie są narzędziem wyboru modeli A — to potwierdzenie po zamknięciu selekcji. Nie trzeba organizować dodatkowych ręcznych splitów ani adnotacji.

## Lista modeli i wykonanie

Zarejestruj przed pierwszym score faktycznie zamknięte galerie, checkpointy A oraz **stałe referencje**: polskie MetaCLIP-S16/M16 i EmbeddingGemma2 jako większe jakościowe odniesienie. Przypnij dokładne rewizje z `reports/series_2026-10-23/reference_eval_registration_v3_20261029.json`. To nowa rejestracja porównania na nowych danych, z uzasadnieniem małego podzbioru referencji zamiast ponownego liczenia całej 20. Bez dodawania/wyjmowania modeli po score. MobileEN nie jest główną konkurencją polską.

Lista naszych modeli: po `A_FINALISTS_FROZEN.json` wszyscy9 zarejestrowanych checkpointów R0/R1/wybranyWiki do mean±SD, z wyraźnie wyróżnionymi trzema ustalonymi reprezentantami. Jeśli guardrail Wiki niezaliczony, nadal pokaż wariant jako diagnostykę, bez nazywania go nowym zwycięzcą. Nie wybieraj „najlepszego seeda na OOD”. Gdy A nie ukończy szkolenia, przygotuj dane i raportblockedByA zamiast zastępować checkpointy.

Po zamrożeniu modeli A i zakończeniu jego CUDA:

1. Zweryfikuj istniejący adapter ProjectorEncoder i wspólny scorer, nie wprowadzaj nowej geometrii. Własne encodery/cechy reuse z właściwymi SHA, FP32weights +BF16autocast, wynikFP32. Przechowuj cache cech bazowych niezależnie od headów; nie koduj9razy tych samych obrazów/textów.
2. Referencje ładuj **po jednej**: wszystkie nowe galerie i oba tryby promptów, zapis wyników/embeddingów, dopiero potem bezpieczne usunięcie własnych wag. Gemma bez audio, tylko vision+text. Zachowaj historyczną precyzję referencji i przypięte procesory, bez arbitralnego ujednolicania po wynikach. Przestrzegaj istniejącego `flock` i budżetów.
3. Dwa kierunki, pełne galerie i multi-positive; R@1/5/10, MRR, NDCG@10. Prerejestrowany pairedbootstrap po komponentach10 000 / seed 20261023, ta sama galeria i query dla każdej pary. Porównaj nowe modele z R0 oraz z referencjami; mean±SD po seedach oddziel od warunkowychCI.
4. W tabelach trzy liczby parametrów: V+T, V, T. Dodatkowe encodery i osobnyhead poza sumą. Dla nas70470528=38587392+31883136; head196992. Użyj `pl_mmb_benchmark.report_parameters`. Podawaj wymiar i istniejące zgodne pomiary efektywności bez ich powtarzania tylko z powodu nowego datasetu.
5. ID/OOD określ per model i zbiór względem treningu projektora. Nowe źródło to kandydat na source-OOD po audycie, nie dowód braku pretreningu. Długie opisy, MT i inne zdjęcia opisz jako różne rodzaje przesunięcia. Konkurenci bez pełnego audytu: ekspozycja nieustalona. Stary Wiki benchmark po dodaniu Wiki nie jest dla nowego modelu OOD.

## Produkty końcowe

Własny `REPORT.md`, JSON i CSV z pełnymi metrykami, pairedCI, liczbami po audycie, source/revision/SHA, licencjami, tokenizacją/truncation, statusem tłumaczeń i kosztów, wyjaśnieniem co jest source-OOD i co pozostaje niepewne. Zero ręcznych nowych etykiet. `B_RESULTS_READY.json` wskazuje finalne pliki/SHA. Nie informuj A o rankingach w toku zamkniętej selekcji; końcowy wynik jest do decyzji o kolejnej, nowej serii. Nie publikuj danych lub modeli na zewnątrz.
