# 06 — Dane, holdout, QA

## 1. Warstwy danych i poziomy licencyjne

| Warstwa | Zawartość | Użycie |
|---|---|---|
| T0 `f-data-v*` | manifest po H2; każdy rekord z polami pochodzenia i `tier ∈ {WEIGHTS, STRICT}` | trening kandydatów (poziom wybrany w H1) |
| T0‑U | pula obrazów bez podpisów po bramce holdoutów (§4) | wyłącznie S1 (cele z nauczyciela) |
| T1 `RESEARCH_ONLY` | źródła dopuszczone w H1 tylko do diagnostyki | eksperymenty z flagą; nigdy inicjalizacja kandydata |
| T2 `EVAL-INTERNAL` | galerie i panele z [02](02_PROTOKOL_EWALUACJI.md) | wyłącznie pomiar |
| T3 `BLOCKED` | wszystko inne, w tym dane modeli NC | nie używać |

Pola pochodzenia per rekord (obowiązkowe): `image_license`, `image_author`, `image_url`, `source_id` (Commons file ID/SHA1, COCO/VG/Flickr ID, VN article path), `text_origin ∈ {native-original, native-recycled, translated, generated}`, `text_model` (alias i, jeśli ustalono, rewizja), `text_quality` (A–D dla Wiki), `grounding` (0–4, jeśli dotyczy), `fingerprints` (SHA bajtów, SHA pikseli, pHash), `component_id`, `tier`.

## 2. Ścieżka dopuszczenia do `f-data-v*`

1. **Pochodzenie i poziom** per rekord ([01 §4–5](01_CEL_ZASADY_LICENCJE.md)).
2. **Guard** wg `training/DATA_SPLITS.md`: fingerprinty oryginału przed resize/crop/JPEG (SHA bajtów, SHA pikseli RGB po EXIF, pHash DCT 32×32 / blok 8×8 / mediana z DC, Hamming ≤ 6); komponenty przechodnie z pełnych metadanych puli; `current_holdouts` z aktualnym rejestrem (v5, potem v6). Kod 2 = stop.
3. **Sygnały nauczyciela** (§5) nadają flagi — nic nie wykluczają automatycznie poza bliskością holdoutu.
4. **Ludzki QA** (§6) kalibruje flagi.
5. **Zamrożenie:** manifest + SHA rejestrów, kodu i wejść + powody wykluczeń; wniosek H2.
6. **Ponowna kontrola** przed treningiem i po uzyskaniu blokady GPU; także po scaleniu źródeł, augmentacji i recyklingu.

„Ready” w pipeline’ie NIE oznacza zgody na trening.

## 3. Wikimedia — reguły

- Allowlista licencji: CC0, CC BY, CC BY‑SA z zgodnymi nazwami i URL; braki, NC, ND i nieznane warianty wykluczone (jak dotąd).
- Jakość tekstu A–D (skala źródła) to inna oś niż grounding 0–4 (skala modelu). Do SEL‑ID (Wiki‑F dev) i do części Commons holdoutu — wyłącznie `native-original` A/B. Do treningu na poziomie WEIGHTS — także recyklowane i tłumaczone z zapisanym `text_model`; na poziomie STRICT — tylko `native-original`.
- Limit w samplerze: ≤ 10 obrazów z jednego komponentu na epokę budżetową; z manifestu niczego się nie usuwa.
- Ochrona kultury: progi filtrów liczone jako kwantyle w warstwach kategorii, nie globalnie.
- Atrybucja (autor, licencja, URL) zostaje przy każdym rekordzie — wejście do H6.
- Oryginały: pobierane osobno od metadanych; fingerprinty liczone z oryginału przed zmniejszeniem (jak w generacji Wiki2).

## 4. Pula obrazów bez podpisów U (do S1)

- Źródła: Wikimedia z allowlisty (pula 417 353 obrazów w statusie), VN (283 012 zweryfikowanych obrazów) — VN tylko na poziomie WEIGHTS i po H1 dla praw do obrazów. Podpisy nie są potrzebne.
- Każdy obraz: lokalny oryginał, fingerprinty, przepustka guarda (także wobec Wiki‑F dev, kandydatów i części testowej holdoutu), `component_id`. Obraz z komponentu mającego kontakt z holdoutem wypada z U.
- Próbkowanie zbalansowane po klastrach osadzeń EG2 (k‑means, 1 000 klastrów) z limitem na klaster; warstwy kategorii Wikimedia chronione kwantylami.
- Rozmiar w FE3: ≥ 50 000; skala U0–U3 badana w FE5a; pobieranie ponad to, co jest lokalnie, wymaga H5.
- Cache widoków: środkowy dla U; dla obrazów par dodatkowo 2 RandomResizedCrop z zapisanymi parametrami (F04).

## 5. Sygnały nauczyciela (EG2 główny; mSigLIP drugi głos po `CLEARED`)

| Sygnał | Jak liczony | Kiedy flaga |
|---|---|---|
| (a) zgodność rangowa | pozycja własnego obrazu wśród 1 000 losowych obrazów tej samej warstwy (źródło × pochodzenie tekstu × kategoria) wg EG2 tekst→obraz | ranga > 50 albo dolne 5% warstwy |
| (b) dryf tłumaczenia | `Δ = s(obraz, PL) − s(obraz, EN)`; `cos(EG2(PL), EG2(EN))` | dolne 2% warstwy |
| (c) bliskość holdoutu | największy cos obrazu do dowolnego obrazu holdoutu/galerii w przestrzeni obrazów EG2; próg kalibrowany na znanych duplikatach i cropach (pary pHash) | powyżej progu → przegląd ręczny; wykluczenie dotyczy wyłącznie nowych kandydatów treningowych |
| (d) false negatives | odsetek par spoza przekątnej w losowych batchach z podobieństwem nauczyciela ≥ mediana podobieństw pozytywów | > 1% → maskowanie takich par w S2 |
| (e) klaster | przypisanie k‑means | balansowanie i limity |

Reguły: progi kwantylowe w warstwach; flaga kieruje do QA; automatyczne wykluczenie tylko dla typów flag z precyzją ≥ 70% zmierzoną w QA; EG2 nigdy nie tworzy etykiet TEST; wartość filtra weryfikuje FE5b (D2 vs D1). Pułapki: filtr tnie różnorodność kulturową; enkodery słabo wykrywają błędy relacji/atrybutów (to robi człowiek); ten sam model jako filtr i nauczyciel podwaja swoje preferencje — dlatego TEST ma wyłącznie ludzkie qrels.

## 6. Ludzki QA (H3)

- Próbka ~600 par warstwowanych po źródle × pochodzeniu tekstu × decylu sygnału (a), z nadreprezentacją dolnych decyli; dodatkowo 100 par odrzuconych przez pipeline (fałszywe odrzucenia) i po 100 podpisów MT z Flickr‑PL i VizWiz‑PL (jakość galerii testowych; przy > 15% błędów semantycznych galeria wypada z roli core w następnej wersji protokołu).
- Podwójna ocena 10% (κ Cohena). Etykiety: poprawny; drobny błąd; zła treść; błąd tłumaczenia zmieniający sens; treść niewidoczna na obrazie; nienaturalna polszczyzna.
- Narzędzie: `reviewer/server.py`. Wyniki: odsetek błędów per warstwa (przedziały Wilsona), precyzja/czułość flag, zgodność człowiek–sędzia pipeline’u (miara samopotwierdzania). Koszt: 6–8 osobogodzin.

## 7. Nowy holdout (TEST + dev)

- **Obrazy:** ~600 zdjęć własnych zespołu (od listopada 2026): polska codzienność, jedzenie, transport, wnętrza, przyroda, napisy; celowo także trudne (noc, ruch, rozmycie, telefon); bez rozpoznawalnych osób i tablic albo ze zgodą; w kopiach roboczych usunięty EXIF‑GPS, fingerprinty z oryginału. Plus ~400 plików Commons z F05.
- **Teksty:** dwóch niezależnych annotatorów na obraz; każdy pisze krótkie zapytanie (2–6 słów, jak do wyszukiwarki) i jedno zdanie opisu; bez wglądu w opisy źródłowe i wyniki modeli; bez LLM/VLM.
- **Jakość galerii:** deduplikacja wizualna w obrębie holdoutu (pHash + EG2) ogranicza false negatives; qrels wyłącznie ludzkie.
- **Podział:** 40% dev / 60% test po komponentach → przy ~1 000 obrazów i ~4 000 tekstów część TEST ma ~2 400 zapytań t2i ([02 §9](02_PROTOKOL_EWALUACJI.md)); część dev wchodzi do SEL‑OOD jako dwa zbiory (zapytania, zdania).
- **Rejestr v6:** nowy plik; aktywacja jawna dla nowych runnerów; po aktywacji ponowny audyt manifestów serii F.
- **Pieczęć:** pliki testu i qrels w katalogu z ograniczonym dostępem, SHA w logu; odczyty wg [02 §7](02_PROTOKOL_EWALUACJI.md). Holdoutu NIE WOLNO używać do treningu ani doboru filtrów, progów i hiperparametrów.
- Koszt: ~50 osobogodzin annotacji; praca równoległa od etapu 1, nie blokuje FE0–FE3.

## 8. Wiki‑F dev

~1 000 obrazów z puli Wikimedia w całych komponentach, teksty `native-original` A/B, warstwowo po kategoriach; bez komponentów Wiki G2 i bez 1 159 obrazów zarezerwowanych; rejestr v5 (F05) przed jakimkolwiek treningiem na tej puli; rola SEL‑ID.

## 9. Korpusy tekstowe do mostu `g_text` i do Fazy 2

| Korpus | Rola | Licencja (do weryfikacji w F02) |
|---|---|---|
| podpisy PL z `f-data-v1` (bez tekstów galerii) | most | jak manifest |
| teksty natywne Wikimedia (A–D) | most | CC per rekord |
| FineWeb‑2 PL (podzbiór) | most | ODC‑By |
| polskie MS MARCO, FiQA, SQuAD v2, TriviaQA, FEVER + 10 trudnych negatywów (od właściciela) | most (opcjonalnie), replay w Fazie 2, panel retencji (części odłożone) | MS MARCO: badawcza (do rozstrzygnięcia w H1); SQuAD CC BY‑SA 4.0; TriviaQA Apache‑2.0; FEVER CC BY‑SA 3.0; FiQA do sprawdzenia |
| mFAQ PL (65 182 par), MAUPQA (3,66 mln wierszy) — lokalnie | most, panel retencji | do weryfikacji |

Most nie wchodzi do eksportu, ale jego dane są „sygnałem treningowym” w rozumieniu poziomu STRICT — status BRIDGE w rejestrze.

## 10. Co zbierać dalej i czego nie

- **Tak:** oryginały i fingerprinty puli Wikimedia (po FE5a i H5); natywne polskie podpisy Commons (strukturalne i opisowe) — priorytet przed tekstem syntetycznym; zdjęcia własne do holdoutu.
- **Warunkowo:** podpisy w stylu krótkich zapytań generowane modelem — tylko po statusie DATA‑GEN generatora i H5 (budżet API); regeneracja opisów VN/Wiki zatwierdzonym generatorem.
- **Nie:** rozszerzanie VN poza to, co ma lokalne oryginały; nowe galerie MT na COCO jako „nowe domeny”; dane z modeli NC w jakiejkolwiek roli.
