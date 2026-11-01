# Prompt A — kontrolowany test nowych VN i Wiki

Pracujesz w repozytorium:
`D:/work/translate-vision-embedder-benchmark/pl-multimodal-benchmark-translator/pl-multimodal-benchmark-translator`.

Przeczytaj i realizuj `reports/data_ood_coordination_20261030/COORDINATION.md`. To kontrakt z równoległą sesją B. Wykonaj pracę, a nie tylko plan. Nie uruchamiaj sesji B ani subagentów. Aktualne zadanie nadaje badaniu nowych danych pierwszeństwo przed dawnym scoutingiem backbone'ów, samplingiem/ARO i zmianami heada.

## Cel i uprawnienia

Sprawdź wpływ większej, audytowanej puli VN oraz dodania Wiki przy tych samych enkoderach TinyCLIP39M/PolDense32M i Linear512→384. Ten prompt **autoryzuje ograniczony pilot Wiki po spełnieniu wszystkich poniższych bramek**; nie proś wtedy ponownie o ogólną zgodę. Nie autoryzuje XM3600, publikacji, FT/LoRA, KD ani zmiany działających pipeline'ów. Użytkownik nie ma czasu na nowe ręczne adnotacje; nie uzależniaj pracy od ich zebrania.

Punkt odniesienia: VN_FULL seedy42/43/44, 5500 kroków; stary finalista seed44, SHA `4bf977049098ca6d8bcc7abc177183aace8c03c5f08e0ecaf011d82aad5b6844`. Zachowaj jego immutable `reports/series_2026-10-23/wiki_dev_zero_shot_pre_wiki_training.json`. Przestrzeń i queryprefix pozostają identyczne.

## Najpierw koordynacja i dane

1. Utwórz własne katalogi z kontraktu. Jako właściciel wspólnego kodu zaimplementuj minimalny helper blokady GPU, atomowych markerów i rezerwacji miejsca; sprawdź realne wykluczanie dwóch procesów i odmowę nadmiernej rezerwacji. Opublikuj `COMMON_READY.json` wraz z SHA i prostym przykładem użycia dla B. Nie zmieniaj wspólnych modułów `src/`.
2. Przeczytaj: `expanded_corpus_metadata_audit_20261029.json`, `expanded_corpus_preparation_decision_20261029.json`, `wiki_dev_guard_enforcement.json`, `wiki_dev_guard_integration.json`, `preservation_before.json`, `vn_full_seeds_posthoc.json`, `vn_full_seeds_training_summary.json`, `evaluation_scope_amendment_20261030.json` w `reports/series_2026-10-23/`. Ostatnia zamrożona pula to132 875 VN i24 125 Wiki; `training_ready_rows=0`. Provisional 96 930 VN / 22 580 Wiki nie są zweryfikowanym train.
3. Domyślnie użyj istniejących surowych kopii pod `C:/Users/jsowa/benchmark-translations/series-2026-10-23/expanded-inputs-20261029/`, sprawdź SHA. Jeśli świadomie wybierzesz nowszy eksport, zapisz decision przed przygotowaniem i odczytaj każdy atomowo publikowany `final_pairs.jsonl` jednym deskryptorem; zachowaj dokładnie odczytane bajty. Nigdy SQLite/WAL. Publikuj `A_INPUTS_FROZEN.json`; nie zmieniaj eksportu w połowie badania.
4. Zweryfikuj mapowanie rekord→oryginalny obraz, dostępność i provenance, raw/pixelSHA, pHash≤6, źródła i teksty EN/PL po normalizacji. Użyj istniejących assetów i metadanych; nie pobieraj ponownie całych zbiorów. Historyczne fingerprinty można reużyć po sprawdzeniu manifestów i dowodów ich weryfikacji; sam cache embeddingów nie jest dowodem tożsamości obrazu. Jedno dekodowanie może służyć wielu hashom. Przy braku lokalnych oryginałów nowych przykładów raportuj brak; nie wstawiaj miniatur ani zamiennych obrazów.
5. Chroń pełne komponenty wszystkich dotychczasowych dev/val/eval: Wiki397 i wszystkie zarezerwowane komponenty, VNdev, COCOval, legacy eval, XTDfull, XMblacklist wyłącznie do audytu. Usuń z nowego train także aliasy starego FULLtrain — stare rekordy pozostają w mieszance, nowe ich kopie nie są nowymi przykładami. Deduplikacja pochodzenia, obrazów, pHash i tekstów jest transytywna. PHash nie zmienia pozytywnych etykiet gold. Nie zmieniaj historycznych train/val.
6. Zachowaj stare VNtrain z oryginalnymi captionami; dokładaj faktycznie nowe, rozłączne komponenty. Nie nadpisuj starych podpisów nowszym recyklingiem pod tym samym ID. Wiki zachowuje provenance native/recycled/translated; nie nazywaj całości ludzkim goldem. Wyjaśnij licencje i atrybucję, bez automatycznej decyzji o publikacji.
7. Cel: cała dostępna zweryfikowana pula, jeśli mieści się w budżecie. Jeżeli pełny audyt/materializacja nie mieszczą się, przed wynikami zarejestruj deterministyczny pilot do 20 000 nowych obrazówVN i 5000 Wiki, bez wyboru po wynikach modelu. Opublikuj uczciwe liczniki; nie czekaj na wszystkie przyszłe eksporty pipeline'u.
8. Zamknij **uniwersum wszystkich obrazów/tekstów dopuszczonych do wszystkich wariantów**, ze starym FULL/VN oraz fingerprintami chronionych zbiorów. Opublikuj `A_UNIVERSE_FROZEN.json` zgodnie z kontraktem. Od tej chwili nie dodawaj treningowych obrazów/captionów. B odrzuci pasujące przykłady OOD, więc możesz trenować bez czekania na wybór datasetów B.

## Bramka Wiki — żadnego obchodzenia historycznego guarda

Sprawdź SHA Wiki dev397 (`e761b644719b8d8795bc56716fa6f67bdc1f5531b9f0e2f76955a1ca91e2c4a3`), guarda i pre-Wiki baseline. Zaimplementuj nowy loader, który odrzuca **wszystkie** rekordy/obrazy z zarezerwowanych komponentów, także image-only. Testy mają wykryć aliasy i pHash, a nie tylko porównać liczniki. Zarejestruj release w swoim katalogu z SHA train/dev/guard/baseline, dowodem testów i niniejszym uprawnieniem użytkownika. Nie zmieniaj starych plików bramek. Historyczne `require_wiki_training_release` kończy się celową blokadą starego runnera nawet po flagach — nie usuwaj tego `raise`; użyj nowego zweryfikowanego loadera i własnego walidatora release w swoim runnerze.

## Badanie, prerejestracja przed pierwszym nowym wynikiem

| Wariant | FULL_CLEAN | VN | Wiki |
|---|---:|---|---:|
| R0 | 50% | 50% stareVN | 0% |
| R1 | 50% | 50% stareVN +nowe rozłączneVN | 0% |
| R2 | 50% | 47% stareVN +noweVN | 3% |
| R3 | 50% | 40% stareVN +noweVN | 10% |

Procenty są udziałami prezentacji par, nie udziałem plików. FULLsloty i ich oryginalne próbkowanie nie zmieniają się. Wiki zastępuje wyłącznie slotyVN. Dokładne liczby dla batch 256 i końcowego 145 rozliczaj deterministycznie między batchami, logując rzeczywiste prezentacje; dla R0/R1 ma to być dokładnie historyczny sampler. Przeczytaj `src/pl_mmb_training/source_ablation.py`: `Sampler`, `train`, PositivePairs i selekcję. VN pozostaje image-balanced, bez zmiany lossu i masek wielopozytywnych.

- Reuse R0 seedy42/43/44, jeśli wszystkie wejścia/precision/konfiguracja są identyczne; bez ponownego legacy testu. Ewentualna niezgodność zatrzymuje porównanie, nie pozwala po cichu trenować innego baseline'u.
- R1/R2/R3: najpierw seed42. Recepta: fresh Linear512→384, oba encodery frozen, FP32weights +BF16autocast, cacheFP32, 5500 kroków, ten sam rzeczywisty budżet prezentacji jak P1.A; batch 256, końcowy 145 na okno, AdamW0.001, weightdecay0.01 tylko dla heada, świeży SigLIPsigmoidloss, brak augmentacji/schedulera. Odtwórz resztę konfiguracji z zatwierdzonego baseline'u, nie zgaduj.
- **Checkpoint w każdym runie wybieraj dokładnie jak P1.A po VNdev**, z tym samym tie-breakiem i częstotliwością. Nie zmieniaj selekcji na Wiki w połowie.
- Rejestracja wyboru wariantu: spośród R2/R3 wybierz większy Wiki-dev t2iR1; remis→mniejszy udziałWiki, potem wcześniejszy zapisany krok. Raportuj spadek VN względem R1; >2 p.p. oznacza niezaliczony guardrail i brak rekomendacji zastąpienia R1, ale nie kasowanie niekorzystnego wyniku.
- Dokończ R1 i jeden wybrany wariantWiki dla seed43/44; seed42 reuse. Dla każdego raportuj mean±SD po trzech seedach. Reprezentant wariantu do jednorazowego nowego OOD: mediana VN-dev t2iR1 z seed42/43/44, przy remisie niższy seed; to zgodna reguła wyboru seedów, odmienna od wyboru wariantu po Wiki. R0 do OOD zachowuje oryginalny seed44.
- Nowych wyników B nie odczytuj, dopóki nie zamkniesz wszystkich runów i selekcji. Nie uzależniaj wyboru od XTD/COCO; traktuj je posthoc. Bez nowego legacyCOCO testu i bez XM3600.
- Oceń już zamrożone galerie VN/Wiki/XTD tym samym adapterem/harnessem, wykorzystuj cache. Dla R2/R3 Wiki to **ID**, dla R0/R1 źródłowo **OOD**; stary wynik przed Wiki zachowaj oddzielnie. Wszystkie warianty mają COCO w treningu. OOD od B oceni później B.
- Każdy wybrany checkpoint: reload, odtworzenie metryk i SHA, skończone cechy, zgodność wymiaru i prefixu. Zachowaj niekorzystne wyniki.

## Zakończenie i handoff

W swoim raporcie pokaż rozliczenie danych/odrzuceń, wszystkie4 warianty, średnie/SD potwierdzonych wariantów, ID/OOD, parametry V+T/V/T (nasze70470528/38587392/31883136; head196992 osobno), rzeczywisty czas i dysk oraz ograniczenia. Opublikuj `A_FINALISTS_FROZEN.json`: lista reprezentantów R0/R1/wybranyWiki, wszystkie 9 checkpointów trzech zamkniętych wariantów jako analiza seedów, pełne SHA i wskazanie które runy uzyskają końcowy OOD. Nie wysyłaj B dowolnych dodatkowych modeli po pierwszym wyniku. Po zakończeniu wszystkich swoich zadań CUDA opublikuj `A_GPU_DONE.json` i zwolnij lock. Nie kasuj plików. Nie publikuj niczego na zewnątrz.
