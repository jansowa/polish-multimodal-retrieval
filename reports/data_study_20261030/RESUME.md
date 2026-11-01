# Sesja A zakończona — historia wznowienia poniżej

Aktualny stan: COMPLETE. Po poleceniu „Kontynuuj prace” wyjaśniono różnicę
batch8/batch32 w BF16; odtworzone batche są bitowo zgodne. Wykonano siedem
zarejestrowanych nowych runów, ocenę zamrożonych galerii i bootstrap komponentów.
Raport: REPORT.md; rozliczenie: final_accounting.json. Wszystkie preservation
SHA i dziewięć checkpointów sprawdzono. Markery A_FINALISTS_FROZEN i A_GPU_DONE
są opublikowane. GPU zwolnione, nie planuje się dalszego CUDA A w tej generacji.
Reprezentanci: R0_seed44, R1_seed44, R3_seed44. R3 wybrany po Wiki dev,
ale spadek VN wobec R1 wyniósł 2.0303 p.p., więc guardrail niezaliczony.
Nie uruchamiać dodatkowych eksperymentów ani zmieniać selekcji po wynikach OOD B.

Poniższy zapis opisuje dawny punkt przerwania i nie jest bieżącą listą zadań.

## Historyczny stan przy wyłączeniu komputera

Użytkownik polecił przerwać prace, aby wyłączyć komputer. Nie uruchamiać pracy
w tle. Ostatni proces `run.py` zakończył się kodem 1 przed pierwszym nowym
treningiem; kontekst `gpu_lock` zwolnił lock. Brak własnego aktywnego CUDA.

Przeczytać oryginalny PROMPT_A_NEW_DATA.md, COORDINATION.md i HANDOFF_FORMAT.md.
Nie uruchamiać B ani subagentów. Autoryzacja ograniczonego pilota pozostaje
warunkowa; nie rozszerzać zakresu i nie pomijać bramek.

Gotowe markery: COMMON_READY, A_INPUTS_FROZEN, A_UNIVERSE_FROZEN (generacja 1).
Sprawdzać SHA, nie nadpisywać markerów ani ich zależności. Ważne dla B:
`HANDOFF.md` wyjaśnia obowiązkową sumę protected_index oraz
normalized_text_guard_artifacts w markerze uniwersum.

Audyt dopuścił 19 888 nowych VN i 28 Wiki. Duże pliki są na C pod
`C:/Users/jsowa/benchmark-translations/data-ood-20261030/A/`.
Wszystkie historyczne preservation SHA i baseline seed44 sprawdzono.
Nowy guard przeszedł 5 testów aliasów/image-only/transytywnego pHash.
Sampler przeszedł pełne 5500 kroków dla seed42/43/44: R0/R1 historycznie
identyczne, FULL=702880, całość=1405780, Wiki3%=42173, Wiki10%=140578.
Ekstrakcja nowych cech zakończona, cache FP32 zapisane. --prepare-only zakończyło
się pomyślnie, R0 reload metryk/lossu przeszedł dla wszystkich trzech seedów.

**Bramka, która zatrzymała start treningów:**
`run.py`, test adapter_cache_parity przed train, assert image_cos>.99999 i
text_cos>.99999. Zakończenie: AssertionError: Encoder/cache parity failed;
comparison stopped. Konkretne wartości cos nie zostały wypisane ani zapisane.
Przy wznowieniu najpierw zapisać obie wartości, wskazać która modalność i
zdiagnozować różnicę względem zatwierdzonego cache. Nie osłabiać progu tylko po
to, by przepchnąć test. Nie trenować zmienionego baseline’u. Obecne środowisko
emitowało ostrzeżenie SentenceTransformer o is_causal=False / transformers4.57.6;
to trop, a nie dowód przyczyny. Nie aktualizować istniejących środowisk.

Interpreter: WSL `/home/jsowa/.venvs/pl-mmb-benchmark/bin/python`.
Repo `.venv/bin/python` nie zawiera torch. Sandbox exec nie uruchamiał procesu
(helper_unknown_error: setup refresh had errors), stosowano zatwierdzane
wywołania require_escalated. Nie zmieniano globalnego safe.directory Git.

Kod własny: experiments/data_study_20261030; helper:
experiments/data_ood_common_20261030. Nie zmieniano src ani historycznych bramek.
Release Wiki jest wersjonowany SHA kodu runnera; po koniecznej poprawce runnera
ponownie sprawdzić release, zachowując poprzednie pliki.

Nie opublikowano A_FINALISTS_FROZEN ani A_GPU_DONE (wymagają ukończenia badania).
A_STATUS wskazuje przerwanie na polecenie użytkownika, nie gotowość OOD.
Nie odczytywano nowych wyników B. Nie uruchomiono R1/R2/R3 ani legacy/XM testów.

Pozostało: rozwiązać bramkę zgodności cache, wykonać wyłącznie prerejestrowane
runy i selekcję, posthoc galerie, CI oraz raport/handoff. final_report.py i
confidence_intervals.py są przygotowane, ale jeszcze nieuruchomione i wymagają
przeglądu przed użyciem. Nie przedstawiać ich planowanych wyników jako wykonanych.
