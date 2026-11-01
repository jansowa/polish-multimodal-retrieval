# S02-REPO-HYGIENE — uporządkowanie zastanych zmian

## Zakres i decyzje
Użytkownik zlecił osobny etap porządkowania Git przed ponowieniem F01.
Zatwierdził oddzielne commity tematyczne, ignorowanie danych roboczych i kopii,
wyłączenie stanów absence z Git bez usuwania ich z dysku, wersjonowanie manifestów
benchmarków/holdoutów oraz zachowanie źródeł przeglądarki przy ignorowaniu wyłącznie
jej wygenerowanego, niezależnego checkoutu Sites. F01 nie rozpoczęto.

## Wynik
Kod, testy, konfiguracje, dokumentacja, rejestry i dowody zapisano tematycznie.
Faktyczna lista commitów przed domknięciem: `commits-before-handoff.json`;
mapowanie plików: `commits.json`, `commit-plan.json`. Tematy obejmują zapis stanu,
pozyskiwanie danych, silnik i badania treningowe, pipeline’y, retrieval/leaderboard,
historyczne eksperymenty, raporty i manifesty, specyfikację oraz integrację pakietów.
Raport i handoff mają osobny commit domknięcia.

`.gitignore` obejmuje bazy SQLite, Parquet, surowe eksporty pipeline’ów, kandydatów
treningowych, kopie logów, eksport całego projektu, cache per rekord, stany wykonania
i lokalne sterowanie. Nie ignorujemy całego `reports/`, manifestów benchmarków ani
rejestrów. Reguły runów pozwalają wersjonować `training/runs/f_series/<run_id>/JOB.json`.
`.gitattributes` chroni bajty plików wiązanych SHA przed konwersją LF/CRLF.

Przeglądarka QA pozostaje wersjonowana: `data_collection/browse.py`,
`src/pl_mmb_collection/browser/`, testy i `data_collection/BROWSER.md`.
`data_collection/browser_site/` pozostał na dysku z własną historią i konfiguracją
Sites. Nie eksportowano ani nie publikowano strony; `nested-site-evidence.json`.

## Weryfikacja
WSL, bez GPU i z modelami offline: główny zestaw CPU — 260 passed, 1 skipped;
guard — 5 passed; kolejka Wiki — 4 passed; OOD CPU — 13 passed; łącznie 282 passed,
1 skipped. Źródło liczb: `verified-test-summary.json` i logi `validation/wsl-*.txt`
z SHA. Wstępne próby Windows miały braki zależności/platformy; właściwe testy WSL
zakończyły się powodzeniem. Pytest i openai doinstalowano do osobnego katalogu
tymczasowego bez zmiany istniejących środowisk (`test-environment.json`).

SHA chronionych plików porównano przed etapem, w checkoutcie i blobach Git,
w tym holdouts.py, obu katalogów, rejestru Wiki i źródeł generacji.
Dowody: `exact-byte-checks.json`, `preservation-verification.json`, `final-preflight.json`.
Stany absence zachowały SHA po wyłączeniu z Git (`runtime-preservation.json`).
Stanów innych sesji nie zmieniano; żywe aktualizacje pominięto w commitach
(`live-state-exclusions.json`). Kontrola AST: bez błędów (`inventory.json`).
Tokenopodobny ciąg w archiwalnym HTML był nazwą klasy CSS.
Kontrole whitespace dopuszczają CR na końcu linii, bo zachowujemy oryginalne CRLF.
Rzeczywiste zastane ostrzeżenia zachowano jawnie (`whitespace-notes.json`,
`validation/*-whitespace.txt`); nie normalizowano oryginałów związanych SHA.
SHA dowodów, logów i raportu: `SHA256SUMS.json`.

## Ograniczenia i następny krok
Nie trenowano, nie oceniano nowych wyników GUARD/TEST, nie dopuszczano nowych danych.
Nie zmieniano zamrożonych generacji i nie zatrzymywano cudzych procesów.
Duże dane pozostały lokalne; Git nie jest ich kopią zapasową.
Nie tworzono gałęzi, worktree, PR ani push.
W trakcie audytu niezależna kolejka Wiki zgłosiła zatrzymanie download_compact.py.
Nie podejmowano napraw; komunikat i SHA: `observed-wiki-failure.json`, problem w BACKLOG.
H-OPS-S01 rozwiązano przez uporządkowanie checkoutu; nie zniesiono warunku czystego
startu. Kolejna sesja: wyłącznie F01 kroki 1–2, według NEXT.md.
