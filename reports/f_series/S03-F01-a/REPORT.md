# S03-F01-a — protokół i podział VizWiz

## Pytanie i zmiana
Zamrożono pomiar serii F przed jej wynikami: protokół MD/JSON, nowy katalog
i połówki istniejącego VizWiz. Zrealizowano wyłącznie zakres wejściowego NEXT
i PLAN; pozostała infrastruktura F01 pozostaje do wykonania.

## Dane i źródła
Źródło wszystkich poniższych liczebności: [summary.json](summary.json), SHA256
`091fff81f885ae4333bbe44a4c26a9d27f700c85945f4235f7bd0bb6abad3976`.
SHA manifestu źródłowego i protokołu są w [assignments.json](../../../protocols/vizwiz-v3-f/assignments.json);
SHA przydziałów, katalogu i nowych artefaktów — w summary.

| Galeria | Obrazy | Zapytania | Krawędzie qrels | Komponenty |
|---|---:|---:|---:|---:|
| VizWiz źródło | 1000 | 4312 | 4329 | — |
| A / SEL-OOD | 500 | 2160 | 2172 | 488 |
| B / GUARD | 500 | 2152 | 2157 | 494 |

Graf uwzględnia wszystkie metadane rejestru, oryginalne komponenty protokołu,
qrels wielopozytywne, teksty EN/PL, tożsamości obrazów i pHash. Posortowane
stabilne identyfikatory komponentów tasuje się seedem zapisanym w assignments,
a całe komponenty przydziela do mniej licznej galerii (remis: A). Nie usunięto
obrazów, tekstów ani qrels; nie zmieniono ich treści ani oryginalnych indeksów
zapisanych jako mapowanie. Każda połówka jest osobną pełną galerią. Historyczny
pełny VizWiz pozostaje pozycją REF i nie zastępuje wyników połówek.

## Protokół
[MD](../../../protocols/protocol-v3-f.md) zachowuje cały normatywny tekst 02;
[JSON](../../../protocols/protocol-v3-f.json) zawiera go w sekcjach oraz zapisuje
maszynowo role, SEL, remisy, kryterium, guardraile, limity odczytów, statystykę,
tożsamość PolDense, koszt i raportowanie. SHA obu plików są w
[detached binding](../../../protocols/protocol-v3-f.SHA256SUMS.json) i summary;
JSON wiąże również SHA MD, źródła specyfikacji, katalogu i rejestru.
Commit zawierający ten raport ma prefiks `[S03-F01-a]`; nie wykonano treningu
ani nowych wyników serii F przed zamrożeniem. Przyszłe configi muszą przypinać
oba SHA protokołu, katalog i właściwy rejestr zgodnie z 07.
W `.gitattributes` wyłączono konwersję końców linii dla protokołów i kodu
związanego nowymi SHA. Kontrola whitespace respektuje istniejące CRLF
(`git -c core.whitespace=cr-at-eol diff --check`), bez normalizacji plików.

## Weryfikacja
Źródło testów i liczników: [validation-evidence.json](validation-evidence.json),
SHA256 `067d542470b5794a5c77ed2005c096c5a5a16a48d45687632165c666ff968121`;
zawiera polecenia WSL, SHA kodu, logów i podsumowań oraz bazowy commit i gałąź.

- Standardowa kontrola `current_holdouts`: `registry_verified` dla niezmienionego v4.
- Powtórne generowanie `--check`: bajtowa zgodność wszystkich artefaktów.
- Pula obrazów/zapytań i multizbiór qrels zachowane; przecięcia komponentów,
  obrazów i zapytań między A/B puste.
- Chronione SHA zgodne z Git HEAD i po wykonaniu; oba istniejące katalogi,
  oba moduły holdoutów i dotychczasowe rejestry zachowane.
- Testy standardowym `unittest`: 8 passed; uszkodzenia qrels, komponentów,
  tekstu i kompletności galerii są odrzucane. Próba użycia pytest wykazała jego
  brak w wymaganym interpreterze; nie instalowano zależności.

## Ograniczenia i decyzja
Fakty: etap zakończony; F01 nadal `in_progress`. Brak ewaluacji modeli,
treningu, GPU i nowych dopuszczeń danych. Nowe odczyty GUARD/TEST: 0/0
(źródło: summary i evidence powyżej). Brak jobów tej sesji w tle.

Podział nie dowodzi braku ekspozycji encoderów w pretrainingu. Domyślny WEIGHTS
nie zastępuje przyszłej decyzji H1 ani audytu/H2 dla nowych danych. Rejestr v4
nadal chroni cały VizWiz; nowy katalog jest przypiętą projekcją tej samej puli,
nie aktywacją nowego rejestru. Nowe zbiory i zmiany SEL wymagają kolejnych
pakietów oraz zapisów decyzji. Sonda PolDense, licznik eksportu, log odczytów,
SEL modeli i panel kompozycyjny nie były wykonywane w tym etapie.

Następny etap: S04-F01-b zgodnie z NEXT — licznik parametrów i dowód tożsamości.
