# Stan i kontynuacja tłumaczeń

Aktualizacja: 2026-10-18. Bieżące liczniki są w plikach JSON podanych niżej.

Pierwszy przebieg zakończył się 18 października o 12:39, po około 18 godzinach.
Rezerwowe przejście nie było jeszcze dopięte; pilotaże kontynuacji uruchomiono
tego dnia około 19:23. W tym pierwszym przebiegu nie udało się zapewnić
nieprzerwanej pracy przez całą dobę.

## Gotowe

| Wariant | Ukończone | Błędy końcowe |
|---|---:|---:|
| MSCOCO_t2i, MMEB, wcześniejszy pełny v2 | 1000/1000 | 0 |
| CIRR, MMEB v3b | 1000/1000 | 0 |
| FashionIQ, MMEB v3b | 1000/1000 | 0 |
| RefCOCO, MMEB v3b | 1000/1000 | 0 |
| ARO Visual Relation v3b | 23937/23937 | 0 |
| ARO Visual Attribution v3b | 28748/28748 | 0 |

Ostatni rekord ARO Attribution (`AROVisualAttribution:19837`) wielokrotnie
kończył się pustą odpowiedzią modelu przy limicie wyjścia. Po zakończeniu
ostatniej próby uzupełniono go tłumaczeniem Codex, z jawnym polem
`model=codex-manual-fallback`; nie przypisano go do Qwen i nie nadpisano żadnego
udanego wyniku. Szczegóły: `reports/aro_manual_fallback.json`.

## Uruchomiona kontynuacja

1. SugarCrepe — wszystkie 7511 par.
2. M-BEIR MSCOCO — pełna pula 24809 opisów wariantu testowego.
3. M-BEIR VisualNews — pełna pula 537568 opisów, z osobnym poprawionym profilem
   tłumaczenia podpisów prasowych. Ten etap czeka na zakończenie dwóch poprzednich.

Razem 569888 kolejnych rekordów. Każdy wybrany wariant obejmuje całość swojego
przypiętego źródła, bez produkcyjnego `--limit`. To zakresy M-BEIR, nie wszystkie
oryginalne splity COCO lub VisualNews. Opisy korpusu pokrywają też wszystkie
24809/19995 zapytań tekstowych tych wariantów; mapowanie jest zapisane w
`metadata.t2i_query_ids`, więc nie trzeba tłumaczyć tego samego opisu ponownie
jako zapytania. Oryginalne JSONL z zapytaniami, pulami obrazów i etykietami
pozostają bez zmian w cache, z SHA256 w planie.

Pobieranie obrazów całego M-BEIR nie jest częścią tej kontynuacji: tłumaczymy
teksty, zachowując oryginalne odwołania do obrazów i relacje trafności.

Wyniki nowych etapów są na C:, aby nie zapełnić D:

```text
C:\Users\jsowa\benchmark-translations\qwen3.8-27b-absence-extension-v3b\
C:\Users\jsowa\benchmark-translations\qwen3.8-27b-visualnews-v3c\
```

Próbki są w osobnych katalogach z końcówką `-pilot`. Jeden przebieg bez
automatycznego review; temperatura 0, produkcyjna współbieżność 4. Zasoby nie są
sztucznie zajmowane powtarzaniem ukończonych przykładów. Praca nie ma limitu doby.

## Status i ponowne uruchamianie

Z katalogu projektu w WSL:

```bash
.venv/bin/python experiments/absence_suite.py status
.venv/bin/python experiments/absence_extension.py status
.venv/bin/python experiments/absence_news_refined.py status
```

Po restarcie komputera/WSL można wznowić obie dalsze kolejki:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -u experiments/absence_extension.py full --wait-for-pilots --detach
OPENBLAS_NUM_THREADS=1 .venv/bin/python -u experiments/absence_news_refined.py full --wait-for-pilots --detach
```

Checkpointy pomijają ukończone ID. Blokady chronią przed równoległym zapisem
tego samego zadania. Zamknięcie terminala lub czatu nie zatrzymuje odłączonych
procesów. Dwa uruchomione w Windows pomocnicze procesy blokują automatyczne
uśpienie podczas kontynuacji i podczas oczekiwania VisualNews; monitor może
się wyłączyć. Po restarcie Windows pomocniki też wymagają ponownego startu.

Pełne stany i brakujące ID:

- `reports/absence_suite_state.json`
- `reports/absence_extension_state.json`
- `reports/absence_news_state.json`

Logi mają nazwy `reports/absence-full-<zbiór>.log`.
Plany zawierają rewizje, SHA256 źródeł i hash polityki tłumaczenia. Zmiana promptu
wymaga osobnego przebiegu; istniejące wyniki pozostają w swoich katalogach.
Na C: kolejka zachowuje 8 GiB rezerwy. Po trwałym błędzie lub braku miejsca
raportuje niekompletność; nie udaje ukończonego zbioru.

## Jakość i jej granice

Przejrzano pilotaże po 40 przykładów, a poprawiony profil VisualNews sprawdzono
oddzielnie na tych samych 40 ID. Doprecyzowanie usuwa m.in. interpretację
„Henry Kissinger left chats…” jako „opuścił rozmowy” i dosłowne tłumaczenie nazwy
grupy „Idiots of Ants”. Wciąż występują niezgrabności językowe, trudne terminy
branżowe i błędy źródłowe; jest to materiał do wstępnego benchmarku.

Raporty oceny: `absence_pilot_assessment.json`,
`absence_extension_assessment.json`, `absence_news_assessment.json` w `reports/`.
Ocena nie jest automatycznym wskaźnikiem poprawności semantycznej.
