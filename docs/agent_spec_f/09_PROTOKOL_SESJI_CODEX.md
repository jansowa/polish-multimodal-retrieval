# 09 — Protokół sesji Codex i prompt stały

Cel: każda sesja ma mały, zamknięty zakres, czyta tylko to, czego potrzebuje, i zostawia po sobie stan, od którego następna sesja startuje bez odtwarzania historii. Pamięć projektu jest w plikach (`STATUS.md`, handoffy, `NEXT.md`), nie w czacie. Właściciel nie zarządza gałęziami ani przeglądami kodu: agent wykonuje etap w całości i kończy go commitem na bieżącej gałęzi.

## 1. Zasady sesji (skrót tej sekcji jest w `AGENTS.md`, §„Protokół sesji”)

1. **Jedna sesja = jeden etap** (pakiet, krok pakietu albo ramię eksperymentu). Zakres etapu wynika z `09 §4` (plan sesji) albo z `reports/f_series/NEXT.md` zapisanego przez poprzednią sesję. Nic spoza zakresu — pomysły i zauważone problemy trafiają do `reports/f_series/BACKLOG.md`.
2. **Etap wykonuje się w całości**, do kryterium „gotowe” z `04`/`05`. Nie zaczynaj następnego etapu w tej samej sesji, nawet jeśli bieżący skończył się szybko.
3. **Czytaj minimalnie:** na starcie `reports/f_series/STATUS.md`, `reports/f_series/NEXT.md`, najnowszy plik w `reports/f_series/handoffs/`, `docs/agent_spec_f/REPO_MAP.md`, sekcję pakietu w `04_PAKIETY_PRACY.md`, sekcję eksperymentu w `05_EKSPERYMENTY.md` (jeśli dotyczy) oraz pliki z tabeli „kolejność czytania” w `00` dla danego typu akcji. Nie czytaj całej specyfikacji ani dużych plików (`results.json`, manifesty, logi) w całości — używaj skryptów, `head`, `jq`; podsumowania zapisuj do plików.
4. **Bez gałęzi, worktree i PR.** Praca na bieżącej gałęzi głównego checkoutu; `git status` czysty na starcie; na koniec sesji jeden commit (lub kilka) z prefiksem `[<etap>]`. Zamrożonych plików (`src/pl_mmb_training/holdouts.py`, stary `retrieval_benchmark/catalog.json`, zamrożone generacje) nie dotykaj.
5. **Zadania GPU są uruchamiane, nie pilnowane.** Job musi być wznawialny i mieć ticket `training/runs/f_series/<run_id>/JOB.json` (polecenie startu i wznowienia, pid, start, szacowany koniec, ścieżki wyników). Po starcie i smoke teście sesja czeka najwyżej 15 minut, potem kończy się handoffem; odbiór wyników to osobny etap („odbiór”), który sam staje w `NEXT.md`.
6. **Bramka ludzka = koniec sesji.** Eskalacja wg `07 §8` zapisana w handoffie i w `NEXT.md` (etap „oczekuje na H…”). Nie obchodź bramki, nie zgaduj decyzji.
7. **Koniec sesji (obowiązkowo, w tej kolejności):** `STATUS.md` → handoff `reports/f_series/handoffs/<SESSION_ID>.md` (≤ 1 strona, §3) → `NEXT.md` (następny etap, §3) → wiersz w `reports/f_series/SESSIONS.md` → commit → odpowiedź ≤ 10 linii.
8. **Logi w czacie ≤ 50 linii;** pełne logi do plików. **Każda liczba w raporcie ma źródło** (ścieżka, SHA); nie przepisuj liczb z pamięci rozmowy.

## 2. Identyfikatory i pliki stanu

- `SESSION_ID = S<nn>-<pakiet>[-<krok>]`, np. `S01-F01-a`, `S14-F07-b`. Numer `nn` rośnie o 1 z każdą sesją (także „odbiór” i „po decyzji”).
- `reports/f_series/STATUS.md` — tablica pakietów (07 §9) + sekcja „Rozbieżności do wyjaśnienia” + liczniki GUARD/TEST; jedno źródło prawdy o stanie.
- `reports/f_series/NEXT.md` — następny etap do wykonania (format w §3). Jeśli prompt nie podaje etapu, agent wykonuje etap z tego pliku.
- `reports/f_series/handoffs/<SESSION_ID>.md` — notatka przekazania.
- `reports/f_series/SESSIONS.md` — rejestr: `SESSION_ID | data | etap | wynik (done / partial / blocked) | handoff | commit`.
- `reports/f_series/BACKLOG.md` — jedna linia na pomysł/problem spoza zakresu.
- `training/runs/f_series/<run_id>/JOB.json` — ticket joba GPU.

## 3. Szablony handoffu i `NEXT.md`

```markdown
# Handoff <SESSION_ID> — <etap> — <data>
## Zrobione (fakty, ze ścieżkami)
## Jak zweryfikowano (testy, polecenia, SHA)
## Działa w tle (run_id, JOB.json, szacowany koniec, jak sprawdzić)
## Pytania do człowieka / bramki
## Zmienione pliki
```

```markdown
# NEXT
SESSION_ID: S02-F01-b
Etap: F01 kroki 4 i 6 (licznik parametrów eksportu, test identyczności PolDense)
Zakres: … (1–3 zdania, odwołania do 04/05)
Gotowe, gdy: …
Do przeczytania: 04 §F01, 02 §5, 07 §3–4
Czeka na człowieka: nie | tak — H…
```

## 4. Plan sesji (pierwsze ~15; każda ≤ pół dnia pracy agenta; później etapy wyznacza `NEXT.md`)

| # | SESSION_ID | Zakres | GPU | Kończy się |
|---|---|---|---|---|
| 0 | S00-BOOT | weryfikacja statusu z `00` względem repo; szkielet `reports/f_series/`, `protocols/`, `licensing/`; `STATUS.md`, `SESSIONS.md`, `BACKLOG.md`, `REPO_MAP.md`; sekcja „Protokół sesji” w `AGENTS.md`; lista rozbieżności | nie | handoff + `NEXT.md` |
| 1 | S01-F01-a | F01 kroki 1–2: protokół v3‑F w `protocols/` z SHA; podział VizWiz A/B; `catalog-f-v3.json`; kontrola spójności rejestru | nie | commit protokołu |
| 2 | S02-F01-b | F01 kroki 4 i 6: licznik parametrów (test 70 667 520), test identyczności PolDense (sonda, referencja), testy jednostkowe | krótko | testy zielone |
| 3 | S03-F01-c | F01 kroki 3 i 5: log odczytów + wrapper (wpis GUARD #0), kalkulator SEL, SEL dla R1 42/43/44 i EG2 | nie | tabela SEL F‑base |
| 4 | S04-F01-d | F01 kroki 7–8: panel kompozycyjny dla R1 i EG2; latencja wież kandydujących; raport F01 | krótko | `STATUS: F01 done` |
| 5 | S05-F02 | rejestr licencyjny v1, dowody, dokumentacja generatora, pokrycie poziomów, wniosek H1 | nie | eskalacja H1 |
| 6 | S06-F03 | FE0: sufity, hubność, prefiks dla PolDense 17/32/68M | krótko | raport FE0 |
| 7 | S07-F05-a | wybór Wiki‑F dev i kandydatów holdoutu; start pobierania oryginałów (ticket) | nie | handoff z ticketem |
| 8 | S08-F05-b | odbiór pobierania; fingerprinty, komponenty, `registry-v5.json`, aktywacja dla nowych runnerów | nie | `STATUS: F05 done` |
| 9 | S09-F04-a | sanity adapterów wież; start cache cech (ticket) | tak | handoff z ticketem |
| 10 | S10-F04-b | start cache nauczyciela EG2 (manifest, U, `eg2_eval/`) (ticket) | tak | handoff z ticketem |
| 11 | S11-F04-c | odbiór obu cache, manifesty SHA, sygnały audytu, EG2 zero‑shot na SEL | krótko | `STATUS: F04 done` |
| 12 | S12-F06 | manifest `f-data-v1`, guard v5, sampler, wniosek H2 | nie | eskalacja H2 |
| 13 | S13-F07-a | FE1 screening seed 42: start ramion A0–A4 (ticket) | tak | handoff |
| 14 | S14-F07-b | odbiór A0–A4; start A5–A8 (ticket) | tak | handoff |
| 15 | S15-F07-c | odbiór A5–A8; start potwierdzeń 43/44 (ticket); raport FE1 i rekord DF1 w kolejnej sesji | tak | `STATUS: F07 done` po raporcie |

## 5. Prompt stały (jedno pole do podmiany: identyfikator etapu)

```text
Repozytorium jansowa/polish-multimodal-retrieval. Obowiązuje docs/agent_spec_f/ (00–09), AGENTS.md i training/DATA_SPLITS.md; przy konflikcie wygrywa reguła bardziej restrykcyjna.
Wykonaj w całości etap: S01-F01-a
Jeśli pole etapu jest puste albo nie ma go w 09 §4, wykonaj etap z reports/f_series/NEXT.md. Zakres etapu, kryterium „gotowe” i pliki do przeczytania ustal sam z 09 §4, NEXT.md oraz sekcji 04/05 tego etapu; czytaj tylko to (plus STATUS.md, najnowszy handoff, REPO_MAP.md).
Pracuj na bieżącej gałęzi głównego checkoutu — nie twórz gałęzi, worktree ani PR. Zrób etap do końca; wszystko spoza zakresu zapisz w BACKLOG.md; następnego etapu nie zaczynaj. Bramka ludzka = eskalacja wg 07 §8 i koniec sesji. Joby GPU: wznawialne, z ticketem JOB.json, bez czekania dłużej niż 15 minut.
Na koniec obowiązkowo: STATUS.md, handoff w reports/f_series/handoffs/, NEXT.md z następnym etapem, wiersz w SESSIONS.md, commit na bieżącej gałęzi. Odpowiedź ≤ 10 linii: zrobione / jak zweryfikowane / co działa w tle / następny etap.
```

Po decyzji człowieka ten sam prompt plus jedna linia na początku: `Decyzja Hn z dnia …: …` (agent wpisuje ją do rejestru/rekordu decyzji i odblokowuje etap, który na nią czekał).

## 6. Sygnały, że sesja jest za duża (przerwij etap, zrób handoff i `NEXT.md` z resztą)

- czytasz trzeci duży plik „żeby zrozumieć kontekst”;
- powstają zmiany w więcej niż jednym pakiecie;
- czekasz na job dłużej niż 15 minut;
- potrzebujesz decyzji człowieka, a kontynuujesz „na założeniach”;
- odpowiedź końcowa nie mieści się w 10 liniach.
