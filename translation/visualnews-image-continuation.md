# VisualNews: kontynuacja z grafiką

Od przełączenia 20.10.2026 nowe tłumaczenia wymagają jednego zweryfikowanego
zdjęcia. Reasoning jest jawnie włączony przez
`chat_template_kwargs.enable_thinking=true`. Współbieżność: 16. Timeout klienta:
connect 75 s, read 3900 s, write 600 s, pool 75 s.

Proces działa w środowisku Windows, niezależnie od restartów WSL:

```powershell
.\.venv-windows\Scripts\python.exe -B -u experiments\visualnews_images_production.py
```

Stare wejścia produkcyjne odmawiają uruchomienia trybu tekstowego po utworzeniu
`reports/visualnews_image_plan.json`. Blokady chronią przed drugim procesem Windows.

Przed każdym przebiegiem czytane są checkpointy VisualNews z `outputs/` i
`C:/Users/jsowa/benchmark-translations/`, włącznie z pilotażami. Ukończone ID
oraz dokładnie zgodne opisy źródłowe są pomijane niezależnie od metody.
Wcześniejsze wyniki pozostają w swoich bazach.

Nowe wyniki trafiają do:

```text
C:/Users/jsowa/benchmark-translations/qwen3.8-27b-visualnews-v3c/mbeir_visualnews/with_image/checkpoint.sqlite3
```

Kolumna `with_image` ma wartość 1; w eksporcie JSONL jest to `true`.
`metadata.translation_inputs` zawiera m.in. `reasoning_enabled`, liczbę tokenów
reasoningu ostatniej odpowiedzi, źródłową ścieżkę i ID zdjęcia, jego SHA256 oraz
przypiętą rewizję repozytorium z obrazami. SQLite jest źródłem aktualnego postępu;
JSONL jest eksportowany po zakończeniu/przerwaniu przebiegu.

Aktualny stan: `reports/visualnews_production_state.json`.
Punkt odniesienia ostatniego wznowienia: `reports/visualnews_resume_baseline.json`.
Nieusuwalna przez zwykłe wznowienie historia: `reports/visualnews_resume_history.jsonl`
oraz datowane pliki `reports/visualnews_resume_*.json`.
Plan i lista brakujących zdjęć: `reports/visualnews_image_plan.json`.
Log: `reports/visualnews_images_production.log`.

Pula M-BEIR zawiera obrazy dla 283012 z 537568 opisów. Pozostałe 254556 opisów
nie ma obrazu w tej puli; nie są wysyłane tekstowo. Dopasowanie odbywa się po
oryginalnym podpisie, normalizacji autora UniIR i ścieżce zdjęcia. Zweryfikowano
zgodność obrazów z oficjalną pulą oraz relacje z testowych qrels.
Nie wylicza się ID zdjęcia przez dodawanie/odejmowanie od ID podpisu.

Cache i audyt obrazów:
`C:/Users/jsowa/benchmark-translations/visualnews-image-cache/manifest.json`.
Samo dodanie obrazów do tej puli wymaga ponownej weryfikacji mapowania i indeksu.

Aby zatrzymać proces bez wysyłania kolejnych rekordów, utwórz plik
`reports/visualnews_image_stop.request`. Proces dokończy requesty będące w toku.
Przed następnym uruchomieniem trzeba świadomie usunąć to żądanie zatrzymania.
