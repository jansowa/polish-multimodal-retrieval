---
title: Polski benchmark tekst obraz
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Polskie modele tekst–obraz, wersjonowane galerie retrieval
---

# Polski benchmark tekst ↔ obraz

Samodzielny leaderboard CPU. Źródłem danych jest wyłącznie `results.json` obok
`app.py`. Nie wymaga modeli, obrazów, tokena ani importów z projektu nadrzędnego.

Uruchomienie: `pip install -r requirements.txt`, następnie `python app.py`.
Cały ten katalog można wysłać jako zawartość repozytorium Hugging Face Space.

Wyniki to frakcje [0,1]; interfejs pokazuje procenty. Jedna tabela prezentuje średni
wynik i osobne zadania. Wynik zadania = średnia T2I/I2T, średnia ogólna wymaga
kompletu wybranych zadań. Pełne XTD10 domyślnie nie dubluje oczyszczonego wariantu
w średniej. W konfiguracji można włączyć kolumny T2I/I2T oraz organiczne,
syntetyczne i MT (oddzielne grupy), a także wybrać zadania do średniej.
Zmiana konfiguracji nie uruchamia nowych obliczeń. Braków nie zastępujemy zerami.
Kolory czerwony–zielony porównują wyniki w obrębie kolumny; parametrów nie kolorujemy.
Nie wyświetlamy kolumn czasu, precyzji, rewizji ani ekspozycji.
Nie mieszamy wyników różnych profili precyzji w jednym rankingu.

`results.json` zawiera wyłącznie zagregowane pomiary i ich identyfikatory/SHA.
Nie zawiera obrazów, podpisów, lokalnych ścieżek ani wag modeli.
Galerie Flickr/VizWiz są lokalnymi audytowanymi podzbiorami; nie deklarujemy
reprodukcji niezmienionych oficjalnych benchmarków. XM3600 PL jest zwykłym zadaniem
leaderboardu: 3597 obrazów po wcześniejszych wykluczeniach. DOCCI PL pozostaje
w przygotowaniu i nie ma jeszcze wyników retrieval.
