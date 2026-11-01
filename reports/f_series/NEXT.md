# NEXT
SESSION_ID: S06-LEADERBOARD-RECEIVE
Etap: Odbiór kolejki referencji zleconej przez właściciela
Zakres: Sprawdzić JOB/STATE, complete/failure/cleanup i PUBLISH_STATUS dla leaderboard-requested-20261101. Zweryfikować komplet komórek z rejestracji, SHA źródeł i leaderboard; naprawić błędy własnej kolejki bez ponownej oceny istniejących komórek. Potwierdzić push HEAD do origin/main. Nie uruchamiać F01-b w tej samej sesji.
Gotowe, gdy: zarejestrowane braki uzupełniono lub dokładnie udokumentowano blokery; prywatne wagi sprzątnięte; leaderboard zwalidowany, commit/push potwierdzone; raport i handoff zaktualizowane.
Do przeczytania: STATUS; handoff S05-LEADERBOARD-PARETO i S04-LEADERBOARD; reports/leaderboard_requested_20261101/REPORT.md, summary.json, registration.json (skryptem); retrieval_benchmark/REQUESTED_QUEUE.md; JOB/STATE/PUBLISH_STATUS i krótkie końcówki logów.
Czeka na człowieka: nie

Po odbiorze wrócić w odrębnej sesji do odłożonego F01-b: licznik parametrów eksportu i tożsamość PolDense według poprzedniego NEXT zachowanego w Git i handoffu S03-F01-a. Zamrożonych plików nie zmieniać.
