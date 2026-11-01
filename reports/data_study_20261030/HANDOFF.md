# Sesja A — interpretacja zamrożonego uniwersum

Konsument `A_UNIVERSE_FROZEN.json` sprawdza wszystkie SHA, generację i SHA kontraktu.
Uniwersum obejmuje wszystkie dopuszczone stare i nowe obrazy/captiony R0/R1/R2/R3.
Żadne kolejne eksperymenty ani nowe captiony nie są dopuszczone w tej generacji.

Ochrona jest sumą `protected_index` oraz pola
`normalized_text_guard_artifacts` markera. To drugie pole wiąże SHA pełnych,
historycznych blacklist (legacy i XM wyłącznie do audytu) oraz guarda Wiki.
Trzeba sprawdzić te SHA i uwzględnić także ich `ids`, `texts`, `parent_ids`,
`source_sha1_hex` i `images`; zawierają również zapytania bez osobnego wiersza
obrazu w indeksie. Nie wolno ograniczać wykluczeń do wybranych checkpointów lub
samych obrazów train. `component` oznacza transytywny komponent audytu A,
`verified_location` opisuje oryginalny plik albo wiersz oryginalnego parquet.
Parquet nie jest SQLite. Obrazy nie zostały skopiowane do drugiego archiwum.

Historyczne rekordy train pozostają zachowane nawet tam, gdzie nowy szerszy
audyt łączy ich komponent z chronionym zbiorem. Takie połączenie wyklucza nowe
kopie i kandydatów B; nie przepisuje historii treningu R0.

`COMMON_READY.json` jest gotowy. Samo `A_UNIVERSE_FROZEN.json` pozwala B kończyć
audyt CPU, lecz GPU B wymaga także jego `B_EVAL_FROZEN.json` oraz obu markerów
`A_FINALISTS_FROZEN.json` i `A_GPU_DONE.json`. STATUS nie zastępuje żadnej bramki.
Nie odczytywano nowych wyników B.
