# Wstępne porównanie ukończonych referencji

Źródło wszystkich liczb: score-snapshot.json; zapisuje SHA256 wejściowego
retrieval_benchmark/space/results.json:
d6bf5a7341406baf20e3fddd5d66785bb2c4227299235f114ccf7e5be85d917e.
Odtworzenie: inspect_scores.py, bez inferencji. Domyślne R@1 leaderboardu:
średnia T2I/I2T, następnie sześciu zadań, bez dublowania pełnego XTD10.
Tryb v2_no_instructions, profil historical-native-v2, pełne pokrycie.

| Model | V+T [M] | Licencja w rejestrze | Średnie R@1 [%] |
|---|---:|---|---:|
| EmbeddingGemma-2 (istniejące odniesienie) | 438.76 | Apache-2.0 | 73.57 |
| M-CLIP XLM-R Large / ViT-B-16Plus | 677.76 | brak deklaracji | 59.69 |
| SigLIP i18n B16 256 | 370.63 | Apache-2.0 | 59.68 |
| Polski SigLIP2 base 32768 | 204.38 | Apache-2.0 | 46.24 |
| Polski SigLIP2 base 16384 | 191.80 | Apache-2.0 | 45.05 |
| Nomic vision v1.5 + text v1.5 | 229.68 | Apache-2.0 | 8.09 |
| BGE-VL-large | 427.62 | MIT | 7.43 |

Najciekawsze jest potwierdzenie SigLIP i18n: zbliżony punktowo do większego
M-CLIP, z deklaracją Apache-2.0. Nie wyprzedza większej EmbeddingGemma-2.
To uzupełnienie istniejących wyników o brakujący XM3600, nie całkowicie nowa
ocena wszystkich zadań. Polski SigLIP2 32768 ma umiarkowany wzrost względem
16384; nie jest przełomem. Nomic/BGE nie wyróżniają się na polskich zadaniach.
Nie badano istotności różnic. Modelom z nieukończonym kompletem nie przypisuje
się zbiorczej pozycji. To porównanie referencji, nie wybór modelu treningowego F.
Licencje są deklaracjami wydawców wag, nie audytem pochodzenia danych.
