# Protokół v2: bez instrukcji i bez normalizacji przed sumowaniem

Identyfikator: `mmeb-v1-no-instructions-raw-sum-cosine-v2`.
Referencja: [MMEB v1, commit 28fdbdd](https://github.com/TIGER-AI-Lab/VLM2Vec/tree/28fdbdd68a66374ec1b53207ad7a24b9979f1ab6).

1. Zachowujemy te same przypięte źródła, tłumaczenia, 1000 przykładów i lokalne
   pule kandydatów. Instrukcję zadania usuwamy z polskiego tekstu przed tokenizacją.
   Dopasowanie odbywa się po pełnym znanym prefiksie, nie po dowolnym dwukropku.
   Dwukropki i cudzysłowy w samym opisie pozostają bez zmian. Nieznany prefiks
   zatrzymuje przebieg zamiast zgadywać granicę opisu.
2. Enkodery CLIP/SigLIP zwracają swoje natywne embeddingi. Nie normalizujemy
   osobno tekstu ani obrazu przed połączeniem. Zapytanie mieszane to `q = t + i`.
3. Wynik kandydata to `dot(candidate, q) / (norm(candidate) * norm(q))`.
   To dokładnie wariant `get_pred(normalization=True)` z oryginalnego kodu.
   **Uwaga o reprodukcji:** oryginalny evaluator udostępnia również
   `normalization=False` (surowy iloczyn skalarny), a argument domyślnie ma wartość
   False. Nie znaleziono opublikowanej komendy baseline'u, która rozstrzygałaby
   ustawienie użyte do tabeli w publikacji. Nasz wybór cosine jest jawny w ID
   protokołu oraz w `similarity` każdego wyniku; nie deklarujemy reprodukcji
   nieudokumentowanych ustawień tabeli angielskiej.
4. Remisy rozstrzyga kolejność kandydatów, jak w oryginalnym `np.argmax`.
   Oznaczony pozytyw na pozycji 0 wygrywa remis. NaN, nieskończoności i zerowe
   embeddingi są odrzucane. Precision@1 jest główną metryką; dostępne są też
   NDCG@10, Recall@5/10 i MRR.

Sumowanie surowych embeddingów i cosine po połączeniu sprawdzono testem
numerycznym względem funkcji referencyjnej. Przypadek testowy daje inną najlepszą
odpowiedź po błędnej normalizacji obu części przed sumowaniem.

## EmbeddingGemma-2

Model ładuje wyłącznie tekst i wizję: `config_kwargs={"audio_config": None}`.
Rzeczywista liczba parametrów wynosi **438760448**. Kod sprawdza limit 500M oraz
brak parametrów o nazwach zawierających `audio`. Checkpoint pozostaje w natywnym
cache HF, choć zawiera również niewykorzystywane wagi audio.

Zapytania obraz+tekst są kodowane natywnie, wspólnie, zgodnie ze zdolnościami
tego modelu. Nie zastępujemy ich sumą osobnych embeddingów. Nie dodajemy instrukcji
MMEB ani prefiksów `SearchQuery`/`Document`. Stosujemy pooling i wymiar 768 z
oficjalnego checkpointu, BF16 oraz domyślny budżet wizji 280 soft tokens.
Wyniki mają `fusion=native-joint` i zapis konfiguracji komponentów.

Biblioteki Gemma są izolowane w `~/.venvs/pl-mmb-modern` (Transformers 5.20.1,
Sentence Transformers 6.1.0, PyTorch 2.14.1 i torchvision 0.29.1 z CUDA 13.0).
Podstawowe zależności są współdzielone przez plik `.pth`, ale nowy stos CUDA jest
zainstalowany oddzielnie. Nie modyfikuje starszego środowiska ani tłumaczeń.
Instalacja: `bash benchmark/setup_modern_wsl.sh`.

## Archiwum i wznawianie

Poprzednie pliki `result.json` i `predictions.jsonl` pozostają niezmienione.
Nowy protokół daje nowe ID wyników. Leaderboard pokazuje wyłącznie bieżący
protokół, a poprzednia tabela jest pod `benchmark/results/index.protocol-v1.html`.
Manifest `preservation_before_protocol_v2.json` zawiera SHA256 wcześniejszych
70 plików wyników i predykcji.

Surowe embeddingi nowych przebiegów są cache'owane pod natywnym
`~/.cache/huggingface/pl_mmb_benchmark/embeddings`, co ułatwia wznowienie i zmianę
samego rankingu bez ponownego kodowania obrazów. Klucz obejmuje model, rewizję,
komponenty, precyzję, limit tekstu i uporządkowane wejścia.
Każdy model i dataset działa w osobnym procesie CUDA, a gotowe wyniki są pomijane
przed utworzeniem kontekstu GPU. Chroni to również przed kumulacją alokacji
sterownika WSL przy przełączaniu wielu modeli w jednym procesie.

Nowe aliasy: `siglip2-pl-32768`, `clip-b32-pl-32768`, `siglip2-naflex`,
`embeddinggemma-2`. NaFlex zachowuje proporcje i domyślny limit 256 patchy;
nie jest wymuszany resize do kwadratu 512×512.
