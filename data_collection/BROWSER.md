# Przeglądarka danych

Strona pokazuje obrazy i istniejące teksty PL/EN, oceny jakości, autorów,
licencje i pełne provenance. Kliknięcie obrazu otwiera pełny rekord i link
do źródła. Można pobrać pojedynczy rekord jako JSON.

## Lokalnie: aktualne wyniki scrapera

Z katalogu głównego projektu:

```bash
python data_collection/browse.py serve --port 8766
```

Adres: **http://127.0.0.1:8766**. Serwer używa standardowej biblioteki Pythona,
bez instalacji frontendu, GPU i zależności treningowych. Domyślnie udostępnia
`outputs/polish_culture_fast` (zbiór główny), wcześniejszy
`outputs/polish_culture` oraz ukończony `outputs/polish_culture_pilot`.
Bezpośredni link do bieżącego zbioru: http://127.0.0.1:8766/?dataset=fast.
Gdy istnieje `outputs/collection_browser/active_dataset.json`, zbiór główny
korzysta ze wskazanego tam katalogu. Kontynuacja od 27.10.2026 znajduje się
w `C:/Users/jsowa/datasets/polish_culture`; lokalna kopia wcześniejszego zakresu
pozostaje na D. Po zmianie wskaźnika zrestartuj serwer przeglądarki.

Serwer czyta SQLite w trybie tylko do odczytu i nie zmienia checkpointu,
manifestu, rekordów ani procesu scrapera. Nie trzeba czekać na eksport Parquet.
Statystyki odświeżają się co 30 sekund, ale galeria zmienia się po świadomym
odświeżeniu, aby nowe rekordy nie przesuwały oglądanych stron.

Inne katalogi można wskazać jawnie:

```bash
python data_collection/browse.py serve --dataset outputs/polish_culture --dataset outputs/inny_zbior
```

## Stronicowanie i filtry

- Numery stron z bieżącym oknem i skrótami do początku/końca.
- Poprzednia/następna strona oraz przejście do wpisanego numeru.
- Rozmiary strony: 12, 24, 48 lub 96 obrazów.
- Filtry: zbiór, język PL/EN/oba, jakość A–D, licencja.
- Wyszukiwanie w pełnym rekordzie, w tym polskie znaki, opisy, kategorie i QID.
- Sortowanie po dacie pozyskania, nazwie encji, rozdzielczości lub losowo.
- Losowanie jest stabilne między stronami; przycisk losowania zmienia seed.
- Stan filtrów i numer strony są zapisane w URL. Działają cofanie przeglądarki
  i ponowne otwarcie tego samego adresu.

Strona nie zawiera tylko losowej próbki. Stronicowanie i filtry obejmują
**cały wybrany zbiór**. Lokalna wersja pobiera z serwera tylko wybraną stronę.
Wersja migawkowa filtruje lekki indeks w przeglądarce, a pełne rekordy doczytuje
dopiero po otwarciu szczegółów. Pusta lista i nieprawidłowy numer strony mają
obsłużone stany; zbyt duży numer jest sprowadzany do ostatniej strony.

Przy filtrowaniu jakości w trybie kilku języków wystarczy wybrana ocena
w jednym z nich. Każdy caption pokazuje swoje źródło i ocenę. D oznacza
kontekst encji/kategorii, który może nie opisywać zawartości zdjęcia.

## Prywatna strona Sites

Opublikowana strona: https://polska-kultura-biblioteka-obrazow.navymite32.chatgpt.site

Strona jest prywatna i przeznaczona dla właściciela. Lokalny widok bieżących
danych pozostaje pod adresem http://127.0.0.1:8766.

Migawka jest eksportowana do `data_collection/browser_site/dist`:

```bash
python data_collection/browse.py export --site-dir data_collection/browser_site
```

Po synchronizacji i publikacji przez Sites strona działa bez lokalnego
serwera i zachowuje prywatny dostęp właściciela. Hosting nie ma dostępu
do lokalnego SQLite. Nowe wyniki wymagają nowego eksportu i publikacji;
strona jawnie pokazuje datę migawki. Obrazy w obu trybach wymagają internetu.

Nie publikujemy bazy SQLite, logów, manifestów scrapera ani surowego cache API.
Eksport zawiera wybrane rekordy datasetu, ich indeks i pliki strony.
Adres i tożsamość hostingu są zapisane w `browser_site/.openai/hosting.json`.

## Pliki i sprawdzenia

Kod: `src/pl_mmb_collection/browser/{data,server,snapshot}.py` i `browser/web/`.
Uruchomiony lokalny serwer zapisuje logi w `outputs/collection_browser/`.
Testy API, filtrów, stronicowania, odczytu i eksportu są w
`tests/data_collection/test_browser.py`. Pełny test interfejsu desktop/mobile
znajduje się w `tests/data_collection/browser_ui_check.mjs`; korzysta z lokalnego
Chrome z DevTools na porcie 9335 i nie wymaga dodatkowych pakietów Node.
