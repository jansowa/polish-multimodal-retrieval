"""Read-only, CPU-only Hugging Face Space. No model or dataset downloads."""
import tempfile
import os
from pathlib import Path
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
import gradio as gr
from view import METRICS, columns, csv_bytes, default_columns, default_tasks, load, organizations, overview, pareto_html, styled, tasks


def column_widths(view):
    return [420 if index == 0 else 220 if header == 'Licencja' else 110
            for index, header in enumerate(view['headers'])]


def build_app():
    document = load()
    profiles = sorted({r['profile'] for r in document['results']})
    initial_organizations = organizations(document)

    def calculate(metric, mode, profile, limit, seeds, controls, visible, average, minimum, selected_orgs):
        return overview(load(), metric, mode, profile, limit, seeds, controls, visible, average,
                        min_parameters=minimum, selected_organizations=selected_orgs)

    def update(*args):
        metric, mode, profile, limit, seeds, controls, visible, average, minimum, selected_orgs, log_scale, known_orgs = args
        snapshot = load()
        current_orgs = organizations(snapshot)
        # Newly arriving models can introduce an organization while the UI is open.
        # Select new organizations only if the user previously selected everything.
        selected_orgs = [o for o in selected_orgs if o in current_orgs]
        if set(selected_orgs) == set(known_orgs):
            selected_orgs = current_orgs
        view = overview(snapshot, metric, mode, profile, limit, seeds, controls, visible, average,
                        min_parameters=minimum, selected_organizations=selected_orgs)
        chart = pareto_html(snapshot, log_scale=log_scale, filtered_view=view)
        return (gr.update(value=styled(view), datatype=['html'] + ['str'] * (len(view['headers']) - 1), column_widths=column_widths(view)),
                f"Pełny średni wynik: **{view['complete']}/{view['total']} modeli** · {len(view['average_tasks'])} zadań. „—” oznacza brak kompletu wyników.",
                chart, gr.update(choices=current_orgs, value=selected_orgs), current_orgs)

    def download(*args):
        view = calculate(*args)
        with tempfile.NamedTemporaryFile(suffix='.csv', prefix='leaderboard-', delete=False) as f:
            f.write(csv_bytes(view))
        return f.name

    with gr.Blocks(title='Polski benchmark tekst ↔ obraz', fill_width=True, css='''
        #leaderboard td:first-child a { overflow-wrap: anywhere; word-break: normal; }
        #leaderboard thead th * { white-space: normal !important; text-overflow: clip !important; }
    ''') as demo:
        gr.Markdown('# Polski benchmark tekst ↔ obraz\nJedna tabela modeli: średni wynik i osobne wyniki każdego zadania. Wyższy wynik = lepiej.')
        metric = gr.Dropdown(METRICS, value='recall@1', label='Metryka')
        with gr.Accordion('Konfiguracja tabeli', open=False):
            visible = gr.CheckboxGroup(choices=[(label, key) for key, label in columns(document).items()],
                                       value=default_columns(document), label='Widoczne kolumny')
            average = gr.CheckboxGroup(choices=[(document['datasets'][key]['title'], key) for key in tasks(document)],
                                       value=default_tasks(document), label='Zadania uwzględniane w średnich')
            gr.Markdown('Domyślnie uwzględniamy wszystkie gotowe zadania; XTD10 pełne jest widoczne osobno, lecz nie dubluje oczyszczonego XTD10 w średniej. Ukrycie kolumny nie zmienia średniej.')
            with gr.Row():
                minimum = gr.Number(value=0, minimum=0, label='Minimum V+T [M]')
                limit = gr.Number(value=0, minimum=0, label='Maksimum V+T [M]; 0 = bez limitu')
            selected_orgs = gr.CheckboxGroup(choices=initial_organizations, value=initial_organizations,
                                            label='Organizacje')
            known_orgs = gr.State(value=initial_organizations)
            with gr.Row():
                seeds = gr.Checkbox(value=False, label='Pokaż wszystkie seedy')
                controls = gr.Checkbox(value=False, label='Pokaż kontrole EN')
            with gr.Row():
                mode = gr.Dropdown(['v2_no_instructions', 'author_recommended'], value='v2_no_instructions', label='Prompty')
                profile = gr.Dropdown(profiles, value='historical-native-v2', label='Profil pomiaru')
        initial = overview(document)
        gr.Markdown('## Jakość a rozmiar modelu\nLinie łączą modele ustanawiające najlepszy wynik przy rosnącym rozmiarze.')
        log_scale = gr.Checkbox(value=False, label='Logarytmiczna oś rozmiaru')
        chart = gr.HTML(value=pareto_html(document, filtered_view=initial))
        board = gr.Dataframe(value=styled(initial), datatype=['html'] + ['str'] * (len(initial['headers']) - 1),
                             interactive=False, wrap=True, elem_id='leaderboard',
                             column_widths=column_widths(initial), pinned_columns=1)
        coverage = gr.Markdown(f"Pełny średni wynik: **{initial['complete']}/{initial['total']} modeli** · {len(initial['average_tasks'])} zadań. „—” oznacza brak kompletu wyników.")
        table_inputs = [metric, mode, profile, limit, seeds, controls, visible, average, minimum, selected_orgs]
        inputs = table_inputs + [log_scale, known_orgs]
        outputs = [board, coverage, chart, selected_orgs, known_orgs]
        refresh = gr.Button('Odśwież wyniki')
        refresh.click(update, inputs, outputs, api_name=False)
        demo.load(update, inputs, outputs, api_name=False)
        for index, control in enumerate(inputs):
            if control is known_orgs:
                continue
            event = control.input if control is selected_orgs else control.change
            event(update, inputs, outputs, api_name='leaderboard' if index == 0 else False)
        with gr.Row():
            export = gr.Button('Eksportuj widok CSV')
            csv_file = gr.File(label='CSV')
            gr.File(value=str(Path(__file__).with_name('results.json')), label='Wszystkie wyniki JSON', interactive=False)
        export.click(download, table_inputs, csv_file, api_name='export_csv')
        with gr.Accordion('Jak czytać benchmark', open=False):
            gr.Markdown('V+T = parametry używanych enkoderów wizji i tekstu. Dodatkowy head jest osobno; audio i inne modalności nie są wliczane. '
                        'T2I: tekst → obraz; I2T: obraz → tekst. Galerie są pełne, z wieloma znanymi pozytywami. '
                        'Wynik zadania = średnia T2I i I2T. Średni wynik = nieważona średnia wybranych zadań; wymagamy kompletu pomiarów. '
                        'Dodatkowe kolumny T2I/I2T uśredniają odpowiedni kierunek tych samych zadań. '
                        'Organiczne = natywne opisy ludzkie i tłumaczenia ludzkie; syntetyczne = opisy generowane/recyklingowane; '
                        'tłumaczenia maszynowe stanowią odrębną grupę. VisualNews ma mieszany pipeline i trafia do syntetycznych, '
                        'Flickr/VizWiz do MT, Wiki/XTD10 do organicznych. Każda średnia grupy wymaga kompletu wybranych zadań tej grupy. '
                        'Kolory są względne w obrębie kolumny: czerwony = niższy, zielony = wyższy wynik, remis = neutralny. Parametrów nie kolorujemy. '
                        'Informacja o danych treningowych (dawniej „ekspozycja”) opisuje, czy model widział dany zbiór/domenę w treningu. '
                        'ID = dane/domena znane z treningu, OOD = inna domena; brak identycznych obrazów sam nie dowodzi OOD. '
                        'Ekspozycja w pretreningu jest nieustalona; szczegółowe oznaczenia są w JSON. '
                        'R3 ma charakter diagnostyczny po niezaliczeniu wcześniejszego warunku zachowania jakości. '
                        'Wiki/VN to wcześniej używane zbiory dev, więc łączna średnia jest opisem tego zestawu, nie niezależnym wynikiem zero-shot. '
                        'Pełne pochodzenie, parametry i warunkowe CI różnic zawiera JSON. '
                        'Licencje pochodzą z kart ocenianych rewizji HF. Dla lokalnych checkpointów podajemy osobno deklaracje encoderów V/T; nie zastępują one deklaracji licencji całego checkpointu. '
                        'XM3600 PL jest zwykłym zadaniem leaderboardu; lokalna wersja ma 3597 obrazów po wcześniejszych wykluczeniach. DOCCI pozostaje w przygotowaniu. Aplikacja nie oblicza embeddingów.')
    return demo


if __name__ == '__main__':
    build_app().launch(server_name=os.environ.get('GRADIO_SERVER_NAME', '0.0.0.0'),
                       server_port=int(os.environ.get('GRADIO_SERVER_PORT', '7860')), share=False)
