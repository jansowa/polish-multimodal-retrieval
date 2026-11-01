"""Optional real HTTP smoke test, run with the standalone Gradio environment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'space'))
from app import build_app
from view import default_columns, default_tasks, load, organizations
from gradio_client import Client
import httpx


def main():
    demo = build_app()
    try:
        demo.launch(server_name='127.0.0.1', server_port=7864, prevent_thread_lock=True, quiet=True, share=False)
        with httpx.Client(trust_env=False) as http:
            assert http.get('http://127.0.0.1:7864/').status_code == 200
            assert http.get('http://127.0.0.1:7864/config').status_code == 200
        client = Client('http://127.0.0.1:7864', verbose=False)
        document = load()
        inputs = ['recall@1', 'v2_no_instructions', 'historical-native-v2', 100, False, False,
                  default_columns(document), default_tasks(document), 0, organizations(document), False, True]
        result = client.predict(*inputs, api_name='/leaderboard')
        # Gradio versions return either a table value or a component update.
        table = lambda response: response[0].get('value', response[0])
        rows = table(result)['data']
        assert len(rows) >= 5, rows
        assert table(result)['headers'][1] == 'Średni wynik [%]'
        assert all(row[2] <= 100 for row in rows)
        assert 'background-color' in str(table(result)['metadata'])
        assert '<svg' in result[2] and 'data-frontier="commercial_frontier"' in result[2]
        assert 'data-model-id="checkpoint:' in result[2]
        assert result[2].count('data-model-id=') == len(rows)
        assert result[2].count('class="model-label"') == len(rows)
        assert all(not any(word in header for word in ['Precyzja', 'Rewizja', 'GPU', 'ms/']) for header in table(result)['headers'])
        for row in table(result)['metadata']['styling']:
            assert all(not row[k] for k in (2, 3, 4)), row
        inputs[6] = ['average', 't2i', 'i2t', 'organic', 'synthetic', 'mt']
        changed = client.predict(*inputs, api_name='/leaderboard')
        assert 'T2I [%]' in table(changed)['headers']
        assert 'Tłumaczenia maszynowe [%]' in table(changed)['headers']
        inputs[7] = ['flickr30k_pl_local_test']
        single = client.predict(*inputs, api_name='/leaderboard')
        assert table(single)['data'] != table(changed)['data']
        assert single[2] != changed[2]
        inputs[10] = True
        logarithmic = client.predict(*inputs, api_name='/leaderboard')
        assert 'skala logarytmiczna' in logarithmic[2]
        inputs[11] = False
        hover_only = client.predict(*inputs, api_name='/leaderboard')
        assert 'class="model-label"' not in hover_only[2]
        assert '<title>' in hover_only[2]
        inputs[11] = True
        inputs[6] = default_columns(document)
        inputs[8] = 50
        inputs[9] = ['jansowa']
        own = client.predict(*inputs, api_name='/leaderboard')
        assert table(own)['data']
        assert all(50 <= row[2] <= 100 for row in table(own)['data'])
        assert own[2].count('data-model-id=') == len(table(own)['data'])
        assert own[2].count('data-model-id="checkpoint:') == len(table(own)['data'])
        inputs[9] = []
        empty = client.predict(*inputs, api_name='/leaderboard')
        assert table(empty)['data'] == []
        assert 'Brak modeli' in empty[2]
        inputs[9] = ['jansowa']
        path = client.predict(*inputs[:10], api_name='/export_csv')
        assert Path(path).read_text(encoding='utf-8-sig').startswith('Model;Średni wynik [%]')
        print('HTTP_SMOKE_OK: table/Pareto identity, filters, persistent labels, hover-only toggle, CSV')
    finally:
        demo.close()


if __name__ == '__main__':
    main()
