"""Optional real HTTP smoke test, run with the standalone Gradio environment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'space'))
from app import build_app
from view import default_columns, default_tasks, load
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
                  default_columns(document), default_tasks(document)]
        result = client.predict(*inputs, api_name='/leaderboard')
        rows = result[0]['data']
        assert len(rows) >= 5, rows
        assert result[0]['headers'][1] == 'Średni wynik [%]'
        assert all(row[2] < 100 for row in rows)
        assert 'background-color' in str(result[0]['metadata'])
        assert all(not any(word in header for word in ['Precyzja', 'Rewizja', 'GPU', 'ms/']) for header in result[0]['headers'])
        for row in result[0]['metadata']['styling']:
            assert all(not row[k] for k in (2, 3, 4)), row
        inputs[6] = ['average', 't2i', 'i2t', 'organic', 'synthetic', 'mt']
        changed = client.predict(*inputs, api_name='/leaderboard')
        assert 'T2I [%]' in changed[0]['headers']
        assert 'Tłumaczenia maszynowe [%]' in changed[0]['headers']
        inputs[7] = ['flickr30k_pl_local_test']
        single = client.predict(*inputs, api_name='/leaderboard')
        assert single[0]['data'] != changed[0]['data']
        path = client.predict(*inputs, api_name='/export_csv')
        assert Path(path).read_text(encoding='utf-8-sig').startswith('Model;Średni wynik [%]')
        print('HTTP_SMOKE_OK: unified table, averages, hidden columns, score colors, uncolored parameters, CSV')
    finally:
        demo.close()


if __name__ == '__main__':
    main()
