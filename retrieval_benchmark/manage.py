"""One entry point for inventory, frozen runs, export and paired comparisons."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrieval_benchmark.core import HERE, CATALOG, identity, local_path, public_check, read, sha, write
from retrieval_benchmark.ingest import configuration, sync


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'local/config.json')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('init', help='Create local paths and pinned model registry; no downloads')
    commands.add_parser('list')
    serve = commands.add_parser('serve', help='Start the standalone leaderboard locally')
    serve.add_argument('--port', type=int, default=7860)
    serve.add_argument('--python', dest='ui_python', help='Interpreter with Gradio installed')
    export = commands.add_parser('sync', help='Import historical/new results and export the single public JSON')
    export.add_argument('--output', type=Path, default=HERE/'space/results.json')
    validate = commands.add_parser('validate')
    validate.add_argument('--results', type=Path, default=HERE/'space/results.json')
    for name in ('plan', 'run'):
        command = commands.add_parser(name)
        command.add_argument('--model', required=True)
        command.add_argument('--datasets', nargs='+', default=['core'])
        command.add_argument('--prompt-modes', nargs='+', default=['v2_no_instructions', 'author_recommended'])
        command.add_argument('--batch-size', type=int, default=64)
    compare = commands.add_parser('compare')
    compare.add_argument('--model', required=True, help='Exact public model_id')
    compare.add_argument('--reference', required=True)
    compare.add_argument('--dataset', required=True)
    compare.add_argument('--direction', choices=['t2i', 'i2t'], default='t2i')
    compare.add_argument('--mode', default='v2_no_instructions')
    compare.add_argument('--profile', default='historical-native-v2')
    compare.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = configuration(args.config if args.config.is_file() else None)
    if args.command == 'init':
        from retrieval_benchmark.runner import initialize
        write(args.config, initialize(config), immutable=True)
        print(args.config)
    elif args.command == 'list':
        print(json.dumps(read(CATALOG), ensure_ascii=False, indent=2))
        print('Models:', ', '.join(config.get('models', {})))
    elif args.command == 'serve':
        default = local_path('C:/Users/jsowa/benchmark-translations/retrieval-benchmark-v1/ui-env/Scripts/python.exe')
        executable = args.ui_python or config.get('ui_python') or (str(default) if os.name == 'nt' and default.is_file() else sys.executable)
        environment = dict(os.environ, GRADIO_SERVER_NAME='127.0.0.1', GRADIO_SERVER_PORT=str(args.port), GRADIO_ANALYTICS_ENABLED='False')
        subprocess.run([executable, '-u', str(HERE/'space/app.py')], env=environment, check=True)
    elif args.command == 'sync':
        print(json.dumps(sync(config, args.output), ensure_ascii=False))
    elif args.command == 'validate':
        doc = public_check(read(args.results))
        print(f"OK: {len(doc['models'])} models, {len(doc['results'])} results; SHA256 {sha(args.results)}")
    elif args.command in ('plan', 'run'):
        from retrieval_benchmark.runner import plan, run_worker
        execution = plan(config, args.model, args.datasets, args.prompt_modes, args.batch_size)
        if args.command == 'plan':
            print(json.dumps(execution, ensure_ascii=False, indent=2))
            return
        sys.path.insert(0, str(ROOT/'src'))
        from pl_mmb_benchmark.models import worker_python
        spec = execution['model_spec']
        executable = (str(Path.home()/'.venvs/pl-mmb-benchmark/bin/python') if 'checkpoint' in spec else
                      worker_python(spec['repository'], sys.executable))
        if str(Path(executable).resolve()) != str(Path(sys.executable).resolve()):
            command = [executable, str(Path(__file__).resolve()), '--config', str(args.config.resolve()),
                       'run', '--model', args.model, '--datasets', *args.datasets, '--prompt-modes',
                       *args.prompt_modes, '--batch-size', str(args.batch_size)]
            subprocess.run(command, check=True)
        else:
            run_worker(config, execution)
    elif args.command == 'compare':
        sys.path.insert(0, str(ROOT/'src'))
        from pl_mmb_benchmark.full_gallery import paired_bootstrap
        sources = read(HERE/'local/source_index.json')['artifacts']
        def find(model):
            matches = [r for r in sources if (r['model_id'], r['dataset_id'], r['prompt_mode'], r['profile']) ==
                       (model, args.dataset, args.mode, args.profile)]
            if len(matches) != 1:
                raise ValueError('Expected exactly one matching source result')
            path = local_path(matches[0]['path'])
            if sha(path) != matches[0]['sha256']:
                raise ValueError('Source artifact changed')
            result = read(path)
            if matches[0].get('equivalent_prompt_mode'):
                result = dict(result, prompt_mode=matches[0]['equivalent_prompt_mode'])
            if result.get('status') == 'complete' and 'public_result' in result:
                scores = path.parent/'scores.json'
                if sha(scores) != result['public_result']['source_sha256']:
                    raise ValueError('Score SHA mismatch')
                result = read(scores)
            return result
        ci = paired_bootstrap(find(args.model), find(args.reference), args.direction,
                              replicates=10000, seed=20261023)
        write(args.output, ci, immutable=True)
        print(args.output)


if __name__ == '__main__':
    main()
