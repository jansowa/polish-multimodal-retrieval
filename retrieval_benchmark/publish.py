"""Publish only the standalone display files, never the parent project."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval_benchmark.core import HERE, public_check, read, sha

FILES = ('app.py', 'view.py', 'results.json', 'requirements.txt', 'README.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, help='owner/space-name')
    parser.add_argument('--upload', action='store_true', help='Actually create/update the remote Space; default is a local preview')
    parser.add_argument('--public', action='store_true', help='Create a new public Space; otherwise a new Space is private')
    args = parser.parse_args()
    space = HERE/'space'
    public_check(read(space/'results.json'))
    plan = {name: dict(bytes=(space/name).stat().st_size, sha256=sha(space/name)) for name in FILES}
    print(json.dumps(dict(repo=args.repo, files=plan, upload=args.upload), indent=2))
    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi()  # Existing hf login / HF_TOKEN; no credentials in config or files.
        api.create_repo(args.repo, repo_type='space', space_sdk='gradio', private=not args.public, exist_ok=True)
        commit = api.upload_folder(repo_id=args.repo, repo_type='space', folder_path=str(space),
                                   allow_patterns=list(FILES), commit_message='Update Polish retrieval benchmark leaderboard')
        print(commit.commit_url)


if __name__ == '__main__':
    main()
