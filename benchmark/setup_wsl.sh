#!/usr/bin/env bash
set -euo pipefail
python3 -m venv "$HOME/.venvs/pl-mmb-benchmark"
"$HOME/.venvs/pl-mmb-benchmark/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cu128
"$HOME/.venvs/pl-mmb-benchmark/bin/pip" install -r benchmark/requirements.txt
