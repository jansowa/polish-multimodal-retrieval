#!/usr/bin/env bash
set -euo pipefail
# Reuse base Python dependencies, but isolate the newer CUDA and HF stack.
base="$HOME/.venvs/pl-mmb-benchmark"
modern="$HOME/.venvs/pl-mmb-modern"
python3 -m venv "$modern"
base_site=$("$base/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
modern_site=$("$modern/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo "$base_site" > "$modern_site/benchmark_dependencies.pth"
"$modern/bin/pip" install 'torch==2.14.1' 'torchvision==0.29.1' --index-url https://download.pytorch.org/whl/cu130
"$modern/bin/pip" install 'transformers==5.20.1' 'sentence-transformers==6.1.0'
