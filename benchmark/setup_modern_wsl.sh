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

# CUDA 13 with the older HF APIs used by CLIP/SigLIP.
legacy="$HOME/.venvs/pl-mmb-legacy-cu130"
python3 -m venv "$legacy"
legacy_site=$("$legacy/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
printf '%s\n%s\n' "$modern_site" "$base_site" > "$legacy_site/benchmark_dependencies.pth"
"$legacy/bin/pip" install 'transformers==4.57.6' 'sentence-transformers==5.7.0' 'huggingface-hub==0.36.2' 'tokenizers==0.22.2'
