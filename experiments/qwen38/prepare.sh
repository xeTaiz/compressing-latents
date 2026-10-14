#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
ENV="$ROOT/.venvs/qwen38"
python3 -m venv "$ENV"
"$ENV/bin/python" -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 torchvision==0.24.1
"$ENV/bin/python" -m pip install --no-cache-dir -r "$ROOT/experiments/qwen38/environment-lock.txt"
"$ENV/bin/python" "$ROOT/experiments/qwen38/experiment.py" prepare "$@"
