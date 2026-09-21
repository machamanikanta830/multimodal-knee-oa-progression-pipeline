#!/usr/bin/env bash
set -euo pipefail
bundle_root="$(cd "$(dirname "$0")/.." && pwd)"
task_venv="${KNEE_OA_COLAB_VENV:-/content/knee_oa_6d_env}"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/content/knee_oa_6d_matplotlib
# Check target hardware before installing anything or touching model data.
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python3 -c 'import re, subprocess; names=subprocess.check_output(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],text=True).splitlines(); assert any(re.match(r"^(?:NVIDIA|Tesla)\s+(A100|L4|T4)\b", name.strip(), re.IGNORECASE) for name in names), "STOP: select an approved NVIDIA A100, L4 or T4 CUDA GPU"'
python3 -c 'import sys; assert (3,12) <= sys.version_info[:2] < (3,15), "STOP: use compatible Python 3.12-3.14 for the exact pinned dependencies"'
python3 -c 'import sys; from pathlib import Path; assert not Path(sys.argv[1]).resolve().is_relative_to(Path(sys.argv[2]).resolve()), "STOP: virtual environment must be outside the immutable bundle"' "$task_venv" "$bundle_root"
python3 -m venv "$task_venv"
if ! "$task_venv/bin/python" -m pip install -r "$bundle_root/requirements.txt"; then
  echo "STOP: exact pinned dependencies are incompatible. Do not substitute versions." >&2
  exit 1
fi
"$task_venv/bin/python" -m pip check
cd "$bundle_root"
"$task_venv/bin/python" scripts/verify_bundle.py --cuda
"$task_venv/bin/python" scripts/run_training.py --device cuda --train --output data/processed/modeling/image/v1/production_cuda
