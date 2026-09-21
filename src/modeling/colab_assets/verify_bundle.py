"""Verify bundle only: no training, model inference, or TEST access."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from modeling.bundle_runtime import main  # noqa: E402 - resolve packaged source first

if __name__ == "__main__":
    raise SystemExit(main())
