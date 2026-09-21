"""Explicit CUDA-only approved training entry; --plan never fits or predicts."""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from modeling import bundle_runtime as runtime  # noqa: E402 - resolve packaged source first


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--output", type=Path, default=runtime.PRODUCTION)
    args = parser.parse_args()
    runtime.require(
        args.output == runtime.PRODUCTION, "Use the approved bundle-relative production directory"
    )
    report = runtime.verify(ROOT)
    os.chdir(ROOT)
    if args.plan:
        print(
            json.dumps(
                {
                    "verified": report,
                    "training_source": str(importlib.util.find_spec("modeling.image").origin),
                    "command": "python scripts/run_training.py --device cuda --train --output "
                    + str(args.output),
                    "fit_performed": False,
                },
                indent=2,
            )
        )
        return 0
    runtime.require(args.train, "Explicit --train required")
    print(json.dumps(runtime.environment(ROOT, cuda=True), indent=2), flush=True)
    runtime.bind(ROOT)
    from modeling import image

    sys.argv = ["modeling.image", "--device", "cuda", "--train", "--output", str(args.output)]
    return image.main()


if __name__ == "__main__":
    raise SystemExit(main())
