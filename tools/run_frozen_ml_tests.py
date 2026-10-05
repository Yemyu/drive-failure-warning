"""Run the E1.0 pytest-style checks with only the project Python runtime.

The analysis environment deliberately has no pytest dependency.  Each test is
still a plain function, so this runner executes the same functions and reports
the count without installing anything into the environment.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "tests/test_frozen_ml_evaluation.py"


def main() -> int:
    spec = importlib.util.spec_from_file_location("test_frozen_ml_evaluation", TEST)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {TEST}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tests = [(name, fn) for name, fn in vars(module).items() if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"TOTAL {len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
