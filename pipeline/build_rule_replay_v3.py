"""Versioned wrapper for a new, provenance-complete rule replay.

The implementation remains the reviewed single-process builder in
``build_rule_replay_v2``.  This wrapper changes only the run identifier and
default output/evidence paths, preserving the historical v2 database.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import build_rule_replay_v2 as implementation


REPLAY_ID = "rule_replay_train_v3"
OUTPUT_DB_DEFAULT = ROOT / "data/derived/rule_replay_train_v3.sqlite"
EVIDENCE_DEFAULT = ROOT / "evidence/q2/feature_replay_closeout_v2/replay_v3"


def build_replay(
    panel_db: Path = implementation.PANEL_DB_DEFAULT,
    feature_db: Path = implementation.FEATURE_DB_DEFAULT,
    output_db: Path = OUTPUT_DB_DEFAULT,
    evidence_root: Path = EVIDENCE_DEFAULT,
    *,
    resume: bool = False,
) -> dict:
    previous = implementation.REPLAY_ID
    implementation.REPLAY_ID = REPLAY_ID
    try:
        return implementation.build_replay(panel_db, feature_db, output_db, evidence_root, resume=resume)
    finally:
        implementation.REPLAY_ID = previous


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=implementation.PANEL_DB_DEFAULT)
    parser.add_argument("--feature", type=Path, default=implementation.FEATURE_DB_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DB_DEFAULT)
    parser.add_argument("--evidence-root", type=Path, default=EVIDENCE_DEFAULT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = build_replay(args.panel, args.feature, args.output, args.evidence_root, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
