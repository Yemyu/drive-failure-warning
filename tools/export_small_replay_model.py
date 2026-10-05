"""Export one frozen linear model into a small, explicit JSON snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys


ROOT = Path(__file__).resolve().parents[1]
MODEL_DATABASE = ROOT / "data/derived/simple_baseline_v2/model_results.sqlite"
MODEL_DATABASE_SHA256 = "f112323774527887686a6471296af036415e45e751416f4a2e4a3574cf16d3c4"
MODEL_NAME = "current_lr"
EXPECTED_FEATURE_COUNT = 16


class ExportError(RuntimeError):
    """The frozen model cannot be exported safely."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: object, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ExportError(f"non-numeric {field}") from exc
    if not math.isfinite(number):
        raise ExportError(f"non-finite {field}")
    return number


def _open_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)


def export_model(database: Path = MODEL_DATABASE) -> dict:
    database = database.resolve()
    if not database.is_file():
        raise ExportError(f"model database missing: {database}")
    actual_sha = sha256(database)
    if actual_sha != MODEL_DATABASE_SHA256:
        raise ExportError(f"model database SHA-256 mismatch: {actual_sha}")
    connection = _open_readonly(database)
    try:
        run = connection.execute(
            "SELECT model,status,feature_count,intercept FROM model_runs WHERE model=?",
            (MODEL_NAME,),
        ).fetchone()
        if run is None:
            raise ExportError("current_lr model run is missing")
        model, status, feature_count, intercept = run
        if int(feature_count) != EXPECTED_FEATURE_COUNT:
            raise ExportError(f"current_lr feature count differs: {feature_count}")
        if status not in {"reused_v1", "complete"}:
            raise ExportError(f"current_lr model status is not frozen: {status}")
        feature_rows = connection.execute(
            "SELECT position,feature_name,source_name,is_missing_indicator "
            "FROM model_feature_map WHERE model=? ORDER BY position",
            (MODEL_NAME,),
        ).fetchall()
        if len(feature_rows) != EXPECTED_FEATURE_COUNT or [int(row[0]) for row in feature_rows] != list(range(EXPECTED_FEATURE_COUNT)):
            raise ExportError("current_lr feature map is not a contiguous 16-column map")
        stats = {
            str(row[0]): row
            for row in connection.execute(
                "SELECT feature_name,imputation_mean,standardization_mean,standardization_scale,observed_count "
                "FROM preprocessing_stats WHERE model=?",
                (MODEL_NAME,),
            )
        }
        coefficients = {
            str(row[0]): row[1]
            for row in connection.execute(
                "SELECT feature_name,coefficient FROM model_coefficients WHERE model=?",
                (MODEL_NAME,),
            )
        }
        if len(stats) != EXPECTED_FEATURE_COUNT or len(coefficients) != EXPECTED_FEATURE_COUNT:
            raise ExportError("current_lr preprocessing or coefficient count differs")
        features = []
        for position, name, source_name, is_missing_indicator in feature_rows:
            if name not in stats or name not in coefficients:
                raise ExportError(f"current_lr parameter row missing: {name}")
            stat = stats[name]
            scale = _finite(stat[3], f"scale {name}")
            if scale <= 0:
                raise ExportError(f"standardization scale is not positive: {name}")
            features.append(
                {
                    "position": int(position),
                    "feature_name": str(name),
                    "source_name": str(source_name),
                    "is_missing_indicator": int(is_missing_indicator),
                    "imputation_mean": _finite(stat[1], f"imputation mean {name}"),
                    "standardization_mean": _finite(stat[2], f"standardization mean {name}"),
                    "standardization_scale": scale,
                    "observed_count": int(stat[4]),
                    "coefficient": _finite(coefficients[name], f"coefficient {name}"),
                }
            )
        return {
            "schema_version": 1,
            "model": MODEL_NAME,
            "model_run_status": str(status),
            "feature_count": EXPECTED_FEATURE_COUNT,
            "intercept": _finite(intercept, "intercept"),
            "source_database": database.relative_to(ROOT).as_posix(),
            "source_database_sha256": actual_sha,
            "features": features,
        }
    finally:
        connection.close()


def _new_output(raw: str) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(ROOT) or resolved == ROOT:
        raise ExportError("output must be a new file inside the project")
    if candidate.exists() or candidate.is_symlink():
        raise ExportError(f"output already exists: {candidate}")
    if not resolved.parent.is_dir():
        raise ExportError("output parent directory does not exist")
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export frozen current_lr parameters")
    parser.add_argument("--output", default="examples/small_replay/current_lr.json")
    args = parser.parse_args(argv)
    try:
        output = _new_output(args.output)
        value = export_model()
        encoded = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n").encode("utf-8")
        with output.open("xb") as stream:
            stream.write(encoded)
        print(f"wrote {output.relative_to(ROOT)}")
    except (ExportError, OSError, sqlite3.Error) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
