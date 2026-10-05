"""Print a compact, read-only view of the frozen Q3 results."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_OUTPUT_BYTES = 1024 * 1024
EXPECTED_MODELS = ("current_lr", "history_lr", "history_hgb_v1")
EXPECTED_DENOMINATORS = (2000, 1000, 500)
CAPACITY_LABELS = {2000: "0.05%", 1000: "0.1%", 500: "0.2%"}
EXPECTED_ALERTS = {2000: 765, 1000: 1509, 500: 3018}
EXPECTED_EVENT_HITS = {
    "current_lr": {2000: 55, 1000: 65, 500: 69},
    "history_lr": {2000: 65, 1000: 72, 500: 73},
    "history_hgb_v1": {2000: 63, 1000: 72, 500: 75},
}
EXPECTED_REPEAT = {
    "current_lr": {"devices": 226, "first_alerts": 226, "later_alerts": 1283, "first_events": 28, "later_events": 37, "events": 65},
    "history_lr": {"devices": 295, "first_alerts": 295, "later_alerts": 1214, "first_events": 22, "later_events": 50, "events": 72},
    "history_hgb_v1": {"devices": 349, "first_alerts": 349, "later_alerts": 1160, "first_events": 19, "later_events": 53, "events": 72},
}


class ReaderError(RuntimeError):
    """A frozen result cannot be trusted or does not satisfy its contract."""


@dataclass(frozen=True)
class Binding:
    relative_path: str
    sha256: str
    kind: str


DEFAULT_BINDINGS: dict[str, Binding] = {
    "q3_db": Binding(
        "data/derived/q3_scoring_amended_v2/attempt_001/q3_scoring_amended_v2.sqlite",
        "4f22ed507eb11b2a464fa267b6fb761b4ee4f287765d0f1c9838344dd0099388",
        "sqlite",
    ),
    "q3_manifest": Binding(
        "evidence/q3/scoring_amended_v2/attempt_001/q3_scoring_amended_v2_complete_manifest_v1.json",
        "87e6c9bcb48c6c720bf12731bebfb8f256096dbda3bc2b522ccb303862432f72",
        "json",
    ),
    "tree_db": Binding(
        "data/derived/nonlinear_baseline_v1/attempt_006/nonlinear_baseline.sqlite",
        "99632ace9a99d3d39d1e32e4dbf6fff7b322a9198271086efb420ccce901c2c8",
        "sqlite",
    ),
    "tree_audit": Binding(
        "evidence/nonlinear_baseline_v1/attempt_006/independent_audit.json",
        "f635de54a93e5fb6cc7d154586cb0cf73aa8936de042f4174f5d2cc870d5d725",
        "json",
    ),
    "capacity_db": Binding(
        "data/derived/alert_capacity_v1/evaluation_001/capacity_evaluation.sqlite",
        "8aec3cb4d2e8172d3e1d39243a618cf65b91c8001f17adc0cf1bec3806c00888",
        "sqlite",
    ),
    "capacity_manifest": Binding(
        "evidence/alert_capacity_v1/evaluation_001/capacity_complete_manifest_v1.json",
        "c6eab6a1b20b40c1a89fec313b8b2e18267e8d63848c7581f303a5f32ee584e9",
        "json",
    ),
    "capacity_review": Binding(
        "evidence/alert_capacity_v1/results_review.json",
        "439c6d2885b5e1884fb66b48f0ffbd441451a1abb1405dd91578919f2ce3b773",
        "json",
    ),
    "repeat_review": Binding(
        "evidence/alert_repeat_analysis_v1/review.json",
        "49d2144bab0377d4a0b01933e9018b9aadda1eade0adf19d097b20172e558e67",
        "json",
    ),
}


@dataclass
class ResultSnapshot:
    q3_metrics: dict[str, dict[str, Any]]
    tree_metrics: dict[str, dict[str, Any]]
    capacity_metrics: dict[tuple[str, int], dict[str, Any]]
    repeat_analysis: dict[str, dict[str, Any]]
    input_sha256: dict[str, str]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(root: Path, binding: Binding) -> Path:
    candidate = (root / binding.relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise ReaderError(f"bound input escapes project root: {binding.relative_path}")
    return candidate


def _validate_bindings(root: Path, bindings: Mapping[str, Binding]) -> dict[str, Path]:
    root = Path(root).resolve()
    paths: dict[str, Path] = {}
    for key, binding in bindings.items():
        path = _path(root, binding)
        if not path.is_file():
            raise ReaderError(f"missing bound input: {binding.relative_path}")
        actual = sha256(path)
        if actual != binding.sha256:
            raise ReaderError(f"SHA-256 mismatch for {binding.relative_path}: {actual}")
        if binding.kind == "sqlite":
            for suffix in ("-wal", "-shm"):
                if Path(str(path) + suffix).exists():
                    raise ReaderError(f"SQLite sidecar must be empty or absent: {path}{suffix}")
        paths[key] = path
    return paths


def _open_readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReaderError(f"cannot read JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReaderError(f"JSON input is not an object: {path}")
    return value


def _require_tables(connection: sqlite3.Connection, names: Sequence[str], source: str) -> None:
    existing = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = [name for name in names if name not in existing]
    if missing:
        raise ReaderError(f"{source} is missing tables: {', '.join(missing)}")


def _require_model_metrics(connection: sqlite3.Connection, models: Sequence[str], source: str) -> dict[str, dict[str, Any]]:
    _require_tables(connection, ("model_metrics", "model_event_summary"), source)
    result: dict[str, dict[str, Any]] = {}
    for model in models:
        row = connection.execute("SELECT * FROM model_metrics WHERE model=?", (model,)).fetchone()
        if row is None:
            raise ReaderError(f"{source} is missing model_metrics row: {model}")
        events = connection.execute("SELECT COUNT(*) FROM model_event_summary WHERE model=?", (model,)).fetchone()[0]
        if int(events) != 126:
            raise ReaderError(f"{source} has {events} event rows for {model}; expected 126")
        result[model] = dict(row)
    return result


def _close_all(connections: Sequence[sqlite3.Connection]) -> None:
    for connection in connections:
        connection.close()


def _close_float_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def validate_capacity_snapshot(connection: sqlite3.Connection) -> dict[tuple[str, int], dict[str, Any]]:
    """Validate all nine capacity paths and return their metric rows."""
    _require_tables(connection, ("metrics", "selections", "daily", "alert_outcomes", "event_summary"), "capacity evaluation")
    rows = [dict(row) for row in connection.execute("SELECT * FROM metrics")]
    expected_keys = {(model, denominator) for model in EXPECTED_MODELS for denominator in EXPECTED_DENOMINATORS}
    actual_keys = {(str(row["model"]), int(row["denominator"])) for row in rows}
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ReaderError(f"capacity metric paths differ; missing={missing}, extra={extra}")
    result = {(str(row["model"]), int(row["denominator"])): row for row in rows}
    for (model, denominator), row in result.items():
        if row["capacity_label"] != CAPACITY_LABELS[denominator]:
            raise ReaderError(f"capacity label differs for {model}/{denominator}")
        if int(row["alerts"]) != EXPECTED_ALERTS[denominator]:
            raise ReaderError(f"capacity alert count differs for {model}/{denominator}")
        if int(row["event_total"]) != 126 or int(row["opportunity_total"]) != 126:
            raise ReaderError(f"capacity event denominator differs for {model}/{denominator}")
        selected = int(connection.execute("SELECT COUNT(*) FROM selections WHERE model=? AND denominator=?", (model, denominator)).fetchone()[0])
        if selected != int(row["alerts"]):
            raise ReaderError(f"selection count differs for {model}/{denominator}")
        daily = int(connection.execute("SELECT COUNT(*) FROM daily WHERE model=? AND denominator=?", (model, denominator)).fetchone()[0])
        if daily != 85:
            raise ReaderError(f"decision-day count differs for {model}/{denominator}: {daily}")
        outcome_count = int(connection.execute("SELECT COUNT(*) FROM alert_outcomes WHERE model=? AND denominator=?", (model, denominator)).fetchone()[0])
        if outcome_count != int(row["alerts"]):
            raise ReaderError(f"alert outcome count differs for {model}/{denominator}")
        event_count = int(connection.execute("SELECT COUNT(*) FROM event_summary WHERE model=? AND denominator=?", (model, denominator)).fetchone()[0])
        opportunity_count = int(connection.execute("SELECT COALESCE(SUM(opportunity),0) FROM event_summary WHERE model=? AND denominator=?", (model, denominator)).fetchone()[0])
        hit_count = int(connection.execute("SELECT COALESCE(SUM(hit),0) FROM event_summary WHERE model=? AND denominator=?", (model, denominator)).fetchone()[0])
        if (event_count, opportunity_count, hit_count) != (126, 126, EXPECTED_EVENT_HITS[model][denominator]):
            raise ReaderError(f"capacity event summary differs for {model}/{denominator}")
        if int(row["event_hits"]) != hit_count or not _close_float_equal(row["event_recall"], hit_count / 126.0):
            raise ReaderError(f"capacity event metric differs for {model}/{denominator}")
        if int(row["known_hit_alerts"]) + int(row["known_no_hit_alerts"]) + int(row["unknown_alerts"]) != int(row["alerts"]):
            raise ReaderError(f"capacity outcome counts do not sum for {model}/{denominator}")
    return result


def validate_repeat_analysis(value: Mapping[str, Any], capacity_metrics: Mapping[tuple[str, int], Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    repeat = value.get("repeat_analysis")
    if not isinstance(repeat, dict):
        raise ReaderError("repeat review has no repeat_analysis object")
    if set(repeat) != set(EXPECTED_MODELS):
        raise ReaderError("repeat review model set differs")
    for model, expected in EXPECTED_REPEAT.items():
        item = repeat.get(model)
        if not isinstance(item, dict):
            raise ReaderError(f"repeat review row missing: {model}")
        by_occurrence = item.get("by_occurrence")
        if not isinstance(by_occurrence, dict):
            raise ReaderError(f"repeat review has no occurrence rows: {model}")
        first = by_occurrence.get("1")
        if not isinstance(first, dict):
            raise ReaderError(f"repeat review has no first occurrence: {model}")
        later = [row for key, row in by_occurrence.items() if key != "1" and isinstance(row, dict)]
        first_alerts = int(first.get("alerts", -1))
        later_alerts = sum(int(row.get("alerts", 0)) for row in later)
        first_events = int(first.get("first_captured_events", 0))
        later_events = sum(int(row.get("first_captured_events", 0)) for row in later)
        if {
            "devices": int(item.get("devices", -1)),
            "first_alerts": first_alerts,
            "later_alerts": later_alerts,
            "first_events": first_events,
            "later_events": later_events,
            "events": int(item.get("captured_events", -1)),
        } != expected:
            raise ReaderError(f"repeat review values differ for {model}")
        metric = capacity_metrics[(model, 1000)]
        if int(metric["event_hits"]) != int(item["captured_events"]):
            raise ReaderError(f"repeat review and capacity event counts differ for {model}")
        if first_alerts + later_alerts != 1509 or first_events + later_events != int(item["captured_events"]):
            raise ReaderError(f"repeat review occurrence totals do not sum for {model}")
    return {model: dict(repeat[model]) for model in EXPECTED_MODELS}


def _validate_manifests(manifests: Mapping[str, Mapping[str, Any]], bindings: Mapping[str, Binding]) -> None:
    q3 = manifests["q3_manifest"]
    if q3.get("status") != "complete" or q3.get("database_sha256") != bindings["q3_db"].sha256:
        raise ReaderError("Q3 completion manifest is not bound to the frozen database")
    tree = manifests["tree_audit"]
    if tree.get("status") != "pass" or int(tree.get("event_total", -1)) != 126:
        raise ReaderError("tree audit is not a passing 126-event audit")
    if set(tree.get("metrics", {})) != set(EXPECTED_MODELS):
        raise ReaderError("tree audit is missing a required model metric")
    capacity = manifests["capacity_manifest"]
    if capacity.get("status") != "complete" or capacity.get("output_sha256", {}).get("database") != bindings["capacity_db"].sha256:
        raise ReaderError("capacity completion manifest is not bound to the frozen database")
    if set(capacity.get("models", [])) != set(EXPECTED_MODELS) or set(capacity.get("denominators", [])) != set(EXPECTED_DENOMINATORS):
        raise ReaderError("capacity completion manifest model or denominator set differs")
    if int(capacity.get("event_total", -1)) != 126 or int(capacity.get("opportunity_total", -1)) != 126:
        raise ReaderError("capacity completion manifest event denominator differs")
    review = manifests["capacity_review"]
    if review.get("status") != "pass" or int(review.get("source_alert_rows", -1)) != 15876 or int(review.get("event_rows", -1)) != 1134:
        raise ReaderError("capacity review evidence is not the expected passing review")
    repeat = manifests["repeat_review"]
    if repeat.get("scope") != "saved_output_review_and_descriptive_repeat_analysis":
        raise ReaderError("repeat review scope differs")


def _validate_cross_source(q3: Mapping[str, Mapping[str, Any]], tree: Mapping[str, Mapping[str, Any]], capacity: Mapping[tuple[str, int], Mapping[str, Any]]) -> None:
    sources = {"current_lr": q3["current_lr"], "history_lr": q3["history_lr"], "history_hgb_v1": tree["history_hgb_v1"]}
    mapping = {
        "eligible_device_days": "eligible_device_days",
        "alerts": "alerts",
        "known_hit_alerts": "known_hit_alerts",
        "known_no_hit_alerts": "known_no_hit_alerts",
        "unknown_alerts": "unknown_alerts",
        "known_outcome_precision": "known_outcome_precision",
        "early_event_hits_ge2_days": "early_event_hits_ge2",
        "early_event_hits_ge3_days": "early_event_hits_ge3",
        "earliest_lead_count": "earliest_lead_count",
        "earliest_lead_median": "earliest_lead_median",
        "earliest_lead_q25": "earliest_lead_q25",
        "earliest_lead_q75": "earliest_lead_q75",
        "repeated_alert_devices": "repeated_alert_devices",
        "max_alerts_per_device": "max_alerts_per_device",
        "minimum_alert_gap_days": "minimum_alert_gap_days",
    }
    for model, source in sources.items():
        actual = capacity[(model, 1000)]
        for source_field, capacity_field in mapping.items():
            left, right = source.get(source_field), actual.get(capacity_field)
            if isinstance(left, float) or isinstance(right, float):
                equal = _close_float_equal(left, right)
            else:
                equal = left == right
            if not equal:
                raise ReaderError(f"0.1% source metric differs for {model}: {source_field}")
        if source.get("event_opportunity_total") != actual.get("opportunity_total") or source.get("event_hits") != actual.get("event_hits"):
            raise ReaderError(f"0.1% event metric differs for {model}")
        if not _close_float_equal(source.get("event_recall_at_opportunity"), actual.get("event_recall")):
            raise ReaderError(f"0.1% event recall differs for {model}")


class ResultReader:
    def __init__(self, root: Path = PROJECT_ROOT, bindings: Mapping[str, Binding] | None = None):
        self.root = Path(root).resolve()
        self.bindings = dict(bindings or DEFAULT_BINDINGS)

    def read(self) -> ResultSnapshot:
        expected_keys = set(DEFAULT_BINDINGS)
        if set(self.bindings) != expected_keys:
            raise ReaderError("result input binding set differs")
        paths = _validate_bindings(self.root, self.bindings)
        before = {key: sha256(path) for key, path in paths.items()}
        manifests = {key: _json(paths[key]) for key in ("q3_manifest", "tree_audit", "capacity_manifest", "capacity_review", "repeat_review")}
        _validate_manifests(manifests, self.bindings)
        connections: list[sqlite3.Connection] = []
        try:
            q3_connection = _open_readonly(paths["q3_db"])
            tree_connection = _open_readonly(paths["tree_db"])
            capacity_connection = _open_readonly(paths["capacity_db"])
            connections.extend((q3_connection, tree_connection, capacity_connection))
            q3_metrics = _require_model_metrics(q3_connection, ("current_lr", "history_lr"), "Q3 result")
            tree_metrics = _require_model_metrics(tree_connection, EXPECTED_MODELS, "tree result")
            capacity_metrics = validate_capacity_snapshot(capacity_connection)
            _validate_cross_source(q3_metrics, tree_metrics, capacity_metrics)
            repeat_analysis = validate_repeat_analysis(manifests["repeat_review"], capacity_metrics)
        finally:
            _close_all(connections)
        after = {key: sha256(path) for key, path in paths.items()}
        if before != after:
            raise ReaderError("a bound input changed while results were being read")
        return ResultSnapshot(q3_metrics, tree_metrics, capacity_metrics, repeat_analysis, after)


def _pct(value: Any) -> str:
    return f"{float(value) * 100:.2f}%"


def _metric(snapshot: ResultSnapshot, model: str, denominator: int) -> Mapping[str, Any]:
    return snapshot.capacity_metrics[(model, denominator)]


def render_markdown(snapshot: ResultSnapshot, *, output_directory: Path | None = None) -> str:
    lines = [
        "# 硬盘故障提前预警：固定结果总览",
        "",
        "数据来自 Backblaze Drive Stats 的 ST4000DM000，使用 2023 年 Q1/Q2 训练、Q3 评价。预测时点只使用当天及以前的运行记录，目标是未来 7 日内的首次故障标记。Q3 共有 126 个有预警机会的事件；评分期为 2023-07-01 至 09-23，共 85 个决策日。",
        "",
        "Q3 已参与错误分析和方法选择，所以这里是开发期结果，不是新的独立泛化测试。公开 failure 是运营标记，不能直接解释为维修确认的机械故障。",
        "",
        "## 0.1% 检查容量",
        "",
        "| 模型 | 告警总数 | 已知命中 | 已知未命中 | 未知 | 事件捕获 | 事件召回 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    names = {"current_lr": "current_lr", "history_lr": "history_lr", "history_hgb_v1": "history_hgb_v1"}
    for model in EXPECTED_MODELS:
        row = _metric(snapshot, model, 1000)
        lines.append(
            f"| {names[model]} | {row['alerts']} | {row['known_hit_alerts']} | {row['known_no_hit_alerts']} | {row['unknown_alerts']} | {row['event_hits']}/126 | {_pct(row['event_recall'])} |"
        )
    lines.extend(
        [
            "",
            "事件召回按去重事件计算，已知命中／未命中／未知按告警行计算，三者不能互相替代。历史 LR 相对当前 LR 的点差为 +5.56 个百分点，但设备级配对 bootstrap 的 95% 区间为 −0.79 至 11.38 个百分点，跨过零，因此 current_lr 仍是控制模型。",
            "",
            "## 首次提醒与后续提醒",
            "",
            "下表中的告警数是 85 个决策日的累计数。后续提醒并不一定无效：它可能首次进入故障前 7 日窗口。",
            "",
            "| 模型 | 季度首次告警 | 后续告警 | 首次提醒捕获事件 | 后续提醒首次捕获事件 | 合计事件 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model in EXPECTED_MODELS:
        row = snapshot.repeat_analysis[model]
        lines.append(f"| {model} | {row['by_occurrence']['1']['alerts']} | {sum(int(item.get('alerts', 0)) for key, item in row['by_occurrence'].items() if key != '1')} | {row['by_occurrence']['1'].get('first_captured_events', 0)} | {sum(int(item.get('first_captured_events', 0)) for key, item in row['by_occurrence'].items() if key != '1')} | {row['captured_events']} |")
    lines.extend(
        [
            "",
            "## 三档检查容量",
            "",
            "告警数量是评价期累计值，不是每天的数量。容量从 0.1% 增至 0.2% 时，每个模型都增加 1,509 次告警，但净增事件只有 1—4 个；七日冷却会改变具体捕获集合。",
            "",
            "| 模型 | 0.05%：告警／事件 | 0.1%：告警／事件 | 0.2%：告警／事件 |",
            "|---|---:|---:|---:|",
        ]
    )
    for model in EXPECTED_MODELS:
        cells = []
        for denominator in EXPECTED_DENOMINATORS:
            row = _metric(snapshot, model, denominator)
            cells.append(f"{row['alerts']} / {row['event_hits']}/126 ({_pct(row['event_recall'])})")
        lines.append(f"| {model} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.extend(
        [
            "",
            "结果文件：",
            "",
            "- [研究说明](Q3_RESEARCH_RESULT_BRIEF.md)",
            "- [重复告警分析](ALERT_REPEAT_ANALYSIS.md)",
            "- [容量结果](ALERT_CAPACITY_RESULTS.md)",
            "- [容量结果审查](ALERT_CAPACITY_REVIEW.md)",
            "- [完成清单](../evidence/alert_capacity_v1/evaluation_001/capacity_complete_manifest_v1.json)",
            "",
            "这些数字用于说明风险排序与检查资源的取舍，不代表真实维修成本、故障减少或线上部署效果。",
            "",
        ]
    )
    document = "\n".join(lines)
    if output_directory is not None:
        from urllib.parse import quote

        for target in (
            "Q3_RESEARCH_RESULT_BRIEF.md", "ALERT_REPEAT_ANALYSIS.md",
            "ALERT_CAPACITY_RESULTS.md", "ALERT_CAPACITY_REVIEW.md",
            "../evidence/alert_capacity_v1/evaluation_001/capacity_complete_manifest_v1.json",
        ):
            relative = os.path.relpath(PROJECT_ROOT / "reports" / target, output_directory)
            document = document.replace(f"]({target})", f"]({quote(relative, safe='/')})")
    return document


def _output_path(root: Path, raw: str) -> Path:
    root = Path(root).resolve()
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ReaderError("--output must name a new file inside the project")
    if os.path.lexists(str(candidate)):
        raise ReaderError(f"--output already exists: {candidate}")
    if not resolved.parent.is_dir():
        raise ReaderError("--output parent directory does not exist")
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print frozen HDD warning results")
    parser.add_argument("--output", help="write a new Markdown overview inside the project")
    args = parser.parse_args(argv)
    try:
        snapshot = ResultReader().read()
        document = render_markdown(snapshot)
        if args.output is None:
            sys.stdout.write(document)
        else:
            path = _output_path(PROJECT_ROOT, args.output)
            encoded = render_markdown(snapshot, output_directory=path.parent).encode("utf-8")
            if len(encoded) > MAX_OUTPUT_BYTES:
                raise ReaderError("rendered overview exceeds 1 MiB")
            with path.open("xb") as stream:
                stream.write(encoded)
            print(f"wrote {path.relative_to(PROJECT_ROOT)}")
    except (ReaderError, OSError, sqlite3.Error) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
