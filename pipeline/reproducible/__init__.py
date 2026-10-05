"""Small, source-backed execution pieces for the reproducible research run."""

from .current import (
    ReproducibleStopped,
    build_labels,
    build_panel_from_manifest,
    build_current_features,
    run_current_research,
    verify_source_manifest,
    verify_cross_quarter_identity,
)
from .artifacts import ArtifactError, verify_manifest, write_json_exclusive, write_manifest
from .process_contract import ProcessAccessError, authorize_inputs, authorize_bindings_file
from .input_binding import BindingError, create_binding_file, load_binding_file, verify_binding_for_spec
from .training_check import APPROVAL_VERSION, TrainingCheckError, load_approved_training_package, load_training_package, verify_training_package, write_training_package
from .panel_reference import PanelReferenceError, REFERENCE_VERSION, verify_training_reference
from .evaluation_summary import EvaluationSummaryError, summarize_lead_metrics
from .worker import WorkerError, evaluate_worker, run_worker, score_worker, train_worker
from .orchestrator import OrchestratorError, isolated_run
from .supervisor import REQUEST_VERSION as SUPERVISOR_REQUEST_VERSION, SupervisorError, run_supervised
from .source_snapshot import ENTRYPOINT as SOURCE_SNAPSHOT_ENTRYPOINT, SNAPSHOT_VERSION, SourceSnapshotError, create_source_snapshot, verify_source_snapshot
from .panel_orchestrator import PanelOrchestratorError, rebuild_panels
from .resource_guard import ResourceGuard, ResourceLimits, ResourceViolation
from .cancellation import CancellationRequested, CancellationToken, attach_sqlite_progress
from .environment import snapshot as environment_snapshot, write_snapshot as write_environment_snapshot
from .comparison import (
    ABS_TOL as COMPARISON_ABS_TOL,
    REL_TOL as COMPARISON_REL_TOL,
    ComparisonError,
    numeric_summary,
    scalar_equal,
)

__all__ = [
    "ReproducibleStopped",
    "build_labels",
    "build_panel_from_manifest",
    "build_current_features",
    "run_current_research",
    "verify_source_manifest",
    "verify_cross_quarter_identity",
    "ArtifactError",
    "verify_manifest",
    "write_manifest",
    "write_json_exclusive",
    "ProcessAccessError",
    "authorize_inputs",
    "authorize_bindings_file",
    "BindingError",
    "create_binding_file",
    "load_binding_file",
    "verify_binding_for_spec",
    "TrainingCheckError",
    "APPROVAL_VERSION",
    "load_approved_training_package",
    "load_training_package",
    "verify_training_package",
    "write_training_package",
    "PanelReferenceError",
    "REFERENCE_VERSION",
    "verify_training_reference",
    "EvaluationSummaryError",
    "summarize_lead_metrics",
    "WorkerError",
    "evaluate_worker",
    "run_worker",
    "score_worker",
    "train_worker",
    "OrchestratorError",
    "isolated_run",
    "SUPERVISOR_REQUEST_VERSION",
    "SupervisorError",
    "run_supervised",
    "SOURCE_SNAPSHOT_ENTRYPOINT",
    "SNAPSHOT_VERSION",
    "SourceSnapshotError",
    "create_source_snapshot",
    "verify_source_snapshot",
    "PanelOrchestratorError",
    "rebuild_panels",
    "ResourceGuard",
    "ResourceLimits",
    "ResourceViolation",
    "CancellationRequested",
    "CancellationToken",
    "attach_sqlite_progress",
    "environment_snapshot",
    "write_environment_snapshot",
    "COMPARISON_ABS_TOL",
    "COMPARISON_REL_TOL",
    "ComparisonError",
    "numeric_summary",
    "scalar_equal",
]
