"""Conservative retained-output scope for R, including failed and old attempts."""
from pipeline.reproducible.artifacts import bound_path
from pipeline.reproducible.runtime_context import project_root


def cumulative_roots(output):
    root=project_root()
    # All local execution records are counted, including older unrelated runs.
    # This avoids inferring membership from a run name and undercounting it.
    candidates=[bound_path(output,'R output'),*(bound_path(root/p,'R budget root') for p in (
        'data/raw/r_validation_v1','data/derived/r_validation_v1','evidence/r_validation_v1',
        'data/raw/q1_2024_validation_v1','data/derived/q1_2024_validation_v1','evidence/q1_2024_validation_v1',
        'configs/q1_2024_validation_v1.json','tools/run_q1_validation.py',
        'handoff/runs','pipeline/r_validation','configs/r_validation_v1.json','tools/run_r_validation.py'))]
    roots=[]
    for path in sorted(set(candidates),key=lambda p:(len(p.parts),str(p))):
        if not any(path.is_relative_to(parent) for parent in roots): roots.append(path)
    return tuple(roots)
