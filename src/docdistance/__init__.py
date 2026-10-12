from importlib.metadata import PackageNotFoundError, version

from docdistance import config  # noqa: F401  (sets up logging + paths on import)
from docdistance.bootstrap import init
from docdistance.distance import (
    DistanceResult,
    SolverNotOptimal,
    SourceConditionedResult,
    StructuralResult,
    closeness,
    compute_distance,
    compute_source_conditioned,
    compute_structural,
    grounding_blend,
    grounding_residual,
    matched_order_gap,
    opw_cost,
    opw_gap,
    opw_plan,
    order_alignment,
    partial_matching,
    rwmd,
    smd,
    structure_displacement,
    transport_plan,
    unsupported_count,
    wcd,
)
from docdistance.pipeline import (
    DocDistance,
    semantic_distance,
    source_conditioned_distance,
    structural_distance,
)
from docdistance.settings import NotInitializedError

try:
    __version__ = version("docdistance")
except PackageNotFoundError:  # running from source, not installed
    __version__ = "0.0.0"

__all__ = [
    "DistanceResult",
    "DocDistance",
    "NotInitializedError",
    "SolverNotOptimal",
    "SourceConditionedResult",
    "StructuralResult",
    "__version__",
    "closeness",
    "compute_distance",
    "compute_source_conditioned",
    "compute_structural",
    "grounding_blend",
    "grounding_residual",
    "init",
    "matched_order_gap",
    "opw_cost",
    "opw_gap",
    "opw_plan",
    "order_alignment",
    "partial_matching",
    "rwmd",
    "semantic_distance",
    "smd",
    "source_conditioned_distance",
    "structural_distance",
    "structure_displacement",
    "transport_plan",
    "unsupported_count",
    "wcd",
]
