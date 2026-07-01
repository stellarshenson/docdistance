from importlib.metadata import PackageNotFoundError, version

from docdistance import config  # noqa: F401  (sets up logging + paths on import)
from docdistance.bootstrap import init
from docdistance.distance import (
    DistanceResult,
    SourceConditionedResult,
    StructuralResult,
    closeness,
    compute_distance,
    compute_source_conditioned,
    compute_structural,
    grounding_blend,
    grounding_residual,
    opw_cost,
    opw_gap,
    opw_plan,
    order_alignment,
    rwmd,
    smd,
    structure_displacement,
    transport_plan,
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
    "DocDistance",
    "DistanceResult",
    "SourceConditionedResult",
    "StructuralResult",
    "NotInitializedError",
    "init",
    "semantic_distance",
    "structural_distance",
    "source_conditioned_distance",
    "compute_distance",
    "compute_source_conditioned",
    "compute_structural",
    "grounding_residual",
    "grounding_blend",
    "smd",
    "transport_plan",
    "opw_plan",
    "opw_cost",
    "opw_gap",
    "order_alignment",
    "structure_displacement",
    "wcd",
    "rwmd",
    "closeness",
    "__version__",
]
