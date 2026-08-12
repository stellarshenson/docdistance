from pathlib import Path
import sys

from dotenv import load_dotenv
from loguru import logger

########### SETUP ###############

# set up logger - INFO by default (DEBUG only via the CLI --verbose flag), sink to stderr so
# stdout stays clean for --json / --result-only output
logger.remove()
logger.add(sys.stderr, colorize=True, level="INFO")

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end="", file=sys.stderr), colorize=True, level="INFO")
except ModuleNotFoundError:
    pass

########## VARIABLES ############

# Load environment variables from .env file if it exists
load_dotenv()

# paths
PROJ_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJ_ROOT / "models"

# log current root dir (debug so it never pollutes machine-readable stdout)
logger.debug(f"PROJ_ROOT path is: {PROJ_ROOT}")

########## MODELS ###############

# segmenter (wtpsplit SaT) and the mmBERT statement encoders, by backend
SAT_MODEL = "sat-3l-sm"
MMBERT_TORCH_MODEL = "jhu-clsp/mmBERT-base"
MMBERT_OPENVINO_LOCAL = MODELS_DIR / "02-mmbert-openvino-int8"
MMBERT_OPENVINO_HF = "stellars/mmBERT-base-openvino-int8"

# source-conditioned grounding axis (D_grd): cross-encoder reranker + NLI entailment head
RERANKER_TORCH_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_OPENVINO_LOCAL = MODELS_DIR / "03-reranker-openvino-int8"
RERANKER_OPENVINO_HF = "stellars/bge-reranker-v2-m3-openvino-int8"
NLI_TORCH_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
NLI_OPENVINO_LOCAL = MODELS_DIR / "04-nli-openvino-int8"
NLI_OPENVINO_HF = "stellars/mdeberta-v3-base-mnli-xnli-openvino-int8"

# grounding hyperparameters (nb05 operating point; E03-H11 gate, E03-H14 blend)
RERANK_TOP_K = 3  # source statements fused into the NLI premise per document statement
RERANK_MAX_TOKENS = 256  # reranker / NLI tokenizer truncation
RERANK_PAIR_BATCH = 256  # reranker / NLI inference batch
GROUNDING_BLEND_ALPHA = 0.75  # E03-H14: alpha * D_sel + (1 - alpha) * D_grd

# model registry: key -> HF repo, torch repo, local mirror subdir (under the init models dir and
# the S3 docdistance/ prefix). init resolves each required key local -> S3 -> HF.
MODEL_REGISTRY = {
    "mmbert": {
        "hf": MMBERT_OPENVINO_HF,
        "torch": MMBERT_TORCH_MODEL,
        "subdir": "mmbert-openvino-int8",
    },
    "sat": {"hf": SAT_MODEL, "torch": SAT_MODEL, "subdir": "sat-3l-sm"},
    "reranker": {
        "hf": RERANKER_OPENVINO_HF,
        "torch": RERANKER_TORCH_MODEL,
        "subdir": "reranker-openvino-int8",
    },
    "nli": {"hf": NLI_OPENVINO_HF, "torch": NLI_TORCH_MODEL, "subdir": "nli-openvino-int8"},
}

# which models each distance mode requires; init pulls only these
MODE_MODELS = {
    "wmd": ["mmbert", "sat"],
    "wmd-wrt-source": ["mmbert", "sat", "reranker", "nli"],
}


def configure_logging(verbose: bool = False) -> None:
    """Re-point loguru at stderr at INFO, or DEBUG when ``verbose`` - the CLI calls this first.

    stderr keeps stdout reserved for the result so ``--json`` and ``--result-only`` stay machine-parseable.
    """
    level = "DEBUG" if verbose else "INFO"
    logger.remove()
    try:
        from tqdm import tqdm

        logger.add(
            lambda msg: tqdm.write(msg, end="", file=sys.stderr), colorize=True, level=level
        )
    except ModuleNotFoundError:
        logger.add(sys.stderr, colorize=True, level=level)
