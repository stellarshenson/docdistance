"""High-level document-distance API.

Two entry styles:

- :class:`DocDistance` - load the models once, score many pairs (the pipeline-integration entry)
- :func:`document_distance` / :func:`source_conditioned_distance` - one-shot convenience that loads
  and scores in a single call

Inputs are raw text or a path to a text/markdown file (auto-detected). A leading markdown ``# `` title
line is stripped from files so the document title does not count as a statement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from docdistance import distance as _core
from docdistance.distance import (
    DEFAULT_THRESHOLD,
    DistanceResult,
    SourceConditionedResult,
)
from docdistance.encoders import Segmenter, load_encoder


def _load_body(path: Path) -> str:
    lines = path.read_text().splitlines()
    return "\n".join(ln for ln in lines if not ln.startswith("# ")).strip()


def _read(doc: str | Path) -> str:
    """Resolve a document argument: an existing file path is read, anything else is treated as text."""
    if isinstance(doc, Path):
        return _load_body(doc)
    if isinstance(doc, str):
        try:
            p = Path(doc)
            if p.exists() and p.is_file():
                return _load_body(p)
        except OSError:
            pass
        return doc.strip()
    raise TypeError(f"document must be str or Path, got {type(doc).__name__}")


class DocDistance:
    """Reusable pipeline - construct once (models load here), then call :meth:`distance` per pair."""

    def __init__(self, backend: str = "openvino", offline: bool = True, device: str | None = None):
        self.backend = backend
        self.segmenter = Segmenter(offline=offline)
        self.encoder = load_encoder(backend, offline=offline, device=device)

    def embed(self, doc: str | Path) -> np.ndarray:
        """Segment then embed a document into L2-normalized statement vectors ``[n, dim]``."""
        statements = self.segmenter.split(_read(doc))
        if not statements:
            raise ValueError("document produced no statements")
        return self.encoder.encode(statements)

    def distance(
        self,
        a: str | Path,
        b: str | Path,
        *,
        anisotropy: bool = False,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> DistanceResult:
        return _core.compute_distance(
            self.embed(a), self.embed(b), anisotropy=anisotropy, threshold=threshold
        )

    def distance_wrt_source(
        self,
        a: str | Path,
        b: str | Path,
        source: str | Path,
        *,
        anisotropy: bool = True,
    ) -> SourceConditionedResult:
        return _core.compute_source_conditioned(
            self.embed(a), self.embed(b), self.embed(source), anisotropy=anisotropy
        )


def document_distance(
    a: str | Path,
    b: str | Path,
    *,
    backend: str = "openvino",
    anisotropy: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
    offline: bool = True,
    device: str | None = None,
) -> DistanceResult:
    """Symmetric Statement Mover's Distance between documents ``a`` and ``b`` (loads models, then scores)."""
    return DocDistance(backend=backend, offline=offline, device=device).distance(
        a, b, anisotropy=anisotropy, threshold=threshold
    )


def source_conditioned_distance(
    a: str | Path,
    b: str | Path,
    source: str | Path,
    *,
    backend: str = "openvino",
    anisotropy: bool = True,
    offline: bool = True,
    device: str | None = None,
) -> SourceConditionedResult:
    """Source-conditioned distance d(A, B | S): selection divergence + each document's distance to S."""
    return DocDistance(backend=backend, offline=offline, device=device).distance_wrt_source(
        a, b, source, anisotropy=anisotropy
    )
