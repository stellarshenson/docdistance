# docdistance

Semantic distance between two documents via Statement Mover's Distance - optimal transport over mmBERT statement embeddings, after Kusner et al. 2015 (*From Word Embeddings To Document Distances*). A thin frontend to the library; the SOTA docs carry the mechanics, benchmarks, and validation.

- **Input** - two documents, raw text or a file path
- **Output** - an SMD distance, a 0..1 closeness, a verdict, and the statement alignment
- **Use** - agentic document conversion and extraction pipelines, where token logits are unavailable and KL divergence cannot be computed
- **Unit** - statement-level and position-invariant, with an interpretable transport plan

## Theory

A document distance grounded in embeddings and optimal transport, not surface overlap.

- **WMD** - Word Mover's Distance (Kusner et al. 2015) casts document similarity as optimal transport between embedded tokens
- **SMD** - this project lifts it to statements: segment, embed, transport between the two statement clouds
- **Beyond cosine** - whole-document cosine collapses when the same claims sit in a different place or order; statement-level transport is position-invariant
- **Metric** - the ground cost `√(2 − 2cos)` on L2-normalized embeddings is a metric, so the document distance is one too
- **Logit-free** - an embedding-grounded alternative where token probabilities (KL divergence) are unavailable, as in frontier-model pipelines

## Method

Three stages; the transport plan is the interpretable by-product.

1. **Segment** - split each document into atomic statements with the SAT (Segment Any Text) segmenter
2. **Embed** - encode each statement with the mmBERT contextual encoder (mean-pooled, L2-normalized)
3. **Compare** - optimal transport between the two statement clouds (Statement Mover's Distance), optionally unbalanced so added or missing statements are scored, not force-matched

- **Closeness** - `1 − SMD/√2`, on a 0..1 scale
- **Source-conditioned** - a variant `d(A, B | S)` re-bases the transport onto a shared source `S` and reads off a selection axis and a grounding axis

## Usage

The library is the product; install once, then call it.

```python
from docdistance import document_distance

result = document_distance("report_v1.md", "report_v2.md")
print(result.closeness)  # 0..1 similarity, 1 - SMD/sqrt(2)
print(result.verdict)    # "similar" | "not similar"
```

```bash
make install                                   # environment, package, Jupyter kernel
docdistance install                            # download + cache the models (once)
docdistance distance a.md b.md                 # rich, coloured verdict
docdistance distance a.md b.md --json          # machine-readable JSON
```

- **Offline after install** - distance calls run fully offline once the models are cached
- **Backend** - `--backend openvino|torch`, default `openvino` (CPU INT8)
- **Full API and flags** - `docdistance --help` and the SOTA docs

## Documentation

The SOTA documents explain how it works in detail; this README only introduces it.

- `docs/wmd-docdistance-solution-sota.md` - source-free distance: design, mechanism, performance, validation
- `docs/wmd-wrt-source-docdistance-solution.md` - source-conditioned distance `d(A,B|S)`
- `docs/mmbert-quantization-solution.md` - the INT8 / FP8 statement encoder
- `references/papers/from-word-embeddings-to-document-distances.md` - WMD paper digest (Kusner et al. 2015)

> **Note**: Scaffolded with the [copier-data-science](https://github.com/stellarshenson/copier-data-science) template.
