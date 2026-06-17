# docdistance-estimator

Compute a meaningful semantic distance between two documents using Word Mover's Distance (WMD) and Optimal Transport, following Kusner et al. 2015 (*From Word Embeddings To Document Distances*). The intended use is agentic document conversion and extraction pipelines that run through frontier models, where token-level logits are unavailable and KL divergence cannot be computed directly - WMD provides an embedding-grounded distance instead.

## Why this exists

Whole-document cosine similarity is too coarse when two executive summaries carry the same claims in different places, in a different order, or with content added or dropped. Optimal transport lifts the comparison to individual statements: it matches each statement in one document to its best counterpart in the other regardless of position, and the transport plan itself reveals what moved, what was added, and what was dropped.

## Approach

The distance is computed in stages:

1. **Segment** each document into atomic statements with the SAT (Segment Any Text) sentence segmenter
2. **Embed** each statement with a contextual encoder
3. **Compare** the two statement clouds with optimal transport (statement-level Word Mover's Distance), optionally unbalanced so added or missing statements are scored explicitly rather than force-matched

Position-invariance, statement-level granularity, and an interpretable alignment are what distinguish this from a single document embedding.

## Validation

The distance is implemented in `notebooks/04-kj-wmd-document-distance.ipynb` and validated on an executive-summary fixture set built from one IBM AI-adoption article - a gold tier (faithful summaries written under shared rules) plus two adversarial tiers (information loss and information noise). Statement Mover's Distance ranks every gold summary closer to the anchor than every adversarial one with zero ordering mistakes. The design and conclusions are in `docs/wmd-docdistance-solution.md`, with a source-conditioned variant (`d(A,B|S)`) in `docs/wmd-wrt-source-docdistance-solution.md`.

## Notebooks

- `notebooks/01-kj-document-segmentation.ipynb` - stage 1: splits a source PDF into statements with the `sat-3l-sm` model (PyTorch, GPU), writing `data/interim/01-statements.parquet`
- `notebooks/02-kj-mmbert-quantization.ipynb` - stage 0: quantizes the mmBERT statement encoder and emits the best model per target (CPU OpenVINO INT8, GPU torchao FP8)
- `notebooks/03-kj-mmbert-throughput-saturation.ipynb` - GPU batch-saturation sweep for the encoder (throughput knee, per-core CPU optimum)
- `notebooks/04-kj-wmd-document-distance.ipynb` - stage 3: the statement-level distance (WCD / RWMD / SMD), scored across the fixture set and validated against the gold anchor

## Encoder quantization performance

The mmBERT statement encoder is quantized for two deployment targets. All rows normalized to CPU FP as the 1.0x baseline (full detail in `docs/mmbert-quantization-solution.md`; shipped CPU model: [`stellars/mmBERT-base-openvino-int8`](https://huggingface.co/stellars/mmBERT-base-openvino-int8)).

| config | ms/sentence | sentences/sec | speedup |
|---|---|---|---|
| CPU FP (base, full precision) | 30.6 | 33 | 1.0x |
| CPU OpenVINO INT8 | 21.4 | 47 | 1.4x |
| GPU bf16 eager (raw base) | 0.84 | 1196 | 37x |
| GPU bf16 compiled | 0.44 | 2281 | 70x |
| **GPU FP8 + compile** | **0.39** | **2588** | **79x** |

GPU FP8 is ~2.2x over the raw GPU base and ~55x over the shipped CPU INT8 model, at near-lossless fidelity (GPU 0.999, CPU 0.98 vs FP32). GPU rows are throughput at batch 128 / seq 128; CPU rows are per-sentence latency at small batch, so the cross-device multiple is directional, not a like-for-like benchmark.

## Quick Start

```bash
make install
```

## Makefile Targets

- `make install` - Create environment and install package
- `make test` - Run tests
- `make lint` / `make format` - Check / fix code style
- `make build` - Build distributable wheel
- `make clean` - Remove compiled files and caches
- `make .env` / `make .env.enc` - Decrypt / encrypt environment secrets
- `make help` - Show all available targets

## Best Practices

- **Notebooks**: Name with number prefix, initials, description - `01-jqp-data-exploration.ipynb`
- **Data**: Keep `raw/` immutable, use `interim/` for transforms, `processed/` for final datasets
- **Source code**: Refactor reusable notebook code into `src/docdistance_estimator/` modules
- **Models**: Store trained models in `models/` with clear naming

## References

- `references/papers/from-word-embeddings-to-document-distances.md` - digest of the WMD paper (Kusner et al. 2015)
- `docs/wmd-docdistance-solution.md` - source-free distance design, implementation, and results
- `docs/wmd-wrt-source-docdistance-solution.md` - source-conditioned distance design (`d(A,B|S)`)

## Project Organization

```
├── Makefile           <- Makefile with convenience commands
├── README.md          <- The top-level README for developers
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump
│
├── models             <- Trained and serialized models
├── notebooks          <- Jupyter notebooks
├── pyproject.toml     <- Project configuration and dependencies
├── references         <- Data dictionaries, manuals, explanatory materials
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures
├── tests              <- Test files
└── src
    └── docdistance_estimator   <- Source code for this project
        ├── __init__.py
        ├── config.py      <- Configuration variables
        ├── dataset.py     <- Data download/generation scripts
        ├── features.py    <- Feature engineering code
        ├── modeling
        │   ├── predict.py <- Model inference
        │   └── train.py   <- Model training
        └── plots.py       <- Visualization code
```

> **Note**: Scaffolded with the [copier-data-science](https://github.com/stellarshenson/copier-data-science) template.
