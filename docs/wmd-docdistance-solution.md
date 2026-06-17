# WMD Document Distance with mmBERT Embeddings

Design for measuring semantic distance between two documents by adapting Word Mover's Distance (Kusner et al. 2015, digest in `references/papers/from-word-embeddings-to-document-distances.md`) to use mmBERT embeddings instead of word2vec. The paper's optimal-transport formalism is kept verbatim; only the embedded unit and the embedding model change.

## Method

Each document is a weighted point cloud; the distance is the minimum transport cost to morph one cloud into the other - the paper's Earth Mover's Distance, unchanged.

- **Mapping** - swap the unit and the embedding, keep the transport

| WMD (paper) | this design |
|---|---|
| point = word | point = statement (from nb 01) or pooled word |
| embedding `x_i` = word2vec | `x_i` = mmBERT embedding (quantized encoder) |
| weight `d_i` = nBOW term frequency | `d_i` = uniform `1/n` (or length / salience) |
| cost `c(i,j) = ‖x_i − x_j‖` | `c(i,j) = ‖x_i − x_j‖` on L2-normalized mmBERT |
| solve EMD LP → WMD | same LP → Statement Mover's Distance |

- **Transport LP** - minimize `Σ T_ij · c(i,j)` subject to `Σ_j T_ij = d_i`, `Σ_i T_ij = d'_j`, `T ≥ 0`
- **Solver** - exact EMD for small clouds; entropic Sinkhorn (Cuturi 2013, cited by the paper) for speed and a differentiable variant
- **Output** - scalar distance plus the transport plan `T`, which is the statement-to-statement alignment

## Unit choice

The single decision that shapes the rest - what a "point" is.

- **Statements** (recommended) - cloud of statement embeddings from nb 01; matches the segmenter + encoder already built, position-invariant statement alignment, few points so exact OT is cheap, serves the "claims in different places" use case
- **Words** - pool mmBERT subwords to per-word vectors; literal Word Mover's Distance, more points, slower, less aligned to the use case
- **Hierarchical (HOTT)** - document → statement → word, two-level OT where the statement-to-statement cost is itself a mini-WMD; most faithful, heaviest

## Ground cost

Use cosine semantics, realized as Euclidean on L2-normalized embeddings - this preserves the metric property the paper requires (WMD is a metric only if the ground cost is a metric).

- **Identity** - on unit vectors `‖x − y‖ = √(2 − 2·cos(x,y))`, so Euclidean-on-normalized ranks pairs identically to cosine

| ground cost | cosine ranking | metric | use |
|---|---|---|---|
| `1 − cos` | yes | no | avoid |
| `2 − 2cos` (squared Euclidean) | yes | no | avoid |
| `√(2 − 2cos)` (Euclidean on normalized) | yes | yes | recommended |
| `arccos(cos)` (angular) | yes | yes | valid, costlier |

- **Rule** - `1 − cos` and squared Euclidean break the triangle inequality, so they void the metric guarantee even though they rank like cosine

## Lower bounds

The paper's bounds carry over and gain a clean interpretation with statement embeddings.

- **WCD** - `‖Xd − Xd'‖`, the distance between the two documents' mean-pooled embeddings; with uniform weights this is exactly the whole-document-cosine baseline, now the cheap floor (`O(dp)`)
- **RWMD** - relax one constraint so each statement moves to its nearest counterpart; a greedy statement alignment, tight lower bound for pruning (`O(p²)`)

## Why it generalizes whole-document cosine

The whole-document cosine baseline is not discarded - it is the single cheapest lower bound of this exact transport, so the design strictly subsumes it.

- **WCD = document cosine** - with uniform statement weights, `‖Xd − Xd'‖` is the distance between the two mean-pooled document embeddings, the baseline already judged too coarse for executive summaries with claims in different places
- **SMD = its refinement** - the full transport recovers the statement-level structure mean-pooling discards; WCD is the floor, Statement Mover's Distance is the answer
- **RWMD = greedy alignment** - the one-sided relaxation is a fast nearest-statement matching, the prefetch-and-prune bound sitting between WCD and exact SMD
- **Clean drop-in** - the optimal-transport machinery is the paper's verbatim; only the embedded unit (statement vs word) and the embedding model (mmBERT vs word2vec) change

## Design decisions

- **Unit** - statements / words / hierarchical
- **Cost** - Euclidean on L2-normalized mmBERT (metric-safe cosine)
- **Weights** - uniform `1/n` vs statement-length or salience
- **Balanced vs unbalanced OT** - exact WMD moves all mass; unbalanced leaves added or dropped statements unmatched for a penalty, yielding an omission / hallucination signal
- **Direction** - one symmetric distance, or directional precision (every output statement supported) and recall (every source statement covered)

## Recommended configuration

- Statement-level units from nb 01
- Euclidean cost on L2-normalized mmBERT embeddings (quantized encoder, FP8 on GPU / OpenVINO INT8 on CPU)
- Uniform weights
- Unbalanced OT, transport plan exposed as the alignment
- WCD + RWMD as prefetch-and-prune lower bounds

## Implementation and results

Implemented in `notebooks/04-kj-wmd-document-distance.ipynb` - segment with `sat-3l-sm`, embed with mmBERT (mean-pooled, L2-normalized), SMD by exact optimal transport - and validated on the eleven executive-summary fixtures against the gold anchor.

- **Perfect tier ordinality** - zero ordering mistakes: every gold summary is closer to the reference gold than every adversarial one; the two 3-sweep golds are nearest, the gold tier holds 73-82% closeness, the adversarial sets 68-72%
- **One distance, one verdict** - SMD is the answer, reported as a distance plus a bounded closeness `1 − SMD/√2` (1 = identical, 0 = orthogonal clouds) and a similar / not-similar call against a threshold set by the gold cluster's own spread
- **Bounds nest in value and cost** - `WCD ≤ RWMD ≤ SMD` holds throughout; exact SMD is ~0.08 ms/pair at statement scale and entropic Sinkhorn is slower here, so exact OT is the default and Sinkhorn is dropped
- **Boundary margin is modest** - the gap at the gold/adversarial boundary is ~0.01 SMD (mean ratio ~1.15x); the ordering is clean and correct, just not wide
- **What SMD responds to** - the two adversarial sets are different rules with no expected order between them, and both land outside the gold band; SMD responds to shared statement content, so the info-noise set (which kept the source's numbers) sits slightly closer than the info-loss set (which stripped them) - numeric retention correlates with closeness
- **Next** - the source-conditioned variant in `wmd-wrt-source-docdistance-solution.md` adds a grounding axis (is added content supported by the source) on top of this selection-based distance

## Status

- Implemented and validated in `notebooks/04-kj-wmd-document-distance.ipynb`
- Pending - lift the distance into reusable `src/docdistance_estimator/distance.py` (WCD, RWMD, exact SMD) for the pipeline
