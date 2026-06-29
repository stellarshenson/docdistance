# WMD Structure-Distance with mmBERT (SOTA)

## Abstract

A second, structure-sensitive metric beside the Statement Mover's Distance, for pipelines that must tell *content drift* from *rearrangement*. SMD is deliberately position-invariant - reorder a document's statements and it barely moves. This doc adds a structural axis by fusing a positional channel into the transport ground cost: **position-augmented Wasserstein**, the exact optimal transport between the two statement clouds under a cost that combines the semantic chord distance with normalized statement position. It is a true metric (0% triangle violations), monotone in the amount of rearrangement, wide-dynamic-range, defined on unequal-length pairs, and rides the same embeddings and solver as SMD - no second model. The structural distance reduces to SMD at λ=0, so the two are reported side by side: semantic (how far in meaning) and structural (how far in arrangement). The transport plan is the structural mapping - which statement moved where. Evidence: batch E08, [`experiments/wmd-structure-distance-experiments.md`](experiments/wmd-structure-distance-experiments.md), hypothesis E08-H44.

## Problem

SMD measures shared content but is blind to arrangement, and the use case needs both.

- **SMD is position-invariant by design** - it transports statement clouds, symmetric over the cloud; the same claims in a different order score as near-identical (the content axis, intentionally)
- **Agentic conversion rearranges** - PDF/HTML → markdown pipelines reorder statements, move them across sections, and re-compose blocks while preserving the content; SMD cannot see that
- **A metric is required** - the structural distance must satisfy the triangle inequality to threshold, rank and cache like SMD, and stay interpretable enough to name what moved
- **The barycentric read is not a metric** - the first attempt (E07, the τ-from-`T` disorder) recovers the order signal but is a barycentric projection: 4% triangle violations and a narrow, saturating range that ties a model-free greedy-1-NN baseline

## Solution

Add a positional channel to the transport ground cost, so the same exact-OT machinery yields a structure-sensitive metric.

- **Distance interpretation** - exact transport between the two statement clouds under a cost that fuses semantic distance and normalized position; `0` for identical arrangement and content, larger = more rearranged or more divergent, read as `closeness = 1 − d/√2`
- **Generalizes SMD** - at λ=0 the cost is purely semantic and the distance *is* SMD; λ>0 turns on order-sensitivity, so the two distances are the same object at two settings of one knob
- **Both axes side by side** - report SMD (semantic, order-invariant) and position-augmented Wasserstein (structure-aware, order-sensitive); when content is preserved (SMD ≈ 0) but the structural distance is large, the arrangement changed
- **Structural mapping recovered** - the transport plan couples A's statements to B's; each statement's induced target position, read off the plan, names the movers - the structural analogue of the content transport map
- **Headline result** - a true metric (0% triangle violations against the τ-read's 4%), monotone in displacement (Spearman 1.00), dynamic range scaling with the positional weight λ to 8.8 at λ=0.5, and defined on unequal-length pairs (E08-H44)

## Pipeline

Two documents in → two closeness numbers (semantic, structural) and a structural mapping; the structure axis reuses the SMD embeddings and solver.

- **Segment** - `sat-3l-sm` into statements (nb 01); the transport unit
- **Embed** - mmBERT, mean-pooled and L2-normalized; the same vectors SMD uses
- **Position** - each statement carries its normalized index `pos_i = i/(n−1) ∈ [0,1]`, comparable across documents of any length
- **Semantic cost** - `d_sem = √(2 − 2cos)`, the metric-safe chord distance (the SMD ground cost)
- **Structural cost** - fuse semantic and positional into one metric `M̃ = √((1−λ)·d_sem² + λ·d_pos²)`, with `d_pos = |pos_i − pos_j|`
- **Distances** - SMD = exact OT under `d_sem` (λ=0); position-augmented Wasserstein = exact OT under `M̃` (λ>0); both via POT `ot.emd2`, uniform weights `1/n`
- **Readout** - semantic closeness `1 − SMD/√2`, structural closeness `1 − d/√2`, and the transport plan as the statement-to-statement mapping with per-statement displacement

## Mechanism

The structural distance is optimal transport on statements lifted to (embedding, position) points; the fused ground cost is a metric by construction, so the Wasserstein distance is a metric, and the solve stays exact.

**The positional channel.** SMD's blindness to order is structural: its ground cost reads only the embedding. Attach each statement's normalized position as a second coordinate and measure it with the absolute-difference metric `d_pos(i,j) = |pos_i − pos_j|`, `pos_i = i/(n−1)`. Normalizing to `[0,1]` makes position comparable across documents of different length - the property the equal-length rank statistics (footrule) lack, and the reason this distance is defined on real, unequal-length conversion pairs.

**The fused cost is a metric.** Combine the semantic and positional metrics in `ℓ2` with a trade-off `λ ∈ [0,1]`:

$$
\tilde{M}(i, j) = \sqrt{(1-\lambda)\, d_{\text{sem}}(i, j)^{2} + \lambda\, d_{\text{pos}}(i, j)^{2}}
$$

For any two metrics on the same points and non-negative weights, this `ℓ2` combination is itself a metric: it is the Euclidean norm of the vector `(√(1−λ)·d_sem, √λ·d_pos)`, so the triangle inequality follows from Minkowski's inequality, and it is zero only when both coordinates agree (same content and same position). Each statement is thus a point in the product space (unit embedding, position), and `M̃` is a genuine metric on that space.

**The distance.** With that cost, the structural distance is the minimum-cost plan moving one cloud's uniform mass onto the other's - identical in form to SMD, only the ground cost differs:

$$
d_{\lambda}(A, B) = \min_{T \ge 0} \sum_{i=1}^{n_A} \sum_{j=1}^{n_B} T_{ij}\, \tilde{M}(i, j)
\quad \text{s.t.} \quad
\sum_{j} T_{ij} = \tfrac{1}{n_A}, \;\; \sum_{i} T_{ij} = \tfrac{1}{n_B}
$$

Because `M̃` is a metric, `d_λ` is a `1`-Wasserstein distance on distributions over the product space, hence a true metric - measured at **0% triangle violations**. At `λ = 0` the cost collapses to `d_sem` and `d_λ = SMD` exactly (reproduced to `1.3e-12`), so SMD is the semantic special case and `d_λ` the order-sensitive generalization.

**Why the range is wide.** The positional cost scales with displacement magnitude: a statement matched to a far-away position pays `|Δpos|`, continuously, so `d_λ` grows smoothly with how far content moved rather than saturating. This is the property the rank normalizers lack - the normalized footrule and off-diagonal mass tie the naive baseline at full scramble (range ratio ~6-7, and saturating), while `d_λ`'s range ratio scales with `λ` to 8.8 at `λ = 0.5`.

**The structural mapping (interpretability).** The transport plan `T` is the structural alignment. For each A-statement `i`, its induced target position is the transport-weighted mean of where its mass lands:

$$
\tau(i) = \frac{\sum_{j} T_{ij}\, \text{pos}_j}{\sum_{j} T_{ij}}, \qquad \delta(i) = \tau(i) - \text{pos}_i
$$

`δ(i)` is the displacement of statement `i` - the statements with large `|δ|` are the movers ("the intro claim at position 0.05 is matched in B near 0.80, displaced +0.75"). So the structural number arrives with a per-statement account of what moved where, the structural analogue of SMD's `--transport-map-json`. The mapping is read in `O(n)` off the already-computed plan, no second alignment.

**The solver - exact EMD.** As with SMD this is a linear program, solved exactly by the network-simplex (`ot.emd2`), returning the true optimal cost and a sparse plan; no entropic blur, no `ε`, deterministic. The structural distance is convex and globally optimal - in contrast to the Fused Gromov-Wasserstein alternative, a non-convex quadratic program whose Gromov term makes it both harder to solve and, empirically, prone to collapse at extreme rearrangement (see FAQ).

## Performance

Batch E08 on the structure fixture (7 summary bases, 12-14 statements each; 6 displacement bins × 12 seeds for the reorder sweep; the byte-identical reorder upper bound plus 11 cross-summary diffuse pairs). Full evidence in [batch E08](experiments/wmd-structure-distance-experiments.md).

| measure | position-augmented Wasserstein (E08-H44, shipped) | τ-from-`T` footrule (E07, prior) |
|---|---|---|
| triangle-inequality violations | 0% | ~4% |
| metric | yes | no (barycentric projection) |
| monotone in displacement (Spearman) | 1.00, zero inversions | 1.00 (by construction) |
| dynamic-range ratio | 6.7 (λ=0.25), 8.8 (λ=0.5) | 6.4 |
| SMD recovered at λ=0 | yes (max `|Δ|` 1.3e-12) | n/a |
| defined on unequal-length pairs | yes | only as a soft barycentric position |

- **It is a metric, the τ-read is not** - the decisive win is the triangle inequality (0% vs 4%), so the structural distance thresholds, ranks and caches like SMD
- **Wide, non-saturating range** - the range ratio scales with the positional weight `λ`; at `λ = 0.5` it reaches 8.8, clearing both E07 normalizers (footrule 6.4, off-diagonal mass 7.4)
- **Two-axis separation** - on a reordered-but-faithful pair SMD stays ≈ 0 (content preserved) while `d_λ` rises with the reorder, so the pair `(SMD, d_λ)` separates "what changed" from "how it is arranged"
- **The rejected alternatives** - positional Fused-GW (E08-H45) is a metric but collapses at extreme reorder; the optimal-assignment footrule (E08-H46) is a metric only on equal-length pairs; the displacement-weighted anti-monotone mass (E08-H48) is wide-range but not a metric (6.7% violations); the offline anisotropy lever (E08-H47) is inert on an already-sharp `T`

## Setup

- **Hardware** - embed on the RTX 5000 Ada (32 GB, sm_89); OT solves on the AMD Ryzen Threadripper PRO 7975WX, single pair, ~12-14 statements per document
- **Models** - `sat-3l-sm` segmenter and `mmBERT-base` encoder (torch CUDA for the embed; the deployable path is the OpenVINO INT8 CPU encoder, identical to SMD)
- **Regime** - raw single-pair embeddings (no anisotropy step), the production single-pair regime; the anisotropy lever was tested (E08-H47) and found inert because raw `T` is already concentrated
- **Pipeline timed** - segment → embed → SMD (λ=0) and position-augmented Wasserstein (λ>0) on the same plan machinery

## Methods of measurement

- **Dynamic-range ratio** - `(d at full scramble − d at one adjacent swap) / paraphrase-floor sd`, the resolution preserved across displacement; reported per `λ`
- **Triangle-inequality rate** - sampled triples `(X, Y, Z)`, the fraction violating `d(X,Z) ≤ d(X,Y) + d(Y,Z)` within numerical tolerance
- **Monotonicity** - Spearman `ρ(displacement, d_λ)` with inversion count across the 6 bins, mean over 12 seeds per bin
- **SMD recovery** - `max |d_{λ=0} − SMD|` over all fixture pairs
- **Latency** - per-pair OT solve time, exact EMD, the same measurement as SMD

## Throughput and footprint

The structure axis adds no model and reuses the SMD solver; its cost is one extra exact-OT solve plus `O(n)` arithmetic for the mapping.

| stage | cost |
|---|---|
| structural cost matrix `M̃` | `O(n_A·n_B)` numpy, negligible |
| position-augmented Wasserstein (`ot.emd2`) | ~0.08 ms/pair at 12×12, same LP as SMD |
| structural mapping (`τ`, `δ`) | `O(n)` over the plan |

- **No second model** - reuses the mmBERT statement embeddings already computed for SMD; the only added compute is one more `ot.emd2` solve under `M̃`
- **Same scaling as SMD** - exact EMD is `O(n³ log n)`; at statement scale (~12-55 points) it is sub-millisecond, so the encoder dominates end-to-end exactly as for SMD
- **Footprint** - identical to the SMD stack (`sat-3l-sm`, mmBERT INT8, POT); no new dependency

## Limitations

- **λ selection deferred** - the semantic/positional balance `λ` is a design knob (headline `λ = 0.25`); its principled selection is deferred to a second-article fixture, since the single-article cohort offers no held-out split
- **Structure-aware, not a decoupled pure axis** - `d_λ` fuses content and arrangement in one number (it rises on either), and is read beside SMD to separate them; a fully decoupled pure-structure metric that is *also* wide-range and defined on unequal-length pairs remains open - the pure reads tested were non-metric (τ-footrule, anti-mass) or equal-length-only (optimal-assignment footrule)
- **Absolute-position sensitivity** - `d_pos` penalizes absolute position, so a pure shift (an insertion at the top that displaces every later statement) reads as disorder; Fused-GW's translation-invariance is the property to graft, but FGW collapses at extreme reorder, so it is not yet usable
- **Single-article fixture, upper bound** - every number is on one IBM article's synthetic perturbations and the byte-identical reorder upper bound; the result is *promising*, not *confirmed*, until it replicates on a second article and on genuinely diffuse unequal-length conversion pairs
- **Programme gate still open** - the real-conversion-pair kill-gate (E07-H28: do agentic pipelines actually restructure-while-preserving-content) needs ≥10 real pairs that do not yet exist; the structural axis ships only if that gate passes

## FAQ

- **Why not Gromov-Wasserstein for order?** - GW compares intra-document distance matrices and is invariant to a pure reorder (an isometry → GW = 0), so it is blind to order. Rebuilding it as a *positional* Fused-GW (E08-H45) makes it rise mid-range and stay translation-invariant, but it **collapses at full scramble** as the Gromov isometry solution reasserts - a metric, but not a monotone order measure. GW is the relational-rewrite instrument, not the order one
- **Why not the footrule or a rank statistic?** - the barycentric τ-footrule is not a metric (4% triangle violations) and saturates; the optimal-assignment footrule *is* a metric but only on equal-length pairs, undefined when a conversion changes the statement count. Position-augmented Wasserstein is a metric *and* defined on unequal-length pairs
- **Is it really a metric?** - yes: the `ℓ2` combination of two metrics is a metric (Minkowski), and the `1`-Wasserstein distance on a metric ground cost is a metric; measured 0% triangle violations
- **Do we recover the structural mapping plan?** - yes: the transport plan is the statement-to-statement structural alignment, and each statement's induced position `τ(i)` and displacement `δ(i)` name the movers; the readout is `O(n)` over the already-computed plan
- **Why a positional channel, not a second model?** - it reuses the SMD embeddings and the same exact-OT solver, adds no model and no second alignment, and stays the same `O(n³ log n)` cost as SMD
- **Why exact EMD, not Sinkhorn?** - identical to SMD: at statement scale exact EMD is faster and unbiased, and it returns the sparse plan the mapping needs; the entropic blur would smear the very displacements we read

## Implementation

- **Notebooks** - `notebooks/experiments/E08-kj-structure-distance-metric.ipynb` (batch E08, the five structural mechanisms and their verdicts), `notebooks/12-kj-structure-distance-e2e.ipynb` (the end-to-end showcase: two documents → semantic and structural distance + the structural mapping)
- **Experiments** - `docs/experiments/wmd-structure-distance-experiments.md` (E07 the barycentric read, E08 the metric formulation; E08-H44 is the surviving mechanism)
- **Functions** - `cost_matrix` (`d_sem = √(2−2cos)`), `smd` / `ot.emd2` (the OT solve), `closeness` (`1 − d/√2`); the structural cost `M̃` and the induced-position read are a few lines of numpy over the shipped plan, not yet packaged in `src/`
- **Status** - the structural axis is a confirmed experiment result (E08-H44), not yet wired into the `docdistance` pipeline or CLI; production integration (a `structure_distance` beside `document_distance`, and a structural transport-map JSON) is the next step, gated on the E07-H28 real-pair precondition
- **References** - WMD (Kusner et al. 2015), Fused Gromov-Wasserstein (Vayer et al. 2019), Gromov-Wasserstein (Mémoli 2011), Order-Preserving OT (Su & Hua 2017); digests under `../references/papers/`

## Conclusions

A second metric beside SMD that scores arrangement, built from the same embeddings and solver.

- **One object, two settings** - SMD is `d_λ` at `λ = 0` (semantic, order-invariant); position-augmented Wasserstein is `d_λ` at `λ > 0` (structure-aware, order-sensitive); reported together they separate content drift from rearrangement
- **A true metric** - `ℓ2`-fused ground cost → Wasserstein metric, 0% triangle violations, monotone, wide-dynamic-range, and defined on unequal-length pairs - the properties the barycentric τ-read and the rank statistics could not all hold at once
- **Interpretable** - the transport plan is the structural mapping; each statement's induced position and displacement name what moved, `O(n)` over the plan
- **Operating point** - exact, metric, deterministic, no second model; one extra `ot.emd2` solve on the SMD embeddings, sub-millisecond at statement scale
- **Honest status** - promising on one article and the byte-identical upper bound; the `λ` selection, the cross-fixture replication, and the real-conversion-pair gate (E07-H28) are the open items before it ships

## Bibliography

- <span id="ref1">ref1 Kusner, Sun, Kolkin, Weinberger. *From Word Embeddings To Document Distances*. ICML 2015. Digest `../references/papers/from-word-embeddings-to-document-distances.md`</span>
- <span id="ref2">ref2 Vayer, Chapel, Flamary, Tavenard, Courty. *Fused Gromov-Wasserstein Distance for Structured Objects*. 2019/2020 (arXiv 1811.02834)</span>
- <span id="ref3">ref3 Mémoli. *Gromov-Wasserstein Distances and the Metric Approach to Object Matching*. Foundations of Computational Mathematics, 2011</span>
- <span id="ref4">ref4 Su, Hua. *Order-Preserving Wasserstein Distance for Sequence Matching*. CVPR 2017</span>
- <span id="ref5">ref5 Mu, Viswanath. *All-but-the-Top: Simple and Effective Postprocessing for Word Representations*. ICLR 2018</span>
- <span id="ref6">ref6 Flamary et al. *POT: Python Optimal Transport*. JMLR 2021 (`ot.emd2`, `ot.fused_gromov_wasserstein2`)</span>
