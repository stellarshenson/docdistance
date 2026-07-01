# WMD Structure-Distance with mmBERT (SOTA)

## Abstract

A second, structure-sensitive number beside the Statement Mover's Distance, for pipelines that must tell *content drift* from *rearrangement*. SMD is deliberately position-invariant - reorder a document's statements and it barely moves. The structure axis adds *how a document is arranged*, as a single number that is **content-invariant by construction**: a faithful reword with order preserved reads ~0, a reorder with content preserved reads large. The shipped mechanism is the **OPW order-gap** - the order-preserving Wasserstein cost minus SMD - which subtracts the content-optimal transport cost so only the cost of order violation remains. It is reported as a bounded closeness on the same `1 − d/√2` scale as SMD. The decisive evidence is batch E11, which stress-tested it against the earlier position-augmented Wasserstein metric on a true paraphrase and a second article: the metric fuses content and arrangement and reads a reword at 73.5% of a full-scramble distance, while the order-gap reads it at 0.5%. Evidence: [`experiments/wmd-structure-distance-experiments.md`](experiments/wmd-structure-distance-experiments.md), hypotheses E10-H55 (mechanism) and E11 (decision).

## Problem

SMD measures shared content but is blind to arrangement, and the use case needs a single number for arrangement that does not move when only the wording changes.

- **SMD is position-invariant by design** - it transports statement clouds, symmetric over the cloud; the same claims in a different order score as near-identical (the content axis, intentionally)
- **Agentic conversion rearranges** - PDF/HTML → markdown pipelines reorder statements, move them across sections, and re-compose blocks while preserving the content; SMD cannot see that
- **One number, not two** - the consumer is an engineer who must be told *documents differ, and differ in arrangement*; a second choose-between metric is cognitive load they cannot adjudicate, so the structure axis is a single read
- **Content-invariance is the hard requirement** - the structure number must isolate arrangement from meaning: a reword (content changed at the surface, order preserved) must read ~0, otherwise it conflates the two axes and the read is meaningless
- **The metric that was dropped** - position-augmented Wasserstein (E08-H44) is a true metric but fuses semantic and positional cost into one distance, so a faithful reword reads as large as a real reorder (E11) - a metric on the wrong quantity, not the structure number

## Solution

Read the **extra transport cost that an order constraint adds over the content-optimal plan**: zero when the content already matches in order, positive when content forces inversions.

- **Distance interpretation** - `gap = OPW − SMD`, the order-preserving Wasserstein cost minus the order-free SMD; `0` when a reword keeps order, larger as more content is reordered, read as `structure_closeness = 1 − gap/√2`
- **Content-invariant by construction** - SMD is subtracted, so the content-matching cost cancels and only the temporal prior's order penalty remains; a reword keeps order, so OPW ≈ SMD → gap ≈ 0
- **Translation-invariant** - OPW reads fractional statement rank `i/N`, not absolute position, so a uniform shift (a header inserted, everything offset, relative order intact) is invisible to it - the failure mode the absolute-position metric had
- **One number, bounded** - reported on the library's `1 − d/√2` closeness scale, `[0,1]`, the same readout as semantic closeness; the pair `(SMD-closeness, structure-closeness)` separates "how far in meaning" from "how far in arrangement"
- **Headline result** - content-invariance 0.5% of a full-scramble reading (the metric leaks 73.5%), monotone in displacement (Spearman 1.00, no collapse), translation-invariant (a pure shift reads 0.001 of range), and replicated across two articles (E11)

## Pipeline

Two documents in → two closeness numbers (semantic, structural); the structure axis reuses the SMD embeddings and one extra regularized-transport solve.

- **Segment** - `sat-3l-sm` into statements (nb 01); the transport unit
- **Embed** - mmBERT, mean-pooled and L2-normalized; the same vectors SMD uses
- **Ground cost** - `d_sem = √(2 − 2cos)`, the metric-safe chord distance (the SMD ground cost)
- **SMD** - exact OT under `d_sem` (POT `ot.emd2`), the content-optimal transport cost, uniform weights `1/n`
- **OPW** - the same content cost regularized by two temporal terms (an inverse-difference-moment reward toward the diagonal and a Gaussian temporal prior), solved by log-stabilized Sinkhorn; the order-preserving transport cost
- **Order-gap** - `gap = OPW − SMD`, the extra cost the order constraint forces beyond the content-optimal plan
- **Readout** - semantic closeness `1 − SMD/√2`, structure closeness `1 − gap/√2`, reported side by side

## Mechanism

The structure number is the gap between two transport costs over the same statement clouds: the order-free optimum (SMD) and the order-preserving optimum (OPW). Subtracting cancels the content term, isolating arrangement.

**The order-preserving transport.** OPW (Su & Hua 2017) keeps the content ground cost `D = √(2 − 2cos)` but biases the coupling toward the reading-order diagonal with two terms over fractional ranks `i/N`, `j/M`: an inverse-difference-moment reward `S` that rewards near-diagonal couplings, and a Gaussian temporal prior `P` centred on the diagonal. The coupling is the entropic-regularized optimum:

$$
\mathrm{OPW}(A, B) = \langle T^{\star}, D \rangle, \qquad
T^{\star} = \arg\min_{T \in U(a, b)} \; \langle T,\, D - S \rangle \; + \; \lambda_2 \, \mathrm{KL}\!\left(T \,\|\, P\right)
$$

with `S_{ij} = λ₁ / ((i/N − j/M)² + 1)`, `P_{ij} ∝ exp(−l(i,j)² / 2σ²)`, and `l(i,j)` the distance of `(i,j)` from the diagonal. Defaults `λ₁ = 50`, `λ₂ = 0.1`, `σ = 1.0`. The Sinkhorn kernel `K = P · exp((S − D)/λ₂)` overflows naively (`S` reaches 50 while `D ≤ √2`), so it is built in log space with a single global max subtracted - a transport-invariant rescale, so the result is the exact `λ₂ = 0.1` OPW, only numerically stable.

**The order-gap is the structure signal.** SMD is the minimum transport cost under `D`; any other coupling pays at least as much, so:

$$
\mathrm{gap}(A, B) = \mathrm{OPW}(A, B) - \mathrm{SMD}(A, B) \;\ge\; 0
$$

When the content-optimal plan is already in order (a reword, or identical content), `T^{\star} ≈` the SMD plan and `gap ≈ 0`. When content is reordered, the temporal prior forces `T^{\star}` onto content-suboptimal couplings, and the gap grows with the inversions - to 0.44 at full scramble against ~0.002 for a reword, a 200× separation.

**Why it is content-invariant.** A faithful reword changes the embeddings slightly (mmBERT cosine ~0.98), which raises SMD to ~0.2 - but it raises OPW by the *same* amount, because order is preserved and the order-optimal plan tracks the content-optimal one. The subtraction cancels that shared content cost, leaving the gap near zero. This is the property the position-augmented metric could not hold: its fused cost `√((1−λ)d_sem² + λ d_pos²)` reduces to `≈ 0.87·SMD` when order is preserved, so it tracks content drift directly.

**The bounded readout.** The gap is a difference of two 1-Wasserstein costs under a ground metric bounded by `√2`, so `gap ∈ [0, √2]` and it ships on the library's existing closeness scale:

$$
\mathrm{structure\_closeness}(A, B) = 1 - \frac{\mathrm{gap}(A, B)}{\sqrt{2}} \in [0, 1]
$$

This is a fixed analytic rescale by a constant - stateless (no batch statistics, no calibration, identical for a single pair or a million), the same convention SMD and the dropped metric use. The trade is a compressed practical range: the gap reaches only ~0.34·√2, so closeness sits in `[0.66, 1.0]`; the raw gap is the higher-resolution internal form.

**The solver - Sinkhorn, not exact EMD.** Unlike SMD, OPW is an entropic-regularized transport (the KL prior is a divergence), so the gap is a **score, not a metric** - it carries a few percent triangle violations (4.5% measured). This is registered honestly: for a pairwise *arrangement, not meaning* read the triangle inequality is never invoked, so the score-ness costs nothing the use case needs.

## Performance

Batch E11 on the two-article structure fixture: IBM exec-summaries and the Wergeland *Impact of AI on Society* curriculum (three section bases); the byte-identical reorder pool (6 displacement bins) for monotonicity, a synthetic pure-shift for translation-invariance, and a **true paraphrase** for content-invariance - opus-mt EN→DE→EN back-translation (per-statement reword, mmBERT cosine 0.98) plus same-model a/b regenerations. Full evidence in [batch E11](experiments/wmd-structure-distance-experiments.md).

| gate | OPW order-gap (E10-H55, shipped) | position-augmented Wasserstein (E08-H44, dropped) |
|---|---|---|
| content-invariance (paraphrase ÷ scramble-top) | **0.005** | 0.735 |
| translation-invariance (pure shift ÷ scramble-top) | 0.001 | 0.417 |
| scramble-monotone (Spearman, both articles) | 1.00, no collapse | 1.00, no collapse |
| triangle-inequality violations | 4.5% (score) | 0% (metric) |
| reword reads as | ~0 (0.0017 raw) | a structure change (0.149 raw) |

- **Content-invariance is the decider, and the order-gap wins it** - a true reword reads 0.5% of a full-scramble distance; the metric reads 73.5%, almost as much as a real reorder - it cannot isolate arrangement
- **Translation-invariant** - a uniform shift reads 0.001 of range (index-intrinsic); the metric fires at 0.42 (absolute position)
- **Monotone and replicated** - Spearman 1.00 rising through the top bin on both articles (scramble top-bin 0.439 IBM / 0.396 Wergeland), so the order signal is real and not article-specific
- **The honest cost** - a score, not a metric (4.5% triangle, the KL prior), and the a/b-regeneration control reads ~0.04 (independent regenerations carry genuine minor content/order differences) versus ~0.002 for a pure reword - real signal, not failure

## Why the metric was dropped

The earlier ship was E08-H44, chosen for its metricity before content-invariance could be tested. E11 tested it on a real paraphrase and reversed the decision.

- **It fuses content and arrangement** - `M̃ = √((1−λ)d_sem² + λ d_pos²)` rises on either, and with order preserved (`d_pos = 0`) it is `≈ 0.87·SMD`, so a reword (0.149) reads as large as a reorder (0.166)
- **Its metric is on the wrong quantity** - the triangle inequality is real, but it holds for *content + arrangement fused*, not arrangement; metricity does not rescue a number that cannot answer "is this rearranged?"
- **One-number design forbids the workaround** - H44 only separates arrangement when read beside SMD (two numbers, infer the difference), which is the cognitive load the single-number design rejects
- **Kept, not deleted** - H44 remains a valid content-aware distance for callers who already read SMD; it is simply not the structure number

## Setup

- **Hardware** - embed on the RTX 5000 Ada (32 GB, sm_89); OT solves on the AMD Ryzen Threadripper PRO 7975WX, single pair, ~12-15 statements per document
- **Models** - `sat-3l-sm` segmenter and `mmBERT-base` encoder (torch CUDA for the embed; the deployable path is the OpenVINO INT8 CPU encoder, identical to SMD); opus-mt EN-DE/DE-EN is a fixture-build tool only, not a serving dependency
- **Regime** - raw single-pair embeddings (no anisotropy), the production single-pair regime
- **Pipeline timed** - segment → embed → SMD (exact EMD) and OPW (Sinkhorn) on the same statement clouds, then the gap and closeness

## Methods of measurement

- **Content-invariance** - the structure reading on the paraphrase pairs (back-translation + a/b regenerations) normalized by the full-scramble top-bin mean; the deciding gate, lower is better
- **Translation-invariance** - the structure reading on a synthetic pure shift (positions translated by a constant, content and order fixed) normalized by the scramble top bin
- **Monotonicity** - Spearman `ρ(displacement, gap)` with a no-collapse check at the top bin, averaged over bases, reported per article
- **Triangle-inequality rate** - sampled triples, the fraction violating `d(X,Z) ≤ d(X,Y) + d(Y,Z)`; the metric-vs-score check
- **Latency** - per-pair Sinkhorn solve (100 iterations) plus the exact EMD for SMD

## Throughput and footprint

The structure axis adds no serving model and one extra regularized-transport solve beside SMD.

| stage | cost |
|---|---|
| ground cost `D` | `O(n_A·n_B)` numpy, negligible |
| SMD (`ot.emd2`) | exact EMD, sub-millisecond at statement scale |
| OPW (log-stabilized Sinkhorn, 100 iters) | `O(n_A·n_B)` per iteration, sub-millisecond at ~12-15 statements |
| gap + closeness | `O(1)` over the two costs |

- **No second serving model** - reuses the mmBERT statement embeddings already computed for SMD; the only added compute is the OPW Sinkhorn solve
- **Footprint** - identical to the SMD stack (`sat-3l-sm`, mmBERT INT8, POT); no new serving dependency (opus-mt is fixture-build only)

## Limitations

- **A score, not a metric** - the KL temporal prior makes the gap a divergence (4.5% triangle violations), so it cannot index, cluster, or cache at corpus scale; it is a pairwise *arrangement* read
- **The metric + invariant + monotone combo is unsolved** - a structure read that is simultaneously a metric, translation-invariant, and scramble-monotone remains open (E08-H45 had metric + translation-invariance but collapsed at full scramble)
- **Compressed closeness range** - the bounded readout `1 − gap/√2` uses only ~⅓ of the scale (closeness sits in `[0.66, 1.0]`); the raw gap is the higher-resolution form for internal thresholds
- **OPW hyperparameters fixed, not tuned** - `λ₁ = 50`, `λ₂ = 0.1`, `σ = 1.0` are the Su & Hua / E10 defaults; their principled selection on a held-out fixture is deferred
- **Single second article** - cross-fixture replication holds on IBM and one Wergeland article (three section bases); a broader article set is the next confidence step
- **Paraphrase control is a back-translation, not an LLM rewrite** - opus-mt back-translation is a faithful per-statement reword (cosine 0.98), but a genuine LLM paraphrase that also splits/merges statements is untested; the a/b-regeneration residual (~0.04) flags that near-duplicate regenerations are not perfectly zero
- **Real-conversion-pair evidence pending** - the E07-H28 question (do agentic pipelines actually restructure-while-preserving-content) needs ≥10 real pairs that do not yet exist; the structure axis is validated on the synthetic reorders and the back-translation paraphrase, and real-conversion evidence is the next confidence step
- **Displacement is ill-defined for `n ≠ m` with a real drop** - `structure_displacement` rank-transforms a bijective alignment, so when A and B differ in statement count because a statement is genuinely dropped (not reordered), the alignment is non-bijective and the per-statement displacement is not well-defined; documented as a limitation, not solved. The aggregate `order_gap` / `structure_closeness` is unaffected - the OT handles the unbalanced marginals

## FAQ

- **Why subtract SMD instead of reading OPW directly?** - OPW alone still contains the content-matching cost, so it rises on content drift; subtracting SMD cancels that shared term, leaving only the order penalty. The gap, not OPW, is the content-invariant structure signal
- **Why not the position-augmented metric (E08-H44)?** - it fuses content and position into one distance, so it inherits SMD's content sensitivity and reads a reword as large as a reorder (E11: 73.5% of a full scramble). It is a metric on *content + arrangement*, not arrangement; the order-gap isolates arrangement at the cost of metricity
- **Is it really content-invariant?** - measured: a back-translation reword reads 0.0017 (0.5% of a full scramble) across both articles, against the metric's 0.149 (73.5%); the subtraction of SMD is what cancels the content cost
- **Why is it a score and not a metric?** - OPW carries a KL temporal prior, a divergence, so the gap has no triangle inequality (4.5% violations measured). For a pairwise read this costs nothing - the triangle inequality is only invoked by corpus-scale indexing
- **Why not Gromov-Wasserstein for order?** - GW compares intra-document distance matrices and is invariant to a pure reorder (an isometry → GW = 0), so it is blind to order; its positional Fused-GW variant (E08-H45) is translation-invariant but collapses at full scramble. GW is the relational-rewrite instrument, not the order one
- **How is the score normalized - batch or curve?** - neither; `1 − gap/√2` is a fixed analytic rescale by a constant (the ground-cost ceiling), stateless and per-pair, the same convention as SMD closeness. Batch-max normalization would make the score depend on the reference set; a logistic squash would need calibration - both rejected
- **Do we recover what moved?** - yes, `structure_displacement` names the movers per statement. It is read off the crisp exact-EMD alignment (`order_alignment`, an `ot.emd` plan with an `eps` positional tie-break), not the entropic OPW plan, so a moved statement gets a sharp integer shift rather than smeared mass; the OPW plan drives only the aggregate `order_gap`

## Implementation

- **Notebooks** - `notebooks/experiments/E11-kj-structure-sota-decision.ipynb` (the H44-vs-H55 decision, the deciding gates and the verdict), `notebooks/experiments/E10-kj-structure-sensitive-distance.ipynb` (E10-H55 the mechanism, in its original batch), `notebooks/11-kj-structure-fixture.ipynb` (the two-article fixture with the back-translation paraphrase control)
- **Experiments** - `docs/experiments/wmd-structure-distance-experiments.md` (E07 the barycentric read, E08 the metric, E10 the order-gap mechanism, E11 the decision; E10-H55 is the surviving shipping mechanism)
- **Functions** - `cost_matrix` (`d_sem = √(2−2cos)`), `smd` / `ot.emd2` (the SMD solve), the OPW Sinkhorn `opw_plan` / `opw_cost` and the gap `opw_gap` (`= max(0, opw_cost − smd)`), the crisp `order_alignment` / `structure_displacement`, `closeness` (`1 − d/√2`); internals `_round_to_polytope` (Altschuler 2017 rounding, so the OPW plan is a valid coupling and the gap is genuinely `≥ 0`) and `_aligned_plan` (an `eps` positional tie-break so duplicate statements do not invent displacement); constants `OPW_LAMBDA1 = 50.0`, `OPW_LAMBDA2 = 0.1`, `OPW_SIGMA = 1.0` - all shipped in `src/docdistance/distance.py`
- **Status** - shipped. The OPW order-gap is packaged in `src/docdistance/distance.py` (`opw_gap`, with `opw_plan` / `opw_cost` and the crisp `order_alignment` / `structure_displacement`) and surfaced as `structure_closeness` via `DocDistance.structural_distance` / `.structural_distance_with_details` and the CLI `distance-structural --details-json`, which report a per-statement `displacement` and `moved` (order) beside the top-level `smd`, `order_gap`, and `structure_closeness`. The structure axis is first-class and validated on the synthetic reorders and the back-translation paraphrase (E11)
- **References** - OPW (Su & Hua 2017), order-constrained OT (Lim et al. 2022), WMD (Kusner et al. 2015), Fused Gromov-Wasserstein (Vayer et al. 2019); digests under `../references/papers/`

## Conclusions

A single structure number beside SMD that scores arrangement and is blind to wording, built from the same embeddings and one extra transport solve.

- **The order-gap is the structure axis** - `OPW − SMD` subtracts the content-optimal cost, so a reword reads ~0 and a reorder reads large; it answers *arrangement, not meaning* as one number
- **It beat the metric on the gate that matters** - content-invariance: the order-gap leaks 0.5% of a reword into the structure axis, the position-augmented metric leaks 73.5%; metricity did not survive a real paraphrase
- **Bounded and stateless** - reported as `1 − gap/√2 ∈ [0,1]`, a fixed analytic rescale on the library's closeness scale, deterministic for a single pair
- **Operating point** - content-invariant, translation-invariant, scramble-monotone, replicated across two articles; a score not a metric (4.5% triangle), which a pairwise read never penalizes
- **Status** - shipped and validated, confirmed on two articles and a back-translation paraphrase; the OPW hyperparameter selection, a broader article set, a genuine LLM-rewrite control, and real-conversion-pair evidence (E07-H28) are the open confidence steps

## Bibliography

- <span id="ref1">ref1 Su, Hua. *Order-Preserving Wasserstein Distance for Sequence Matching*. CVPR 2017</span>
- <span id="ref2">ref2 Lim, Wynter, Lim. *Order Constraints in Optimal Transport*. ICML 2022 (arXiv 2110.07275)</span>
- <span id="ref3">ref3 Kusner, Sun, Kolkin, Weinberger. *From Word Embeddings To Document Distances*. ICML 2015. Digest `../references/papers/from-word-embeddings-to-document-distances.md`</span>
- <span id="ref4">ref4 Vayer, Chapel, Flamary, Tavenard, Courty. *Fused Gromov-Wasserstein Distance for Structured Objects*. 2019/2020 (arXiv 1811.02834)</span>
- <span id="ref5">ref5 Cuturi. *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*. NeurIPS 2013</span>
- <span id="ref6">ref6 Flamary et al. *POT: Python Optimal Transport*. JMLR 2021 (`ot.emd2`, Sinkhorn)</span>
