# WMD Structure-Distance with mmBERT (SOTA)

## Abstract

A second, structure-sensitive number beside the Statement Mover's Distance, for pipelines that must tell *content drift* from *rearrangement*. SMD is deliberately position-invariant - reorder a document's statements and it barely moves. The structure axis adds *how a document is arranged*, as a single number that is **content-invariant by construction**: a faithful reword with order preserved reads ~0, a reorder with content preserved reads large. The shipped mechanism is the **OPW order-gap** - the order-preserving Wasserstein cost minus SMD - which subtracts the content-optimal transport cost so only the cost of order violation remains. It is reported as a bounded closeness on the same `1 − d/√2` scale as SMD. The decisive evidence is batch E11, which stress-tested it against the earlier position-augmented Wasserstein metric on a true paraphrase and a second article: the metric fuses content and arrangement and reads a reword at 61.5% of a full-scramble distance, while the order-gap reads it at 0.4%. Evidence: [`experiments/wmd-structure-distance-experiments.md`](../experiments/wmd-structure-distance-experiments.md), hypotheses E10-H55 (mechanism) and E11 (decision).

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
- **Headline result** - content-invariance 0.4% of a full-scramble reading (the metric leaks 61.5%), monotone in displacement (Spearman 1.00, no collapse), translation-invariant (a pure shift reads 0.001 of range), and replicated across two articles (E11)

## Pipeline

Two documents in → two closeness numbers (semantic, structural); the structure axis reuses the SMD embeddings and one extra regularized-transport solve.

- **Segment** - `sat-3l-sm` into statements (nb 01); the transport unit
- **Embed** - mmBERT, mean-pooled and L2-normalized; the same vectors SMD uses
- **Ground cost** - `d_sem = √(2 − 2cos)`, the metric-safe chord distance (the SMD ground cost)
- **SMD** - exact OT under `d_sem` (POT `ot.emd`), the content-optimal transport cost, uniform weights `1/n`
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

**The solver - Sinkhorn, not exact EMD.** Unlike SMD, OPW is an entropic-regularized transport (the KL prior is a divergence), so the gap is a **score, not a metric** - it carries a substantial triangle-violation rate (15% measured on the E11 fixtures, `reports/E11-structure-sota-decision.json` `.triangle.H55`). This is registered honestly: for a pairwise *arrangement, not meaning* read the triangle inequality is never invoked, so the score-ness costs nothing the use case needs.

## Performance

Batch E11 on the two-article structure fixture: IBM exec-summaries and the Wergeland *Impact of AI on Society* curriculum (three section bases); the byte-identical reorder pool (6 displacement bins) for monotonicity, a synthetic pure-shift for translation-invariance, and a **true paraphrase** for content-invariance - opus-mt EN→DE→EN back-translation (per-statement reword, mmBERT cosine 0.98) plus same-model a/b regenerations. Full evidence in [batch E11](../experiments/wmd-structure-distance-experiments.md).

| gate | OPW order-gap (E10-H55, shipped) | position-augmented Wasserstein (E08-H44, dropped) |
|---|---|---|
| content-invariance (paraphrase ÷ scramble-top) | **0.004** | 0.615 |
| translation-invariance (pure shift ÷ scramble-top) | 0.001 | 0.459 |
| scramble-monotone (Spearman, both articles) | 1.00, no collapse | 1.00, no collapse |
| triangle-inequality violations | 15% (score) | 0% (metric) |
| reword reads as | ~0 (0.0011 raw) | a structure change (0.132 raw) |

- **Content-invariance is the decider, and the order-gap wins it** - a true reword reads 0.4% of a full-scramble distance; the metric reads 61.5%, more than half of a full reorder - it cannot isolate arrangement
- **Translation-invariant** - a uniform shift reads 0.001 of range (index-intrinsic); the metric fires at 0.46 (absolute position)
- **Monotone and replicated** - Spearman 1.00 rising through the top bin on both articles (scramble top-bin 0.416 IBM / 0.396 Wergeland), so the order signal is real and not article-specific
- **The honest cost** - a score, not a metric (15% triangle, the KL prior), and the a/b-regeneration control reads ~0.02 (independent regenerations carry genuine minor content/order differences) versus ~0.001 for a pure reword - real signal, not failure

## Why the metric was dropped

The earlier ship was E08-H44, chosen for its metricity before content-invariance could be tested. E11 tested it on a real paraphrase and reversed the decision.

- **It fuses content and arrangement** - `M̃ = √((1−λ)d_sem² + λ d_pos²)` rises on either, and with order preserved (`d_pos = 0`) it is `≈ 0.87·SMD`, so a reword (0.132) reads as large as a reorder between the third and the fourth of the six displacement bins (0.107 and 0.148 on the ibm article)
- **Its metric is on the wrong quantity** - the triangle inequality is real, but it holds for *content + arrangement fused*, not arrangement; metricity does not rescue a number that cannot answer "is this rearranged?"
- **One-number design forbids the workaround** - H44 only separates arrangement when read beside SMD (two numbers, infer the difference), which is the cognitive load the single-number design rejects
- **Kept, not deleted** - H44 remains a valid content-aware distance for callers who already read SMD; it is not the structure number

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
| SMD (`ot.emd`) | exact EMD, sub-millisecond at statement scale |
| OPW (log-stabilized Sinkhorn, 100 iters) | `O(n_A·n_B)` per iteration, sub-millisecond at ~12-15 statements |
| gap + closeness | `O(1)` over the two costs |

- **No second serving model** - reuses the mmBERT statement embeddings already computed for SMD; the only added compute is the OPW Sinkhorn solve
- **Footprint** - identical to the SMD stack (`sat-3l-sm`, mmBERT INT8, POT); no new serving dependency (opus-mt is fixture-build only)

## Calibration of the structural verdict

The verdict compares `structure_closeness` with `STRUCTURE_THRESHOLD`: `similar` at or above it, `not similar` below it. The shipped value `0.90` was placed on the ibm ladder of the stored fixture statements, encoded with the torch backend (E12-H59), and holds on that corpus only: on the wergeland ladder it mis-sides rung 2 (worked example below). On the ibm summary files the command removes the title line and encodes with OpenVINO INT8 by default, and reads 3 of 55 un-reordered pairs below `0.90` with `input_format="plain"` and 4 with the default reader, lowest 0.897057; see the bullet "Readings of this document and the reader of a file" under Limitations (`DEF-DOCS-18`). A different corpus needs its own value, passed as `--threshold` / `threshold=`.

**Per-corpus procedure.**

1. Build the k-swap ladder of the corpus: reorder each base document with a rising number of statement swaps, group the reorders into rungs by displacement, and read the mean `order_gap` of each rung over its reorders and base documents. The fixture uses swap counts 0 to 40 with 14 seeds each, grouped into 6 rungs
2. Collect the un-reordered pairs of the same corpus: pairs that differ in wording or in content selection and carry no deliberate reorder, such as independent summaries of one source and paraphrases
3. State which rungs must read `not similar` (on the fixture: rung 2 and deeper; rung 1 is a light reorder that must read `similar`), convert every reading to `structure_closeness = 1 − order_gap/√2`, and sweep the threshold from 0.85 to 0.99 in steps of 0.005
4. Take the band that keeps every rung and every un-reordered pair on its side. Its lower edge is the highest closeness among the rungs that must read `not similar`, and the edge itself is excluded. Its upper edge is the lowest closeness among the un-reordered pairs and the rungs that must read `similar`, and the edge itself is included. Set the threshold inside the band. An empty band means no threshold separates the two sets on that corpus

**Worked example - the E12-H59 sweep over two articles.** Closeness values from `reports/E12-H59-structure-threshold-transfer-metrics.json`:

| article | rung 1 | rung 2 | lowest un-reordered pair | band |
|---|---|---|---|---|
| ibm | 0.970245 | 0.892817 | 0.906144 (of 66 pairs) | (0.892817, 0.906144] |
| wergeland | 0.976651 | 0.908048 | 0.973676 (of 6 pairs) | (0.908048, 0.973676] |

- **ibm** - the swept thresholds 0.895, 0.900 and 0.905 keep every rung and every un-reordered pair on its side; `0.90` has 0.006144 of headroom below the lowest un-reordered pair
- **wergeland** - the swept thresholds 0.910 to 0.970 do; `0.90` reads rung 2 (closeness 0.908048) as `similar`, so it mis-sides 1 of the 12 rungs of the two ladders. It flags 0 of the 72 un-reordered pairs
- **Cross-article band** - empty: the wergeland lower edge 0.908048 lies 0.001904 above the ibm upper edge 0.906144, and 0 of the 29 swept thresholds is valid on both articles
- **Scope** - two articles; the wergeland upper edge rests on 6 un-reordered pairs

**Verdict under deletions.** `order_gap` is computed with uniform marginals on both documents. When the statement counts differ, the order-preserving plan places the surplus mass off the diagonal, so a deletion that keeps the order of the remaining statements raises `order_gap` (E12-H62, 280 order-preserving deletions of 1 to 4 statements on 10 base documents of 11 to 14 statements). The reference for a pair whose order is kept is 0.005, the 95th percentile of `order_gap` on the paraphrase pairs:

- **Reading** - median `structure_closeness` 0.9493 / 0.9052 / 0.9141 / 0.8902 at 1 / 2 / 3 / 4 deleted statements, pooled over head, tail and interior deletions, and a median of 0.8219 for 3 statements deleted at the tail. Both 0.8902 and 0.8219 are below `0.90`, so a content drop without any reorder can read `not similar`
- **What ships for it** - `matched_order_gap`, reported beside `order_gap` from the release after 1.1.5: the order gap over the statements both documents hold (the one-to-one pairs of `partial_matching`, ground cost below `MATCH_CUTOFF` 0.325). On the same 280 deletions it reads at or below 0.005 in 280 of 280 cases, largest reading 6.101e-04, where `order_gap` reads above 0.005 in 280 of 280 (E13-H70). The value is the identity floor of the shorter document: `opw_gap(B, B)` up to float32 rounding of the zero costs (at most 7.813e-06 measured, E15-H79)
- **What does not change** - `order_gap`, `structure_closeness` and the verdict keep their 1.1.5 values. The verdict is read from `order_gap` only, so the caveat above still applies to it. No length correction of `order_gap` ships
- **Reading the two together** - an order-preserving deletion with the surviving statements unchanged reads `n_matched` equal to the statement count of the shorter document and `matched_order_gap` at the identity floor, whatever `structure_closeness` reads. Deleted-only cases against reordered-and-deleted cases separate at AUC 0.999903 on `matched_order_gap` and 0.950980 on `order_gap` (E13-H70)
- **Scope of the 280-case reading** - documents of 11 to 14 statements without repeated statements. The limits measured outside it are in [Limitations](#limitations)

## Limitations

- **A score, not a metric** - the KL temporal prior makes the gap a divergence (15% triangle violations), so it cannot index, cluster, or cache at corpus scale; it is a pairwise *arrangement* read
- **The metric + invariant + monotone combo is unsolved** - a structure read that is simultaneously a metric, translation-invariant, and scramble-monotone remains open (E08-H45 had metric + translation-invariance but collapsed at full scramble)
- **Compressed closeness range** - the bounded readout `1 − gap/√2` uses only ~⅓ of the scale (closeness sits in `[0.66, 1.0]`); the raw gap is the higher-resolution form for internal thresholds
- **OPW hyperparameters fixed, not tuned** - `λ₁ = 50`, `λ₂ = 0.1`, `σ = 1.0` are the Su & Hua / E10 defaults; their principled selection on a held-out fixture is deferred
- **Single second article** - cross-fixture replication holds on IBM and one Wergeland article (three section bases); a broader article set is the next confidence step
- **Paraphrase control is a back-translation, not an LLM rewrite** - opus-mt back-translation is a faithful per-statement reword (cosine 0.98), but a genuine LLM paraphrase that also splits/merges statements is untested; the a/b-regeneration residual (~0.04) flags that near-duplicate regenerations are not perfectly zero
- **Real-conversion-pair evidence pending** - the E07-H28 question (do agentic pipelines restructure-while-preserving-content in the field) needs ≥10 real pairs that do not yet exist; the structure axis is validated on the synthetic reorders and the back-translation paraphrase, and real-conversion evidence is the next confidence step
- **Displacement is ill-defined for `n ≠ m` with a real drop** - `structure_displacement` rank-transforms a bijective alignment, so when A and B differ in statement count through a dropped statement rather than a reordered one, the alignment is non-bijective and the per-statement displacement is not well-defined. Since 1.1.5 the reporting layer flags this case per statement: when the aligned pair's ground cost exceeds `0.6` (`no_counterpart_cost`) the record nulls `displacement_norm`, sets `moved = false` and raises `no_counterpart` - the ill-defined value is withheld, not invented. `matched_displacement` is the rank shift among the matched statements only, `null` for a statement without a match, and `moved` is read on it. The aggregate `order_gap` / `structure_closeness` is affected as well: a deletion that keeps the order raises `order_gap` (see [Calibration of the structural verdict](#calibration-of-the-structural-verdict))
- **Readings of this document and the reader of a file** - the readings on the structure fixture, which include the calibration of `0.90` (E12-H59), were taken on the stored statements of `data/processed/structure-fixture/statements.json`: `notebooks/11-kj-structure-fixture.ipynb` segmented each ibm summary per blank-line paragraph and kept its title line as statement 0, which the package removes from a file with every reader, so they are the output of no reader (`DEF-DOCS-18`). On the 55 pairs of ibm summary files the command reads 3 pairs below `0.90` with `input_format="plain"` and 4 with the default reader, the lowest at 0.897057, where the calibration reads the lowest un-reordered ibm pair at 0.906144 and flags 0 of 72 pairs. After 1.1.5 the default reader of a `.md` file is the markdown reader. On the 55 ibm pairs the default call against `input_format="plain"`: the order gap differs in 27 pairs by at most 0.0329, `structure_closeness` in 26 pairs by at most 0.0232, `matched_order_gap` in 11 pairs by at most 0.0326, and 1 of 55 verdicts at `0.90` changes (`reports/E13-campaign/checks/default-reader/diff.py`). The 11 summaries hold a title line and prose only, so these 55 pairs are not a bound for a `.md` file with list, heading or table lines; the ibm source article reads 48 statements with the default reader and 70 with the plain reader, and no order reading of such a document under both readers is stored. `--input-format plain` restores the 1.1.5 reading of a file; it does not return the fixture readings
- **Identity floor of the order gap** - `opw_gap(P, P)` is 0.0 for one statement and not zero otherwise, and it grows with the statement count: at most 8.583e-04 on the 10 base documents of 11 to 14 statements, 5.182e-03 at 30 statements and 6.786e-03 at 53 (E13-H70, one document), medians of 0.00655 and 0.00995 at 300 statements (E13-H74). `matched_order_gap` of a pure deletion reads that floor of the shorter document, so the paraphrase reference 0.005 is exceeded from about 30 statements: 63 of 120 pure deletions on prefixes of the 53-statement document read above 0.005
- **The order gap grows with the count of displaced statements and does not level off beyond the kernel reach** - the OPW kernel is exactly 0.0 beyond 54 positions at 300 statements, and the order gap keeps growing past that reach: with one block of 10 statements moved by 5 / 50 / 60 / 200 positions the median reads 0.03228 / 0.12482 / 0.14324 / 0.41615 (E13-H74, Refuted: the prediction was a change of at most 5% from 60 to 200, measured +190.5%). The reading is linear in the count of displaced statements (slope 0.5833 per displaced fraction, R² 0.999): the 10 block statements cost 0.02023 at 10 positions and 0.02027 at 200, and the growth comes from the statements the block passes (0.11992 → 0.39295)
- **`matched_order_gap` reads the matched statements only** - a statement without a pair below `MATCH_CUTOFF` leaves the reading. On the 66 pairs of unequal statement count that the structure fixture holds without a constructed edit (51 of them pairs of ibm summaries) a mean of 4.68 statements per pair is matched, against 8.24 statements in the shorter document; 19 pairs hold at most one matched pair and the median `matched_order_gap` is 0.0000 (E15-H79). On independently written documents it therefore reads the order of a few statements and does not replace `order_gap`. It is not bounded by `order_gap`: it reads above it in 15 of those 66 pairs, and the [worked example](../example-map-interpretations.md) reads 0.3168 over 4 matched statements against an `order_gap` of 0.2185
- **Repeated statements, the 5% rank step and the cutoff margin** - three limits of the one-to-one matching and of `moved`, measured in E15-H79 and E15-H80 and not solved; the readings are in the [API reference](../api-reference.md#matching-and-the-matched-order-gap)

## FAQ

- **Why subtract SMD instead of reading OPW directly?** - OPW alone still contains the content-matching cost, so it rises on content drift; subtracting SMD cancels that shared term, leaving only the order penalty. The gap, not OPW, is the content-invariant structure signal
- **Why not the position-augmented metric (E08-H44)?** - it fuses content and position into one distance, so it inherits SMD's content sensitivity and reads a reword as large as a reorder (E11: 61.5% of a full scramble). It is a metric on *content + arrangement*, not arrangement; the order-gap isolates arrangement at the cost of metricity
- **Is it really content-invariant?** - measured: a back-translation reword reads 0.0011 (0.4% of a full scramble) across both articles, against the metric's 0.132 (61.5%); the subtraction of SMD is what cancels the content cost
- **Why is it a score and not a metric?** - OPW carries a KL temporal prior, a divergence, so the gap has no triangle inequality (15% violations measured). For a pairwise read this costs nothing - the triangle inequality is only invoked by corpus-scale indexing
- **Why not Gromov-Wasserstein for order?** - GW compares intra-document distance matrices and is invariant to a pure reorder (an isometry → GW = 0), so it is blind to order; its positional Fused-GW variant (E08-H45) is translation-invariant but collapses at full scramble. GW is the relational-rewrite instrument, not the order one
- **How is the score normalized - batch or curve?** - neither; `1 − gap/√2` is a fixed analytic rescale by a constant (the ground-cost ceiling), stateless and per-pair, the same convention as SMD closeness. Batch-max normalization would make the score depend on the reference set; a logistic squash would need calibration - both rejected
- **Do we recover what moved?** - yes, `structure_displacement` names the movers per statement. It is read off the crisp exact-EMD alignment (`order_alignment`, an `ot.emd` plan with an `eps` positional tie-break), not the entropic OPW plan, so a moved statement gets a sharp integer shift rather than smeared mass; the OPW plan drives only the aggregate `order_gap`. After 1.1.5 the `moved` flag is read on the one-to-one matched statements, not on that alignment: a statement is `moved` when it is matched, its rank among the matched statements changed (`matched_displacement != 0`), `|matched_displacement / n_matched| > 0.05` and its record is not `no_counterpart` (E15-H80). A statement inserted or removed without a match changes no rank and flags nothing. On a document that repeats a statement this does not always hold: `moved` flagged a kept statement in 4 of 120 pure deletions of a 62-statement document with 9 repeated statements (E15-H80; bullet "Limit: repeated statements" of the [API reference](../api-reference.md#matching-and-the-matched-order-gap))

## Implementation

- **Notebooks** - `notebooks/experiments/E11-kj-structure-sota-decision.ipynb` (the H44-vs-H55 decision, the deciding gates and the verdict), `notebooks/experiments/E10-kj-structure-sensitive-distance.ipynb` (E10-H55 the mechanism, in its original batch), `notebooks/11-kj-structure-fixture.ipynb` (the two-article fixture with the back-translation paraphrase control)
- **Experiments** - `docs/experiments/wmd-structure-distance-experiments.md` (E07 the barycentric read, E08 the metric, E10 the order-gap mechanism, E11 the decision; E10-H55 is the surviving shipping mechanism)
- **Functions** - `cost_matrix` (`d_sem = √(2−2cos)`), `smd` / `ot.emd` (the SMD solve), the OPW Sinkhorn `opw_plan` / `opw_cost` and the gap `opw_gap` (`= max(0, opw_cost − smd)`), the crisp `order_alignment` / `structure_displacement`, `closeness` (`1 − d/√2`), the one-to-one matching `partial_matching` and the order gap over its pairs `matched_order_gap` (scores that depend on a calibrated cutoff, not metrics); internals `_round_to_polytope` (Altschuler 2017 rounding, so the OPW plan is a valid coupling and the gap is `≥ 0` by construction) and `_aligned_plan` (an `eps` positional tie-break so duplicate statements do not invent displacement); constants `OPW_LAMBDA1 = 50.0`, `OPW_LAMBDA2 = 0.1`, `OPW_SIGMA = 1.0`, `STRUCTURE_THRESHOLD = 0.90`, `MATCH_CUTOFF = 0.325`, `MATCH_POSITION_EPS = 1e-6` - all shipped in `src/docdistance/distance.py`
- **Status** - shipped. The OPW order-gap is packaged in `src/docdistance/distance.py` (`opw_gap`, with `opw_plan` / `opw_cost` and the crisp `order_alignment` / `structure_displacement`) and surfaced as `structure_closeness` via `DocDistance.structural_distance` / `.structural_distance_with_details` and the CLI `distance-structural --details-json`. Since 1.1.5 each per-statement record carries the raw `displacement` beside the length-normalized `displacement_norm = target_index/n_b − index/n_a`; the record also names the aligned pair's ground `cost`, the content-optimal plan's `semantic_target_index` with `targets_differ` (true = moved AND reworded), and `no_counterpart` for the drop case - all beside the top-level `smd`, `order_gap`, and `structure_closeness`. After 1.1.5 the result also carries `matched_order_gap` and `n_matched`, each record carries `matched_displacement`, and `moved` reads `matched AND matched_displacement != 0 AND |matched_displacement / n_matched| > 0.05 AND NOT no_counterpart` (`moved_tolerance` 0.05, E15-H80) in place of the 1.1.5 rule `|displacement_norm| > 0.05`; `moved` and `n_moved` are the only shipped fields whose values changed. The structure axis is first-class and validated on the synthetic reorders and the back-translation paraphrase (E11)
- **References** - OPW (Su & Hua 2017), order-constrained OT (Lim et al. 2022), WMD (Kusner et al. 2015), Fused Gromov-Wasserstein (Vayer et al. 2019); digests under `../references/papers/`

## Conclusions

A single structure number beside SMD that scores arrangement and is blind to wording, built from the same embeddings and one extra transport solve.

- **The order-gap is the structure axis** - `OPW − SMD` subtracts the content-optimal cost, so a reword reads ~0 and a reorder reads large; it answers *arrangement, not meaning* as one number
- **It beat the metric on the gate that matters** - content-invariance: the order-gap leaks 0.4% of a reword into the structure axis, the position-augmented metric leaks 61.5%; metricity did not survive a real paraphrase
- **Bounded and stateless** - reported as `1 − gap/√2 ∈ [0,1]`, a fixed analytic rescale on the library's closeness scale, deterministic for a single pair
- **Operating point** - content-invariant, translation-invariant, scramble-monotone, replicated across two articles; a score not a metric (15% triangle), which a pairwise read never penalizes
- **Status** - shipped and validated, confirmed on two articles and a back-translation paraphrase; the OPW hyperparameter selection, a broader article set, a genuine LLM-rewrite control, and real-conversion-pair evidence (E07-H28) are the open confidence steps

## Bibliography

- <span id="ref1">ref1 Su, Hua. *Order-Preserving Wasserstein Distance for Sequence Matching*. CVPR 2017</span>
- <span id="ref2">ref2 Lim, Wynter, Lim. *Order Constraints in Optimal Transport*. ICML 2022 (arXiv 2110.07275)</span>
- <span id="ref3">ref3 Kusner, Sun, Kolkin, Weinberger. *From Word Embeddings To Document Distances*. ICML 2015. Digest `../references/papers/from-word-embeddings-to-document-distances.md`</span>
- <span id="ref4">ref4 Vayer, Chapel, Flamary, Tavenard, Courty. *Fused Gromov-Wasserstein Distance for Structured Objects*. 2019/2020 (arXiv 1811.02834)</span>
- <span id="ref5">ref5 Cuturi. *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*. NeurIPS 2013</span>
- <span id="ref6">ref6 Flamary et al. *POT: Python Optimal Transport*. JMLR 2021 (`ot.emd`, Sinkhorn)</span>
