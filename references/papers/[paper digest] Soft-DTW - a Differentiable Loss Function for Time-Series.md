# Soft-DTW: a Differentiable Loss Function for Time-Series - Digest

**Authors**: Marco Cuturi (CREST, ENSAE, Université Paris-Saclay), Mathieu Blondel (NTT Communication Science Laboratories)
**Venue**: ICML 2017 - 34th International Conference on Machine Learning, PMLR 70
**Source**: [Soft-DTW: a Differentiable Loss Function for Time-Series](https://arxiv.org/abs/1703.01541)

## Overview

Turns dynamic time warping into a differentiable loss by softening its minimum. DTW compares two time series by the minimum-cost monotonic alignment between them, computed by dynamic programming; it handles variable lengths and is robust to shifts and dilations, but the hard minimum makes it non-differentiable and prone to bad optima as a loss. Soft-DTW replaces the minimum over alignments with a soft-minimum (LogSumExp at temperature `γ`), giving a smooth objective whose value and gradient both compute in quadratic time.

## Key Facts

- DTW = `min` over all monotonic alignment paths of the path cost; soft-DTW = soft-min over the same paths, summing the influence of every alignment
- The soft-min is `−γ log Σ exp(−·/γ)`; as `γ → 0` soft-DTW recovers exact DTW, larger `γ` is smoother
- Value and gradient computable in quadratic time and space by dynamic programming
- Alignment is order-preserving by construction - a warping path is monotone non-decreasing in both indices, so it cannot cross in time
- Soft-DTW is not a true distance - it can be negative and is not minimized when the two series are equal (the bias the follow-up divergence paper fixes)

## Relevance To This Project

The monotonic warping path is exactly the order constraint that Statement Mover's Distance lacks - SMD's optimal transport matches statements by content with no regard for sequence, so a reorder leaves it unchanged, whereas a DTW-style alignment must respect reading order and therefore pays when statements move. Hypothesis E10-H58 uses this order-preserving alignment cost on the position-indexed statement-embedding sequences (with the project's `√(2 − 2cos)` ground cost) and reads `softDTW − SMD` as the order penalty. The non-distance caveat here is the reason H58 does not use raw soft-DTW directly but the soft-DTW divergence variant (companion paper, Blondel-Mensch-Vert 2021), which restores a clean zero at equality so a faithful paraphrase scores near zero.
