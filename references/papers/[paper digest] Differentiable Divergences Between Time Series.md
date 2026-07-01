# Differentiable Divergences Between Time Series - Digest

**Authors**: Mathieu Blondel, Arthur Mensch, Jean-Philippe Vert (Google Research, Brain team; École Normale Supérieure)
**Venue**: AISTATS 2021 - 24th International Conference on Artificial Intelligence and Statistics, PMLR 130
**Source**: [Differentiable Divergences Between Time Series](https://arxiv.org/abs/2010.08354)

## Overview

Fixes the bias that stops soft-DTW from being a proper divergence. Soft-DTW is differentiable and handles variable lengths, but because of the entropic bias it can be negative and is not minimized when the two series are equal, so it cannot serve as a clean discrepancy. This paper defines the soft-DTW divergence, proves it is a valid divergence under conditions on the ground cost, and adds a sharp variant that removes the entropic bias entirely.

## Key Facts

- Soft-DTW divergence `D(x, y) = sdtw(x, y) − ½ sdtw(x, x) − ½ sdtw(y, y)` - the debiasing subtracts each series' self-similarity
- Proven a valid divergence under conditions on the ground cost - non-negative and minimized if and only if the two time series are equal (identity of indiscernibles)
- A sharp variant further removes entropic bias
- Computed by dynamic programming in the same quadratic complexity as soft-DTW
- Significant accuracy gains over both DTW and soft-DTW on time-series averaging and 84 classification datasets

## Relevance To This Project

This paper makes the E10-H58 order-gap well-posed. The hypothesis needs a monotonic-alignment cost that is zero when two statement sequences carry the same content in the same order, so that a faithful paraphrase reads near zero and only a genuine reorder registers; raw soft-DTW fails this because it is not minimized at equality. The soft-DTW divergence gives exactly the property required - non-negative, zero if and only if equal - so the structural signal `softDTW_divergence − SMD` is a proper non-negative score: order-constrained matching minus the order-free optimal-transport matching. The project records it honestly as a divergence (no triangle inequality), a structure score rather than a metric, consistent with the optimal-transport-correctness rule that an approximation is never dressed up as the exact metric.
