# Order-Preserving Wasserstein Distance for Sequence Matching - Digest

**Authors**: Bing Su (Institute of Software, Chinese Academy of Sciences), Gang Hua (Microsoft Research)
**Venue**: CVPR 2017 - IEEE Conference on Computer Vision and Pattern Recognition (extended as Order-Preserving Optimal Transport, TPAMI 2018)
**Source**: [Order-Preserving Wasserstein Distance for Sequence Matching](https://openaccess.thecvf.com/content_cvpr_2017/html/Su_Order-Preserving_Wasserstein_Distance_CVPR_2017_paper.html)

## Overview

Casts the distance between two ordered sequences as an optimal-transport problem, then adds two temporal regularizers so the transport respects reading order rather than matching instances freely. Plain OT (Sinkhorn) ignores position and a free coupling can pair late items with early ones; DTW respects order strictly but breaks on local order reversals. OPW sits between them - it keeps OT's soft, variable-length matching while penalizing couplings that cross in time, and is solved by the same matrix-scaling iteration as Sinkhorn.

## Key Facts

- Views sequence instances as samples of an unknown distribution and matches them with entropic OT over a feature ground cost
- **Inverse difference moment (IDM) regularization** - rewards transport with local homogeneous structure, `1 / ((i/N − j/M)² + 1)`, concentrating mass near the temporal diagonal
- **KL-divergence prior regularization** - a prior `P_ij` (a narrow Gaussian band around `i/N ≈ j/M`) penalizes transport between far temporal positions, preventing long-range order violations
- Optimized efficiently by the matrix-scaling (Sinkhorn) algorithm; handles variable-length sequences and local temporal distortion
- Outperforms DTW variants and unregularized smoothed OT on sequence classification across several datasets
- The temporal regularizers (the KL prior in particular) make OPW a regularized divergence, not a strict metric

## Relevance To This Project

OPW is the canonical "structure equals order" optimal-transport mechanism, and the strategy table in the structure experiments already names it. It is the basis for hypothesis E10-H55, which reads the **order-gap** `OPW − SMD` - the extra transport cost the order regularizers add beyond the content-optimal SMD plan - as the structural signal: near zero when the content-optimal coupling is already monotone (a faithful paraphrase) and large when content forces order-crossing transport (a reorder). The paper's own caveat is carried forward honestly - because the KL prior makes OPW a divergence, the project registers the gap as a structure score, not a metric, never conflating it with the exact-metric position-augmented Wasserstein (E08-H44).
