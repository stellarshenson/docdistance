# Fused Gromov-Wasserstein Distance for Structured Objects - Digest

**Authors**: Titouan Vayer, Laetitia Chapel, Rémi Flamary, Romain Tavenard, Nicolas Courty (Univ. Bretagne-Sud / Univ. Côte d'Azur / Univ. Rennes, CNRS, IRISA)
**Venue**: 2019 (arXiv 1811.02834, stat.ML; published in Algorithms, MDPI)
**Source**: [Fused Gromov-Wasserstein Distance for Structured Objects](https://arxiv.org/abs/1811.02834)

## Overview

Defines a single distance that compares structured objects on both their features and their internal structure at once. The Kantorovich-Wasserstein distance compares the features of elements but treats them independently; the Gromov-Wasserstein distance compares only the relations between elements (the structure) and discards the features. Fused Gromov-Wasserstein (FGW) combines the two through a weight `α ∈ [0,1]`, and the paper proves it is a genuine distance with interpolation and geodesic properties.

## Key Facts

- Wasserstein term = feature cost `M_ij` between elements; Gromov term = structure-distortion functional `Σ_ijkl T_ij T_kl |C_A[i,k] − C_B[j,l]|^q` over intra-object relation matrices `C_A`, `C_B`
- FGW(α) interpolates - `α=0` recovers Wasserstein (features only), `α=1` recovers Gromov-Wasserstein (structure only)
- Proves metric properties, interpolation between W and GW, geodesic properties, and a finite-sample concentration result
- GW compares relations, so it is invariant to isometries of the structure space - a relabeling or reordering that preserves the pairwise relation matrix leaves GW unchanged
- The discrete case reduces to a non-convex quadratic program over the coupling

## Relevance To This Project

FGW is the theoretical anchor for two hypotheses. It explains the E08-H45 result on data: positional Fused-GW collapses at extreme reorder because a reorder of identical statements is an isometry of the position line, so the re-optimized GW coupling escapes to the trivial isometric solution and the structure cost falls back toward zero. Hypothesis E10-H54 keeps the same Gromov structure-distortion functional but **pins the coupling to the SMD content plan** instead of re-optimizing it, so the isometry escape is structurally impossible - the content correspondence is forced, and its positional distortion becomes a translation-invariant, non-collapsing order signal. The paper's GW term is exactly the quantity H54 computes; its proof that GW reads relations, not absolute position, is precisely the translation-invariance the project wants and the reason re-optimized GW is the wrong tool for order.
