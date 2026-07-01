# Kendall Tau Sequence Distance: Extending Kendall Tau from Ranks to Sequences - Digest

**Authors**: Vincent A. Cicirello (Computer Science, Stockton University)
**Venue**: 2019 technical report (arXiv 1905.02752, cs.DM; published EAI Transactions on Industrial Networks and Intelligent Systems, 2020)
**Source**: [Kendall Tau Sequence Distance: Extending Kendall Tau from Ranks to Sequences](https://arxiv.org/abs/1905.02752)

## Overview

Generalizes Kendall tau distance from permutations to arbitrary sequences over a finite alphabet, where symbols may repeat. An edit distance is the minimum-cost sequence of edit operations transforming one structure into another; on permutations, Kendall tau distance equals the number of pairwise inversions, which is the edit distance under the adjacent-swap operation (bubble-sort distance). The paper defines Kendall tau sequence distance as the minimum number of adjacent swaps to turn one sequence into another and gives two efficient algorithms.

## Key Facts

- Kendall tau on permutations = number of pairwise element inversions = adjacent-swap edit distance
- Extends to strings / sequences with repeated symbols, the case earlier partial-ranking extensions did not cover
- Defined as the minimum number of adjacent swaps to transform one sequence into the other
- Two O(n log n) algorithms provided, with open-source Java reference implementations
- A metric on sequences (the adjacent-swap edit distance satisfies the metric axioms)

## Relevance To This Project

This supplies the metric foundation for the two metric-grade hypotheses in the E10 batch. E10-H56 reads the Kendall-tau disagreement between two documents' induced source-order rank sequences (each statement assigned a soft source position via the project's coverage alignment), and E10-H57 encodes each document as a string of shared-codebook symbols in reading order and reads the Kendall-tau sequence distance between the two strings - the symbols repeat, so it is exactly the sequence case this paper solves rather than the permutation special case. Both are true metrics by this construction, the property the E08-H37 barycentric `τ`-footrule lacked (it violated the triangle inequality because the projection is not a genuine permutation). The adjacent-swap interpretation also gives the structural read a direct meaning - the count of local reorderings needed to reconcile the two arrangements.
