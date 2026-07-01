# Speeding up Word Mover's Distance and its variants via properties of distances between embeddings - Digest

**Authors**: Matheus Werner, Eduardo Laber (Departamento de Informática, PUC-Rio, Rio de Janeiro, Brazil)
**Venue**: arXiv:1912.00509v2 [cs.CL], 8 May 2020 (ECAI 2020)
**Source**: [Speeding up Word Mover's Distance and its variants via properties of distances between embeddings](https://arxiv.org/abs/1912.00509)

## Overview

The paper speeds up WMD and its relaxation RWMD by exploiting one empirical fact about word embeddings: every word is close to only a handful of others, and the distances to all the rest cluster tightly around a single large value. Treating those far distances as one constant `cmax` collapses the dense transportation problem into a sparse one and shrinks the distance cache from quadratic to linear in vocabulary size. The resulting Rel-WMD and Rel-RWMD match the test error of the originals while running several times faster across ten datasets.

## The Problem

RWMD already cuts WMD's super-cubic transport solve down to `O(|D|·|D'|)`, but its real bottleneck is the ground-cost computation `O(|D|·|D'|·d)` and the cache that avoids recomputing it. Caching all pairwise word distances costs `O(n²)` memory, which is prohibitive for a large vocabulary `n` - and each distance evaluation itself is hundreds of operations for typical embedding dimension `d`. For applications that compute the same word-pair distance repeatedly (kNN classification), caching is essential yet infeasible at scale. The paper attacks this memory-and-recompute wall rather than the transport solve itself.

## Key Facts

- Builds on Kusner et al.'s WMD and its relaxation RWMD `max(Σ Dᵢ minⱼ c, Σ D'ⱼ minᵢ c)` - the two one-sided transport relaxations
- Empirical observation - distances between word2vec embeddings (Google News, 300d) concentrate near `[1.2, 1.4]`, roughly Normal; only a few words sit close to any given word
- Assumption - split the vocabulary around each word `w` into `RELATED(w)` (the `r` nearest) and `UNRELATED(w)`, with every unrelated distance set to one constant `cmax`
- `cmax` is the average of all the non-cached distances; the cache `C` keeps only the `r` nearest words per word, `O(n·r)` space instead of `O(n²)`
- Ten datasets, two tasks; methods implemented in C++ (Eigen, OR-Tools) on a single i7-6700 core; code at `github.com/matwerner/fast-wmd`

## How The Speed-up Works

A preprocessing pass builds the cache `C`: for each word it finds the `r` closest words and their distances (expected-linear-time QuickSelect), routes the rest into an accumulator that yields `cmax`. For large vocabularies the `O(n²·d)` build is itself reduced by first k-means clustering the embeddings and only searching within a cluster, with the optimal cluster count `k = √(n/I)` derived by minimizing total preprocessing time.

With the cache in place, two new distances follow. **Rel-WMD** keeps the full transportation problem but replaces the swarm of `cmax`-cost edges with a single bucket variable per word (`Tᵢ,ₜ` and `Tₜ,ⱼ`), so the bipartite graph is sparse for small `r` and the solve is much faster. **Rel-RWMD** applies the same cost structure to the RWMD relaxation and computes in `O((|D|+|D'|)·r)` time by hashing each document and walking only each word's `r` related entries - a large gain over RWMD's `O(|D|·|D'|)` when documents are bigger than `r`. The linear-time RWMD(L) cache matrix also fills faster and sparser under the assumption.

## Results

- Document classification (kNN, eight datasets) - Rel-RWMD's average test error 19.75% is on par with RWMD (19.94%) and WMD (20.26%), and beats Cosine (23.36%) and Word Centroid Distance (24.59%)
- Rel-RWMD(L) is on average 4.7x faster than RWMD(L) - the second fastest - driven by a ~10x faster cache-matrix build
- Related-document triplets (ARXIV, WIKIPEDIA) - Rel-RWMD(S) holds the error and runs 3x (Wikipedia) to 27x (Arxiv) faster than RWMD(S); after preprocessing, distance evaluation alone is 25-60x faster
- A small `r` suffices - test error at `r ≥ 16` matches the cross-validated `r`, and even `r = 1, 2` stays close, so `r` can be fixed rather than tuned

## Relevance To This Project

This paper sits in the WMD lower-bound and relaxation lineage that `docdistance` already carries as WCD and RWMD, and its core insight - embedding distances are structured, so most of the cost matrix is redundant and can be pruned or constant-folded - is the same reasoning behind those cheap bounds. The direct speed mechanics are less applicable here because `docdistance` runs exact optimal transport (`ot.emd2`) over a small set of statement embeddings per document, not over a large word vocabulary, so the quadratic-cache wall this paper removes is not the project's bottleneck. The transferable idea is the relaxation-as-lower-bound discipline and the empirical-geometry argument for when an approximation is safe: should statement counts ever grow large enough that exact OT hurts, Rel-RWMD-style sparsification of the cost matrix is the principled next lever, kept honest as a documented lower bound rather than conflated with exact distance.
