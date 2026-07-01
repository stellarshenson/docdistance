# All-but-the-Top: Simple and Effective Postprocessing for Word Representations - Digest

**Authors**: Jiaqi Mu, Pramod Viswanath (University of Illinois at Urbana-Champaign)
**Venue**: ICLR 2018
**Source**: [All-but-the-Top: Simple and Effective Postprocessing for Word Representations](https://arxiv.org/abs/1702.01417)

## Overview

The paper introduces a simple, counter-intuitive postprocessing for off-the-shelf word embeddings: subtract the common mean vector and project away the few top dominant principal directions - "all-but-the-top". This renders representations more isotropic and consistently improves both intrinsic and extrinsic tasks across embedding methods, hyperparameters, and languages - word similarity, concept categorization, analogy, semantic textual similarity, and text classification all gain.

## The Observation

Every embedding tested (word2vec, GloVe, RAND-WALK, TSCCA, CBOW, Skip-gram) shares two geometric defects that distort the semantic geometry.

- **Large common mean** - all word vectors share a nonzero mean `µ` whose norm is roughly 1/6 to 1/2 of the average word-vector norm, so the cloud is off-center
- **Anisotropy** - after removing the mean, variance concentrates in a few dominant directions: the normalized PCA spectrum decays near-exponentially over the first directions (~10 of d=300) then stays flat, so a handful of directions dominate every vector
- **Implication** - all words share the same common vector and the same dominant directions, which sway every representation identically and carry no discriminative signal

## The Algorithm

A three-line postprocessing applied once to the whole vocabulary (Algorithm 1).

1. Compute the mean `µ = (1/|V|) Σ_w v(w)` and center: `ṽ(w) = v(w) − µ`
2. Take the PCA components `u_1 … u_d` of the centered vectors
3. Remove the top-D directions: `v'(w) = ṽ(w) − Σ_{i=1..D} (u_iᵀ v(w)) u_i`

- **One hyperparameter** - only D, the number of directions nulled
- **Rule of thumb** - `D ≈ d/100` works uniformly across languages, embeddings, and tasks (about 3 for the standard d=300, though the spectrum shows roughly 10 visibly dominant directions)
- **Counter-intuitive** - ordinary denoising removes the weakest directions; here removing the strongest ones is what purifies the geometry

## Why It Works

The dominant directions are an artifact, not signal, and removing them restores the isotropy the embedding's own generative model assumes.

- **Top directions encode frequency** - the leading PCA coefficients correlate with a word's unigram probability, so the nulled directions carry frequency, not meaning
- **Isotropy / self-normalization** - under the RAND-WALK model (Arora et al. 2016) the partition function `Z(c) = Σ_w exp(cᵀv(w))` should be roughly constant for any unit `c`; the paper measures isotropy as `I = min_c Z(c) / max_c Z(c) ∈ [0,1]`
- **The two steps are the isotropy fix** - a first-order approximation of `I = 1` forces zero mean (step 1); a second-order approximation forces a flat singular spectrum (step 2, removing the top directions)
- **Measured effect** - isotropy rises sharply: word2vec 0.70 → 0.95, GloVe 0.065 → 0.60

## Results

Postprocessing helps consistently and does not hurt on average.

- **Word similarity** - +1.7% average over 7 datasets
- **Concept categorization** - +2.8%, +4.5%, +4.3% on three datasets
- **Word analogy** - +0.5% semantic, +0.2% syntactic, +0.4% total (smaller, because the analogy subtraction already cancels part of the common component)
- **Semantic textual similarity** - +4% average over 21 datasets (sentence = averaged word vectors, cosine scored)
- **Text classification** - CNN + 3 RNNs over 2 embeddings and 5 datasets improves on 34 of 40 instances, +2.85% average

## Limitations and Notes

- D must be set; `d/100` is a heuristic and the optimal D varies with the representation, its dimension, and the downstream task
- The operation is on dense low-dimensional embeddings (positive and negative entries), distinct from earlier top-component removal on positive cooccurrence matrices
- Related: Arora et al. 2017 (SIF) nulls only the first principal component but computes it per-dataset; all-but-the-top removes vocabulary-wide directions instead

## Relevance To This Project

This is the reference behind the anisotropy-removal step in the statement-distance pipeline. mmBERT statement embeddings are anisotropic exactly like word2vec and GloVe - pairwise cosines bunch at 0.7-0.9, compressing the `√(2 − 2cos)` transport cost matrix and flattening the distance. Subtracting the dominant principal component from the pooled statement embeddings and re-L2-normalizing de-bunches the cosines and widens the distance dynamic range (experiment E01: DR 0.057 → 0.180, a 3.2x gain at zero ordinality violations), while preserving the metric (re-normalized vectors keep Euclidean = metric-safe cosine). The "top directions encode frequency" finding explains why it helps here: the shared dominant direction carries no discriminative content, so removing it is pure signal gain.
