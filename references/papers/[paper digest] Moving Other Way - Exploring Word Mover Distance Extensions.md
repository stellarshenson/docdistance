# Moving Other Way: Exploring Word Mover Distance Extensions - Digest

**Authors**: Ilya S. Smirnov, Ivan P. Yamshchikov (LEYA Lab, Yandex; HSE University, Russia)
**Venue**: COMPLEXIS 2022 - 7th International Conference on Complexity, Future Information Systems and Risk, SciTePress, pp. 92-97 (DOI 10.5220/0011096900003197)
**Source**: [Moving Other Way: Exploring Word Mover Distance Extensions](https://arxiv.org/abs/2202.03119)

## Overview

A short position paper that probes two independent ways to improve the Word Mover's Distance without sacrificing its interpretability or cheapness: re-weighting the bag-of-words to favour rare words, and computing the metric in non-Euclidean word-vector geometries. The authors run vanilla WMD and several variants across six document-classification datasets and report that no single extension dominates - each wins on the data whose statistics match its assumption, which is the paper's main message rather than a new state-of-the-art.

## The Problem

WMD inherits whatever geometry the underlying word embeddings impose and weights every word by plain term frequency. Both choices are defaults, not derivations. Frequent words (high embedding norm) can dominate the transport even though rare words usually carry the discriminative meaning, and the Euclidean space that word2vec lives in may be a poor model for semantic similarity - hierarchical relations embed more naturally in hyperbolic space, and information-geometric spaces offer a principled cosine. The paper asks whether tilting either choice helps.

## Key Facts

- Frames WMD as the Kantorovich optimal-transport problem between two normalized bag-of-words measures with cost `c(wi,wj) = ||wi - wj||₂`
- Tests five transport variants and three embedding geometries, all on vanilla WMD machinery so differences are attributable to the change alone
- Six datasets - TWITTER, IMDB, AMAZON, CLASSIC, BBCSPORT, OHSUMED; kNN classification error, neighbours tuned on a held-out split
- Embeddings - word2vec (Google News, 300d) plus self-trained Poincaré (hyperbolic) and alpha (information-geometric) embeddings on the text8 corpus
- A position paper - it maps promising directions, it does not claim a winner

## The Two Extension Axes

The frequency axis re-weights or re-normalizes the bag-of-words so rare words count for more, motivated by the finding that embedding norm correlates with corpus frequency. Five schemes are compared: vanilla WMD; **WMD-TF-IDF** (TF-IDF weights on both bags); **WRD** - Word Rotator's Distance (Yokoi et al. 2020), which swaps the Euclidean cost for `1 - cos(wi,wj)` and scales each word's mass by its embedding norm; **OPT1**, which divides the final WMD by a coefficient rewarding rare matching words shared by both documents; and **OPT2**, which rebalances the bag with a TF-IDF-inspired `aᵢ · log(d/||wᵢ||)` factor.

The geometry axis keeps vanilla WMD but changes the ground cost: Euclidean `||wi - wj||₂`; **hyperbolic** distance on the Poincaré unit ball, `cosh⁻¹(1 + 2||wi-wj||² / ((1-||wi||²)(1-||wj||²)))`; and the **tangent space of the probability simplex**, a Fisher-information cosine `wᵢᵀI(p)wⱼ / (||wᵢ||_I ||wⱼ||_I)` from alpha embeddings.

## Results

- WMD-TF-IDF and WRD give small but consistent gains over vanilla WMD across most datasets; WRD (the cosine-plus-norm distance) is the steadiest improver
- OPT1's crude post-hoc division is the best method on OHSUMED - medical abstracts dense with rare terms - but is poor elsewhere, where frequent-word matches dominate; OPT2 is mixed
- On TWITTER and IMDB, word2vec in Euclidean space generally beats Poincaré and alpha embeddings across dimensions
- Non-Euclidean geometries surface only as outliers: alpha embeddings win on TWITTER at dimension 300, and Poincaré embeddings win on IMDB at dimension 5, hinting they capture semantics better in low dimensions

## Conclusions and Limitations

The authors conclude that word-frequency-aware weighting and better optimal-transport mechanics are the more promising directions, while non-Euclidean geometries need large samples and more study before any verdict. The study is explicitly preliminary - small subsamples of two datasets for the geometry comparison, a single hyperparameter `α = 1` for the simplex cost, and self-trained embeddings on a modest corpus - so the numbers are illustrative rather than conclusive.

## Relevance To This Project

The paper is a catalogue of the exact levers `docdistance` already pulls or could pull. Its strongest finding - that the cosine-based Word Rotator's Distance (`1 - cos`, mass scaled by embedding norm) is the most reliable improvement over Euclidean WMD - validates this project's choice of a cosine-grounded statement cost `√(2 - 2cos)` rather than raw Euclidean distance. The frequency-weighting result (rare words carry the discriminative signal) is the word-level analogue of the project's coverage and selection axes that up-weight statements a source uniquely supports. The non-Euclidean experiments are out of scope here - `docdistance` operates on whitened mmBERT statement embeddings, not word2vec - but the paper's central caution carries over: an extension only helps where the data's statistics match its assumption, so any added axis must be justified per-corpus, not adopted wholesale.
