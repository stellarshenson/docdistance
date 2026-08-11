# Improving Sequence-to-Sequence Learning via Optimal Transport (2019) - Digest

**Source**: [Improving Sequence-to-Sequence Learning via Optimal Transport](https://arxiv.org/abs/1901.06283)

## Overview

Sequence-to-sequence models are trained by maximum likelihood estimation, which scores the next word given the previous ground-truth words. That objective is word-level: it models local syntactic patterns and has no term that preserves the meaning of the sequence as a whole. The paper adds a sequence-level supervision built on optimal transport, computing a soft matching between the generated sequence's word embeddings and the reference's, and penalising the transport cost between them. The authors show the resulting objective can be read as a Wasserstein gradient flow driving the model distribution towards the ground-truth sequence distribution. The OT term is used as a regulariser alongside cross-entropy rather than replacing it. Evaluated across machine translation, abstractive summarization and image captioning, it produces consistent but modest gains over strong baselines, without the variance of reinforcement-learning approaches or the instability of adversarial ones.

## Key Mechanism

- The training objective is `L = L_MLE + gamma * L_seq` (plus an optional `L_copy` term); the OT loss supplements cross-entropy and never replaces it
- `L_seq` is an optimal-transport cost between the sequence of generated word embeddings and the sequence of reference word embeddings, with cosine-based ground cost between embedding pairs
- The paper contrasts three matching schemes explicitly (its Figure 1): hard matching (position `i` to position `i`), soft bipartite matching, and OT matching. OT matching lets a word's mass distribute across several counterparts, which is what makes a reworded sequence cheap
- Solved with an IPOT-style iterative approximation rather than an exact linear program, so the term stays differentiable and trainable
- Interpreted theoretically as a Wasserstein gradient flow matching the model to the ground-truth sequence distribution, which is the paper's justification for the term rather than a post-hoc heuristic
- Sidesteps the two standard sequence-level alternatives: RL methods suffer high policy-gradient variance and need control variates or self-critic baselines, and adversarial methods depend on a fragile mini-max balance

## Main Findings

- Machine translation, BLEU on VI-EN: GNMT baseline 20.7 / 23.8, plus `L_seq` 21.9 / 25.4, plus both terms 21.9 / 25.5
- Machine translation, BLEU on EN-VI: baseline 23.8 / 26.1, plus `L_seq` 24.4 / 26.5, plus both 24.5 / 26.9
- DE-EN: baseline 29.0 / 29.9 rising only to 29.2 / 30.1 with both terms - the smallest gain in the paper
- EN-DE on NT2015: 28.0 BLEU against the same-architecture GNMT at 27.6 and a Transformer at 27.3, and competitive with the 28.4 reported by Vaswani et al. using a considerably more elaborate design
- Abstractive summarization, ROUGE-1/2/L on Gigaword: Seq2Seq 33.4 / 15.7 / 32.4, plus `L_seq` 35.8 / 17.5 / 33.7, plus both 36.2 / 18.1 / 34.0. The +2.8 ROUGE-1 is the largest single gain reported
- DUC2004: 28.0 / 9.4 / 24.8 rising to 30.1 / 10.1 / 26.0 with both terms
- Image captioning on COCO: Show and Tell 29.5 BLEU-4 / 87.1 CIDEr rising to 30.1 / 90.0; Show, Attend and Tell 33.1 / 99.1 rising to 33.8 / 102.9
- The method is reported robust to the choice of the weighting hyperparameter gamma
- Gains are consistent in direction across every task but small in magnitude, typically 0.2 to 1.6 BLEU and up to 2.8 ROUGE-1. No result in the paper approaches an order-of-magnitude improvement over the cross-entropy baseline
- The baselines are honest rather than weak (GNMT, Show and Tell, Show Attend and Tell), which makes the small gains more credible and also caps how much can be claimed

## Key Takeaways

- Optimal transport is usable as a differentiable training signal for text generation, and this is the canonical citation for that claim
- The transport term works because a soft many-to-many matching prices a reworded sequence cheaply while a hard positional matching cannot. That is the same property that makes transport useful for scoring documents, arrived at from the training side
- An approximate, entropically-smoothed transport solve is required for training. Exact network-simplex transport is not differentiable and cannot serve as a loss, so a scoring pipeline that deliberately uses exact transport cannot be converted into a training loss without changing the solver
- The OT term supplements cross-entropy rather than replacing it. Anyone proposing transport as an alternative to likelihood training should note that this paper does not do that
- Treat the reported gains as evidence of direction, not magnitude. A team citing this work to justify transport-based feedback should quote 2.8 ROUGE-1 on Gigaword, not an unquantified claim of superiority
- The comparison in this paper is OT-augmented MLE against plain MLE. It is not a comparison of optimal transport against KL divergence as competing feedback signals, and it cannot be cited for one

## Relevance To This Project

- Cited in `docs/medium/docdistance-measuring-document-drift.md` as prior work for using transport as a training signal. It is the strongest published support for the direction of that section, and it explicitly does not support any "order of magnitude better than KL" claim
- Confirms the differentiability constraint that matters for this project: the shipped path uses exact EMD via network simplex, which cannot be a loss. This paper's IPOT approximation is the route if transport ever becomes a training objective here
- Does not bear on the scoring pipeline itself. This project uses transport to measure finished documents, not to train a generator

## Tags

- #optimal-transport
- #sequence-to-sequence
- #training-objective
