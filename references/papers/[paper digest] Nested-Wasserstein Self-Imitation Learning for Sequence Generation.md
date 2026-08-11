# Nested-Wasserstein Self-Imitation Learning for Sequence Generation (2020) - Digest

**Source**: [Nested-Wasserstein Self-Imitation Learning for Sequence Generation](https://proceedings.mlr.press/v108/zhang20b.html)

## Overview

Reinforcement learning is the standard way to push a sequence model past maximum-likelihood training, but the rewards used are usually n-gram overlap scores such as BLEU or ROUGE. Those rewards carry little semantic information, so they bias the model towards surface matching, and they arrive only once the whole sentence exists, which makes exploration slow. The paper proposes a nested-Wasserstein distance for matching distributions of sequences semantically, and builds a self-imitation learning framework around it that pulls historical high-reward sequences out of a replay buffer instead of relying only on fresh policy samples. The authors show the method approximately performs proximal policy optimization with Wasserstein trust regions. Two variants are evaluated, WSIL-D and WSIL-I, across unconditional text generation, video captioning and image captioning, with consistent but modest gains and one striking demonstration of why overlap rewards fail.

## Key Mechanism

- **Nested Wasserstein** - an outer optimal transport between two sets of sequences, whose ground cost is itself an inner optimal transport between the word embeddings of a sequence pair. The nesting is what lets it compare a distribution of sequences rather than one pair
- **Self-imitation** - a replay buffer retains previously generated high-reward sequences, and the model is rewarded for matching that buffer's distribution. This directly attacks the forgetting problem, where a good trajectory is lost unless it happens to be resampled
- **Reward structure** - the nested-Wasserstein term supplements a conventional reward rather than replacing it, giving a dense semantic signal alongside a sparse task reward
- **Theoretical framing** - the update is shown to approximate proximal policy optimization with a Wasserstein trust region, which is the paper's justification for stability
- **Two variants** - WSIL-D compares against the buffer's sequences directly, WSIL-I uses an indirect formulation; neither dominates across all tasks

## Main Findings

- The clearest result is a reward-comparison table on two sequence-matching cases. On case C1 the scores are BLEU 36.8, ROUGE-L 50.0, CIDEr 163.7, naive matching 84.1, Wasserstein 76.3. On case C2 they are BLEU **0.0**, ROUGE-L 35.8, CIDEr 55.9, naive 42.5, Wasserstein **80.1**
- That C2 column is the paper's core argument in one line: a pair with no n-gram overlap scores zero under BLEU while the Wasserstein reward scores it 80.1, because the embeddings are close even though no tokens coincide
- Video captioning on MSR-VTT (BLEU-4 / METEOR / ROUGE-L / CIDEr): MLE 39.2 / 27.8 / 59.8 / 46.6, MIXER 40.2 / 27.9 / 60.8 / 50.3, SCST 40.7 / 27.9 / 61.6 / 51.3, WSIL-D **42.5 / 29.0 / 62.4 / 52.1**, WSIL-I 41.6 / 28.4 / 62.0 / 52.2
- The MSR-VTT gain over the strongest baseline, SCST, is +1.8 BLEU-4 and +1.1 METEOR
- Image captioning on COCO is mixed and the paper does not hide it: SCST reaches 32.1 BLEU-4 against WSIL-D 31.8 and WSIL-I 32.0, so the baseline wins on BLEU. WSIL leads on CIDEr, 107.4 and 107.6 against SCST 105.5
- The same COCO table lists the OT method of Chen et al. 2019 at 31.0 BLEU-4 / 94.7 CIDEr, below both WSIL variants on CIDEr
- Unconditional text generation is scored on the test-BLEU against self-BLEU trade-off, where quality and diversity pull against each other. The paper introduces `F1-BLEU`, the harmonic mean of BLEU and `1 - self-BLEU`, to report both at once, and reports WSIL ahead of the MLE model on F1-BLEU-4
- Baselines without the Wasserstein reward achieve lower self-BLEU only by sacrificing test-BLEU, which is the trade the combined metric is designed to expose
- Human evaluation is included as a separate table, since the authors' own argument is that automatic overlap metrics are a poor proxy

## Key Takeaways

- Wasserstein distance works as a reward signal in an RL loop, not only as a differentiable loss term. This is the closer analogue to feeding a transport cost back into a generation system than a gradient-based OT loss is
- The BLEU 0.0 against Wasserstein 80.1 result is the cleanest published demonstration that overlap-based scores collapse on paraphrase while transport-based scores do not. It is worth citing on its own
- Dense semantic reward plus a replay buffer addresses a specific failure of sparse-reward sequence RL: a good generation is forgotten unless resampled. Any team building a feedback loop over generated documents inherits that problem
- Gains are real but small, roughly 1 to 2 points, and inconsistent across tasks. WSIL loses to SCST on COCO BLEU-4. Cite it for the mechanism, not for a magnitude
- The nested construction is expensive: an inner transport per sequence pair inside an outer transport over sets. Budget for that before adopting it at scale
- Like the rest of this literature, the comparison is against RL and likelihood baselines. There is no head-to-head against KL divergence as a competing feedback signal

## Relevance To This Project

- Cited in `docs/medium/docdistance-measuring-document-drift.md` as prior work for Wasserstein distance used as a reward signal. It is the better citation than Chen et al. for the agentic-feedback framing, because a reward in a loop is closer to this project's use than a differentiable training loss
- The C2 result (BLEU 0.0, Wasserstein 80.1) is independent confirmation of the article's argument that overlap-based measures fail exactly where paraphrase is the normal case
- Does not bear on the shipped scoring path. This project never trains a generator

## Tags

- #optimal-transport
- #reinforcement-learning
- #sequence-generation
