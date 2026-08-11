# Fine-Grained Human Feedback Gives Better Rewards for Language Model Training (2023) - Digest

**Source**: [Fine-Grained Human Feedback Gives Better Rewards for Language Model Training](https://papers.neurips.cc/paper_files/paper/2023/hash/b8c90b65739ae8417e61eadb521f63d5-Abstract-Conference.html)

## Overview

Reinforcement learning from human feedback converts a human preference between two outputs into a single scalar reward for the whole sequence. For long-form text that signal is thin: it says one output was preferred without saying which sentence was false, which clause was off-topic, or what was missing. The paper introduces FINE-GRAINED RLHF, which makes the reward fine-grained along two independent axes. The first is density, emitting a reward after each segment rather than once at the end. The second is type, running several reward models for distinct error categories rather than folding everything into one number. Evaluated on detoxification and long-form question answering, the method beats both supervised fine-tuning and conventional preference-based RLHF, converges in fewer steps, and lets model behaviour be tuned by reweighting the individual reward models. Data, collected human feedback and code are released.

## Key Mechanism

- **Density axis** - a reward is emitted after every segment (a sentence, or a sub-sentence span), so credit is assigned to the span that caused the problem instead of to the whole generation
- **Type axis** - separate reward models for distinct error categories. In the long-form QA task these are factual incorrectness, irrelevance, and information incompleteness, each with its own annotated feedback and its own model
- **Composability** - because the reward models are separate, their weights can be changed at training time to produce different behaviours from the same setup. This is a property a single scalar reward structurally cannot have
- **Detoxification setup** - toxicity is the only error category, so only the density axis is exercised. The PERSPECTIVE API supplies the reward, and a sentence's reward is the change in toxicity attributable to generating that sentence
- **Training** - standard policy optimisation against the combined fine-grained rewards; the architecture of the RL loop is unchanged, only the reward structure differs
- The reward models were trained on 2.8K examples, which is a small annotation budget for the reported effect

## Main Findings

- On the RealToxicityPrompts test set, FINE-GRAINED RLHF with a sentence-level reward attains **both the lowest toxicity and the lowest perplexity** among all methods compared, while holding n-gram diversity at a similar level. Improving fluency and reducing toxicity at once is the notable part, since the two usually trade against each other
- **Sample efficiency**: during training, toxicity drops substantially faster than under holistic reward while perplexity stays low. The paper attributes this directly to reward density rather than to reward quality
- The additional cost is roughly **1% of training time** relative to preference RLHF. The gain does not come from a materially more expensive setup
- On long-form QA, FINE-GRAINED RLHF outperforms both supervised fine-tuning and preference RLHF **on every error type measured** - factual errors, irrelevance, repetition and incoherence
- Preference RLHF does substantially reduce factual errors relative to the initial policy, so the baseline here is genuinely strong rather than a straw man
- In human pairwise comparison, FINE-GRAINED RLHF was rated better than preference RLHF in **30.5%** of all examples, despite preference RLHF being trained directly on the preference data those human judgements come from
- The authors attribute the advantage to the training signal being more specific and more localized, which is a claim about where the reward points rather than how large it is
- Automatic evaluation uses RougeLSum alongside per-error-type human evaluation, and the paper leans on the human evaluation because its own thesis is that aggregate automatic scores hide which aspect changed

## Key Takeaways

- Where feedback points matters independently of how accurate it is. Two systems with the same underlying judgement quality perform differently if one localizes the error and the other does not
- The two axes are separable and worth treating separately when designing a feedback loop: density (how often) and type (what kind). Most systems vary only one
- Separate reward models per error type give a control surface a scalar cannot: behaviour is retuned by reweighting, without collecting new feedback
- The 1% overhead figure is the practical argument. Fine-grained feedback is not an expensive luxury relative to holistic RLHF, so the reason not to adopt it is annotation effort, not compute
- A scoring method that already returns per-unit, typed output is most of the way to a fine-grained reward. The gap is turning localized findings into a training or refinement signal, not producing them
- The comparison is fine-grained against holistic feedback. It says nothing about which underlying distance or divergence should compute the reward, and it cannot be cited for a claim about optimal transport against KL

## Relevance To This Project

- Cited in `docs/medium/docdistance-measuring-document-drift.md` to support the general claim that per-unit feedback beats a single holistic score when refining language models. It is the strongest published support for that section
- Structurally parallel to this project's output: the transport map returns a per-statement cost plus a typed distinction between meaning drift (`changed`) and position drift (`moved`), which is density plus type in this paper's terms
- Supports the mechanism only. This project has no training loop, no reward model, and no measurement of a refinement loop, so the parallel is architectural rather than empirical

## Tags

- #rlhf
- #fine-grained-feedback
- #reward-modeling
