# SELF-REFINE: Iterative Refinement with Self-Feedback (2023) - Digest

**Source**: [SELF-REFINE: Iterative Refinement with Self-Feedback](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html)

## Overview

Language models rarely produce their best output on the first attempt, and the usual remedies are expensive: train a dedicated refinement model on domain-specific data, or collect human annotations for a reward model. SELF-REFINE removes both requirements. One language model plays three roles in a loop - it generates an initial output, then critiques that output, then rewrites it using its own critique, repeating until a stopping condition. No supervised training data, no additional training, no reinforcement learning, and no second model. The authors evaluate across seven tasks spanning dialogue response generation, code optimization, code readability, sentiment reversal, acronym generation, constrained generation and mathematical reasoning, using GPT-3.5, ChatGPT and GPT-4. Outputs are preferred over conventional one-step generation by both humans and automatic metrics, improving roughly 20% absolute on average. The paper's central practical finding is not that self-critique works, but that it only works when the feedback is specific and actionable.

## Key Mechanism

- **One model, three roles** - the same LLM is generator, feedback provider and refiner, differentiated only by prompt. This is what makes the method standalone and test-time only
- **The loop** - generate an initial output, produce feedback on it, feed both the output and the feedback back in to produce a revision, repeat. Prior refinement work trained a separate refiner on domain data; this does not
- **Actionable feedback is the load-bearing requirement** - the authors define actionable explicitly as feedback containing "a concrete action that would likely improve the output". Feedback that only judges quality does not drive a useful revision
- **Specific feedback** - alongside actionability, feedback must identify what in particular to change, not offer a global verdict
- **Few-shot prompted** - feedback and refinement are elicited with examples rather than fine-tuning, so the method transfers across tasks by swapping prompts
- **Test-time only** - no weights are updated anywhere in the procedure, which is the operational difference from every RLHF-style approach

## Main Findings

- Across all seven tasks, SELF-REFINE outputs are preferred over one-step generation by both human evaluators and automatic metrics, improving **approximately 20% absolute on average** in task performance
- Applied to GPT-4, the improvement ranges **5% to 40% absolute** depending on the task. That spread is wide, and it is the number to quote rather than the average when arguing about a specific application
- Code optimization with GPT-4: the share of programs successfully optimized rises from **27.3% to 36.0%**, an absolute gain of 8.7%
- Code generation tasks improve by **up to 13% absolute** even against strong code models
- SELF-REFINE improves over the base model at every model size tested and also beats the previous state of the art across the evaluated tasks
- GPT-4 is used as a proxy judge alongside human evaluation, with reported agreement against human preference of **82%** on sentiment reversal, **71%** on dialogue response generation and **68%** on acronym generation. The authors report those correlations rather than assuming the proxy is sound, and 68% is weak enough to matter
- Human evaluation is run directly on dialogue response generation, code readability improvement and sentiment reversal rather than relying only on the proxy
- The gains depend on the base model being strong enough to critique itself usefully. The method inherits whatever blind spots the model already has, since no external signal ever enters the loop

## Key Takeaways

- Iterative critique-and-revise produces real gains at test time with no training, no labelled data and no second model. For a team that cannot fine-tune, this is the cheapest available lever
- The actionability requirement is the transferable lesson. A critique that says an output is worse does not improve it; a critique that names a concrete change does. Any feedback signal fed into a refinement loop should be judged against that bar first
- The 5% to 40% spread means task fit dominates. Do not budget for the 20% average on a task you have not measured
- Self-critique has a structural ceiling: the model cannot see errors it cannot detect. An external, grounded signal is not made redundant by this method, it is complementary to it
- The reported proxy-judge agreement, as low as 68%, is a caution for anyone using an LLM as an evaluator in their own loop. The paper measures this rather than assuming it
- Nothing here compares one distance or divergence against another. The contribution is the loop structure, not the metric inside it

## Relevance To This Project

- Cited in `docs/medium/docdistance-measuring-document-drift.md` for the claim that iterative self-critique improves output across a range of tasks, supporting the shape of the transport-map feedback loop without supporting any magnitude claim
- The actionability requirement is the sharpest argument for this project's output format. A single closeness score is not actionable by this paper's own definition; a per-statement transport map naming which claim drifted, by how much, and whether it moved or changed meaning satisfies it
- Complementary rather than overlapping. Self-critique has no access to ground truth, whereas this project compares against an actual source document, which is exactly the blind spot the paper's method cannot cover

## Tags

- #iterative-refinement
- #self-feedback
- #llm-evaluation
