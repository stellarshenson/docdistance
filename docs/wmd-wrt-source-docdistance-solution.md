# WMD Document Distance With Respect to a Common Source

Design for measuring the distance between two documents `A` and `B` when both are known to derive from one source `S` - a source-conditioned distance `d(A, B | S)`. It is a variant of the symmetric design in `wmd-docdistance-solution-sota.md`: the optimal-transport skeleton is reused, but the transport is re-based onto the source and the ground cost becomes a grounding score from an entailer and a cross-encoder. The motivating case is two executive summaries of the same article.

## Why condition on the source

A symmetric `d(A, B)` ignores `S`, and that blindness is the failure the test fixtures expose.

- **Different faithful selection reads as distance** - two summaries that pick different but equally valid source facts transport far apart, penalized for a difference that is not an error
- **Fluent fabrication reads as closeness** - a faithful summary and an off-source fabrication that share vocabulary transport near, rewarded for content the source never contained
- **Fix** - compare each document to `S` first, then compare what each does to `S`; grounding in the source replaces surface overlap as the basis

## Method

Project both documents onto the source, then compare the projections - transport `A → S` and `B → S`, never `A → B` directly.

- **Segment** - `S`, `A`, `B` each split into statements with the nb 01 SAT segmenter
- **Score grounding** - build cost matrices `C^A` (`n_A × n_S`) and `C^B` (`n_B × n_S`) where `C^A_{ik} = 1 − g(a_i, s_k)` and `g ∈ [0,1]` is the grounding of statement `a_i` in source statement `s_k`
- **Transport** - solve unbalanced OT for each document against `S`, yielding plans `T^A`, `T^B`
- **Coverage profile** - the column marginal `cov_A(k) = Σ_i T^A_{ik}` is the mass of source statement `k` that `A` represents; same for `B`
- **Grounding residual** - the unbalanced slack (document mass that reaches no source statement) plus NLI-flagged contradiction mass is the off-source signal per document
- **Output** - the two profiles and the two residuals collapse to a two-axis distance below

## Grounding scorer

A hybrid: the cross-encoder finds the alignment, the entailer grades it. Cosine alone cannot tell a faithful paraphrase from a contradiction that shares vocabulary.

| role | model | output | why this model |
|---|---|---|---|
| alignment cost | cross-encoder reranker | relevance `r(a_i, s_k)` | joint attention scores compression / paraphrase where one summary statement fuses several source sentences |
| grounding grade | NLI entailer | `P(entail)`, `P(neutral)`, `P(contradict)` for `S ⊨ a_i` | directional label cosine lacks - separates faithful, unsupported addition, and contradiction |

- **Combination** - `g(a_i, s_k) = r(a_i, s_k) · P(entail)(s_k → a_i)`; relevance localizes the source evidence, entailment confirms the claim follows from it
- **Failure-mode separation** - *entail* is faithful, *neutral* is an unsupported addition (off-source forecast), *contradict* is a wrong number or hallucination
- **Concrete hooks** - `stellars/bge-reranker-v2-m3-openvino-int8` (cross-encoder) and `stellars/mdeberta-v3-base-mnli-xnli-openvino-int8` (NLI), both multilingual and CPU-INT8, matching the mmBERT CPU story
- **No retraining** - both run as the published INT8 artifacts; all adaptation is usage-level. The reranker relevance logit is the alignment cost directly; the NLI head is read by its `id2label` entailment index
- **NLI premise aggregation** - a summary statement can be entailed only by several source statements jointly (compression), so score a pairwise source × summary matrix and aggregate (max or top-k over source) rather than single-premise NLI, which would miss multi-source support (the SummaC pattern)
- **Direction** - premise = source statement, hypothesis = summary statement; faithfulness runs `S → a_i`
- **Numeric-entailment caveat** - general-domain NLI is weak on quantitative claims (`a significant share` vs `42%`), which is exactly the Set 1 degradation; NLI carries the contradiction / unsupported-addition signal (Set 2), while the cross-encoder and the selection axis `D_sel` carry the number-stripping case

## Modifications to the base SMD

Three surgical changes turn the symmetric Statement Mover's Distance into the source-conditioned one.

| base SMD (`wmd-docdistance-solution-sota.md`) | this design |
|---|---|
| transport `A → B` | transport `A → S` and `B → S` |
| cost `c(i,j) = √(2 − 2cos)` on mmBERT | cost `c(a_i, s_k) = 1 − g(a_i, s_k)` from reranker + NLI |
| unbalanced slack = omission / hallucination on the pair | slack + contradiction mass = per-document grounding residual vs `S` |
| output = one scalar + alignment plan | output = two coverage profiles over `S` + two residuals → two-axis distance |

## Two-axis output

The conditioned distance reads off two interpretable axes a single scalar would conflate.

- **Selection divergence `D_sel`** - distance between the coverage profiles `cov_A` and `cov_B` over the shared source statements; captures same source, different picks
- **Grounding divergence `D_grd`** - distance between the per-document grounding residuals (ungrounded mass, contradiction mass); captures one document drifting off source while the other stays faithful
- **Report** - the pair `(D_sel, D_grd)`; an optional scalar is a tunable blend `α·D_sel + (1−α)·D_grd` when a single threshold is needed

## Metric property

The non-metric scorers are confined to the projection onto a fixed reference, so the document-to-document distance stays a metric - the guarantee the project requires.

- **Inner cost is not a metric, and need not be** - `g` from NLI / cross-encoder is asymmetric with no triangle inequality, but it is a transport cost to a fixed `S`, not a distance between `A` and `B`
- **Outer comparison stays metric** - `D_sel` compares two coverage profiles over the same source-statement space with a metric ground cost (`√(2 − 2cos)` on mmBERT), so it satisfies the metric axioms; `D_grd` is a distance between residual vectors
- **Net** - conditioning on `S` buys back the metric on `d(A, B | S)` that a raw NLI cost would void

## Validation against the fixtures

The two-axis output is expected to place the `data/interim/exec-summaries/ibm-ai-adoption` tiers on separable axes.

| pair | `D_sel` | `D_grd` | reason |
|---|---|---|---|
| gold ↔ gold | low | low | similar source coverage, both fully grounded |
| Set 1 (info-loss) ↔ gold | high | low–moderate | stripped figures leave source statements uncovered; what remains is vague but not fabricated |
| Set 2 (info-noise) ↔ gold | moderate | high | fabricated forecasts and wrong numbers raise ungrounded and contradiction mass |

- **Discriminative gain** - a symmetric WMD collapses Set 1 and Set 2 onto one number; the source-conditioned axes name *why* each is far

## Design decisions

- **Scorer** - hybrid reranker + NLI (chosen) / cross-encoder only / NLI only / mmBERT cosine
- **Grounding combination** - product `r · P(entail)` vs gated (reranker top-k, then NLI grade) vs entailment-only argmax
- **Residual definition** - unbalanced slack alone vs slack plus explicit contradiction mass
- **Selection metric** - OT on coverage profiles (metric, recommended) vs Jensen-Shannon / L2 on the normalized profiles
- **Weights** - uniform `1/n` vs statement length or salience

## Recommended configuration

- Statement-level units from nb 01 for `S`, `A`, `B`
- Pipeline (all INT8, used as-is) - `sat-3l-sm` segmenter → `mmBERT-base` encoder → `bge-reranker-v2-m3` cross-encoder → `mdeberta-v3-base-mnli-xnli` NLI, every stage published as OpenVINO INT8 on the `stellars` account, so the chain runs end-to-end on CPU with no FP32 downloads
- Hybrid grounding - `bge-reranker-v2-m3` alignment graded by `mdeberta-v3-base-mnli-xnli` entailment
- Unbalanced OT `A → S` and `B → S`, transport plans exposed as the source alignment
- Two-axis output - `D_sel` via metric OT on the coverage profiles (mmBERT cosine cost), `D_grd` via the residual delta
- mmBERT quantized encoder (FP8 on GPU / OpenVINO INT8 on CPU) for the outer selection metric

## Status

- Design only; not yet implemented
- Builds on `wmd-docdistance-solution-sota.md` - same OT skeleton, re-based onto the source
- Planned build - a source-conditioned mode in `src/docdistance_estimator/distance.py` plus a notebook scoring the fixture matrix on both axes
