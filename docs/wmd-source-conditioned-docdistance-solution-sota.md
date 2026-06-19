# Source-Conditioned Document Distance d(A,B|S) (SOTA)

## Abstract

A document distance for two documents that share a known source `S` - two summaries of an article, two extractions of a report. A symmetric `d(A,B)` is source-blind and conflates two opposite failures, so `d(A,B|S)` re-bases the Statement Mover's Distance[<sup>ref1</sup>](#ref1) onto the source and reads off two interpretable axes:

- **`D_sel`** - selection: does a document cover the same source content; a metric axis, `0/24` ordinality violations
- **`D_grd`** - grounding: is what it says supported by the source; the relevance-gated residual closes the per-document gold intrusion

Their blend orders fabrication above information-loss where a source-blind scalar inverts them, and the `D_sel` pre-pass removes anisotropy by default - widening its dynamic range ~7.4x at `0` violations, with an opt-out flag. Validated on the IBM AI-adoption executive-summary fixtures (batches E02-E04, [`experiments/wmd-docdistance-experiments.md`](experiments/wmd-docdistance-experiments.md)); the design runs in [`../notebooks/05-kj-source-conditioned-distance.ipynb`](../notebooks/05-kj-source-conditioned-distance.ipynb) and the library is validated end-to-end in [`../notebooks/09-kj-docdistance-api-e2e.ipynb`](../notebooks/09-kj-docdistance-api-e2e.ipynb). This is the conclusion doc, the experiments log its evidence.

## Problem

A symmetric `d(A,B)` is source-blind, and on documents that share a source that blindness conflates two opposite failures.

- **Faithful selection reads as distance** - two summaries that pick different but equally valid source facts transport far apart, penalized for a difference that is not an error
- **Fluent fabrication reads as closeness** - a faithful summary and an off-source fabrication that share vocabulary transport near, rewarded for content the source never contained
- **One scalar cannot name why** - on the fixtures the symmetric SMD scores information-loss (Set 1, 0.452) and information-noise (Set 2, 0.406) almost level and inverts their severity, so it cannot say whether two documents differ because one dropped content or because one fabricated it

## Solution

Compare each document to the source first, then compare what each does to the source - never `A → B` directly.

- **Distance interpretation** - two axes read off the source: `D_sel` (selection divergence over the source coverage) and `D_grd` (grounding divergence from the source), reported as the pair `(D_sel, D_grd)` with an optional blended scalar `α·D_sel + (1−α)·D_grd`
- **Why and how to use** - when two documents share a known source and the question is *why* they diverge - selection drift versus fabrication - not just how far; source plus two documents in → the two axes and the source alignment
- **Headline result** - `D_sel` separates every adversarial tier above every gold (`0/24` violations); the relevance-gated `D_grd` drops gold intrusions to `0`; the blend orders Set 2 (fabrication) above Set 1 (info-loss) at `0/28` where the symmetric scalar inverts them
- **Deterministic** - exact OT plus INT8 / fp16 inference, no sampling; identical tier verdicts on CPU INT8 and GPU fp16 despite different absolute values

## Pipeline

Source plus two documents in → two coverage profiles, two grounding residuals, and the `(D_sel, D_grd)` pair.

- **Segment** - `S`, `A`, `B` each split into statements with `sat-3l-sm`[<sup>ref6</sup>](#ref6) (nb 01); statements are the unit, so the comparison is position-invariant
- **Embed** - mmBERT[<sup>ref5</sup>](#ref5), mean-pooled and L2-normalized; OpenVINO INT8 on CPU, torch fp16 on GPU
- **Selection axis `D_sel`** - each document's statements are softly assigned to the source statements, giving a coverage profile (a distribution over `S`); `D_sel` is the metric optimal-transport distance between the two documents' coverage profiles, ground cost `√(2 − 2cos)` on mmBERT
- **Grounding axis `D_grd`** - per summary statement, the cross-encoder `bge-reranker-v2-m3`[<sup>ref3</sup>](#ref3) finds the top-3 aligning source statements joined into one premise, and the `mdeberta-v3-base-mnli-xnli`[<sup>ref4</sup>](#ref4) entailer grades whether the statement follows; the ungrounded mass, gated by reranker relevance, is the residual and `D_grd` is its distance from the gold anchor
- **Blended scalar** - min-max each axis and combine `α·D_sel + (1−α)·D_grd` (operating point `α = 0.75`) when a single threshold is needed
- **Backends** - the whole chain runs OpenVINO INT8 on CPU (the published `stellars/*` artifacts) or torch fp16 on GPU; one flag drives all four models

## Mechanism

Three surgical changes turn the symmetric Statement Mover's Distance into the source-conditioned one, and the metric guarantee survives them.

- **Re-based transport** - the symmetric design transports `A → B`; here transport runs `A → S` and `B → S` so the comparison is mediated by the source, never direct surface overlap
- **Selection divergence `D_sel`** - the coverage profile `cov_A(k)` is the share of source statement `k` that `A` represents (a soft nearest-source assignment, normalized to a distribution over `S`); `D_sel` is exact OT between `cov_A` and `cov_B` with the `√(2 − 2cos)` ground cost on the source-statement embeddings - same source, different picks shows up here
- **Grounding score** - `g(a_i, s_k) = r(a_i, s_k) · P(entail)(s_k → a_i)`: the reranker relevance localizes the source evidence, the NLI entailment confirms the claim follows; premise = source statement, hypothesis = summary statement
- **Joint-premise aggregation** - a faithful summary statement fuses several source sentences, so single-premise NLI mis-grades compression; the entailer scores the top-3 reranked source statements joined into one premise (the multi-premise SummaC pattern[<sup>ref2</sup>](#ref2)), which moved information-loss from 0.206 down to gold level (0.130) while holding fabrication at 0.232 (E02 R1 → R2)
- **Relevance-gated residual (E03-H11)** - the shipped grounding residual is `ungrounded_gated = mean_i (1 − entail_i)·(1 − max_k r(a_i, s_k))`; gating the ungrounded mass by max reranker relevance distinguishes faithful compression (high relevance, low entailment) from fabrication (low on both), closing the per-document gold intrusion the raw entailment residual leaves open
- **Blended scalar (E03-H14)** - `α·D_sel + (1−α)·D_grd` over the min-max axes at `α ∈ [0.6, 0.9]` orders gold < Set 1 < Set 2 with `0/28` violations and Set 2 above Set 1, the correct grounding severity
- **Metric property preserved** - the NLI / cross-encoder grounding cost is asymmetric and non-metric, but it is only ever a transport cost to a fixed reference `S`, not a distance between `A` and `B`; the outer comparison `D_sel` is metric OT over coverage profiles with a metric ground cost, so `d(A,B|S)` keeps the triangle inequality the project requires

## Performance

Validated on the `data/interim/exec-summaries/ibm-ai-adoption` fixtures - 11 executive summaries of one article plus the source, in three tiers (gold faithful, Set 1 info-loss, Set 2 info-noise), scored against the gold anchor. Numbers from batches E02/E03 (CPU INT8), E04 (GPU fp16) and nb05. The `D_sel` row is the raw axis; the shipped default adds the anisotropy pre-pass (E04-H15), which widens the range ~7.4x and preserves this ordering.

| axis | gold | Set 1 | Set 2 | check |
|---|---|---|---|---|
| `D_sel` selection | 0.023 | 0.060 | moderate | `0/24` ordinality violations |
| `D_grd` R2 ungated | 0.084 | 0.130 | 0.232 | 2 gold intrude (per document) |
| `D_grd` H11 relevance-gated | 0.084 | 0.141 | 0.236 | gold intrusions `2 → 0` |
| blend `α ∈ [0.6,0.9]` | low | mid | high | `0/28`, Set 2 > Set 1 |
| symmetric SMD (baseline) | 0.287 | 0.452 | 0.406 | `0/28` but Set 1 > Set 2 (inverted) |

- **Selection axis is clean** - `D_sel` ranks every adversarial above every gold with zero violations; it is the metric, ship-ready half
- **The relevance gate is the grounding-axis fix** - it turns `D_grd` from a tier-level fabrication flag into a per-document discriminator, closing the gold intrusion at no extra cost (free re-weighting of signals already computed)
- **The blend's win is ordering, not separability** - the symmetric SMD also clears `0/28` gold-vs-adversarial here, but inverts the severity; the conditioned blend orders fabrication above information-loss, which a source-blind scalar cannot
- **Default resolution pre-pass (E04-H15)** - anisotropy removal (all-but-the-top, `k = 1`) over the pooled {A, B, source} statements before the coverage profile widens the `D_sel` dynamic range ~7.4x at `0` violations; resolution only, not a sharper boundary (the gold/adversarial contrast is unchanged), so it sharpens thresholding without changing the verdict - on by default on the source-conditioned path (`anisotropy=False` to opt out), the same lever the symmetric distance offers (E01-H3)

## Setup

- **Pipeline** - `sat-3l-sm` segmenter → `mmBERT-base` encoder → `bge-reranker-v2-m3` cross-encoder → `mdeberta-v3-base-mnli-xnli` NLI; every stage published as OpenVINO INT8 on the `stellars` account, so the chain runs end-to-end on CPU with no FP32 downloads, or as the original FP weights in torch fp16 on GPU
- **Grounding direction** - premise = source statement, hypothesis = summary statement; entailment read by the model's `id2label` entailment index (class 0); `TOP_K = 3` source statements per joint premise
- **Quantization** - the INT8 IRs are CPU / Intel-only (OpenVINO); the GPU path uses the original FP weights in fp16; a reproducible, grounding-calibrated INT8 mDeBERTa is owned in-project ([`../notebooks/model-quantization/Q02-kj-deberta-int8-smoothquant.ipynb`](../notebooks/model-quantization/Q02-kj-deberta-int8-smoothquant.ipynb), NNCF SmoothQuant, entailment parity 0.993)
- **Reproducibility** - deterministic apart from natural fp16 reduction-order variance on GPU; the tier verdicts are device-independent

## Methods of measurement

- **Ordinality violations `V`** - count of (gold, adversarial) pairs where the gold is not strictly nearer the anchor; `0` is clean
- **Gold intrusions** - gold documents landing at or above the lowest Set 2 score on `D_grd`; the per-document grounding-quality measure
- **Severity order** - whether Set 2 (fabrication) sits above Set 1 (info-loss); `Set2>Set1` is the correct grounding severity, `Set1>Set2` the inversion
- **Tier means** - per-axis mean over each tier, the coarse separation; the per-document checks above are the sharper test
- **All measurement-only** - none of the scorers is trained or tuned on the fixtures; the models run as published artifacts

## Throughput and footprint

- **CPU INT8 end-to-end** - the shipped grounding chain is ~67 s/pair, ~99% the reranker full grid (`n_summary × 70` pairs); the often-quoted ~109 s/pair includes a diagnostic R1 single-premise NLI sweep the shipped relevance-gate + R2 joint-premise chain never runs (R2 NLI ~0.7 s), so the reranker grid is the entire CPU cost
- **Length-bucketing, reranker grid (E06-H25)** - sorting the pairs by length and tightening `max_length` to the per-call 100th-percentile cuts the CPU INT8 reranker grid 47.7% end-to-end over all 11 documents (860.4 s → 449.8 s, 8960 pairs) at score Spearman 0.99994 / D_grd Spearman 1.0, 0 intrusion change - portable, zero correctness risk
- **Length-bucketing, statement encoder (E06-H25)** - the same lever wired into `OpenVINOEncoder` / `TorchEncoder` reclaims 51% of the padded-token volume (waste 2.59x → 1.27x over 299 statements) for a 32.8% CPU cut at cosine 0.99999 embedding fidelity ([notebook 10](../notebooks/10-kj-encoder-length-bucketing.ipynb), measured under CPU load; the relative cut holds as both arms contend equally)
- **GPU fp16** - the reranker grid runs ~63x faster on an RTX 5000 Ada (107.5 s CPU INT8 → 1.72 s), and the whole 11-document signal build drops from 534 s to ~16 s; the tier verdicts are identical to the CPU INT8 chain
- **Footprint** - the grounding cross-encoders are ~1-2 GB FP and ~300-570 MB INT8 each; `D_sel` alone (mmBERT only) is sub-second per pair and carries the metric, shippable selection axis without the grounding chain
- **Cost reading** - the grounding axis is a heavy diagnostic, not a cheap metric; `D_sel` is the cheap always-on half, `D_grd` the expensive on-demand half

## Limitations

- **One fixture** - all evidence is one article and one degradation design; the blend's severity win needs a second source before it is trusted as a single scalar (the open gate)
- **Grounding cost stands** - the reranker is load-bearing: the bi-encoder cosine neither shortlists faithfully (recall@10 of top-3 is 0.58) nor replaces relevance (Spearman 0.40), and E04 confirmed it from the other side - a distilled same-family `bge-reranker-base` recalls only 0.55 of the v2-m3 top-3 (Spearman 0.70) and a tiny MiniLM cross-encoder cascade, though far closer (Spearman 0.98, 2.3x), still drops 13% of the top-3 evidence at `m = 15` and adds one gold intrusion, so the cost is structural; E05 then ruled out two multilingual cross-encoders (mmarco-MiniLM 118M, jina-reranker-v2 278M - recall@3 0.45 / 0.63 of the v2-m3 top-3), late-interaction MaxSim (0.47) and a learned-sparse proxy, all below the 0.90 bar (E03-H12/H13, E04-H17/H18, E05-H20/H21/H22 refuted) - six scorer classes; E06 then ruled out a *trained* multilingual late-interaction ColBERT (jina-colbert-v2, recall@3 0.47, on top of the untrained proxy) and a *trained* learned-sparse bge-m3 (recall@15 0.82 but Spearman −0.77, the ranking inverted) (E06-H26/H27 refuted) - eight scorer classes, the cross-encoder is irreplaceable and the trained-scorer replacement direction is now closed, not open; the one shipping speed win is length-bucketing (E06-H25, 47.7% end-to-end at numerically identical scores, now wired into the statement encoder), source clustering a 36% safe ceiling, neither reaching 50% alone
- **Numbers are un-verifiable here** - general NLI is weak on quantitative claims and the contradiction signal is near-dead even on fabricated forecasts; a numeric verifier was defeated by the source's figure density (82 figures match fabricated numbers by coincidence), so `D_grd` rides on the ungrounded component, not contradiction (E03-H10 refuted)
- **`D_grd` not yet in the library** - the shipped library covers the selection axis; the grounding axis lives in the notebooks (E02/E03, nb05) pending the cross-fixture check and a wired-in `D_grd`

## FAQ

- **Why not the symmetric SMD?** - it conflates and even inverts the two failure modes on common-source documents; it answers "how far" but not "why"
- **Why two axes instead of one number?** - selection drift and grounding failure are orthogonal; a single scalar cannot separate a faithful re-selection from a fabrication, the two axes can, and the blend is offered only when a threshold is needed
- **Why a reranker and an NLI, not cosine?** - cosine cannot tell a faithful paraphrase from a contradiction that shares vocabulary; the reranker localizes the source evidence, the entailer supplies the directional faithful / unsupported / contradiction label
- **Why a joint premise?** - compression fuses several source sentences into one summary statement, so single-premise NLI mis-grades it; top-k joint-premise aggregation is the SummaC fix, required not optional
- **Why gate the residual by relevance?** - faithful compression has high source relevance but low single-statement entailment, fabrication has both low; gating by `1 − max_k r` keeps only genuinely off-source mass, closing the gold intrusion
- **CPU or GPU?** - both give identical tier verdicts; run CPU INT8 where a GPU is absent (the published artifacts), GPU fp16 for the ~63x reranker speed-up
- **Is `d(A,B|S)` still a metric?** - yes for the selection axis: the non-metric grounding cost is confined to transport against a fixed `S`, and the outer `D_sel` comparison is metric OT over coverage profiles
- **Can it be made faster or sharper?** - sharper is the default: the `D_sel` pre-pass removes anisotropy on by default, widening its dynamic range ~7.4x at `0` violations (E04-H15; `anisotropy=False` to opt out); faster on CPU partially: length-bucketing the reranker pairs cuts 47.7% end-to-end at numerically identical scores (E06-H25, portable, zero risk, now wired into the statement encoder), but the reranker model is irreplaceable - eight scorer classes (bi-encoder, distilled and multilingual cross-encoders, untrained and trained late-interaction, untrained and trained learned-sparse) fail to recall its top-3 (E03/E04/E05/E06, including trained jina-colbert-v2 and bge-m3 sparse), so the grounding floor stands and GPU fp16 (~63x) remains the big speed answer

## Implementation

- **Selection axis (shipped)** - `coverage_profile`, `selection_divergence` and `compute_source_conditioned` in `src/docdistance/distance.py`, exposed through `DocDistance.distance_wrt_source` and the one-shot `source_conditioned_distance(a, b, source)`; returns `d_sel` plus each document's residual to the source
- **Grounding axis (notebook)** - the reranker × NLI chain, the H11 relevance-gate and the H14 blend are implemented and validated in [`../notebooks/05-kj-source-conditioned-distance.ipynb`](../notebooks/05-kj-source-conditioned-distance.ipynb) and the E02/E03 experiment notebooks, not yet folded into the library
- **Default resolution pre-pass** - the source-conditioned path runs the E04-H15 all-but-the-top (`k = 1`) on `D_sel` by default; pass `anisotropy=False` to `compute_source_conditioned` / `DocDistance.distance_wrt_source` (or `--no-anisotropy`) to opt out
- **CLI** - `docdistance distance-wrt-source A B --source S` drives the selection axis with `--json` / `--gpu` / `--backend`
- **Quantization** - `Q01` (mmBERT) and `Q02` (mDeBERTa SmoothQuant) under `notebooks/model-quantization/` own the in-project INT8 IRs
- **Next** - wire the H11 relevance-gated `D_grd` and the H14 blend into the library after the cross-fixture validation gate; the open speed lever is a larger cascade `m` or a better-matched pre-filter (E04-H18 near-miss)

## Conclusions

- **Ships** - `D_sel`, a metric selection axis, `0/24` violations, sub-second per pair on CPU INT8
- **Ships as the grounding-axis definition** - the H11 relevance-gated ungrounded mass, a per-document discriminator at no extra cost over the R2 residual
- **Offered with a caveat** - the H14 blended scalar (`α = 0.75`) orders the failure modes correctly where the symmetric distance inverts them, pending a second source
- **Default resolution pre-pass (E04)** - anisotropy removal on the conditioned `D_sel` (`k = 1`, on by default, opt-out via `anisotropy=False`) widens dynamic range ~7.4x at `0` violations; resolution, not a sharper boundary, so it sharpens thresholding without changing the verdict - the same lever the symmetric distance offers optionally (E01-H3)
- **Not shipped** - the numeric verifier (defeated by source figure density), the bi-encoder reranker-cost levers (E03) and the E04 cross-encoder speed levers (a distilled replacement drops half the evidence, a cascade is the closest but breaks a guardrail at `m = 15`); the cross-encoder is irreplaceable and its cost stands
- **Net** - conditioning on the source separates selection from grounding and names why two documents diverge, the result a symmetric scalar cannot give; the selection axis is production-ready, the grounding axis is a validated heavy diagnostic awaiting cross-fixture confirmation

## Bibliography

<a name="ref1"></a>**[ref1]** Kusner, Sun, Kolkin, Weinberger. *From Word Embeddings To Document Distances*. ICML 2015. Digest: [`../references/papers/from-word-embeddings-to-document-distances.md`](../references/papers/from-word-embeddings-to-document-distances.md). The optimal-transport skeleton this design re-bases onto the source.

<a name="ref2"></a>**[ref2]** Laban, Schnabel, Bennett, Hearst. *SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization*. TACL 2022. The multi-premise NLI aggregation pattern (score source × summary, aggregate over source) the joint-premise grounding uses.

<a name="ref3"></a>**[ref3]** BAAI. *bge-reranker-v2-m3* (multilingual cross-encoder reranker). INT8: `stellars/bge-reranker-v2-m3-openvino-int8`. The alignment-cost scorer.

<a name="ref4"></a>**[ref4]** Laurer. *mDeBERTa-v3-base-mnli-xnli* (multilingual NLI). INT8: `stellars/mdeberta-v3-base-mnli-xnli-openvino-int8`. The grounding-grade entailer.

<a name="ref5"></a>**[ref5]** JHU CLSP. *mmBERT-base* (multilingual encoder). The statement encoder for the selection axis.

<a name="ref6"></a>**[ref6]** Minixhofer, Pfeiffer, Vulić. *Where's the Point? Self-Supervised Multilingual Punctuation-Agnostic Sentence Segmentation* (SaT / `sat-3l-sm`). The statement segmenter.
