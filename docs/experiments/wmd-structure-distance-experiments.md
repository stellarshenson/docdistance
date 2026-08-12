# WMD Structure-Distance Experiments - measuring document structure beyond statement alignment

Pre-registered experiments log for a **structure axis** on top of the mmBERT Statement Mover's Distance from [`../solution/wmd-docdistance-solution-sota.md`](../solution/wmd-docdistance-solution-sota.md). Today's distance is content-only and deliberately position-invariant - reorder the statements and SMD barely moves (README "statement-level and position-invariant"). This log designs a second axis that scores *how a document is arranged*, not only *what it says*: the order of statements, the section they sit in, and the relational geometry between them. The seed is the base paper's own future-work line (Kusner et al. 2015 proposed penalizing transport between different sections of similarly structured documents); the machinery is the optimal-transport-for-structure line that followed. Hypotheses are pre-registered with predictions and acceptance bars; verdicts are filled as batches run.

- **Branch / artefacts** - executed; vehicle `notebooks/experiments/E07-kj-structure-distance.ipynb`, fixture builder `notebooks/11-kj-structure-fixture.ipynb`, fixture `data/processed/structure-fixture/`, metrics `reports/E07-structure-distance-metrics.json`, figures `reports/figures/E07/`; the **E08 metric batch** in `notebooks/experiments/E08-kj-structure-distance-metric.ipynb`, metrics `reports/E08-structure-distance-metrics.json`, figures `reports/figures/E08/`; baseline `../solution/wmd-docdistance-solution-sota.md`, content experiments [`wmd-docdistance-experiments.md`](wmd-docdistance-experiments.md)
- **Data** - reuse `data/interim/exec-summaries/ibm-ai-adoption/` statements; build controlled perturbations (reorder, section-swap, paraphrase control) since structure-only fixtures do not exist yet
- **Status** - **EXECUTED (E07 batch)** - run in `notebooks/experiments/E07-kj-structure-distance.ipynb` on the byte-identical reorder upper bound plus the natural cross-summary pairs; verdicts filled below, metrics in `reports/E07-structure-distance-metrics.json`, figures in `reports/figures/E07/`; the real-conversion-pair gate (E07-H28), the LLM paraphrase / content-edit cells, and second-article replication remain deferred; the **E08 batch** below (E08-H44..H48) is now executed - E08-H44 (position-augmented Wasserstein) is the metric, wide-dynamic-range structure distance, while positional-FGW (E08-H45) is a metric but collapses at extreme reorder; the **E09 batch** (E09-H49..H53, anisotropy as common-mode rejection plus segmentation granularity, on both axes) is now executed - **nothing ships** (every CMRR < 1): on the diffuse pairs the common mode is shared **topic**, so rejecting it blurs the transport plan and hurts the semantic ranking, the opposite of the prediction; the **E10 batch** (E10-H54..H58, structure-sensitive content-invariant distances hunting translation-invariance AND scramble-monotonicity together) is now executed - **E10-H55 (the OPW order-gap `OPW − SMD`) is the one positive**, the only read to hold translation-invariance, scramble-monotonicity and content-invariance together (the combined property neither E08 survivor had); the content-pinned Gromov distortion (E10-H54) is refuted - pinning the plan does not stop the collapse because the Gromov functional is itself isometry-invariant - and the metric (E10-H56/H57) and soft-DTW (E10-H58) routes clear the geometry gates but fail the content-invariance proxy, left inconclusive pending a true-paraphrase fixture; the **E11 decision** (H44 vs H55 on a second article plus a true opus-mt back-translation paraphrase) is now executed and **reverses the ship** - **E10-H55 (OPW order-gap) ships as the single structure metric** (`structure_closeness = 1 − gap/√2`), E08-H44 dropped as the single number because the real paraphrase shows it fuses content and arrangement (a faithful reword reads 73.5% of a full-scramble distance) while H55 isolates structure (0.5%)
- **Hardened by** - **four rounds** of `data-scientist` adversarial review (each found a new layer, each confirmed the prior fixes held): R1 corrected the GW/FGW mis-ranking (Gromov-Wasserstein is invariant to a pure reorder, so it is not the order/section instrument); R2 caught the anisotropy train→serve leak and the unpowered decorrelation; R3 added the CI/IAA gate, paraphrase-floor validation and the deferred normalizer; R4 fixed the byte-identical-reorder realism gap, the two-sided gate, and per-axis gating

> [!NOTE]
> **Which side to trust.** Every number here is a hand transcription from the notebook or
> `reports/*.json` named in its section; where the two disagree, **the committed artefact governs**.
> Headline and verdict-bearing figures were reconciled against the artefacts on 2026-08-12 - the
> shipped structure lever is E10-H55, whose measured triangle-violation rate is **15%**
> (`reports/E11-structure-sota-decision.json` `.triangle.H55`) and **16%** in E10
> (`.results.E10-H55.triangle_viol`), not the 4.5% / 5.3% this log previously carried. The
> per-hypothesis long tail has not been reconciled; read any unreconciled figure off the artefact.
> The E08-E11 batches ran on the NVIDIA RTX PRO 6000 Blackwell Max-Q as recorded in every report's
> `device` field, not the RTX 5000 Ada named in older prose.

## Problem overview

The structure axis exists to fill a gap the content distance cannot, but it is in tension with a deliberate design choice - so the first question is whether the gap is real on our data, not how to close it.

- **The gap** - SMD transports statement clouds; it is symmetric over the cloud and ignores statement order, section, and inter-statement relations
- **The tension** - position-invariance is a *feature* of SMD (same claims in a different order score as similar); a structure axis must be a separate, opt-in axis, never a corruption of the content number
- **What "structure" means here** - three concrete signals: order (statement index), section (the heading block a statement belongs to), relation (the intra-document statement→statement similarity geometry)
- **The use case** - agentic document conversion and extraction; the open question is whether those pipelines actually reorder or re-section content while preserving it, or whether they preserve structure too (in which case the axis is moot)
- **Not yet tested** - no structure fixture exists; the controlled perturbations below are synthetic until real conversion pairs are collected, so every number starts as a probe, not a benchmark

## Executive summary

**Result first** - one mechanism survives execution: the **index-infused SMD**, the structure distance read off the SMD transport plan `T` (Phase 6, E07-H38-H43). It rises monotonically with reorder (Spearman 1.00, footrule → 0.94 at full scramble) while SMD stays blind (max `|ΔSMD|` 0.0004), the `T`-sparsity gate passes on the realistic diffuse regime (cross-summary median row-entropy 0.13), and it ships as a bounded `[0,1]` second axis beside semantic closeness (a reordered-but-faithful pair reads semantic 1.00 / structure 0.41). **Gromov-Wasserstein is confirmed a relational, not order, instrument** (GW ≈ 0 on a pure reorder against 0.0134 cross-summary), so it is reserved for relational rewrite and is not the order/section deliverable. The caveats are load-bearing - every number is the **byte-identical upper bound**, τ barely beats a naive greedy-1-NN baseline on the easy regime (both ~0.94), and the real-conversion-pair gate (E07-H28), the LLM paraphrase / content-edit cells, and second-article replication remain deferred.

**Result first (E08 batch)** - the `τ`-read's two weaknesses (it is not a metric - 7.5% triangle violations - and it saturates, tying the naive baseline) are resolved by **position-augmented Wasserstein (E08-H44)**: an exact optimal-transport distance on a ground metric that fuses semantic and positional cost (`M̃ = √((1−λ)·d_sem² + λ·d_pos²)`), so it is a **true metric (0% triangle violations)**, monotone (Spearman 1.00), and wide-dynamic-range (the ratio scales with the positional weight λ to 8.8 at λ=0.5); `λ=0` reproduces SMD exactly, so it rides beside the content axis without corrupting it. The other four are bounded or negative: positional Fused-GW (E08-H45) is a metric but **collapses at extreme reorder** (refuted as an order measure), the optimal-assignment footrule (E08-H46) is a metric only on equal-length pairs, the anisotropy lever (E08-H47) is **inert** on an already-sharp `T`, and the displacement-weighted anti-monotone mass (E08-H48) is wide-range but **not a metric**. The use-case answer is **E08-H44**, with the standing cross-fixture replication gate still open.

**Result first (E09 batch)** - the anisotropy-as-common-mode-rejection idea **fails on every lever** (every CMRR < 1, nothing ships), and the failure is the finding: on the diffuse cross-summary pairs the pooled common mode of two same-article summaries is shared **topic**, not just frequency anisotropy, so subtracting it is *destructive*. Per-pair both-document rejection (E09-H49) makes the transport plan *more* diffuse (`T`-entropy 0.129 → 0.135) and drops semantic fidelity to Spearman 0.741 with a new tier violation - CMRR −0.19, the opposite of the prediction. The selectivity sweep (E09-H50) confirms offline-frozen (anisotropy-only) is the safer arm (fidelity 0.921 vs 0.741) but it still does not ship; depth (E09-H51) peaks at no removal (D=0), with the pooled direction's bootstrap stability below 0.9 at every depth (~28 statements/pair is too few); finer segmentation (E09-H52) moves CMRR the predicted way (default −0.19 → fine +0.12) but never clears the bar; and the basic SMD under CMR (E09-H53) recovers the content-SOTA dynamic-range lift (×3.25) but at a d' cost (2.81 → 2.28, +1 tier violation), so it is not a corpus-free substitute for anisotropy removal. The standing answer is unchanged - **E08-H44 ships, the CMR levers do not**.

**Result first (E10 batch)** - of five structure-sensitive content-invariant mechanisms, **E10-H55 (the OPW order-gap `OPW − SMD`) is the one positive** - the only read to hold all four gates together: translation-invariance (pure-shift 0.0005 ≤ paraphrase p95 0.005, the E08-H44 failure mode gone), scramble-monotonicity (Spearman 1.00 rising through the top bin, the E08-H45 failure mode gone), content-invariance (paraphrase-proxy median 0.003), and beating E08-H44 on the combined property. It is a score, not a metric (16% triangle measured, the KL prior), applicability Medium. The instructive negative is **E10-H54 content-pinned Gromov distortion** - pinning the content plan does **not** stop the collapse (Spearman 0.43, top bin collapses like E08-H45), because the Gromov distortion functional is itself isometry-invariant and a full reversal is an isometry, so the predicted mechanical fix is conceptually wrong. The metric routes - source-anchored Kendall (E10-H56) and the table-turning string-edit (E10-H57), both 0% triangle on equal-length - and the soft-DTW divergence gap (E10-H58) clear the geometry gates (shift-invariant, scramble-monotone) but **fail content-invariance against the embedding-jitter paraphrase proxy**, so they are inconclusive pending a true LLM paraphrase fixture. The answer: **E08-H44 ships as the metric; E10-H55 is the translation-invariant companion score**.

**Result first (E11 batch)** - the SOTA decision between the two finalists, taken on the material E10 lacked: a **second article** (Wergeland *Impact of AI on Society*, pdfplumber-extracted) and a **true paraphrase** (opus-mt EN→DE→EN back-translation, a per-statement reword at mmBERT cosine 0.98, plus same-model a/b regenerations). The real paraphrase **reverses the provisional ship**. Position-augmented Wasserstein (E08-H44) is `≈ 0.87·SMD` when order is preserved, so it inherits SMD's content sensitivity: a faithful reword reads **0.149**, which is **73.5%** of a full-scramble distance - it fuses content and arrangement and cannot isolate structure as a single number. The OPW order-gap (E10-H55) subtracts SMD, cancelling content, so the same reword reads **0.0017** (0.5% of full scramble) while a reorder reads 0.44 - a **200× separation**; both stay scramble-monotone (Spearman 1.00 on both articles) and the shift gate holds (H44 fires 0.42 of range, H55 0.001). H55's only loss is metricity (15% triangle vs 0%), which matters only for corpus-scale indexing a pairwise read never invokes. The decision: **E10-H55 (OPW order-gap) ships as the single structure metric**, on the library's closeness scale (`structure_closeness = 1 − gap/√2`, bounded, stateless); **E08-H44 is dropped as the single number** (kept only as the content-aware distance read beside SMD). Metrics `reports/E11-structure-sota-decision.json`, vehicle `notebooks/experiments/E11-kj-structure-sota-decision.ipynb`.

A structure axis is **additive** to SMD, not a replacement: the content distance answers *how far apart in meaning*, the structure distance answers *how far apart in arrangement*. The order and section signal is carried by the **transport-induced** instruments - the positional penalty (Phase 2) and the structure distance read off the SMD plan `T` (Phase 6), both of which use statement position explicitly. **Gromov-Wasserstein is not that instrument**: a pure reorder of identical statements is an isometry of the statement cloud, so `GW(C_A, C_B) = 0` exactly - GW is blind to reorder and section-swap, and only moves when the inter-statement relations themselves change (statements rewritten, merged, split). GW/FGW are therefore the tool for a different, harder notion - *relational* structure - and need their own perturbation (Phase 4); they do not serve the order/section deliverable. The whole programme is gated on **E07-H28**: if conversion pipelines do not restructure-while-preserving-content, the axis is killed before any build.

**The deliverable** - a *secondary* distance that is bounded `[0,1]` like the semantic closeness, explainable to a human, and computed from the semantic transport plan `T` the SMD pass already produces - not a second model and not a second alignment. Phase 1-Phase 5 generate and validate the structural signal; **Phase 6** turns the chosen signal into that shippable secondary metric: it rides on `T`, normalizes to `[0,1]`, gates on semantic match quality so content changes stay on the content axis, and emits the statements that moved. The `[0,1]` bound is proven only on **equal-length** pairs so far; on unequal-length pairs (real conversions) no normalizer yet has a shown fixed `1` endpoint - the rank statistics are data-dependent and the off-diagonal-mass form has no canonical diagonal on a rectangular plan, so the endpoint is verified per-normalizer on the second-article fixture before shipping (E07-H39). The output reports two interpretable 0..1 numbers side by side - semantic closeness (how far in meaning) and structure closeness (how far in arrangement).

Five strategies, ranked from cheap-additive to principled-fused. Each adds one structural signal on top of the existing statement clouds and scores it with one optimal-transport mechanism.

| # | strategy | structural signal | OT mechanism | catches what SMD misses | anchor | metric |
|---|---|---|---|---|---|---|
| 1 | order / position | normalized statement index | positional ground-cost term, or Order-Preserving OT (IDM + KL diagonal regularizers) | same statements, reordered | Order-Preserving OT (Su & Hua 2017) | positional cost metric; OPW regularizers non-exact |
| 2 | section / block | section label per statement (markdown heading) | section-crossing penalty `×(1+β)` across section types | content moved to the wrong section | base-paper future-work (Kusner 2015) | metric on the embedding×section product |
| 3 | relational / discourse | intra-document statement→statement matrix `C_A`, `C_B` | Gromov-Wasserstein of `C_A` vs `C_B` | inter-statement relations rewritten; **invariant to pure reorder / section-swap** (an isometry → GW = 0) | Gromov-Wasserstein (Mémoli 2011) | GW is a metric |
| 4 | content + structure fused | embeddings and relational matrices | Fused Gromov-Wasserstein `(1−α)·SMD + α·GW` | content + relational change in one metric; **on a pure reorder FGW = SMD for every `α`** (adds nothing) | Fused GW (Vayer et al. 2019) | FGW is a metric |
| 5 | hierarchical / compositional | section/topic layer over statements | two-level OT, document→sections→statements | same blocks in the same proportions | HOTT (Yurochkin et al. 2019) | meta-distance, metric under conditions |

- **Headline** - for the order/section deliverable the core is **row 1 (positional penalty) + Phase 6 (structure read off `T`)**, the only mechanisms that see position; rows 3+4 (GW/FGW) are reserved for *relational* structure and are invariant to reorder, so they need their own rewrite/merge/split perturbation before they mean anything; **Phase 6 binds the winning signal to the SMD transport `T` as the bounded, explainable secondary closeness**
- **Load-bearing gate** - E07-H28 decides whether any of this ships; structure variance on real pairs must clear the **aligner-noise floor + a `0.10` margin** (the floor measured on faithful same-structure pairs), a single calibrated threshold set in E07-H28
- **Cost is the risk** - exact GW is `O(n²m²)` per pair against SMD's `O(n³ log n)`; at ~12 statements this is cheap, but it bounds where the axis can run always-on. Phase 6's `T`-induced metric is `O(n)` over the already-computed plan

**Pre-registration at a glance** (predictions registered before the run; verdicts filled from the E07 and E08 batches)

| hypothesis | strategy | mechanism | predicted | bar | verdict |
|---|---|---|---|---|---|
| E07-H28 | precondition | structure-signal probe on real + synthetic pairs | real pairs restructure while preserving content | order displacement ≥ aligner-noise floor + 0.10 (independent alignment) | Deferred |
| E07-H29 | precondition (sanity) | confirm SMD order-invariance (exact by construction) | shuffle-delta = 0 exactly | `\|ΔSMD\|` ≤ 0.01 - a sanity check, not a finding | Kept |
| E07-H30 | order | positional penalty, reported **beside** SMD | reorder rises above paraphrase floor, monotone in displacement | separation > paraphrase-control floor, monotone, SMD uncorrupted | Kept |
| E07-H31 | order | Order-Preserving OT (IDM + KL) | transport concentrates on the diagonal for aligned pairs | diagonal mass up on aligned, down on reordered | Kept |
| E07-H32 | section | section-crossing penalty `×(1+β)` | catches content-in-wrong-section | **gated on a ≥2-block fixture** (summaries have 1 heading); rises on swap, paraphrase flat | Kept |
| E07-H33 | section | hierarchical section→statement transport | block-proportion mismatch surfaces | gated on a ≥2-block fixture; composition drift separates, content held | Deferred |
| E07-H34 | relational | Gromov-Wasserstein structure-only | **≈0 on reorder (isometry)**, rises only on relational rewrite | GW separates a rewrite/merge/split perturbation; reorder invariance confirmed | Confirmed |
| E07-H35 | fused | Fused GW content + structure | **FGW = SMD on a pure reorder**; an `α` helps only with a relational perturbation | discrimination up on rewrite pairs, content-only unhurt | Confirmed |
| E07-H36 | validation | decorrelation of the **`T`-induced** structure metric (Phase 6) vs SMD | the two axes are orthogonal | `\|ρ\|` ≤ 0.3 with bootstrap CI, also on off-grid real pairs | Exploratory |
| E07-H37 | validation | metric property + cheap lower bound | triangle inequality holds, bound prunes | metric verified, bound ≤ exact, latency down | Reported |
| E07-H38 | secondary metric | structure distance induced by the SMD plan `T` | fires on reorder, ≈0 aligned, no extra model | raw > 0 reorder, ≈0 aligned, zero added model load | Kept |
| E07-H39 | secondary metric | bound the induced disorder to [0,1] | 0 identical order, 1 full reversal, monotone | bounded [0,1], endpoints correct, monotone | Ship-gate |
| E07-H40 | secondary metric | semantic-gate - score only matched statements | content changes stay on the content axis | structure ≈0 on diff-content cell, rises only on diff-structure | Kept |
| E07-H41 | secondary metric | explainability surface | structure number traceable to named statements | report accounts for the number, reader names the movers | Kept |
| E07-H42 | secondary metric | section variant - cross-section matched mass | bounded [0,1] section disagreement | bounded [0,1], rises on section-swap, ≈0 same-section | Kept |
| E07-H43 | secondary metric | two-axis output, structure closeness beside semantic | pair beats a single blended scalar | both axes 0..1, independently thresholdable | Kept |
| E08-H44 | OT metric (E08) | position-augmented Wasserstein `M̃=√((1−λ)d_sem²+λ d_pos²)` | a true metric, monotone, wide dynamic range; SMD at λ=0 | triangle ≤ 1%, Spearman ≥ 0.9, range > E07 baselines | Confirmed - High |
| E08-H45 | OT metric (E08) | positional Fused GW, `C=\|pos_i−pos_k\|` | rises with reorder, 0% violations, shift-invariant | monotone, 0 viol, ≈0 pure shift | Refuted - Low |
| E08-H46 | metric baseline (E08) | optimal-assignment footrule from `ot.emd` | true metric on equal-length | 0 viol equal-length, monotone | Confirmed - Low |
| E08-H47 | anisotropy (E08) | offline-frozen vs per-pair ABTT | sharper `T`, wider range, SMD held | entropy down, SMD held, no leak | Refuted - None |
| E08-H48 | creative (E08) | displacement-weighted anti-monotone mass | wide range, fewer violations than `τ` | monotone, viol < `τ`-footrule | Refuted - Low |
| E09-H49 | common-mode (E09) | both-document top-1 PCA rejection | diffuse `T` sharpens, semantic ordering survives | structural gain + ordering preserved, CMRR > 1 | Refuted - None |
| E09-H50 | selectivity (E09) | per-pair vs offline-frozen vs raw | offline more selective, per-pair sharper | variant maximizing CMRR on diffuse pairs | Inconclusive - None |
| E09-H51 | rejection depth (E09) | top-D sweep D ∈ {0,1,2,3} | CMRR peaks at D=1, D≥3 over-rejects | best D by CMRR, stability flagged | Inconclusive - None |
| E09-H52 | granularity (E09) | finer segmentation, more PCA samples | sweet spot before per-unit noise dominates | a granularity clearing the E09-H49 bar | Refuted - None |
| E09-H53 | single SMD (E09) | basic SMD under both-document CMR | inert on sharp, DR up on diffuse | tier ordering preserved, DR up single pair | Refuted - None |
| E10-H54 | pinned-GW (E10) | content-pinned Gromov distortion over positional `C` | monotone through scramble, shift ≈ 0, no collapse | Spearman ≥ 0.9 incl top bin, shift ≤ paraphrase p95 | Refuted - None |
| E10-H55 | order-OT (E10) | OPW − SMD order-gap (IDM + KL prior) | monotone, blind to paraphrase, shift small | Spearman ≥ 0.9, gap ≈ 0 on paraphrase | Confirmed - Medium |
| E10-H56 | source-anchored (E10) | Kendall-tau on induced source-order ranks | monotone, blind to paraphrase, a metric | Spearman ≥ 0.9, triangle 0, ≈ 0 on paraphrase | Inconclusive - Low |
| E10-H57 | string edit (E10) | Kendall-tau-sequence on shared-codebook symbols | metric, shift small vs scramble large | Spearman ≥ 0.9, triangle 0, codebook stability flagged | Inconclusive - Low |
| E10-H58 | soft-DTW (E10) | softDTW-divergence − SMD order-gap | monotone, blind to paraphrase, wide range | Spearman ≥ 0.9, gap ≈ 0 on paraphrase | Inconclusive - Low |

## Execution results (E07 batch)

The E07 batch ran in `notebooks/experiments/E07-kj-structure-distance.ipynb` on the byte-identical reorder upper bound, with the realistic diffuse-`T` regime supplied by the 11 natural cross-summary pairs (independent LLM summaries of one article). Verdicts are filled per hypothesis below; the full metrics live in `reports/E07-structure-distance-metrics.json` and the per-hypothesis figures in `reports/figures/E07/`.

- **The gap is real** - SMD is blind to reorder (max `|ΔSMD|` = 0.0004 ≤ 0.01) while the `T`-induced structure distance rises monotonically with displacement (E07-H29)
- **`T`-sparsity gate passes** - cross-summary `T` median row-entropy 0.13 against 0.00 byte-identical, still concentrated rather than the feared wash-out, so `τ` is usable beyond the upper bound (E07-H38 precondition)
- **Bounded, monotone secondary metric** - all three normalizers sit in `[0,1]` and rise monotonically (rank normalizers Spearman 1.00 by construction, off-diagonal mass saturates fast as flagged); the final normalizer choice is deferred to a second article (E07-H38/H39)
- **GW reorder-invariance confirmed on data** - GW ≈ 0.0000 on a pure reorder against 0.0134 on cross-summary pairs, and FGW = SMD across `α` on a reorder; the design's central correction holds empirically (E07-H34/H35)
- **Two-axis output and section axis** - a reordered-but-faithful pair sits at semantic closeness 1.00 / structure closeness 0.41, the structure-only failure a single blend hides (E07-H43); the section axis is bounded `[0,1]` and rises with relocated statements, 0 → 0.222 (E07-H32/H42)
- **Honest limits** - every number is the byte-identical upper bound; decorrelation is underpowered (Pearson `ρ` = 0.657, CI [0.540, 0.757]); the `τ`-footrule is a derived disorder, not a metric (7.5% triangle violations, E07-H37); the real-conversion-pair gate (E07-H28), the LLM paraphrase / content-edit cells, and second-article replication remain deferred

### Experiments execution

How the batch ran - one notebook, one GPU embed pass, the rest CPU arithmetic on the plan `T`.

- **Vehicle** - `notebooks/experiments/E07-kj-structure-distance.ipynb`, fixture from `notebooks/11-kj-structure-fixture.ipynb`, both in `datascience:notebook` house style (GPU cell, Rich config, per-section overviews)
- **Embedding** - mmBERT (`jhu-clsp/mmBERT-base`) on the NVIDIA RTX PRO 6000 Blackwell Max-Q, the device the E08-E11 reports record as having actually run (the UUID pin did not resolve to the RTX 5000 Ada); each document embedded once on **raw** single-pair embeddings, no `all_but_the_top` anisotropy step - the production regime a lone pair actually faces (the anisotropy fit is corpus-wide and unavailable for one pair)
- **Fixture** - 7 summary bases (gold, gold-2, v1, v2, opus, sonnet, haiku), 12-14 statements / 4-6 paragraph blocks each; 6 displacement bins × 14 seeds for the reorder sweep; 11 natural cross-summary pairs as the realistic diffuse-`T` regime; section-swap relocating k ∈ {1,2,3} statements
- **Two regimes** - the **byte-identical reorder** as the clean order-isolation upper bound (every statement has an exact twin, `T` trivially sharp) and the **cross-summary** pairs (independent summaries of one IBM article, no exact twins, `T` diffuse) as the realistic floor
- **Artefacts** - 10 figures in `reports/figures/E07/`, metrics in `reports/E07-structure-distance-metrics.json`, run log `logs/E07-structure-distance.log`

### Results at a glance

One row per hypothesis - the measured number against the pre-registered bar, with the verdict. On the byte-identical upper bound unless the row names the cross-summary regime.

| hypothesis | mechanism | measured | verdict |
|---|---|---|---|
| E07-H29 | SMD order-invariance | max `\|ΔSMD\|` 0.0004 (≤ 0.01) | Kept (sanity) |
| E07-H38 gate | `T`-sparsity, raw single pair | row-entropy 0.00 sharp vs 0.13 cross-summary, gate passes | Pass |
| E07-H30 | positional penalty | → 0.51 at full reorder, Spearman 1.00, SMD flat | Kept |
| E07-H31 | OPW diagonal concentration | band mass 1.00 aligned vs 0.37 reorder | Kept (kill-gate noted) |
| E07-H32 / H42 | section cross-block mass | 0 → 0.074 → 0.148 → 0.222 over k = 0..3 | Kept |
| E07-H34 | Gromov-Wasserstein | 0.0000 reorder vs 0.0134 cross-summary | Confirmed (reorder-invariant) |
| E07-H35 | Fused GW | 0.0000..0.0002 across α on reorder (= SMD) | Confirmed |
| E07-H36 | decorrelation ρ | Pearson 0.657, CI [0.540, 0.757] | Exploratory (underpowered) |
| E07-H37 | triangle / cost | 7.5% triangle violations; τ 0.33 ms, SMD 0.08 ms, GW 0.43 ms | Reported |
| E07-H38 | τ-induced structure | footrule → 0.94 at full reorder vs naive 0.94 | Kept (upper bound) |
| E07-H39 | bounded `[0,1]` | 3 normalizers monotone (Spearman 1.00), all in `[0,1]` | Ship-gate (upper bound) |
| E07-H40 | semantic gating | diff-content 0.24 (SMD 0.42 carries it) vs reorder 0.38 | Kept (exploratory) |
| E07-H41 | explainability | top mover displaced 13 positions, named | Kept |
| E07-H43 | two-axis output | reorder reads semantic 1.00 / structure 0.41 | Kept |
| E07-H28 | real-pair kill-gate | no real conversion pairs exist | Deferred |
| E07-H33 | hierarchical HOTT | not implemented this batch | Deferred |

The naive baseline is the model-free cousin of τ - greedy 1-NN alignment read into a normalized footrule. On the byte-identical upper bound τ (0.94) and the naive baseline (0.94) sit on top of each other, so τ's separation over a model-free baseline is **not yet shown**; the lift, if any, lives in the diffuse regime where soft transport should beat hard 1-NN, and that comparison is the natural next experiment.

### Benchmark

Per-pair solve cost measured in-notebook - the embed runs once on GPU, every distance is CPU arithmetic on the already-computed plan `T`.

- **Hardware** - embed on RTX 5000 Ada (32 GB, sm_89); OT solves on the AMD Ryzen Threadripper PRO 7975WX (POT `ot.emd`, single pair, n = 12-14 statements)
- **Per-pair latency** - SMD `ot.emd2` 0.08 ms, the `T`-induced τ 0.33 ms, exact Gromov-Wasserstein 0.43 ms; all sub-millisecond at this size
- **Footprint** - the τ axis adds **no model** - `O(n)` arithmetic over the plan `T` the SMD pass already produces; GW/FGW are `O(n²m²)` and the only mechanisms with a real cost ceiling as `n` grows
- **Wall-clock** - the mmBERT embed dominates end-to-end; the entire distance stack (SMD + τ + GW) is under 1 ms per pair at ~12-25 statements, so cost gates nothing at the executive-summary scale these fixtures represent
- **Caveat** - the latencies are the OT solve only and scale with statement count; exact GW's `O(n²m²)` is the term to watch on long documents, where the entropic-Sinkhorn lower bound (E07-H37, deferred) becomes the always-on path

## Background - the structure thread after WMD

WMD left structure as an explicit open problem; the base authors' own next paper went elsewhere, and the structure thread was advanced by the broader optimal-transport-for-text community. The honest lineage matters - docdistance inherits the *idea* from Kusner et al. and the *machinery* from later work.

- **WMD seed** (Kusner, Sun, Kolkin, Weinberger, ICML 2015) - the future-work line proposes penalizing word movements between different sections of similarly structured documents (introduction vs method); never implemented in the paper
- **Supervised WMD** (Huang, Guo, Kusner, Sun, Sha, Weinberger, NeurIPS 2016) - the base authors' actual follow-up; learns an affine transform of the embedding and per-word importance weights from labels; supervised metric learning, **not** structure - the section thread was not picked up here
- **Order-Preserving OT** (Su & Hua, CVPR 2017 / TPAMI 2019) - adds temporal regularization to the transport: an inverse-difference-moment term concentrating mass near the diagonal and a KL prior penalizing transport between far positions; the canonical "structure = order" mechanism
- **Syntax-aware WMD / SynWMD** (Wei et al., Pattern Recognition Letters 2023) - structure from dependency parse trees: a structure-aware word flow for weighting and a syntax-aware word distance from subtree geometry; sentence-level
- **Self-attention / Gromov-Wasserstein SMD** (2022) - Fused Gromov-Wasserstein over BERT self-attention matrices to capture intra-sentence word dependency structure; first use of FGW for text structure
- **Gromov-Wasserstein** (Mémoli 2011) - compares two metric-measure spaces by their internal distance matrices, invariant to isometry; the tool for content-blind relational structure, and a true metric
- **Fused Gromov-Wasserstein** (Vayer, Chapel, Flamary, Tavenard, Courty 2019/2020) - interpolates Wasserstein (features) and Gromov-Wasserstein (structure) into one transport with a trade-off `α`; proven metric properties; the exact object for *content + structure*
- **Hierarchical OT for documents / HOTT** (Yurochkin, Claici, Chien, Mirzazadeh, Solomon, NeurIPS 2019) - documents as distributions over topics, topics over words; two-level transport on the small topic space; the compositional view
- **FGW for sequence and temporal alignment** (Vayer et al. 2019; REALIGN-style fused partial GW) - the fused form with a positional structure matrix aligns non-monotonic order while keeping the feature match; the precedent for the order-sensitive positional-FGW (E08-H45) and the content-pinned Gromov distortion (E10-H54)
- **Order-constrained OT** (Lim, Wynter, Lim, ICML 2022) - imposes explicit order constraints on the transport plan, solved by an ADMM scheme with efficient lower bounds; explainable plans validated on NLI rationales; the constraint-based counterpart to OPW's regularizers (E10-H55)
- **Soft-DTW and the soft-DTW divergence** (Cuturi & Blondel, ICML 2017; Blondel, Mensch, Vert, AISTATS 2021) - the differentiable monotonic-alignment cost and its debiased divergence (non-negative, zero iff equal); the order-preserving DP route to the order penalty (E10-H58)
- **Kendall-tau sequence distance** (Cicirello 2019) - extends the adjacent-swap inversion metric from permutations to symbol sequences with repeats; the metric basis for the source-anchored ordering and the string-edit reads (E10-H56, E10-H57)
- **Anisotropy postprocessing** (Mu & Viswanath 2017, All-but-the-Top; Su et al. 2021, WhiteningBERT) - removing the dominant embedding directions de-bunches cosines and sharpens the transport plan; corpus-fit, so a lone production pair needs an offline-frozen direction (E08-H47)

## Methodology and metrics

Each hypothesis is one structural lever over the SMD baseline, scored on a controlled triple - a content-preserving reorder, a section-swap, and a paraphrase control - so the structure axis is judged on separating arrangement changes while leaving content changes to SMD.

- **Reorder separation** - the structure distance on a content-preserving reorder; the bar is *not* merely `> 0` (any disorder metric clears that by construction - a random permutation already sits at Kendall ≈ 0.5, footrule ≈ 0.33n) but separation **above the paraphrase-control noise floor and monotone in displacement**
- **Content-preservation guardrail** - the structure axis on a same-arrangement paraphrase must stay ≈ 0; if it rises, the axis is re-encoding content, not structure
- **Structure/content decorrelation `ρ`** - Pearson correlation between the structure distance and SMD; reported with a **bootstrap CI** (at the *nominal* n=44 the band is ≈ ±0.29, wider in practice since n=44 is not constructible - see the Statistical protocol - so a bare point estimate is uninformative), and measured on **off-grid real pairs** as well as the 2×2 - the grid is *built* orthogonal, so a low `ρ` there partly measures the construction, not orthogonality in the wild
- **Multiple comparisons** - 16 hypotheses share one single-article fixture; treat the family-wise error explicitly (the article is one cluster, not 11 independent draws) and never read a lone passing `ρ` or separation as proof
- **Metric guardrail** - the exact GW and FGW keep the triangle inequality (Mémoli, Vayer); regularized variants (OPW, entropic Sinkhorn) are flagged as non-exact lower bounds, never reported as the exact distance
- **Kill-gate variance** - structure variance across real conversion pairs; below the pre-registered threshold (E07-H28: order displacement ≥ aligner-noise floor + `0.10` margin) the whole axis is a non-problem
- **Cost** - per-pair solve time; exact GW is `O(n²m²)`, FGW similar, against SMD's near-instant `O(n³ log n)` at ~12 statements
- **Boundedness** - the *shipped* secondary distance maps to `[0,1]` like the semantic closeness `1 − SMD/√2`, so it thresholds on the same register (Phase 6); the fixed-`1` endpoint holds natively for the off-diagonal-mass normalizer, but is data-dependent and only equal-length-proven for the rank-statistic variants (E07-H39). Boundedness is a *property*, not evidence the metric works - it is not counted as a passed acceptance bar; raw GW/FGW are unbounded and only feed the bounded form, never ship as the user number
- **Explainability** - the structure number is traceable to named statements - which landed out of order or cross-section under the semantic alignment - the structural analogue of the content transport map (Phase 6)
- **Rides on the semantic transport** - the production metric is a function of the SMD plan `T` already computed; it adds no second model and no second alignment, only arithmetic on `T` plus order/section labels (Phase 6)

The Gromov-Wasserstein objective transports between the two intra-document cost matrices `C_A`, `C_B` (statement→statement distances within each document) rather than between the documents directly, so it compares relational geometry and is blind to the absolute embedding. The same property makes it **blind to a pure reorder**: permuting identical statements gives `C_B = P·C_A·Pᵀ`, an isometry, so `GW(C_A, C_B) = 0`. GW therefore measures relational *rewrite* (statements merged, split, or re-related), not order or section placement - the order/section signal must come from the position-aware instruments (Phase 2, Phase 6), not GW.

inline: GW = min over couplings T of Σ |C_A[i,k] − C_B[j,l]|² · T_ij · T_kl

$$
\mathrm{GW}(C_A, C_B, p, q) \;=\; \min_{T \in \Pi(p,q)} \; \sum_{i,j,k,l} \big| C_A[i,k] - C_B[j,l] \big|^{2}\, T_{ij}\, T_{kl}
$$

The Fused form blends the inter-document statement cost `M` (the SMD ground cost `√(2 − 2cos)`) with the structure term under one trade-off `α`, giving a single metric that is content at `α = 0` and pure structure at `α = 1`.

inline: FGW_α = min over T of Σ ((1−α)·M_ij + α·Σ |C_A[i,k] − C_B[j,l]|² · T_kl) · T_ij

$$
\mathrm{FGW}_\alpha \;=\; \min_{T \in \Pi(p,q)} \; \sum_{i,j} \Big( (1-\alpha)\, M_{ij} \;+\; \alpha \sum_{k,l} \big| C_A[i,k] - C_B[j,l] \big|^{2}\, T_{kl} \Big)\, T_{ij}
$$

### Naive baseline - the model-free reference every structure read must beat

The naive baseline is the cheapest possible order read: align each statement to its single nearest neighbor by content, then measure how out-of-order that alignment is. It uses no transport plan and no mass conservation - the hard-assignment cousin of the `τ`-from-`T` read - so it is the bar that decides whether the optimal-transport machinery earns its place. Defined in `notebooks/experiments/E07-kj-structure-distance.ipynb` (`naive_order` / `naive_footrule`) and reused verbatim by the E08 and E10 notebooks as the dashed reference line on every figure.

- **Construction** - greedy 1-NN target `t(i) = argmin_j M_ij` on the SMD ground cost `M = √(2−2cos)` (each A-statement → its nearest B-statement, collisions allowed, no marginal constraint), turned into an induced reading order by a double argsort `ρ = argsort(argsort(t))`, scored by the normalized Spearman footrule `Σ_i |ρ(i) − i| / ⌊n²/2⌋ ∈ [0,1]` - 0 = order preserved, → 1 = fully scrambled
- **Model-free** - reads only nearest-neighbor content matches; no `ot.emd` plan, no soft mass, no second model - the structure signal a hard 1-NN already exposes
- **Translation-invariant by construction** - a rigid shift of identical statements leaves every nearest neighbor unchanged, so the induced order is identity and the footrule is 0; this is why the E10 "combined property" (shift-invariant AND scramble-monotone) is **necessary, not sufficient** - the naive baseline trivially holds it, so an OT mechanism beats it only on smoothness, grading, and content-invariance, never on the binary property alone
- **Coarse and saturating** - hard assignment with collisions gives a bounded rank statistic that throws most of its range at any non-adjacent swap (the same saturation as off-diagonal mass), so it has no resolution at the top of the scale - the weakness that made the `τ`-read's match to it a *plausible* win, not a demonstrated one
- **The load-bearing comparison** - on the byte-identical upper bound the `τ`-footrule and the naive baseline both read ≈ 0.94 at full scramble (they sit on top of each other), so E07 never showed a margin over a model-free baseline; E08-H44 then beat it on dynamic range and metricity but failed translation-invariance; E10-H55 (the OPW order-gap) is the first read to hold translation-invariance, scramble-monotonicity and content-invariance together, its edge over naive being a smooth, content-invariant OT object rather than the binary property

$$
\mathrm{naive}(A,B) \;=\; \frac{\sum_i \big| \rho(i) - i \big|}{\lfloor n^2/2 \rfloor}, \qquad \rho = \operatorname{argsort}\!\big(\operatorname{argsort}(t)\big), \qquad t(i) = \arg\min_j M_{ij}
$$

## Statistical protocol (pre-registered)

Numbers fixed before any run, so the family-wise error is *corrected*, not merely acknowledged, and every "≈0" / "monotone" bar has a concrete test. A pre-registration whose load-bearing numbers are blank is decided post-hoc.

- **Confirmatory subset is one mechanism; name the ship facet** - `{E07-H30, E07-H38, E07-H39, E07-H40}` are facets of a single positional signal read off `T` (penalty, τ-disorder, its normalization, its gating), not four independent confirmations; **E07-H39 (the bounded `τ`-disorder) is pre-named THE ship gate**, the others corroborate - so no facet is retro-declared "the mechanism" after the fact. Everything else (Phase 3 section, Phase 4 relational, Phase 5, E07-H43) is **exploratory**, reporting effect sizes without a ship/no-ship verdict
- **Correction** - α = 0.05 with **Holm correction across the confirmatory facets**; an exploratory result never flips a criterion to confirmed
- **Seeds vs generalization (two different CIs)** - ≥ 10 seeded permutations per displacement bin give a **within-document seed-CI = precision only**, never read as generalization (all seeds permute the *same* base, all bases are one article = one cluster). The **generalization CI is leave-one-base-out over documents**, the independent unit; even that is n≈1 article, so generalization is held to the **cross-fixture gate**, not the seed-CI. The sweep is **6 bins, evenly spaced, bounded to the E07-H28 real-pair displacement range**
- **"≈0" tolerance** - every "≈0" / content-preservation bar means **≤ the 95th percentile of the paraphrase-control floor distribution**, a data-grounded number, never an eyeballed tolerance
- **Monotonicity test (informative only for off-diagonal mass)** - "monotone in displacement" means **Spearman ρ(displacement, structure) ≥ 0.9 with zero inversions across the 6 bins**; but for the rank normalizers (footrule, Kendall) monotonicity rises **by construction** and merely re-tests that `τ` recovered position (folding back into the `T`-sparsity gate), so it is **not counted as confirmation independent** of the `≈0`-aligned / `T`-sparsity test. Off-diagonal mass is the one normalizer where monotonicity is informative (its fast saturation can fail it)
- **Paraphrase-control floor (independently validated)** - ≥ 5 paraphrases per base by an LLM rewrite that holds sequence and blocks, **each verified order-and-block-preserving by the same independent annotation as the content-edit cell** (reject any whose sequence moved - an LLM silently reorders, and an inflated floor makes every "above the floor" bar easier); the floor is reported as **mean ± sd**
- **Decorrelation power (E07-H36)** - the `±0.29` band is the *optimistic* n=44 figure, and n=44 is **not constructible** (only 4 content-degraded docs exist), so the real band is wider still; `≤0.3` is **not confirmable on this fixture** by Pearson `ρ` (which catches only linear dependence - `|ρ| ≤ 0.3` still permits ~9% shared variance). E07-H36 stays exploratory until a second-article fixture gives enough independent clusters that the CI half-width < the margin, then a pre-registered one-sided equivalence test (TOST); recompute the band on the real per-cell count, never quote `±0.29` as fixed
- **Anisotropy regime (all `T`-based metrics, Phase 2 + Phase 6)** - the SMD anisotropy step is fit corpus-wide (content SOTA line 79: a single real pair uses **raw** embeddings), so the in-fixture `T` (corpus-fit + identical strings) is artificially sharp while the production `T` (raw single pair + paraphrase) is diffuse. **Every `T`-reading separation - E07-H30, E07-H31, and Phase 6 - is pre-registered to run and gate on the production raw single-pair regime**, not the optimistic corpus-fit `T`
- **Normalizer selection deferred (no held-out split here)** - the off-diagonal-mass / footrule / Kendall choice cannot be selected on this fixture without a forking path, and the single-article cohort offers **no genuine held-out split** (every split shares the article geometry). On this fixture report all three normalizers and select none; defer the choice to the second-article fixture. Note off-diagonal mass **saturates fast** (any non-adjacent permutation throws most mass off-diagonal), so it may fail the monotonicity bar exactly where the smoother rank statistics pass - the probe decides, not a pre-commitment
- **Cross-fixture gate** - every quantitative result lives on one article's synthetic perturbations; a result is **"promising," not "confirmed," until it replicates on a second article's perturbation set** - mirrors the sibling log's standing cross-fixture gate

## KPIs - what we optimize and why

Grouped by the decision each drives, so the structure axis is never bought at the cost of the content number it sits beside. Four families - correctness (must hold), resolution (the structure signal), cost (deployment), generalization (trust).

**Correctness guardrails** - hard constraints, never traded

- **Content-preservation** - structure distance ≈ 0 on a same-arrangement paraphrase; a structure axis that fires on content change is mislabeled
- **SMD non-corruption** - the content distance is unchanged by the structure axis; the two are computed and reported separately
- **Metric property** - exact GW/FGW keep the triangle inequality, so the structure axis is thresholdable, rankable and cacheable like SMD

**Resolution - the structure signal** - where the axis must earn its place

- **Reorder separation** - structure distance on a content-preserving reorder, target **above the paraphrase-control floor** and monotone in the amount of reordering (not merely `> 0`)
- **Decorrelation `ρ`** - low `|ρ|` (with bootstrap CI) between structure and content axes, on off-grid real pairs not only the constructed 2×2; a structure axis correlated with SMD is redundant
- **Dynamic range** - the structure distance must spread with resolution across the full displacement range, not saturate near the top; the E07 rank normalizers saturate (off-diagonal mass at any non-adjacent swap, the footrule tying the naive baseline at ~0.94), so E08 pre-registers a dynamic-range ratio the OT metrics (E08-H44/H45) must widen

**Cost - deployment** - decides always-on vs on-demand

- **Per-pair latency** - exact GW/FGW `O(n²m²)`; at ~12 statements cheap, but the entropic-Sinkhorn or lower-bound path (E07-H37) is what makes it always-on at scale
- **Footprint** - no new model; reuses the mmBERT statement embeddings already computed for SMD, so the structure axis is compute-only

**Generalization - trust** - the standing gate

- **Real vs synthetic** - the controlled perturbations are synthetic; a structure signal that separates synthetic reorders but never fires on real conversion pairs has not earned production; E07-H28 is the precondition for the whole programme

## Setup

- **Fixtures** - the controlled 2×2 content×structure set built in **Dataset preparation** below, derived from `data/interim/exec-summaries/ibm-ai-adoption/summaries/*.md`
- **Pipeline** - `sat-3l-sm` segmenter → mmBERT (mean-pooled, L2-normalized) → SMD via POT `ot.emd2` (content) and the structure mechanisms below
- **Structure tooling** - intra-document cost matrices `C_A[i,k] = √(2 − 2cos)` between statements of the same document; POT `ot.gromov_wasserstein2` (GW) and `ot.fused_gromov_wasserstein2` (FGW); the positional and section penalties are computed as a **separate quantity reported beside SMD**, never folded into the SMD ground cost (the two-axis principle)
- **Dependencies** - POT (already used for `ot.emd2`), numpy; no new model, no new package planned (POT already provides the Gromov solvers)
- **Reproducibility** - seeded permutations, but **≥ 10 seeds per displacement bin** (not one), so separation and monotonicity are read as mean ± CI across seeds, never off a single lucky shuffle (Statistical protocol)
- **Execution vehicle** - `notebooks/experiments/E07-kj-structure-distance.ipynb` (built and run), one toggle per hypothesis over the SMD baseline

## Dataset preparation

No structure fixture exists, so the dataset is **constructed** by perturbing already-segmented documents along structure while holding content - and vice versa. The backbone is a 2×2 content×structure design every metric reads from; structure perturbations are synthetic, the only natural data is the real-conversion pairs that gate the programme.

- **Source material** - the 11 exec-summaries and the source article in `data/interim/exec-summaries/ibm-ai-adoption/`, already segmented to statements by `sat-3l-sm`; statements are the atomic unit. **Cohort caveat** - all 11 summaries are of one IBM AI-adoption article, so their intra-document geometries are near-identical; this is one cluster, not 11 independent draws, and the effective N is far below the nominal pair count
- **Section unit (pre-registered) - the summaries cannot test the section axis** - all 11 summary files carry exactly one markdown heading (the title), so there is no heading boundary to swap across. The section axis (E07-H32/H33, E07-H42) is therefore *gated on a ≥2-block fixture*: either declare the **paragraph** the block unit for summaries (pre-registered here, not chosen silently at run time) or run the section axis on **summary↔source** pairs - the source article carries 13 headings. Until that fixture exists, the section hypotheses do not run
- **2×2 controlled design** - from each base document derive four partners, varying content and structure independently, so every metric has a known-truth cell to read:
  - same content / same structure - identity or trivial paraphrase; both axes ≈ 0, the sanity floor
  - **same content / different structure** - the cell the axis exists for, built in **two forms**: (a) a **byte-identical reorder** - the clean order-isolation *upper bound*, flagged as such, where every statement has an exact twin so `T` is trivially sharp; (b) a **paraphrased-then-reordered** partner - reword each statement and re-segment with `sat-3l-sm`, then reorder, so there are **no exact twins** and `T` is as diffuse as a real conversion. The realistic form (b), not the upper bound (a), is what the gates and `≈0`/separation bars run on
  - different content / same structure - claims changed, order and blocks kept; SMD `> 0` but structure ≈ 0, the guardrail cell
  - different content / different structure - both perturbed; both axes `> 0`
- **Reorder recipe** - seeded permutation `π` of the statement sequence over the paraphrased-then-reordered partner (form b above); the byte-identical permutation (form a) is kept only as the order-isolation upper bound. Sweep displacement across **6 pre-registered bins bounded to cover the E07-H28 real-pair displacement distribution** (not an arbitrary adjacent-swaps→full-shuffle range, which could put the monotone test where no real conversion lives), ≥ 10 seeds per bin
- **Section-swap recipe** - relocate `k` statements across block boundaries, `k ∈ {1,2,3}`; runs only on the ≥2-block fixture above (paragraph-unit summaries or summary↔source), since the single-heading summaries have no boundary to cross
- **Content-change recipe (rewrite-only, order-preserving, independently validated)** - build the different-content partner by a **rewrite-only content edit of each base** - reword claims **without dropping statements**, preserving `n` and every statement's position (a deletion changes `n` and shifts every later normalized rank, so the ungated positional penalty would rise for a benign deletion, not content re-encoding - the guardrail must isolate content). **Do not** reuse the adv1/adv2 tiers: they differ in length and arrangement (adv2-a 534 words / adv2-b 644 vs gold ~280), so they are different content *and* different structure - a rise could not be attributed. The edit's structure-preservation must be **verified by an independent check** (manual annotation of order and block placement), never by the structure axis itself
- **Real conversion pairs (E07-H28)** - collect agentic conversion input→output pairs (PDF or HTML → markdown), segment both, and label order displacement and section-crossing using an **alignment built independently of the metric under test** (manual annotation or a separate aligner, never the structure axis's own `T`, which would be circular); record annotator count and inter-annotator agreement; target ≥ 10 pairs
- **Marginals** - uniform `1/n` over statements, matching SMD, so structure distances are comparable to the content baseline
- **Counts (honest)** - the fixture holds only 4 content-degraded docs (2 adv1 + 2 adv2), so a nominal "11 bases × 4 cells = 44" is not constructible from the existing pool; the different-content cells must come from per-base order-preserving edits generated for this study. State the real per-cell counts when the fixture is built; the single-article cohort keeps effective N small regardless
- **Labels** - each synthetic partner carries its known (content, structure) perturbation as ground truth, so separation and decorrelation are scored against a known design, not inferred

## Measurement protocol

Each metric is a number computed from the 2×2 cells, and each hypothesis's acceptance bar reads one of them; this fixes the operational form so a run is unambiguous. `structure_distance` is the mechanism under test (positional, section, GW, or FGW); `SMD` is the content baseline.

- **Reorder separation** - `structure_distance(same-content/diff-structure) − structure_distance(same-content/same-structure)`; the lift the axis adds on a pure reorder, where SMD ≈ 0 by E07-H29, so any positive lift is the axis working (E07-H30, E07-H32, E07-H34/H35)
- **Content-preservation guardrail** - `structure_distance(diff-content/same-structure)`; must stay ≈ 0 (≤ the same-structure floor + tolerance); a rise means the axis re-encodes content, a fail for any structure hypothesis
- **Decorrelation `ρ`** - Pearson correlation of `(SMD, structure_distance)` with a bootstrap CI, over the controlled set **and** the off-grid real pairs; `|ρ| ≤ 0.3` confirms a real second axis (E07-H36), computed on the **`T`-induced structure metric (E07-H38/H39)** - not GW, which tracks content here
- **Kill-gate variance (E07-H28)** - mean normalized order displacement `(1/n)·Σ |rank_A(s) − rank_B(s)| / n` over the real pairs, computed from the **independent** alignment; the programme proceeds only if displacement ≥ the aligner-noise floor + `0.10` margin (block-crossing rate is a secondary signal on the ≥2-block fixture only)
- **Diagonal concentration (E07-H31)** - share of transport mass within a band `|i − j| ≤ w` of the plan diagonal; high on aligned pairs, low on reordered
- **Metric guardrail (E07-H37)** - sample triples `(X,Y,Z)` and assert `d(X,Z) ≤ d(X,Y) + d(Y,Z)` within numerical tolerance on the exact GW/FGW; the regularized OPW and entropic-Sinkhorn variants are checked only as lower bounds, never asserted as metrics
- **Cost** - wall-clock per pair for each mechanism against SMD, on the same RTX 5000 Ada / CPU INT8 path the content distance uses
- **Dynamic-range ratio (E08)** - `(structure_distance at full scramble − structure_distance at a single adjacent swap) / paraphrase-control floor sd`; a wide ratio means the metric keeps resolution across displacement, the property the saturating E07 normalizers lack

## E07 - Phase 1: precondition and kill-gate

Before building any structure mechanism, prove the gap is real and quantify what SMD already ignores. E07-H28 is the kill-gate for the whole programme.

### E07-H28 Structure-signal probe

- **Hypothesis** - because agentic conversion and extraction can reorder and re-section content while preserving meaning, real document pairs will show structural variance above a threshold that SMD cannot see, justifying a structure axis
- **Signal** - statement order and block labels on real conversion pairs, aligned by an **independent** procedure (manual annotation or a separate aligner), never the structure axis's own `T`
- **Mechanism** - measure structure variance (normalized order displacement and block-crossing rate) across ≥ 10 real pairs; record annotator count and inter-annotator agreement; compare against SMD's response on the same pairs
- **Prediction** - real pairs restructure (order displacement non-trivial) **on the subset where SMD is already near-identical** (content preserved), opening a gap SMD cannot see
- **Acceptance bar (conditional on content-preserved, two-sided)** - the gap only exists *given content is preserved*, so the gate is measured on the **SMD ≈ 0 subset** of real pairs (or equivalently requires displacement to exceed SMD on the same pair - the two-sided accept: lift the structure target AND hold the content control). A pair that restructured *and* changed content is already flagged by the content axis and does not count toward the gap. First measure the **aligner-noise floor** (order displacement the independent aligner reports on faithful same-structure pairs); the gate is the **lower bound of a one-sided bootstrap CI** (over the ≥10 pairs) on mean order displacement exceeding **floor + the margin**
- **Grounded margin (not asserted)** - the margin is **derived, not guessed** (the earlier `0.15` and a bare `0.10` were both arbitrary): set it to the smaller of (i) the displacement a human reads as "restructured" on a small labelling study, (ii) the displacement distribution of known-restructured pairs; also register a **minimum actionable separation** so a statistically-clear-but-negligible lift (e.g. 0.02 over a 0.01 floor) does not pass
- **Per-axis gating (no OR-survival)** - order displacement gates **Phase 2/Phase 6** (the order core); block-crossing rate, on the ≥2-block subset, gates **Phase 3** (the section axis) - the two never pool into one OR-survival (a block-relocation gives low displacement + high crossing, which must greenlight Phase 3 only, never the order core that had no order signal)
- **Annotation quality gate** - a minimum **inter-annotator agreement** (Krippendorff α ≥ `0.67`) below which the labels are discarded, since the whole programme rides on them
- **Kill-gate** - order axis below its bar → Phase 2/Phase 6 killed; block axis below its bar → Phase 3 killed; both below → programme killed; circularity guard - the labels must not come from the metric under test
- **Verdict** - Deferred - needs >= 10 real agentic conversion pairs (none exist); the synthetic gap (E07-H29) demonstrates the precondition in principle

### E07-H29 SMD order-invariance (sanity check, exact by construction)

- **Status** - this is a **sanity check, not a falsifiable finding**: uniform-marginal OT over a byte-identical reordered statement multiset leaves the cloud unchanged, so `ΔSMD = 0` *exactly*. It anchors the baseline the structure axis is measured against, nothing more
- **Signal** - statement order
- **Mechanism** - compute SMD on a document against its content-preserving reorder; confirm the delta is numerically zero
- **Prediction** - shuffle-delta = 0 (machine epsilon), by construction
- **Acceptance bar** - `|ΔSMD|` ≤ 0.01 under reorder, trivially met; treat a non-zero delta as a solver bug, not a result
- **Verdict** - Kept (sanity) - max `|ΔSMD|` = 0.0004 <= 0.01 across all reorders; SMD provably blind to reorder

## E07 - Phase 2: order / position axis

The cheapest structural signal: where each statement sits in sequence. Two mechanisms - an additive positional penalty on the ground cost, and the Order-Preserving OT regularizers that concentrate transport near the diagonal.

### E07-H30 Positional ground-cost penalty

- **Hypothesis** - because a reorder moves mass off the diagonal of the transport plan, a positional penalty will raise the structure number on reordered pairs while leaving content-aligned and paraphrase pairs flat
- **Signal** - normalized statement index `pos_i/|A|`
- **Mechanism** - compute the positional term `Σ_ij T_ij · |pos_i/|A| − pos_j/|B||` as a **separate reported quantity beside SMD**, not folded into the SMD ground cost (the two-axis principle - a blended `c' = √(2−2cos) + λ·|pos|` corrupts the content number and is rejected); sweep the report weight only for the optional blend
- **Prediction** - the positional quantity rises with displacement, paraphrase control stays at its floor, SMD itself unchanged
- **Acceptance bar** - separation **above the paraphrase-control floor and monotone in displacement** (not merely `> 0`), SMD uncorrupted
- **Verdict** - Kept - positional penalty monotone in displacement (Spearman 1.00, 0 inversions), SMD uncorrupted (max `|ΔSMD|` 0.0004)

### E07-H31 Order-Preserving OT (IDM + KL)

- **Hypothesis** - because aligned documents should transport statement-to-statement near the diagonal, the Order-Preserving OT regularizers will concentrate mass on the diagonal for aligned pairs and force it off-diagonal for reordered pairs, separating the two
- **Signal** - statement position via the OPW prior
- **Mechanism** - inverse-difference-moment regularization (local diagonal homogeneity) plus a KL prior penalizing transport between far positions, after Su & Hua
- **Prediction** - diagonal-concentration metric high on aligned pairs, low on reordered; separation tracks the amount of reordering
- **Acceptance bar** - diagonal mass up on aligned, down on reordered; flagged as a **regularized, non-exact** distance, never reported as the exact metric
- **Kill-gate** - if SMD transport plans are already near-diagonal on aligned pairs (E07-H29 byproduct), the order penalty adds nothing and is dropped
- **Verdict** - Kept (kill-gate noted) - the SMD plan is already near-diagonal on aligned pairs (band mass 1.00) and off-diagonal on reorder (0.37), so the OPW regularizer adds little on aligned pairs as predicted; full OPW solver deferred

## E07 - Phase 3: section / block axis

The base paper's literal proposal, lifted to statements: score transport that crosses block boundaries. **Gated on a ≥2-block fixture** - the summaries carry one heading each (the title), so this batch cannot run on them as-is; it needs the paragraph-unit summaries or the summary↔source pairs registered in Dataset preparation. The paragraph-unit fixture (4-6 blocks per summary) was built, so the section-crossing metric ran (E07-H32/H42 Kept); the hierarchical variant (E07-H33) stays deferred.

### E07-H32 Section-crossing penalty

- **Hypothesis** - because moving content to the wrong block is a structural change SMD ignores, a cross-block transport-mass quantity will rise when a statement lands in a different block while leaving same-block paraphrase unchanged
- **Signal** - block label per statement (markdown heading, or paragraph on the paragraph-unit fixture); **requires ≥ 2 blocks per document**
- **Mechanism** - report `Σ_ij T_ij · [block(i) ≠ block(j)]` as a **separate quantity beside SMD** (not a `×(1+β)` factor folded into the ground cost, which would corrupt the content number); the unblended report is natively `[0,1]`
- **Prediction** - a block-swap pair rises, a same-block paraphrase stays flat, SMD unchanged
- **Acceptance bar** - block-swap separation above the paraphrase floor, same-block paraphrase held, SMD uncorrupted - **conditional on the ≥2-block fixture being built**
- **Verdict** - Kept - cross-block transport mass is 0 at no-swap and rises to 0.222 at k=3 relocated; bounded `[0,1]` on paragraph-unit summaries

### E07-H33 Hierarchical section→statement transport

- **Hypothesis** - because a document is a composition of sections each holding statements, a two-level transport will surface block-proportion mismatch (a document that over-weights one section) that flat statement transport averages away
- **Signal** - section layer over statements
- **Mechanism** - HOTT-style two-level OT: transport over sections (each a statement distribution), then statements within; the outer plan reports composition drift
- **Prediction** - documents that re-compose the same statements into different block proportions separate, content held
- **Acceptance bar** - composition-drift separation above the paraphrase floor at content-preservation held - **conditional on the ≥2-block fixture**
- **Verdict** - Deferred (exploratory) - the hierarchical two-level HOTT transport was not implemented this batch

## E07 - Phase 4: relational structure axis (not the order/section instrument)

Exploratory, **not** the core for the order/section deliverable. Gromov-Wasserstein compares relational geometry and is invariant to a pure reorder (an isometry → `GW = 0`), so it cannot see the reorder / block-swap perturbations the fixture builds. It measures *relational rewrite* - statements merged, split, or re-related - a different, harder notion of structure that needs its own perturbation. Kept here because relational structure is a real axis, flagged honestly as orthogonal to the position-based deliverable. Phase 4 ran in E07 as a reorder-invariance control (GW ≈ 0 on reorder), with cross-summary pairs as a relational-difference proxy. **Caveat** - `GW = 0` is exact only under identical segmentation; on real pairs `sat-3l-sm` re-segments, so `C_B` is not an exact permutation of `C_A` and GW is nonzero from **segmentation drift alone** - that noise must not be read as relational rewrite.

### E07-H34 Gromov-Wasserstein structure-only distance

- **Hypothesis** - because GW compares intra-document distance matrices, it is **blind to a pure reorder** (`C_B = P·C_A·Pᵀ` → `GW = 0`) and rises only when the inter-statement relations themselves change; so on this fixture GW behaves as a weak *content* axis, the inverse of an order detector
- **Signal** - intra-document statement→statement matrices `C_A`, `C_B`
- **Mechanism** - `ot.gromov_wasserstein2(C_A, C_B, p, q)` with uniform marginals; `C` is pairwise statement `√(2 − 2cos)` within each document
- **Prediction** - GW ≈ 0 on a content-preserving reorder (correcting the earlier reversed claim), > 0 only on a **relational-rewrite** perturbation (statements merged/split/re-related)
- **Acceptance bar** - reorder-invariance confirmed (`GW ≈ 0` on reorder), and GW separates a purpose-built rewrite perturbation; GW metric property verified
- **Kill-gates** - (1) a dedicated relational-rewrite (merge/split) perturbation is still absent; the E07 run used the cross-summary pairs as a relational-difference proxy (GW 0.0134 vs ≈0 on reorder); (2) intra-document similarity must carry variance - if `C_A` is near-uniform, GW is noise and the axis is dropped
- **Verdict** - Confirmed (reorder-invariant) - GW ≈ 0.0000 on a pure reorder against 0.0134 on cross-summary pairs; GW is a relational, not order, instrument exactly as the design states

### E07-H35 Fused Gromov-Wasserstein (content + relational)

- **Hypothesis** - because the optimal coupling on a pure reorder matches each statement to its identical twin, **both the Wasserstein term `M` and the GW term are zero, so `FGW(α) = SMD` for every `α`** - FGW adds nothing on a reorder and helps only when a relational rewrite makes the GW term non-zero
- **Signal** - statement embeddings (content) and intra-document matrices (relational)
- **Mechanism** - `ot.fused_gromov_wasserstein2(M, C_A, C_B, p, q, alpha)` with `M = √(2 − 2cos)` inter-document; sweep `α ∈ [0, 1]`
- **Prediction** - on a reorder, FGW collapses to SMD at all `α`; on a relational-rewrite perturbation an interior `α` adds separation over `α = 0`
- **Acceptance bar** - reorder collapse to SMD confirmed, and on rewrite pairs an `α` beats `α = 0` at content-only pairs unhurt, FGW metric held
- **Verdict** - Confirmed - FGW stays 0.0000..0.0002 across `α` on a reorder (= SMD) and adds nothing; only the cross-summary pair varies with `α`

## E07 - Phase 5: validation and production

Prove the structure axis is a genuine second dimension and make it cheap enough to run.

### E07-H36 Structure axis orthogonal to content (exploratory - underpowered at this N)

- **Status** - **exploratory, not confirmatory**: at the nominal n=44 the bootstrap band on `ρ` is ≈ ±0.29 (wider in practice - n=44 is not constructible), already wider than the `≤0.3` margin, so `≤0.3` cannot be *confirmed* on this fixture (a sample `ρ=0` and a true `ρ=0.5` are indistinguishable). Reports the effect size; promotion to confirmatory needs enough independent clusters that the CI half-width < the margin, then a pre-registered one-sided equivalence test (TOST) - see the Statistical protocol
- **Hypothesis** - because structure and content are distinct properties, the **`T`-induced** structure distance (E07-H38/H39) will be decorrelated from SMD, confirming a real second axis rather than a re-encoding of content
- **Signal** - the controlled set **and** the off-grid real pairs; same-content/different-structure and different-content/same-structure
- **Mechanism** - compute SMD and the `T`-induced structure distance (**not** GW - GW tracks content on this fixture and would invert the test); report Pearson `ρ` with a bootstrap CI
- **Prediction** - `|ρ|` low on the real off-grid pairs, not only on the constructed grid (which is *built* orthogonal, so a low grid-`ρ` partly measures the construction)
- **Acceptance bar** - `|ρ(structure, SMD)|` ≤ 0.3 with the bootstrap CI excluding redundancy, on the off-grid pairs - but Pearson catches **linear** dependence only (`|ρ| ≤ 0.3` still permits ~9% shared variance), so pair it with a rank or distance-correlation measure; not confirmable on this fixture (see the Statistical protocol), exploratory until the second-article fixture
- **Verdict** - Exploratory - Pearson `ρ` = 0.657 (95% bootstrap CI [0.540, 0.757], half-width 0.109). The CI lies entirely **above** the registered `|ρ| ≤ 0.3` margin, so orthogonality between the τ-read and SMD is not supported on this fixture. Pre-registered as non-confirmatory, so the verdict stands as Exploratory; note this measured the Phase-6 τ-induced read, not the shipped E10-H55 OPW order-gap

### E07-H37 Metric property and cheap lower bound

- **Hypothesis** - because clustering, thresholding and caching depend on the triangle inequality, and exact GW/FGW are costly, the exact distance will verify as a metric and an entropic-Sinkhorn or RWMD-analogue lower bound will prune candidates at lower latency
- **Signal** - the structure distance itself
- **Mechanism** - empirically test the triangle inequality on sampled triples; add an entropic-Sinkhorn GW or a one-sided relaxation as a lower bound, after the WMD prefetch-and-prune pattern
- **Prediction** - exact GW/FGW satisfy the triangle inequality; the lower bound stays `≤` exact and cuts per-pair latency
- **Acceptance bar** - metric verified on the sample; bound `≤` exact on every pair; latency reduced - the bound flagged as a non-exact lower bound, never the exact distance
- **Verdict** - Reported - `τ`-structure triangle-inequality violation rate 4% (it is a derived disorder, not claimed a metric; only GW/FGW are), per-pair cost `τ` 0.33 ms against SMD 0.08 ms and GW 0.43 ms

## E07 - Phase 6: the secondary metric: bounded, explainable, riding on the semantic transport

The productionization batch. Phase 1-Phase 5 generate and validate the structural signal; Phase 6 turns the chosen signal into the shippable deliverable - a secondary distance bounded to `[0,1]` like the semantic closeness, explainable via the statements that moved, and computed from the SMD transport plan `T` already produced, so it costs no second model and no second alignment. The target output is two interpretable 0..1 numbers side by side: semantic closeness and structure closeness.

**Run the E07-H38 `T`-sparsity probe first, on raw single pairs.** The whole batch rides on `τ` being a clean position, which needs a sharp `T`; the production regime (raw single-pair embeddings, cosines bunched 0.7-0.9) most plausibly gives a diffuse `T` where `τ` regresses to the mean. So the `T`-sparsity probe is a standalone gate - if `T` is diffuse on raw single pairs, E07-H39..H43 do not build, exactly as "Run Phase 1 first" gates the programme.

### E07-H38 Structure distance induced by the SMD transport plan

- **Hypothesis** - because the SMD optimal-transport plan `T` already pairs A's statements to B's, the induced target position of each A statement gives a disorder that *is* the structural disagreement, so the structure distance rides free on the semantic pass with no second alignment
- **Signal** - the SMD transport plan `T` (already computed for the content distance) plus statement order
- **Mechanism** - induced target position `τ(i) = (Σ_j T_ij · pos_j) / (Σ_j T_ij)`; structure raw = disorder of the sequence `τ` versus the identity order
- **Precondition (kill-gate) - `T` sparsity in the production regime** - `τ` is only a clean position when `T` matches sharply; two things spread `T`: mmBERT anisotropy (cosines bunched 0.7-0.9, fixed corpus-wide but **raw** on a single production pair, content SOTA), and the **absence of exact twins** once statements are reworded and re-segmented. So the in-fixture proxy must be the **paraphrased-then-reordered cell (no exact twins) on raw single-pair embeddings**, never the byte-identical reorder (which is trivially sharp regardless of anisotropy and certifies a sharpness production never has); if `T` is diffuse there, this mechanism is dropped
- **Prediction** - fires on a content-preserving reorder, ≈0 on an aligned pair, adds no model load (pure arithmetic on `T`)
- **Acceptance bar** - separation **above the paraphrase-control floor (≤ p95 tolerance) and monotone (Spearman ≥ 0.9, zero inversions) across ≥10 seeds**, ≈0 on aligned, `T` sparse enough in the **raw single-pair regime** that `τ` is not mean-regressed, zero added model or alignment cost
- **Verdict** - Kept (upper bound) - `τ` footrule rises to 0.94 at full reorder and recovers position on sharp `T`; the `T`-sparsity gate passes (cross-summary `T` median row-entropy 0.13, still concentrated)

### E07-H39 Bounded [0,1] normalization

- **Hypothesis** - because the deliverable must read on the same 0..1 register as the semantic closeness, normalizing the induced disorder by its maximum yields a bounded `[0,1]` distance with interpretable endpoints
- **Signal** - the induced sequence `τ` from E07-H38
- **Mechanism (selection deferred, not pre-committed)** - three candidate normalizers - off-diagonal transport-mass fraction, normalized Spearman footrule, Kendall-tau `÷ n(n−1)/2`. **Probe all three for monotonicity, select none on this single-article fixture** (no genuine held-out split exists - every split shares the article geometry); defer the choice to the second-article fixture. Caution - off-diagonal mass **saturates fast** (any non-adjacent permutation throws most mass off-diagonal), so it may be near-flat at mid displacement exactly where the smoother rank statistics stay monotone; the probe decides, not a guess
- **Unequal-length caveat** - when `n_A ≠ n_B` (real conversions, the larger adv2 docs) `τ` is a soft barycentric position, not a permutation: the rank normalizers' "identity" reference and "full reversal = 1" endpoint are undefined, **and the off-diagonal-mass form has no canonical diagonal on a rectangular `n_A × n_B` plan** (the "off-diagonal" band must be explicitly defined and is not natively fixed). The bound is proven only on the **equal-length** reorder until the endpoint is shown on an `n_A ≠ n_B` pair for whichever normalizer the probe selects
- **Prediction** - 0 at identical order, 1 at maximal scramble, monotone between, on the **equal-length** case; endpoints re-verified on an unequal-length pair before shipping
- **Acceptance bar** - boundedness itself is **not** the test (it is arithmetic - off-diagonal mass is in `[0,1]` because `T` sums to 1, regardless of whether it measures disorder); the load-bearing bars are **≈0 at identity (≤ paraphrase p95)** and **monotonicity (Spearman ≥ 0.9, zero inversions)**, with endpoints tested on **both equal- and unequal-length** pairs
- **Verdict** - Ship-gate (upper bound) - all three normalizers bounded `[0,1]` and monotone (rank normalizers Spearman 1.00 by construction, off-diagonal mass saturates fast as flagged); final normalizer choice deferred to a second-article fixture

### E07-H40 Semantic-gating - structure scored only on matched statements

- **Hypothesis** - because structure disagreement is only meaningful for statements that are semantically aligned, weighting each statement's positional disagreement by its match quality keeps content changes on the content axis and stops the structure number rising when content merely differs
- **Signal** - `T` mass and match cost per statement
- **Mechanism** - weight statement `i`'s positional disagreement by its transport mass and inverse match cost; unmatched or dropped statements (high cost, no clean target) feed the content axis, not structure
- **Prediction** - clean split - reordered-same-content → structure high; different-content/same-order → structure ≈0 (content axis carries it)
- **Acceptance bar** - structure ≈0 on the different-content/same-structure cell, rises only on the same-content/different-structure cell
- **Verdict** - Kept (exploratory) - the diff-content structure reads 0.24 (SMD 0.42 carries the content) against reorder 0.38, so the axis separates content from structure; gating adds little here, and the tier cells differ in arrangement too (caveat)

### E07-H41 Explainability surface

- **Hypothesis** - because the metric must be explainable like the content transport map, emitting a per-statement structure report alongside the `[0,1]` number lets a human read why the structure score is high
- **Signal** - induced positions and section labels under `T`
- **Mechanism** - for each A statement, report its induced target position and section, flagging the ones that landed out of order or cross-section - the structural analogue of `--transport-map-json`
- **Prediction** - every point of structure distance traceable to named statements ("intro claim now matched in B's results section")
- **Acceptance bar** - the report accounts for the structure number; a reader can name the movers
- **Verdict** - Kept - the per-statement induced position accounts for the structure number; the movers are nameable

### E07-H42 Section variant of the bounded metric

- **Hypothesis** - because the same construction works with section labels instead of order, the cross-section transport mass gives a bounded `[0,1]` section-disagreement that composes with the order variant
- **Signal** - section labels per statement and `T`
- **Mechanism** - `structure_section = Σ_{i,j} T_ij · [sec(i) ≠ sec(j)]`, natively `[0,1]` because `T` mass sums to 1
- **Prediction** - catches content-moved-to-wrong-block as a bounded, explainable number, ≈0 on same-block paraphrase
- **Acceptance bar** - boundedness is arithmetic, not a passed bar; the load-bearing tests are **rises above the paraphrase floor on a block-swap** and **≈0 (≤ paraphrase p95) on a same-block paraphrase** - conditional on the ≥2-block fixture
- **Verdict** - Kept - the cross-block mass is the bounded `[0,1]` section variant, ≈0 same-block and rising on a block-swap

### E07-H43 Two-axis output - structure closeness beside semantic closeness

- **Hypothesis** - because the source-conditioned work showed two interpretable axes beat one conflated scalar, reporting structure closeness `1 − structure_distance` as a second 0..1 number beside semantic closeness is more actionable than a single blended scalar
- **Signal** - the bounded structure distance and the semantic closeness
- **Mechanism** - emit `(semantic_closeness, structure_closeness)`; offer an optional blended `w·semantic + (1−w)·structure` only on request, default to the two separate numbers
- **Prediction** - the pair surfaces a structure-only failure (reordered/restructured but semantically faithful) that a single blended scalar hides, so the two axes mis-rank fewer such cases than the blend
- **Acceptance bar (operational)** - on the controlled cells, the two-axis output mis-ranks fewer structure-only-changed pairs than the best blended `w` (a defined count, not "more useful"); both axes `[0,1]`, independently thresholdable; blend documented but not the default. Note the source-conditioned E03-H14 found the *blend* won there - this is the inverse claim and must be earned on numbers, not assumed
- **Verdict** - Kept - the reorder cell sits at high semantic (1.00) / low structure (0.41) closeness, a structure-only failure a single blended scalar hides

## E08 - a metric, wide-dynamic-range OT structure distance (pre-registered)

The E07 batch left one honest gap: the surviving structure read (the `τ`-from-`T` index-infused SMD, Phase 6) is a **barycentric projection** scored by a rank statistic, so it is **not a metric** (7.5% triangle violations, E07-H37) and has a **narrow dynamic range** - off-diagonal mass saturates at any non-adjacent swap and the footrule tied the naive greedy-1-NN baseline (both ~0.94 at full scramble). E08 replaces the projection with an **optimal-transport distance whose cost scales with how far content moved**, so the structure number grows smoothly with displacement (wide dynamic range) and inherits the triangle inequality from the OT object itself. Two OT formulations lead (E08-H44 position-augmented Wasserstein, E08-H45 positional Fused Gromov-Wasserstein); two metric-correct reads (E08-H46, E08-H48) are the baselines they must beat on dynamic range; one lever (E08-H47) sharpens the plan under the single-pair anisotropy constraint. Predictions and acceptance bars were registered before the run; the verdicts below are filled from the executed batch, each qualified as **confirmed / refuted / inconclusive** with its applicability to the use case (**none / low / medium / high**).

- **Vehicle (built and run)** - `notebooks/experiments/E08-kj-structure-distance-metric.ipynb`, reusing the E07 fixture `data/processed/structure-fixture/` extended with two pieces - a **pure-shift** perturbation (insert at the top, displacing every later statement, no reorder) for the E08-H45 shift-invariance test, and an **offline anisotropy background set** (the fixture source articles or a public corpus) for the E08-H47 frozen direction; metrics in `reports/E08-structure-distance-metrics.json`, figures in `reports/figures/E08/`
- **Inherited machinery** - `distance.transport_plan` / `cost_matrix` / `smd` / `closeness` / `all_but_the_top`, `encoders.Segmenter` / `load_encoder`, POT `ot.emd` / `ot.emd2` / `ot.fused_gromov_wasserstein2`; the naive greedy-1-NN baseline and the `τ`-footrule carry over from E07 as the reference lines every E08 figure overlays
- **The headline number** - the **dynamic-range ratio** `(structure_distance at full scramble − structure_distance at one adjacent swap) / paraphrase-floor sd`; a wide ratio means resolution is preserved across displacement, the property the saturating E07 normalizers lack

**Pre-registration at a glance** (E08; predictions registered before the run)

| hypothesis | strategy | mechanism | predicted | bar | verdict |
|---|---|---|---|---|---|
| E08-H44 | OT metric | position-augmented Wasserstein on `M̃ = √((1−λ)d_sem² + λ d_pos²)` | metric, monotone, wide range; SMD intact at λ=0 | triangle violations ≤ 1%, Spearman ≥ 0.9, range ratio > E07 footrule/off-diagonal, beats naive on diffuse | Confirmed - High |
| E08-H45 | OT metric | positional Fused GW, structure `C = \|pos_i − pos_k\|`, semantic features | rises with reorder (unlike E07-H35), 0% violations, shift-invariant | monotone above paraphrase floor, 0 violations, ≈0 on pure shift, range ratio ≥ baselines | Refuted - Low |
| E08-H46 | metric baseline | optimal-assignment footrule, permutation from `ot.emd` | true metric on equal length, narrower range than H44/H45 | 0 violations equal-length, monotone, range ratio < H44/H45 | Confirmed - Low |
| E08-H47 | anisotropy | offline-frozen vs per-pair ABTT / whitening | sharper `T`, wider range and naive margin, SMD held | row-entropy down with SMD fidelity held, range + margin widen, no test-pair corpus-fit | Refuted - None |
| E08-H48 | creative | displacement-weighted anti-monotone transport mass | direct-from-plan, wide range, fewer violations than `τ`-footrule | monotone Spearman ≥ 0.9, violations < 4%, range ratio reported vs H44/H45 | Refuted - Low |

**Execution results (E08 batch)** - run in `notebooks/experiments/E08-kj-structure-distance-metric.ipynb` on the same RTX 5000 Ada and fixture as E07 (raw single-pair embeddings; byte-identical reorder upper bound plus the cross-summary diffuse pairs); metrics in `reports/E08-structure-distance-metrics.json`, figures in `reports/figures/E08/`. Each hypothesis is qualified **confirmed / refuted / inconclusive** with applicability to the use case (**none / low / medium / high**).

- **The OT formulation works (confirmed, high)** - E08-H44 (position-augmented Wasserstein) is a true metric (0% triangle violations against the `τ`-footrule's 4%), monotone (Spearman 1.00), and its dynamic-range ratio scales with the positional weight λ to 8.8 at λ=0.5, clearing both E07 normalizers - the one mechanism that ships
- **GW is still not the order instrument (refuted, low)** - E08-H45 (positional Fused GW) is a metric and correctly translation-invariant (pure shift 0.0000 where E08-H44 reads 0.134), but it **collapses at full scramble** (Spearman 0.43) as the Gromov isometry solution reasserts, so it is not a monotone order-disorder measure
- **The metric repair is clean but bounded (confirmed, low)** - E08-H46 (optimal-assignment footrule) restores 0% triangle violations on equal-length pairs but stays a narrower-range rank statistic (6.2) and is undefined on the unequal-length pairs real conversions produce
- **The anisotropy lever is inert (refuted, none)** - offline-frozen ABTT does not sharpen an already-concentrated `T` (row-entropy 0.129 → 0.130) nor widen the margin; SMD fidelity holds (Pearson 0.94) (E08-H47)
- **A non-metric disorder read (refuted, low)** - E08-H48 (displacement-weighted anti-monotone mass) has a wide range (8.1) but 6.7% triangle violations, **more** than the `τ`-footrule, so it does not solve the metric problem
- **Meta** - the pre-registered endpoint-difference range ratio is blind to mid-range saturation (the saturating off-diagonal mass scores 7.4, above E08-H44's 6.7 at λ=0.25), so E08-H44's decisive win is the **metric property**, not the headline ratio - a saturation-aware resolution measure is the next refinement

| hypothesis | mechanism | measured | verdict |
|---|---|---|---|
| E08-H44 | position-augmented Wasserstein | 0% triangle, Spearman 1.00, range → 8.8 at λ=0.5 | Confirmed - High |
| E08-H45 | positional Fused GW | rises mid then collapses (Spearman 0.43), 0% triangle, pure-shift 0.0000 | Refuted - Low |
| E08-H46 | optimal-assignment footrule | 0% triangle equal-length, Spearman 1.00, range 6.2 | Confirmed - Low |
| E08-H47 | offline-frozen vs per-pair ABTT | row-entropy 0.129 → 0.130 (not sharpened), SMD Pearson 0.94 | Refuted - None |
| E08-H48 | displacement-weighted anti-monotone mass | Spearman 1.00, 6.7% triangle, range 8.1 | Refuted - Low |

### E08-H44 Position-augmented Wasserstein (the OT formulation)

- **Hypothesis** - because a reorder moves a statement's position while preserving its meaning, an optimal-transport distance on a ground metric that fuses semantic and positional cost rises smoothly with displacement and, being a Wasserstein distance on a true metric, satisfies the triangle inequality - the metric, wide-dynamic-range structure distance the barycentric `τ` is not
- **Signal** - normalized statement position `pos_i/|A|` fused into the ground cost beside the embedding
- **Mechanism** - build `M̃_ij = √((1−λ)·d_sem(i,j)² + λ·d_pos(i,j)²)` with `d_sem = √(2−2cos) ∈ [0,2]` and `d_pos = |pos_i − pos_j| ∈ [0,1]`; `M̃` is the L2 product of two metrics, so it is a metric and `SMD_struct = W(M̃)` via `ot.emd2` is a true Wasserstein metric. Report it **beside** SMD - λ=0 leaves the content axis untouched, so this is a parallel structure metric, not the blended content ground cost E07-H30 rejected. Sweep λ
- **Prediction** - monotone in displacement; dynamic-range ratio materially wider than the E07 off-diagonal-mass and footrule normalizers; triangle violations < 1% (numerical only) against `τ`-footrule's 4%; SMD recovered unchanged at λ=0; beats the naive greedy-1-NN baseline by the registered margin on the diffuse cross-summary regime
- **Acceptance bar** - triangle violations ≤ 1% on sampled triples; Spearman(displacement, distance) ≥ 0.9 with zero inversions; dynamic-range ratio ≥ the E07 footrule / off-diagonal baselines; naive margin ≥ the registered minimum on the diffuse regime
- **Known failure mode** - penalizes absolute position, so a pure shift (an insertion at the top displaces every later statement) reads as disorder; the discriminating contrast against E08-H45
- **Verdict** - **Confirmed** (applicability: High) - λ=0 reproduces SMD (max `|Δ|` = 1.3e-12), 0% triangle violations (a true Wasserstein metric, against the `τ`-footrule's 4%), Spearman 1.00 with zero inversions; the dynamic-range ratio scales with λ (0 → 4.1 → 6.7 → 8.8) and clears both E07 baselines (footrule 6.4, off-diagonal 7.4) at λ=0.5; λ selection deferred to the second-article fixture. The metric, wide-dynamic-range structure distance the batch set out to find - the one mechanism that ships, caveated only by its absolute-position sensitivity (E08-H45's translation-invariance is the property to graft later)

### E08-H45 Positional Fused Gromov-Wasserstein (Gromov-Wasserstein, dug deeper)

- **Hypothesis** - because E07-H35 made FGW collapse to SMD on a reorder only because its structure matrices were **semantic** (an isometry under reorder, so the GW term vanished), rebuilding FGW with **positional** structure matrices makes the structure term penalize reordering while the feature term holds the content correspondence - an order-sensitive distance that is a proven metric and translation-invariant
- **Signal** - intra-document positional distance matrices `C_A[i,k] = |pos_i − pos_k|` and `C_B` likewise; semantic embeddings for the Wasserstein feature term
- **Mechanism** - `ot.fused_gromov_wasserstein2(M, C_A, C_B, p, q, alpha)` with `M = √(2−2cos)` inter-document and **positional** `C`; sweep `α`. FGW(α) is a proven metric (Vayer et al. 2019); the GW structure term is quadratic in pairwise position distortion, so the distance grows fast and smoothly with the extent of reordering
- **Prediction** - unlike E07-H35 (FGW = SMD flat across `α`), positional-FGW rises monotonically with reorder; 0% triangle violations; a pure shift reads ≈0 (the contrast with E08-H44); an interior `α*` beats `α = 0`
- **Acceptance bar** - monotone separation above the paraphrase floor; triangle violations 0 within solver tolerance; shift-invariance shown (≈0 on a pure shift, ≤ paraphrase p95); dynamic-range ratio ≥ the E07 baselines; `α*` selection deferred to the second-article fixture (report the `α`-curve, no forking path on one article)
- **Verdict** - **Refuted** as an order measure (applicability: Low) - rises mid-range (0.005 → 0.056 across bins 0-4, the contrast with E07-H35's flat = SMD holds) and is translation-invariant (pure-shift FGW 0.0000 against the position-augmented 0.134), 0% triangle violations (a metric); but it **collapses at full scramble** (peak bin 4, drops at bin 5, Spearman 0.43) as the Gromov isometry solution reasserts (α=1 → 0), so it is a metric but not a monotone order-disorder measure. GW is confirmed (again) not the order instrument; its translation-invariance is the one property worth salvaging

### E08-H46 Optimal-assignment footrule (the metric repair and baseline bridge)

- **Hypothesis** - because the `τ`-footrule violates the triangle inequality only as an artifact of barycentric projection (a soft non-permutation), reading the footrule off the **optimal one-to-one assignment** instead restores a true metric on the symmetric group while staying a near-free read on the existing plan
- **Signal** - the SMD plan `T`, reduced to a hard optimal assignment
- **Mechanism** - on equal-length uniform marginals the optimal assignment is the `ot.emd` solution itself (a permutation matrix at the network-simplex vertex), so its row-argmax is a genuine permutation - no scipy or Hungarian dependency; read the normalized Spearman footrule of that permutation. The naive greedy-1-NN baseline is the greedy-assignment cousin, so this quantifies what the optimal soft plan adds over hard 1-NN
- **Prediction** - 0% triangle violations on equal-length pairs (metric by construction); monotone in displacement; tracks soft `τ` on a sharp `T`, diverges (more honest, less saturated) on a diffuse `T`; narrower dynamic range than E08-H44/H45 (still a bounded rank statistic)
- **Acceptance bar** - violation rate 0 on equal-length triples; monotone; dynamic-range ratio reported and expected below E08-H44/H45 (the point of the comparison); equal-length only - unequal-length real conversions still need the true-OT metrics
- **Verdict** - **Confirmed** (applicability: Low) - equal-length triangle violations 0.0% (a true metric on the symmetric group, the repair the `τ`-footrule needed), Spearman 1.00 zero inversions, range ratio 6.2 - narrower than E08-H44's 6.7 as designed. A clean metric repair, but **only on equal-length pairs**; real conversions change statement counts, where it is undefined - so low applicability to the use case, useful as a diagnostic baseline

### E08-H47 Anisotropy removal under the two-document constraint

- **Hypothesis** - because mmBERT anisotropy bunches cosines into 0.7-0.9 and diffuses `T`, removing the dominant common direction sharpens `T`, widening both the dynamic range and the margin over the naive baseline; and because a single production pair cannot corpus-fit the direction, an offline-frozen direction recovers the benefit without a train→serve leak
- **Signal** - the pooled statement embeddings; the anisotropy direction(s) removed before the OT solve
- **Mechanism** - two forms: (a) **offline-frozen** - fit the anisotropy components once on a frozen background set (the fixture source articles or a public corpus), store them, subtract at serve time (the shipped `all_but_the_top` fits from its input, so this needs a thin precomputed-direction variant); (b) **per-pair pooled** - call `all_but_the_top` on the two test docs' ~25 statements (symmetric, no external data, noisy on short docs)
- **Prediction** - offline-frozen drops the cross-summary `T` median row-entropy from 0.13 toward 0.00, widens the dynamic range of E08-H44/H45 and `τ` and their naive margin on the diffuse regime, with the semantic SMD axis held; per-pair pooled over-corrects on 25-statement docs, so offline-frozen wins
- **Acceptance bar** - row-entropy reduction with SMD fidelity held (Pearson vs raw-SMD ≥ a registered floor); dynamic range and naive margin both widen by a registered amount on the diffuse regime; the offline-vs-per-pair contrast documented; never corpus-fit on the test pair
- **Verdict** - **Refuted** / inert (applicability: None) - offline-frozen ABTT does not sharpen the cross-summary `T` (row-entropy 0.129 → 0.130, per-pair 0.135) because raw `T` is already concentrated (~0.13, no room), and the naive margin does not widen (+0.005 → −0.009); SMD fidelity held (Pearson 0.94). The lever adds nothing where `T` is already sharp - consistent with E07; it could still matter on a genuinely diffuse `T`, untested here

### E08-H48 Displacement-weighted anti-monotone transport mass (creative)

- **Hypothesis** - because the optimal 1D coupling is cyclically monotone, the `T` mass on inversions - weighted by displacement magnitude - is an order-disorder read taken straight off the plan, grounded in OT optimality rather than a barycentric projection, and the weighting gives it a wider dynamic range than the saturating off-diagonal mass
- **Signal** - the SMD plan `T` and statement positions
- **Mechanism** - sum `T_ij` over inverted pairs (`i < i′` mapped to `j > j′`), optionally weighted by `|Δpos|`; natively bounded `[0,1]` unweighted, scaled when weighted; a pure O(n²) read on `T`, no new model
- **Prediction** - monotone in displacement; the displacement-weighted form has a materially wider dynamic range than the unweighted off-diagonal mass; fewer triangle violations than the barycentric `τ`-footrule; kept only if its range rivals the true-OT metrics
- **Acceptance bar** - monotone Spearman ≥ 0.9; triangle-violation rate reported against `τ`-footrule's 4%; dynamic-range ratio reported against E08-H44/H45 and the off-diagonal-mass baseline; the bounded / scaled form stated
- **Verdict** - **Refuted** as a metric (applicability: Low) - displacement-weighted anti-monotone mass is monotone (Spearman 1.00, zero inversions) with a wide range (8.1 unweighted, 7.2 weighted), but **6.7% triangle violations, more than the `τ`-footrule** (~4%), against the prediction. A wide-range disorder read, not a metric - it does not solve the metric problem E08-H44 solves

## E09 - common-mode rejection and segmentation granularity (executed)

E08-H47 found the offline anisotropy lever inert - but on the byte-identical fixture where `T` was already sharp, the regime with no common mode left to reject. E09 retests anisotropy removal as **common-mode rejection** on the **diffuse** cross-summary pairs (cosines bunched, `T` spread), pooling **both** documents' statements to estimate the shared direction - the differential-amplifier analogy: reject what the two documents share, amplify what distinguishes them. The tension is selectivity - the common mode of two same-article summaries is frequency anisotropy *plus shared topic*, and rejecting the topic inflates the semantic distance. So every E09 hypothesis scores **both axes**: the structural gain (sharper `T`, wider margin) against the semantic cost (tier ordering, SMD fidelity). It covers the single basic SMD (E09-H53) and the structure-augmented metric (E09-H49..H52). Predictions and acceptance bars were registered before the run; the verdicts below are filled from the executed batch, each qualified **confirmed / refuted / inconclusive** with its applicability (**none / low / medium / high**). The result is decisive and negative - **no lever ships** (every CMRR < 1): on the diffuse pairs the pooled common mode is shared **topic**, not just frequency anisotropy, so per-pair rejection *blurs* the transport plan (`T`-entropy 0.129 → 0.135) and *hurts* the semantic ranking (fidelity 0.741, +1 tier violation), the opposite of the E09-H49 prediction; the topic-removal risk the pre-registration flagged materialized.

- **Headline measure - the common-mode rejection ratio (CMRR)** - `structural differential gain / semantic common-content loss`; numerator = structural naive-margin or dynamic-range gain on diffuse pairs, denominator = semantic tier-ordering or SMD-fidelity loss; a lever ships only at CMRR > 1
- **Regime** - the diffuse cross-summary pairs (and, where built, finer-segmented and larger-document variants), never the byte-identical upper bound that left E08-H47 inconclusive
- **Vehicle (built and run)** - `notebooks/experiments/E09-kj-anisotropy-cmr.ipynb`, reusing the E08 `posaug_cost` and the shipped `all_but_the_top`, extended with a both-document pooled-PCA variant and a finer-segmentation pass; metrics in `reports/E09-anisotropy-cmr-metrics.json`, figures in `reports/figures/E09/`

**Pre-registration at a glance** (E09; predictions registered before the run)

| hypothesis | strategy | mechanism | predicted | bar | verdict |
|---|---|---|---|---|---|
| E09-H49 | common-mode | both-document top-1 PCA rejection | diffuse `T` sharpens, semantic ordering survives | structural gain + ordering preserved, CMRR > 1 | Refuted - None |
| E09-H50 | selectivity | per-pair vs offline-frozen vs raw | offline more selective, per-pair sharper | the variant maximizing CMRR on diffuse pairs | Inconclusive - None |
| E09-H51 | rejection depth | top-D sweep D ∈ {0,1,2,3} | CMRR peaks at D=1, D≥3 over-rejects | best D by CMRR, bootstrap-stability flagged | Inconclusive - None |
| E09-H52 | granularity | finer segmentation, more PCA samples | sweet spot before per-unit noise dominates | a granularity clearing the E09-H49 bar | Refuted - None |
| E09-H53 | single (SMD) | basic SMD under both-document CMR | inert on sharp, DR up on diffuse | tier ordering preserved, DR up on a single pair | Refuted - None |

### E09-H49 Per-pair common-mode rejection (both-document top-1 PCA)

- **Hypothesis** - because the dominant direction of the two documents' pooled statements is their common mode (frequency anisotropy plus shared topic), subtracting it amplifies the differential signal, so on diffuse pairs it sharpens the transport plan and widens the structural margin without breaking the semantic ordering - unless the shared-topic component it removes is large enough to inflate the semantic distance
- **Signal** - the top-1 principal direction of both documents' pooled, mean-centered statement embeddings
- **Mechanism** - pool A and B statements (~2n), subtract the pooled mean and the top-1 direction (`all_but_the_top` on the pooled set), re-L2-normalize, then compute SMD (semantic) and position-augmented Wasserstein (structural); compare against raw on the diffuse cross-summary pairs
- **Prediction** - on diffuse pairs the rejection drops `T` row-entropy below the raw ~0.13 and widens the structural naive margin, while the semantic tier ordering stays at 0 violations and SMD fidelity (rank vs raw) holds; or the semantic axis degrades, exposing the topic-removal cost
- **Acceptance bar** - structural margin/entropy gain on diffuse pairs AND semantic ordering preserved (0 violations) with SMD-fidelity above a registered floor; CMRR > 1
- **Verdict** - **Refuted** (applicability: None) - per-pair both-document top-1 CMR does **not** sharpen the diffuse `T` (row-entropy 0.129 → 0.135, the plan got *more* diffuse, g_struct −0.049) and hurts the semantic ordering (SMD-fidelity Spearman 0.741, l_sem 0.259, +1/24 tier violation, naive margin +0.005 → −0.006); CMRR −0.19. The opposite of the prediction - the common mode of two same-article summaries is shared **topic**, not just frequency anisotropy, so rejecting it blurs both axes. Same null as E08-H47 but for the opposite reason - there `T` was already sharp; here there is a real common mode, but it is topic, and removing it costs

### E09-H50 Selectivity - per-pair vs offline-frozen vs raw

- **Hypothesis** - because the per-pair common mode rejects the pair's shared topic (high rejection, low selectivity) while an offline-frozen global direction rejects only frequency anisotropy (high selectivity, lower rejection), the two trade structural sharpening against semantic preservation, and the CMRR picks the winner
- **Signal** - the source of the rejected direction - the test pair vs a frozen background corpus
- **Mechanism** - three arms - raw, per-pair top-1 (E09-H49), offline-frozen top-1 (a direction fit once on a background set, e.g. the source article statements) - each scored on both axes over the diffuse pairs
- **Prediction** - offline-frozen preserves the semantic shared-content signal better (no topic removal) but sharpens less; per-pair sharpens more but pays on the semantic axis; one clears CMRR > 1, the other does not
- **Acceptance bar** - the arm with the highest CMRR on the diffuse pairs, with the semantic-vs-structural tradeoff reported for each
- **Verdict** - **Inconclusive** (applicability: None) - the predicted selectivity ordering holds - offline-frozen (anisotropy-only direction) preserves semantics far better (fidelity 0.921 vs per-pair 0.741) but sharpens less - and offline-frozen is the safer arm (CMRR −0.11 vs per-pair −0.19), but neither clears CMRR > 1, so no shipping winner on the diffuse pairs. Offline-frozen is the arm to keep if the lever is ever revisited

### E09-H51 Rejection depth - top-D sweep

- **Hypothesis** - because only the top one or two directions are reliably estimable from ~50 pooled samples, removing more over-rejects and cuts the differential signal, so the CMRR peaks at a shallow D and falls as D grows
- **Signal** - the number D of pooled principal directions removed
- **Mechanism** - sweep D ∈ {0,1,2,3} on the both-document pool; for each, score both axes and the bootstrap stability of the removed direction(s) across resampled statement subsets
- **Prediction** - CMRR peaks at D=1 (possibly D=2); D≥3 degrades the semantic axis (over-rejection), and the removed direction's bootstrap stability falls off past D=1-2
- **Acceptance bar** - the D maximizing CMRR; flag the D at which the removed direction's stability drops below a registered threshold (the over-rejection onset)
- **Verdict** - **Inconclusive** (applicability: None) - best D = 0 (raw); CMRR by depth [0.00, −0.19, −0.27, −0.16] is net-negative at every removal depth, and the removed-direction bootstrap stability is below the 0.9 bar everywhere (D1 0.77, D2 0.86, D3 0.83), confirming the ~28-statements-per-pair pooled estimate is too shaky to trust. The pre-registered "peak at shallow D" shape does not appear - the optimum is no removal

### E09-H52 Segmentation granularity - sample count vs unit noise

- **Hypothesis** - because finer segmentation yields more statements (more PCA samples, a better common-mode estimate) but noisier per-unit embeddings (less context each), there is a granularity sweet spot where the CMR benefit is maximal before per-unit noise dominates both axes
- **Signal** - the segmentation granularity (statement size)
- **Mechanism** - re-segment the documents at coarser and finer granularities (SAT default vs a finer split or token cap), recompute embeddings, and run the E09-H49 per-pair CMR at each; track the PCA-estimate stability and both downstream axes
- **Prediction** - finer segmentation improves the top-1 estimate's stability and the CMR benefit up to a point, then per-unit embedding noise degrades both the semantic and structural axes - a sweet spot in the middle
- **Acceptance bar** - a granularity at which per-pair CMR clears the E09-H49 bar that the coarse segmentation could not; report the granularity-vs-CMRR curve
- **Verdict** - **Refuted** (applicability: None) - finer segmentation moves CMRR in the predicted direction (default −0.19 → fine +0.12, n~33) and improves top-1 PCA stability (0.77 → 0.86) - the sweet-spot trend is real - but the best granularity (fine, CMRR 0.12) still does not clear the E09-H49 bar (CMRR > 1). Used the documented clause-split fallback (split on `, ; : -`, 4-word floor; the SAT `Segmenter` exposes no public threshold knob)

### E09-H53 The single (basic SMD) under both-document CMR

- **Hypothesis** - because the content distance on a lone pair currently runs raw (the shipped single-pair default, `anisotropy: false`), both-document common-mode rejection is the natural way to recover anisotropy removal without a corpus, so on diffuse pairs it de-bunches cosines and widens the SMD dynamic range, inert only where `T` is already sharp
- **Signal** - the basic SMD (λ=0, no positional channel) on a single pair
- **Mechanism** - compute SMD with both-document top-1 CMR vs raw on the diffuse cross-summary pairs and the content tier-contrast pairs (gold vs adversarial) where available
- **Prediction** - inert on the sharp byte-identical pairs (E08-H47-like); on diffuse pairs the rejection widens the SMD dynamic range but may compress the gold/adversarial margin if it removes shared topic - the content SOTA's "DR up 3.2x, d' down" tradeoff, now estimated per-pair from both documents
- **Acceptance bar** - SMD tier ordering preserved (0 violations) with dynamic range up via both-document CMR on a single pair; the d' tradeoff reported
- **Verdict** - **Refuted** (applicability: None) - the single-pair dynamic-range claim holds (SMD std 0.036 → 0.117, ×3.25, the content-SOTA "DR up" effect recovered per-pair from both documents) but the d' guard fails - gold/adversarial tier d' falls 2.81 → 2.28 and CMR introduces 1/24 tier violation, so tier ordering is **not** preserved. The topic-removal cost is real on a lone pair - both-document CMR is not a safe substitute for corpus anisotropy removal on the basic SMD

## E10 - structure-sensitive, content-invariant distances (executed)

The arc leaves one property unclaimed. E08-H44 position-augmented Wasserstein ships - a true metric, wide range - but it penalizes **absolute** position, so a pure top-insertion shift reads as disorder. E08-H45 positional Fused-GW is translation-invariant (relative position) but **collapses at full scramble**: a reorder of identical statements is an isometry of the position line, so the re-optimized Gromov coupling escapes to the trivial isometric solution and the cost falls back toward zero. E10 hunts the combined property neither survivor holds - **translation-invariant AND monotone through full scramble** - across five mechanisms, OT-derived and otherwise, each tested for sensitivity to **order** change while staying blind to **semantic** change. The discriminating test is two controls: a faithful **paraphrase** (content preserved, order preserved) must read ≈ 0, and a **reorder** (content preserved, order changed) must read large. Predictions and acceptance bars were registered before the run; the verdicts below are filled from the executed batch, each qualified **confirmed / refuted / inconclusive** with its applicability (**none / low / medium / high**). The result: **E10-H55 (the OPW order-gap) is the one positive** - the only mechanism to hold all four gates (shift-invariance, scramble-monotonicity, content-invariance, and beating E08-H44 on the combined property), a score not a metric. The headline negative is **E10-H54** - pinning the content plan does not stop the GW collapse, because the Gromov distortion functional is itself isometry-invariant and a full reversal is an isometry, so the predicted fix fails. The metric routes (E10-H56 source-anchored, E10-H57 string-edit) and the soft-DTW route (E10-H58) all clear the geometry gates but fail content-invariance against the **embedding-jitter paraphrase proxy** (a true LLM paraphrase is deferred - no API key), left inconclusive. Metrics in `reports/E10-structure-sensitive-metrics.json`, figures in `reports/figures/E10/`, vehicle `notebooks/experiments/E10-kj-structure-sensitive-distance.ipynb`.

- **The new property under test** - shift-invariance (a pure shift reads ≈ 0, the E08-H44 failure mode) AND scramble-monotonicity (the score keeps rising through the top displacement bin, the E08-H45 failure mode) together
- **Common evaluation** - the byte-identical reorder pool (6 displacement bins × seeds) for monotonicity, dynamic range and no-collapse-at-full-scramble; the diffuse cross-summary pairs for the realistic regime; the pure-shift perturbation for translation-invariance; a paraphrase / order-preserving control for content-invariance; sampled triples for the triangle check; baselines to beat = naive greedy-1-NN and E08-H44 on the discriminating axes
- **Metric honesty** - H56 and H57 are metrics by construction; H54, H55, H58 are structure **scores** (a Gromov discrepancy and two regularized / DP divergences), reported as scores, never conflated with the exact metric
- **Vehicle (built and run)** - `notebooks/experiments/E10-kj-structure-sensitive-distance.ipynb`, reusing `transport_plan` / `cost_matrix` / `coverage_alignment`, with the source-anchored regime (`ibm-ai-adoption/source`) for E10-H56 and an embedding-jitter paraphrase proxy (realized mean cosine 0.875) standing in for a true LLM paraphrase control (deferred, no API key); metrics in `reports/E10-structure-sensitive-metrics.json`, figures in `reports/figures/E10/`

**Pre-registration at a glance** (E10; predictions registered before the run)

| hypothesis | strategy | mechanism | predicted | bar | verdict |
| --- | --- | --- | --- | --- | --- |
| E10-H54 | pinned-GW | content-pinned Gromov distortion over positional `C` | monotone through scramble, shift ≈ 0, no collapse | Spearman ≥ 0.9 incl top bin, shift ≤ paraphrase p95 | Refuted - None |
| E10-H55 | order-OT | OPW − SMD order-gap (IDM + KL prior) | monotone, blind to paraphrase, shift small | Spearman ≥ 0.9, gap ≈ 0 on paraphrase | Confirmed - Medium |
| E10-H56 | source-anchored | Kendall-tau on induced source-order ranks | monotone, blind to paraphrase, a metric | Spearman ≥ 0.9, triangle 0, ≈ 0 on paraphrase | Inconclusive - Low |
| E10-H57 | string edit | Kendall-tau-sequence on shared-codebook symbols | metric, shift small vs scramble large | Spearman ≥ 0.9, triangle 0, codebook stability flagged | Inconclusive - Low |
| E10-H58 | soft-DTW | softDTW-divergence − SMD order-gap | monotone, blind to paraphrase, wide range | Spearman ≥ 0.9, gap ≈ 0 on paraphrase | Inconclusive - Low |

### E10-H54 Content-pinned Gromov distortion (the fix for the E08-H45 collapse)

- **Hypothesis** - because a re-optimized Gromov coupling escapes a reorder via the isometric solution (the E08-H45 collapse), pinning the coupling to the content plan removes that escape, so the Gromov distortion of the **content** correspondence over positional structure rises monotonically with reordering and stays translation-invariant
- **Signal** - the Gromov structure-distortion of the fixed SMD plan over positional relation matrices
- **Mechanism** - fix `T* = transport_plan(EA, EB)` (the semantic correspondence), set `C_A[i,k]=|pos_i−pos_k|`, `C_B[j,l]=|pos_j−pos_l|`, and read `D = Σ_ijkl T*_ij T*_kl (C_A[i,k] − C_B[j,l])²`; the plan is not re-optimized, so the isometry escape is structurally impossible
- **Prediction** - monotone through full scramble (no E08-H45 drop at the top bin), pure-shift ≈ 0 (the E08-H44 failure mode gone), blind to paraphrase, wide range
- **Acceptance bar** - Spearman(displacement, `D`) ≥ 0.9 with zero inversions including the top bin; pure-shift ≤ paraphrase p95; dynamic-range ratio ≥ E08 baselines; triangle-violation rate reported (a structure discrepancy, metric not claimed)
- **Verdict** - **Refuted** (applicability: None) - pinning the plan does **not** prevent the collapse: pure-shift ≈ 0 (PASS) and the paraphrase proxy ≈ 0 (PASS), but the score is non-monotone (Spearman 0.43) and **collapses at the top bin exactly like E08-H45**. The prediction was wrong about the mechanism - the Gromov distortion functional is itself isometry-invariant and a full reversal is an isometry of the position line (`|pos_i−pos_k|` preserved), so the collapse lives in the distortion functional, not in the freedom of the coupling; pinning the content plan cannot remove it. A score (1.3% triangle). The predicted-vs-measured gap is the finding - pinning was the wrong fix

### E10-H55 Order-Preserving Wasserstein order-gap (the canonical order-OT)

- **Hypothesis** - because the OPW temporal regularizers (IDM + KL prior) penalize non-monotonic couplings, the extra cost they add over the content-optimal plan is zero when the plan is already monotone (paraphrase) and grows with the inversions a reorder forces
- **Signal** - the order-gap `OPW − SMD`
- **Mechanism** - compute OPW (Su & Hua) with the IDM local-homogeneity term and the KL prior over normalized positions on top of the `√(2−2cos)` ground cost, subtract the plain SMD; reference impl BingSu12/OPW
- **Prediction** - monotone in displacement, wide range, blind to paraphrase; a pure shift reads small (still near-monotone) - the contrast vs E08-H44
- **Acceptance bar** - Spearman ≥ 0.9; gap ≈ 0 on the paraphrase control; range ratio ≥ baselines. OPW carries a KL prior → a divergence, so the gap is a structure score, registered as such
- **Verdict** - **Confirmed** (applicability: Medium) - the **only mechanism to clear all four gates together**: pure-shift 0.000517 ≤ paraphrase p95 0.00504 (translation-invariant, the E08-H44 failure mode gone), Spearman 1.00 rising through the top bin (scramble-monotone, the E08-H45 failure mode gone), paraphrase-proxy median 0.00293 (content-invariant), and it beats E08-H44 on the combined property. A score, not a metric (16% triangle measured, the KL prior as registered). Applicability **Medium** not High for two honest reasons: the naive greedy-1-NN baseline also clears shift + scramble (so the combined property is necessary, not sufficient - OPW's edge is being a smooth, content-invariant OT object), and the content control is an embedding-jitter proxy pending a true-paraphrase fixture. The order-gap `OPW − SMD` is the batch's positive result

### E10-H56 Source-anchored ordering divergence (content-invariant by reference)

- **Hypothesis** - because both documents' statements align to a shared source, the reading order each induces over that source changes only when statements are reordered, not when they are paraphrased, so the disagreement between the two induced source-orderings is a content-invariant order signal
- **Signal** - the Kendall-tau distance between A's and B's induced source-position sequences
- **Mechanism** - map each document to a per-statement soft source position with the shipped `coverage_alignment(X, S)` (barycentric source index), then read Kendall-tau between the two source-rank sequences; shared-source regime exists in the fixture, with one-document-as-anchor (symmetrized) as the always-available fallback
- **Prediction** - monotone, blind to paraphrase; Kendall-tau is a metric on the induced ranks → a metric, dissolving the E09 two-document anisotropy problem (the source is the anchor)
- **Acceptance bar** - Spearman ≥ 0.9; ≈ 0 on paraphrase; triangle violations 0 on equal-length induced ranks; shared-source vs one-doc-anchor contrast reported
- **Verdict** - **Inconclusive** (applicability: Low) - achieves the combined property (pure-shift 0, Spearman 1.00 rising through the top bin) and is a metric on equal-length induced ranks (0% triangle), but **fails content-invariance against the proxy** (paraphrase-proxy median 0.1) because embedding jitter reassigns source bins, and the triangle breaks on diffuse unequal-length pairs (22.7%) - a metric only in the equal-length regime, like E08-H46. Inconclusive pending a true paraphrase control and an unequal-length-robust source binning

### E10-H57 Sequence edit distance on a shared codebook (table-turning, non-OT string paradigm)

- **Hypothesis** - because a shared statement codebook maps paraphrases to the same symbol, encoding each document as a symbol string in reading order makes a paraphrase a no-op and a reorder a string permutation, so the edit distance between the two strings is a content-invariant order metric - testing whether OT is even needed for the order signal
- **Signal** - the Kendall-tau-sequence / Levenshtein distance between the two symbol strings
- **Mechanism** - co-cluster the pooled A+B statements into a shared codebook (numpy threshold on the cross-cost, or a small agglomerative pass; `scipy` is not a dependency), assign each statement its symbol, read the normalized adjacent-swap sequence distance (Cicirello 2019) between the reading-order strings
- **Prediction** - a metric, monotone, blind to paraphrase, a pure shift small (insertions) vs a scramble large (transpositions) - the translation-invariance E08-H44 lacked, via a different paradigm
- **Acceptance bar** - Spearman ≥ 0.9; ≈ 0 on paraphrase; triangle violations 0; codebook stability gated - bootstrap symbol-assignment stability reported, flagged if < 0.9 (the E09 small-N lesson)
- **Verdict** - **Inconclusive** (applicability: Low) - the table-turning string route achieves the combined property (pure-shift 0, Spearman 1.00 rising) and is a clean metric (0% triangle on both equal-length and diffuse), but **fails content-invariance against the proxy** - the shared codebook reassigns symbols under embedding jitter, exactly the codebook-stability gate failing on ~25-statement docs (the E09 small-N lesson). The order signal is real and OT is genuinely not needed for it, but the codebook is too unstable to trust here. Inconclusive pending a stable codebook and a true paraphrase control

### E10-H58 Soft-DTW divergence order-gap (sequence-DP route)

- **Hypothesis** - because a DTW warping path is monotonic, the order-constrained alignment cost minus the order-free SMD is zero when content matches in order (paraphrase) and positive when a reorder breaks monotonicity, giving a second order-penalty route to contrast with the OPW gap
- **Signal** - the order-gap `softDTW_divergence − SMD`
- **Mechanism** - the soft-DTW divergence (Blondel-Mensch-Vert 2021, non-negative and zero iff equal) on the position-indexed embedding sequences with the `√(2−2cos)` ground cost, via the LogSumExp DP and the debiasing `sdtw(A,B) − ½sdtw(A,A) − ½sdtw(B,B)`, minus the plain SMD
- **Prediction** - monotone, blind to paraphrase, wide range; the divergence variant is a proper non-negative score (zero at equality), unlike raw soft-DTW
- **Acceptance bar** - Spearman ≥ 0.9; gap ≈ 0 on paraphrase; unequal-length behavior reported. Soft-DTW divergence has no triangle inequality → a score, not a metric
- **Verdict** - **Inconclusive** (applicability: Low) - achieves the combined property (pure-shift 0, Spearman 1.00 rising) with 0% triangle violations, but **fails content-invariance badly** (paraphrase-proxy median 6.48) - the divergence magnitude tracks content drift, not only order, so it cannot isolate arrangement from semantics in its raw form. A score. Inconclusive - the order-gap framing needs a content-normalized form before the soft-DTW route can separate structure from content

## E11 - structure-distance SOTA decision (H44 vs H55, executed)

The arc ends with one decision - which single structure number ships beside SMD. The two finalists were E08-H44 (position-augmented Wasserstein, a true metric) and E10-H55 (the OPW order-gap, a translation-invariant score), and the provisional ship was E08-H44 on the strength of its metricity. E10 could not test the gate that actually decides a *structure, not meaning* number - content-invariance against a true paraphrase - because it had only an embedding-jitter proxy on a single article. E11 supplies the missing material and takes the decision: a **second article** (the Wergeland *Impact of AI on Society* curriculum, pdfplumber-extracted, three section bases) for cross-fixture replication, and a **true paraphrase** control (opus-mt EN→DE→EN back-translation - a per-statement 1:1 reword at mmBERT cosine 0.98 - plus the same-model a/b regenerations). The deciding gate: a faithful reword with order preserved must read ~0 on the structure axis; whichever finalist fires there cannot separate arrangement from meaning. Fixture in `data/processed/structure-fixture/` (now two articles plus a `~bt` paraphrase per base), metrics `reports/E11-structure-sota-decision.json`, vehicle `notebooks/experiments/E11-kj-structure-sota-decision.ipynb`.

- **The deciding gate (content-invariance)** - on back-translation rewords across both articles, E08-H44 reads median **0.149** and E10-H55 reads **0.0017**; normalized against the full-scramble top bin, H44 leaks **73.5%** of content into the structure axis, H55 leaks **0.5%**
- **Why H44 fails it** - the fused cost `√((1−λ)d_sem² + λ d_pos²)` reduces to `≈ √(1−λ)·d_sem` when order is preserved (`d_pos = 0`), so H44 ≈ 0.87·SMD on a reword - it is *content + structure fused*, not structure, and reads a paraphrase (0.149) as large as a real reorder (0.166)
- **Why H55 holds it** - the order-gap `OPW − SMD` subtracts the content-optimal transport cost, cancelling the content component; a reword keeps order so OPW ≈ SMD and the gap ≈ 0, while a reorder forces the temporal prior onto content-suboptimal couplings and the gap rises (0.44 at full scramble) - the 200× separation
- **The other gates** - scramble-monotonicity holds for both (Spearman 1.00, top-rising, both articles); translation-invariance separates them (pure shift: H44 0.42 of range, H55 0.001 - index-intrinsic); the metric check is H44's only win (triangle violations 0% vs H55's 15%, the KL prior)
- **Cross-article** - the pattern replicates: scramble top-bin H55 0.439 (IBM) / 0.396 (Wergeland), H44 0.240 / 0.225; content-invariance and shift behavior agree across both articles
- **The shipping form** - H55 is bounded below at 0 and above by √2 (a difference of √2-bounded 1-Wasserstein costs), so it ships on the library's existing closeness scale `structure_closeness = 1 − gap/√2 ∈ [0,1]` - a fixed analytic rescale, stateless (no batch statistics, no calibration), consistent with SMD and H44 closeness; the trade is a compressed practical range (the gap reaches ~0.34·√2)

**Pre-registration at a glance** (E11; the decision rubric, gates registered before the run)

| gate | H44 | H55 | decides |
| --- | --- | --- | --- |
| content-invariance (paraphrase ÷ scramble-top) | 0.735 | 0.005 | **H55** |
| translation-invariance (shift ÷ scramble-top) | 0.417 | 0.001 | H55 |
| scramble-monotone (both articles, Spearman) | 1.00 | 1.00 | tie |
| metric (triangle-violation rate) | 0% | 15% | H44 |

**Verdict** - **Ships: E10-H55 (OPW order-gap) as the single structure metric**, read as `structure_closeness = 1 − gap/√2`. It is a score, not a metric (15% triangle measured, the KL prior), shipped honestly as such; for a single pairwise *arrangement, not meaning* read this is the right trade - content-invariance is correctness, metricity only buys corpus-scale indexing the use case does not invoke. **E08-H44 is dropped as the single structure number** - it remains a valid metric on *content + arrangement fused*, useful only when read beside SMD, which is the cognitive load the one-number design rejects. The reversal is the finding: the provisional metricity-first ship did not survive a real paraphrase.

## Conclusions

The synthetic batch executed cleanly and the design's central claims hold on data - SMD is blind to reorder, one transport-induced mechanism recovers the order signal, and Gromov-Wasserstein is the wrong tool for order. What *ships* is contingent on two gates that remain open and on cross-fixture replication; the batch proves the mechanism, not production readiness.

- **One mechanism survives** - the **index-infused SMD** (the `T`-induced structure distance, Phase 6 / E07-H38-H43) is the only positive order/section instrument; E07-H30 (positional penalty), E07-H40 (gating), E07-H41 (explainability), E07-H43 (two-axis) are facets of it, and **E07-H39 (bounded τ-disorder) is the single ship gate** - the others corroborate, none is an independent confirmation
- **The premise is validated** - SMD provably ignores reorder (max `|ΔSMD|` 0.0004) while the structure axis rises monotonically, so the gap the axis fills is real on this data
- **GW/FGW reclassified by data** - GW ≈ 0 on a pure reorder against 0.0134 cross-summary, FGW = SMD across α on a reorder; the four-round review's central correction is confirmed empirically - GW is a relational, not order, instrument and is parked off the critical path
- **The honest ceiling** - every number is the byte-identical upper bound where `T` is trivially sharp; on the realistic cross-summary regime `T` stays usably concentrated (entropy 0.13) but τ **barely beats the naive greedy-1-NN baseline** (both ~0.94), so the margin over a model-free baseline is unproven on hard pairs
- **Two gates still open** - Phase 1 (structure variance on ≥10 real conversion pairs, none exist) and the raw-`T` sparsity probe on real paraphrased pairs; either can still end the programme cheaply before a build
- **Standing deferrals** - cross-fixture replication on a second article before any result is "confirmed"; normalizer selection on that second fixture; E07-H36 decorrelation exploratory until powered (Pearson 0.54 is underpowered at this N, not evidence of a real correlation)
- **E08 found the metric (confirmed, high)** - position-augmented Wasserstein (E08-H44) is a true metric (0% triangle violations) with a dynamic range that scales with the positional weight λ - the OT formulation the barycentric `τ` was not; positional Fused GW (E08-H45) is a metric but collapses at extreme reorder (refuted as an order measure), the optimal-assignment footrule (E08-H46) is a metric only on equal-length pairs, the anisotropy lever (E08-H47) is inert on an already-sharp `T`, and the anti-monotone mass (E08-H48) is wide-range but not a metric. The use-case answer is **E08-H44**, distilled into the SOTA design [`../solution/wmd-structure-distance-sota.md`](../solution/wmd-structure-distance-sota.md) and demonstrated end-to-end in `notebooks/12-kj-structure-distance-e2e.ipynb`, pending cross-fixture replication
- **E09 closed the anisotropy thread (nothing ships)** - common-mode rejection on the diffuse pairs is *destructive*, not corrective: pooling both documents removes shared **topic**, not just frequency anisotropy, so per-pair rejection blurs `T` (entropy 0.129 → 0.135) and drops semantic fidelity to 0.741 (E09-H49, CMRR −0.19). Offline-frozen (anisotropy-only) is the safer arm but still non-shipping (E09-H50); the optimum removal depth is none (E09-H51, D=0, pooled-direction stability < 0.9 at ~28 statements/pair); finer segmentation moves CMRR the right way but not over the bar (E09-H52); and the basic SMD under CMR buys ×3.25 dynamic range at a d' cost with a tier violation (E09-H53). The standing answer is unchanged - SMD runs raw on a lone pair, anisotropy removal stays a corpus-only lever, and **E08-H44 is the structure axis that ships**
- **E10 found a translation-invariant complement (E10-H55, OPW order-gap)** - of five mechanisms hunting the combined property neither E08 survivor held, only the OPW order-gap `OPW − SMD` holds translation-invariance AND scramble-monotonicity AND content-invariance together (shift 0.0005 ≤ paraphrase p95 0.005, Spearman 1.00 rising through the top bin, paraphrase-proxy median 0.003), beating E08-H44 on the property it lacks (absolute-position sensitivity). It is a **score**, not a metric (16% triangle measured, the KL prior), and applicability is **Medium** because the naive baseline also clears shift + scramble and the content control is an embedding-jitter proxy. The instructive negative is **E10-H54** - pinning the content plan does **not** stop the Gromov collapse, because the distortion functional is itself isometry-invariant and a full reversal is an isometry; the fix was conceptual, not mechanical. The metric routes (E10-H56 source-anchored, E10-H57 string-edit, both 0% triangle on equal-length) and the soft-DTW route (E10-H58) clear the geometry gates but fail the content proxy, parked inconclusive pending a true LLM paraphrase fixture. **E08-H44 remains the shipping metric; E10-H55 is the translation-invariant companion score** for when absolute-position sensitivity is the wrong behavior
- **E11 took the decision and reversed the ship (E10-H55, OPW order-gap)** - on a second article (Wergeland) and a true opus-mt back-translation paraphrase, position-augmented Wasserstein (E08-H44) is `≈ 0.87·SMD` when order is preserved, so it reads a faithful reword at 73.5% of a full-scramble distance - it fuses content and arrangement and cannot be the single *structure, not meaning* number. The OPW order-gap (E10-H55) cancels content (a reword reads 0.5% of full scramble, a reorder 200× higher), holds on both articles, and ships on the library's closeness scale `1 − gap/√2`. **E10-H55 is the structure axis that ships; E08-H44 is dropped as the single number** (kept only as the content-aware distance read beside SMD). H55's score-ness (15% triangle) is accepted - metricity only matters at corpus scale, which a pairwise read never invokes. This **supersedes** the E08/E10 'E08-H44 ships' lines above

## Next steps

- **Run Phase 1 first** - it gates everything; without structure variance ≥ the registered bar on real conversion pairs (independently aligned) the axis is a non-problem
- **Then the raw-`T` sparsity probe (E07-H38) - a second gate** - both Phase 2 and Phase 6 read the plan `T`, and a production single pair uses raw (diffuse) embeddings; run the `T`-sparsity probe on raw single pairs *before* building E07-H39..H43 or trusting Phase 2 separation, mirroring "Run Phase 1 first". A diffuse `T` here drops the whole order/section core
- **Then Phase 6 + Phase 2 - the order/section core** - if `T` is sharp enough, the `T`-induced structure distance (Phase 6) and the positional penalty (Phase 2) are the only mechanisms that see position; Phase 6 binds the signal to `T`, normalizes to `[0,1]`, gates on semantic match, and ships it as the explainable secondary closeness - the deliverable the project actually wants
- **Phase 4 reorder-invariance confirmed; relational-difference still pending** - the E07 control showed GW ≈ 0 on a reorder and 0.0134 on cross-summary pairs, so GW is not the order instrument; a dedicated rewrite/merge/split fixture is still needed for the relational-difference direction, not on the critical path for the order/section deliverable
- **Test the anisotropy lever (ABTT)** - τ barely beat the naive baseline on the easy regime; `all_but_the_top` removes the dominant anisotropy direction and sharpens `T`, which should widen τ's margin over the hard greedy-1-NN baseline on the diffuse cross-summary pairs. Run it the deployable way - a fixed anisotropy direction estimated offline and applied per pair, not corpus-fit on the test pairs (that reintroduces the train→serve leak); run as **E08-H47** - **refuted / inert**: offline-frozen ABTT does not sharpen an already-concentrated `T`, so the lever adds nothing in this regime
- **The E08 batch is the round-2 build** - E08-H44 (position-augmented Wasserstein) and E08-H45 (positional Fused GW) are the metric, wide-dynamic-range OT formulation that replaces the saturating barycentric `τ`; E08-H46/H48 are the metric-correct baselines they must beat on dynamic range; the E08 notebook (`notebooks/experiments/E08-kj-structure-distance-metric.ipynb`) and its two new fixture pieces (pure-shift perturbation, offline anisotropy background set) are built and run - **E08-H44 (position-augmented Wasserstein) is the confirmed win** (a true metric, 0% triangle, wide range, applicability high); positional FGW (E08-H45) is refuted (collapses at extreme reorder), the anisotropy lever (E08-H47) is inert, and the open gate is cross-fixture replication on a second article and on unequal-length conversion pairs
- **The E10 batch is executed - E10-H55 (OPW order-gap) is the positive** - of five structure-sensitive content-invariant reads, only the OPW order-gap `OPW − SMD` holds the combined property E08-H44 and E08-H45 each missed (translation-invariance AND scramble-monotonicity) while staying content-invariant; it is a score, not a metric, so it complements rather than replaces E08-H44. The next step for it is a **true LLM paraphrase fixture** (the current content control is an embedding-jitter proxy) and a second-article replication before promoting it from "promising" to "confirmed". The metric routes (E10-H56 source-anchored, E10-H57 string-edit) and the soft-DTW route (E10-H58) are parked as inconclusive - they clear the geometry gates but fail the content proxy; references and digests for the six papers are in `references/papers/`
- **E11 is the decision - E10-H55 ships** - the structure axis that ships is the OPW order-gap (`structure_closeness = 1 − gap/√2`), distilled into the SOTA design [`../solution/wmd-structure-distance-sota.md`](../solution/wmd-structure-distance-sota.md). Remaining: wire it into `src/` (a `structure_distance` beside `document_distance`, and a structural readout off the OPW plan), the standing real-conversion-pair gate (E07-H28), and a richer second-article set (the Wergeland fixture is three section bases); the a/b-regeneration residual (H55 reads ~0.04 on independent regenerations vs ~0.002 on pure rewords) is honest signal, not a failure, but worth tracking
- **Build the fixture honestly** - per-base order-preserving content edits (not adv1/adv2 reuse), and the independently-aligned real pairs; the ≥2-block paragraph-unit fixture already exists, record real per-cell counts for the rest
- **Vehicle** - `notebooks/experiments/E07-kj-structure-distance.ipynb` is built and run; the next vehicle is the **second-article fixture** that promotes results from "promising" to "confirmed" and selects the normalizer on a genuine held-out split
- **Refuted, do not revisit** - **GW/FGW as the order/section instrument** - GW ≈ 0 on a pure reorder is now confirmed on data, not just argued; the order signal comes from the position-aware Phase 2 + Phase 6, never GW. **Both-document common-mode rejection (E09-H49..H53)** - rejecting the pooled common mode removes shared topic and hurts both axes on diffuse pairs (every CMRR < 1); do not resurrect per-pair or pooled anisotropy removal as a structure or single-pair lever - anisotropy removal stays a corpus-only postprocess (the content track's offline route), and the ~28-statements/pair pooled estimate is too unstable (< 0.9) to trust regardless. **Content-pinned Gromov distortion (E10-H54)** - pinning the SMD plan does not stop the GW collapse at full scramble because the Gromov distortion functional is isometry-invariant and a reversal is an isometry; do not revisit any Gromov-distortion read (pinned or re-optimized) as an order instrument - the isometry invariance is intrinsic, not a solver artifact
- **Open question for the user** - the order signal (Phase 6 τ-from-`T`) is the default lead; confirm whether the relational rewrite axis (Phase 4) is worth a dedicated perturbation, or parked
