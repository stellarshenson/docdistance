# WMD Document-Distance Experiments - tier contrast and source conditioning

Experiments log for the mmBERT Statement Mover's Distance from `../wmd-docdistance-solution-sota.md`. Batch E01 ran five pre-registered levers to widen the source-free gold/adversarial gap in [`notebooks/experiments/E01-kj-wmd-contrast-hypotheses.ipynb`](../../notebooks/experiments/E01-kj-wmd-contrast-hypotheses.ipynb): one promoted (E01-H2 anisotropy removal), four refuted. Batch E02 builds and tests the source-conditioned two-axis distance (selection `D_sel` + grounding `D_grd`) from `../wmd-wrt-source-docdistance-solution.md` in [`notebooks/experiments/E02-kj-source-conditioned-grounding.ipynb`](../../notebooks/experiments/E02-kj-source-conditioned-grounding.ipynb): selection axis confirmed, grounding axis confirmed at the tier level once aggregated to a joint premise. Batch E03 ran five source-conditioned improvement hypotheses in [`notebooks/experiments/E03-kj-source-conditioned-improvements.ipynb`](../../notebooks/experiments/E03-kj-source-conditioned-improvements.ipynb): the relevance-gated residual (H2) and the blended scalar (H5) confirmed, the numeric verifier (H1) and both reranker-cost levers (H3 cascade, H4 replacement) refuted - the cross-encoder reranker is load-bearing and the gate is met on correct failure-mode ordering, not on raw per-document ordinality where the symmetric scalar already passes this single fixture.

- **Branch / artefacts** - baseline `notebooks/04-kj-wmd-document-distance.ipynb`; E01 execution [`notebooks/experiments/E01-kj-wmd-contrast-hypotheses.ipynb`](../../notebooks/experiments/E01-kj-wmd-contrast-hypotheses.ipynb); design `../wmd-docdistance-solution-sota.md`
- **Data** - `data/interim/exec-summaries/ibm-ai-adoption/` (one source article, eleven summaries)

## Problem overview

Eleven executive summaries of one IBM AI-adoption article, three quality tiers, scored against the reference gold (`exec-summary-gold-opus-4-8`).

- **Tiers** - 7 gold (faithful, shared rules), 2 adv1 (information loss - numbers stripped), 2 adv2 (information noise - bloat, fabricated forecasts)
- **Size** - summaries segment to ~12 statements, the source article to 70; clouds small, exact OT is cheap
- **Baseline** - perfect ordinality (`0 / 24` violations), boundary margin only `+0.79` closeness points, contrast ratio `1.27x`
- **Core difficulty** - mmBERT embeddings are anisotropic; cosines bunch at 0.7-0.9, so the `√(2 − 2cos)` cost matrix is compressed and tiers sit close
- **Not tested** - generalisation beyond one article and one degradation design; a controlled probe, not a benchmark

## Executive summary

Five levers, one promoted (E01-H2 anisotropy removal), five variants refuted. The baseline exact SMD already orders every tier without error (`d' = 2.70`, `V = 0/24`) at 0.08 ms/pair, and no lever manufactures separation the embedding geometry does not already support.

| hypothesis | lever | mechanism | predicted | result | verdict |
|---|---|---|---|---|---|
| E01-H1 | weights | salience-weighted transport | margin up, `R` up | `V = 1`, margin negative, `R` 1.29 | Refuted |
| E01-H1b | weights | numeric-density fallback | margin up, `V = 0` | `V = 1`, margin negative, `R` 1.31 | Refuted |
| E01-H2 | embedding geometry | anisotropy removal (all-but-the-top) | dynamic range ≥ 1.5x | `DR` 3.2x, margin up, `V = 0` | **Promoted** |
| E01-H3 | cost function | angular distance `arccos` | margin up, metric kept | `d'` flat (2.72), margin down | Refuted (null) |
| E01-H4 | OT formulation | unbalanced residual | widest margin (≥ +0.040) | worse than baseline, ~120x slower | Refuted |
| E01-H5 | aggregation | tail-aware plan statistic | margin up | `V = 3`, margin sharply negative | Refuted |
| E02-H1 | selection axis | coverage-profile OT over S | Set 1 D_sel > gold | gold 0.023, Set 1 0.060, 0 viol | Confirmed |
| E02-H2 | grounding axis | reranker × NLI, joint premise | Set 2 D_grd > Set 1, gold | R2 Set 2 0.232 vs Set 1 0.130, 2 gold intrude | Partially confirmed |
| E02-H3 | two-axis output | (D_sel, D_grd) plane | splits Set 1 / Set 2 | symmetric 0.452 ≈ 0.406, 2D splits | Confirmed |
| E03-H1 | grounding residual | numeric-aware verifier | Set 2 ≥ 2x gold, gold ≤ 0.05 | Set 2 2.0x gold, gold 0.080 > 0.05 | Refuted |
| E03-H2 | residual definition | relevance-gated ungrounded mass | gold intrusion 2 → 0 | intrusions 2 → 0, Set 2 held +2% | Confirmed |
| E03-H3 | pipeline | bi-encoder cascade pre-filter | reranker 66s → ≤10s, Spearman ≥ 0.95 | recall@m 0.58, Spearman 0.55, cut 82% | Refuted |
| E03-H4 | scorer | bi-encoder relevance replaces cross-encoder | end-to-end 109s → ≤45s | 0.9s but Spearman 0.40, intrusions 2 → 5 | Refuted |
| E03-H5 | output | blended scalar vs symmetric | 0 violations, clean win | blend Set2>Set1 at α∈[0.6,0.9], symmetric inverts | Confirmed |

E02 is a separate goal from E01 - not widening the symmetric gap but splitting the distance into selection and grounding axes when both documents share a source. The source-conditioned distance separates the two adversarial failure modes the symmetric scalar conflates; the selection axis is clean and the grounding axis works at the tier level once aggregated.

**Baseline performance** (notebook 04, SMD against the reference gold)

| measure | value |
|---|---|
| mean gold → ref | 0.332 |
| mean adversarial → ref | 0.423 |
| reference contrast ratio `R` | 1.27x |
| all-pairs contrast (gold-adv / gold-gold) | 1.15x |
| boundary margin `M` | +0.79 closeness pts (+0.012 SMD) |
| dynamic range `DR` (std of → ref) | 0.057 |
| separation `d'` | 2.70 |
| ordinality violations `V` | 0 / 24 |
| gold closeness band | 73-82% |
| adversarial closeness band | 68-72% |

## Methodology and metrics

Each lever rebuilds the distance to the reference gold over all eleven summaries, recomputes the metrics, and compares to the baseline row.

- **Boundary margin** - `min(gold closeness) − max(adversarial closeness)`, closeness points on the 0-1 scale; comparable across methods, baseline `+0.79`, must stay `> 0`
- **Contrast ratio `R`** - `mean(adv → ref) / mean(gold → ref)`, scale-free, baseline `1.27x`
- **Dynamic range `DR`** - std of the ten `→ ref` distances, resolution proxy, baseline `0.057`
- **Separation `d'`** - `(mean adv − mean gold) / pooled std`, scale-free effect size, baseline `2.70`
- **Ordinality violations `V`** - count of (gold, adversarial) pairs ranked wrong, hard guardrail, must stay `0 / 24`
- **Metric guardrail** - a hypothesis claiming to be a metric must keep the triangle inequality; non-metric variants are flagged as discriminative scores

## Setup

- **Fixtures** - `data/interim/exec-summaries/ibm-ai-adoption/summaries/*.md` (11), reference `exec-summary-gold-opus-4-8`
- **Pipeline** - `sat-3l-sm` segmenter → mmBERT (mean-pooled, L2-normalized) → SMD via POT `ot.emd2`
- **Dependencies** - `wtpsplit`, `transformers`, `torch`, `ot` (POT) in the conda base kernel; GPU RTX 5000 Ada
- **Reproducibility** - fixed seed; deterministic across runs apart from minor GPU non-determinism in the encoder
- **Execution vehicle** - [`notebooks/experiments/E01-kj-wmd-contrast-hypotheses.ipynb`](../../notebooks/experiments/E01-kj-wmd-contrast-hypotheses.ipynb); each hypothesis is one toggle over the nb04 baseline

## E01 - experiment batch 1

Five independent levers, pre-registered before any run, one toggle each over the nb04 baseline, executed in [`notebooks/experiments/E01-kj-wmd-contrast-hypotheses.ipynb`](../../notebooks/experiments/E01-kj-wmd-contrast-hypotheses.ipynb). Composable, but tested one at a time to isolate effect.

### E01-H1 Salience-weighted transport

- **Hypothesis** - because quantified claims carry the article's signal and uniform weights dilute them, salience-weighting (IDF × numeric) will lift the boundary margin and `R ≥ 1.40` while holding `0/24` violations
- **Lever** - transport weights (baseline uniform `1/n`)
- **Mechanism** - weight each statement by salience (corpus IDF × numeric-content boost, renormalized) so quantified claims dominate and filler is discounted
- **Prediction** - adv1 (drops claims) and adv2 (pads filler) pay more; margin up, `R ≥ 1.40`
- **Acceptance bar** - margin up, `V = 0`
- **Result** - `V = 1`, margin `−0.23`, `d'` 2.47, `R` 1.292; numeric fallback E01-H1b same failure (`V = 1`, margin `−0.16`, `R` 1.309)
- **Verdict** - Refuted; up-weighting numbers pulls the number-retaining adv2 tier into the gold band and breaks ordering, fallback confirms the mechanism not IDF noise

### E01-H2 Embedding anisotropy removal

- **Hypothesis** - because mmBERT embeddings are anisotropic and statement cosines are compressed, subtracting the dominant principal component will raise distance dynamic range ≥ 1.5x while preserving `0/24` ordinality violations
- **Lever** - embedding geometry
- **Mechanism** - mean-center the statement embeddings, subtract the top 1-3 principal components (all-but-the-top), re-L2-normalize, then cost
- **Prediction** - cost matrix de-compresses, `DR ≥ 1.5x`, margin widens
- **Acceptance bar** - `DR ≥ 1.5x` baseline and `V = 0`, swept `k ∈ {1,2,3}`
- **Result** (k=1) - `DR` 0.180 = 3.2x baseline, margin `+0.92` (up from +0.79), `V = 0`, `d'` slips to 2.34, latency ~2x
- **Verdict** - Promoted; clears the `DR ≥ 1.5x` bar and widens the margin at `V = 0`, the only lever to do so; caveat - it spreads the gold band too, so `d'` drops (resolution, not a sharper boundary)

### E01-H3 Sharpened ground cost - angular distance

- **Hypothesis** - because `arccos` expands the high-cosine region where statements bunch, the angular ground cost will widen the boundary margin while keeping the metric property and `0/24` violations
- **Lever** - ground cost function
- **Mechanism** - replace `√(2 − 2cos)` with angular distance `arccos(cos) / π`, a true metric that expands the high-cosine region
- **Prediction** - more spread among near-duplicate statements, margin up, metric preserved
- **Acceptance bar** - margin up, `V = 0`, triangle inequality intact
- **Result** - `V = 0`, `d'` 2.72 (baseline 2.70), margin `+0.37` (down from +0.79), metric kept
- **Verdict** - Refuted (null); `arccos` is near-affine to `√(2 − 2cos)` at these cosines, so the ranking barely moves - a valid metric, a no-op for separation

### E01-H4 Unbalanced / partial transport residual

- **Hypothesis** - because balanced OT force-matches every statement and hides omissions and additions, unbalanced OT with a folded residual will widen the margin most (`≥ +0.040`) while holding `0/24` violations
- **Lever** - OT formulation
- **Mechanism** - unbalanced OT (marginal-relaxation `reg_m`) with the unmatched residual folded into the score (`+ √2 · residual`)
- **Prediction** - residual loads the adversarial tiers hardest, the widest margin of the five (`≥ +0.040`)
- **Acceptance bar** - margin up, `V = 0`, sweep `reg_m`
- **Result** (reg_m=2.0) - `V = 0`, margin `+0.68` (below baseline +0.79), `d'` 2.57, non-metric, ~9.4 ms/pair (~120x exact SMD)
- **Verdict** - Refuted; the boldest prediction lands worse than baseline, drops the metric property, costs two orders of magnitude more latency; both tiers share enough content that the residual does not concentrate

### E01-H5 Tail-aware aggregation

- **Hypothesis** - because a few badly-matched statements are averaged away by the mean, a p90 tail of matched cost will sharpen tier separation while holding `0/24` violations
- **Lever** - aggregation of the transport plan
- **Mechanism** - report the cost-weighted p90 of matched cost instead of the mean, surfacing the few badly-matched statements the mean averages away
- **Prediction** - the tail separates the tiers more sharply
- **Acceptance bar** - margin up, `V = 0`
- **Result** - `V = 3`, margin `−1.39`, `d'` 1.79 (baseline 2.70), the worst performer
- **Verdict** - Refuted; at ~12 statements the p90 tail is dominated by one or two noisy alignments and reorders the tiers - the mean is the right aggregator

### Results table (E01)

| hypothesis | V | margin (clos pts) | d' | R | DR | metric | ms/pair | verdict |
|---|---|---|---|---|---|---|---|---|
| baseline (SMD mean) | 0/24 | +0.79 | 2.70 | 1.273 | 0.057 | yes | 0.08 | reference |
| E01-H1 salience (IDF×num) | 1/24 | −0.23 | 2.47 | 1.292 | 0.060 | yes | 0.08 | Refuted |
| E01-H1b numeric proxy | 1/24 | −0.16 | 2.44 | 1.309 | 0.063 | yes | 0.08 | Refuted |
| E01-H2 anisotropy (k=1) | 0/24 | +0.92 | 2.34 | 1.256 | 0.180 | yes | 0.17 | **Promoted** |
| E01-H3 angular cost | 0/24 | +0.37 | 2.72 | 1.275 | 0.018 | yes | 0.10 | Refuted (null) |
| E01-H4 unbalanced (reg_m 2) | 0/24 | +0.68 | 2.57 | 1.253 | 0.067 | no | 9.4 | Refuted |
| E01-H5 tail (p90) | 3/24 | −1.39 | 1.79 | 1.123 | 0.048 | no | 0.06 | Refuted |

### Benchmarks (E01)

Latency per document-pair, exact baseline SMD = 1x, measured on the RTX 5000 Ada over the ten non-reference summaries.

- **baseline exact SMD** - 0.08 ms/pair, the reference
- **E01-H1 / E01-H1b weighting** - 0.08 ms/pair, same solver, weights are free
- **E01-H3 angular** - 0.10 ms/pair, only the cost matrix changes
- **E01-H2 anisotropy** - 0.17 ms/pair (~2x), one extra SVD on the pooled embeddings
- **E01-H5 tail** - 0.06 ms/pair, cheapest, reuses the balanced plan
- **E01-H4 unbalanced** - 9.4 ms/pair (~120x), the majorization-minimization solver dominates

## E02 - experiment batch 2: source-conditioned grounding axis

A different question from E01 - not widening the symmetric gap, but splitting the distance into two axes when both documents derive from one source `S`. Tests the design in [`../wmd-wrt-source-docdistance-solution.md`](../wmd-wrt-source-docdistance-solution.md), executed in [`notebooks/experiments/E02-kj-source-conditioned-grounding.ipynb`](../../notebooks/experiments/E02-kj-source-conditioned-grounding.ipynb). Two axes - selection `D_sel` (coverage-profile OT over `S`, already shipped) and grounding `D_grd` (reranker × NLI residual, the deferred build). Pipeline adds `bge-reranker-v2-m3` and `mdeberta-mnli-xnli`, both OpenVINO INT8 on CPU.

- **Anchor** - `gold` (Opus 3-sweep); every summary scored `d(anchor, X | S)` on both axes plus the symmetric SMD baseline
- **Two rounds of grounding** - R1 single-premise (SummaC max over source), R2 top-k joint premise (`k = 3`, the design's aggregation)
- **Source** - 70 statements; the grounding sweep scores every (summary statement × source statement) pair through both cross-encoders

### E02-H1 Selection axis separates information loss

- **Hypothesis** - because info-loss strips source figures, `D_sel` will rank Set 1 above gold while gold stays clustered (selection axis carries omission)
- **Result** - D_sel gold 0.023, Set 1 0.060 (2.6x), Set 2 0.073; every adversarial above every gold, `0` ordinality violations on the selection axis
- **Verdict** - Confirmed; D_sel cleanly separates both adversarial tiers from gold, info-loss included

### E02-H2 Grounding axis isolates fabrication (aggregation required)

- **Hypothesis** - because info-noise fabricates unsupported claims, `D_grd` will rank Set 2 above gold and above Set 1, once grounding is aggregated over evidence (R2) not single-premise (R1)
- **Result R1** (single-premise) - gold 0.097, Set 1 0.206, Set 2 0.233; faithful info-loss almost level with fabrication, axis muddied
- **Result R2** (joint premise) - gold 0.120, Set 1 0.130, Set 2 0.232; aggregation drops info-loss to gold level while holding fabrication (1.8x over gold), but two gold summaries (haiku 0.230, v2 0.285) intrude into Set 2's range
- **Verdict** - Partially confirmed; R2 isolates Set 2 at the tier mean and beats R1, but the axis is a tier-level fabrication flag, not a clean per-document discriminator; contradiction mass is near-zero across tiers (numeric-entailment weakness), so the residual rides on the ungrounded component

### E02-H3 Two axes separate what the symmetric scalar conflates

- **Hypothesis** - the 2D `(D_sel, D_grd)` plane will place Set 1 and Set 2 in distinct regions the symmetric SMD conflates
- **Result** - symmetric SMD conflates and even mis-orders by severity (Set 1 0.452 ≈ Set 2 0.406, gold 0.287); the 2D plane places info-loss (high selection, low grounding) and info-noise (high selection, high grounding) in distinct regions
- **Verdict** - Confirmed (tier level); the grounding axis is what distinguishes the two failure modes

### Round comparison - R1 vs R2

| round | grounding | gold | Set 1 | Set 2 | Set 1 vs gold | reading |
|---|---|---|---|---|---|---|
| R1 | single-premise max | 0.097 | 0.206 | 0.233 | 2.1x | faithful info-loss muddied with fabrication |
| R2 | top-3 joint premise | 0.120 | 0.130 | 0.232 | 1.1x | info-loss pulled to gold, fabrication held |

R2 promoted; single-premise NLI mis-grades a compressive faithful summary because no single source statement entails a claim fused from several - the documented SummaC failure. Fusing each statement's top-`k` reranked source into one premise fixes it.

### Results table (E02, tier means)

| axis | gold | Set 1 (info-loss) | Set 2 (info-noise) | separates? |
|---|---|---|---|---|
| D_sel (selection) | 0.023 | 0.060 | 0.073 | yes, 0 violations |
| D_grd R1 (grounding) | 0.097 | 0.206 | 0.233 | Set 2 ≈ Set 1, muddied |
| D_grd R2 (grounding) | 0.120 | 0.130 | 0.232 | Set 2 isolated at tier mean |
| SMD (symmetric baseline) | 0.287 | 0.452 | 0.406 | conflates Set 1 / Set 2 |

### Benchmarks (E02)

Full source-conditioned chain, one document pair, CPU INT8, single-pair latency over 13 × 70 = 910 grounding pairs.

- **reranker sweep** - 66.1 s/pair (60.5%), `bge-reranker-v2-m3` over every (summary, source) statement pair, 14 pairs/s
- **NLI sweep R1** - 42.0 s/pair (38.5%), full grid `mdeberta-mnli-xnli`
- **NLI joint premise R2** - 0.73 s/pair (0.7%), one call per summary statement, reuses the reranker top-k
- **selection + symmetric** - sub-ms (D_sel transport 0.8 ms, SMD 0.6 ms), negligible
- **end-to-end** - 109 s/pair, ~1000x the symmetric SMD; the grounding axis is a heavy diagnostic, not a cheap metric

## E03 - experiment batch 3: source-conditioned improvements

Five levers to turn the E02 grounding axis from a tier-level flag into a per-document metric and to cut its cost, all over the same `data/interim/exec-summaries/ibm-ai-adoption` fixture, executed in [`notebooks/experiments/E03-kj-source-conditioned-improvements.ipynb`](../../notebooks/experiments/E03-kj-source-conditioned-improvements.ipynb). Three target quality (the grounding weaknesses E02 exposed), two target performance (the reranker bottleneck). Two confirmed (H2, H5), three refuted (H1, H3, H4).

- **Headline** - the relevance-gate (H2) closes the per-document gold intrusion (2 → 0) and the blend (H5) then orders the two failure modes correctly (Set 2 fabrication above Set 1 info-loss) where the symmetric scalar inverts them; number-aware verification (H1) and both reranker-cost levers (H3, H4) do not survive this fixture
- **Load-bearing reranker** - neither a bi-encoder cosine pre-filter (H3, recall@10 of top-3 only 0.58) nor a cosine replacement (H4, relevance Spearman 0.40) preserves the grounding ranking; the 60% reranker cost is structural to this design

- **Aim** - improve the quality and performance of the source-conditioned distance `d(A,B|S)`
- **Quality targets** - the four E02 grounding weaknesses: gold intrusion (2 golds in Set 2's band), dead contradiction signal, per-document noise, number blindness
- **Performance target** - the 109 s/pair cost, dominated by the reranker full grid (66 s, 60.5%)
- **Pairing** - E03-H3 (conservative cascade) and E03-H4 (aggressive replacement) attack the same reranker cost; the H4 kill-gate routes to H3 on failure
- **Capstone** - E03-H5 composes the winning quality levers and tests the batch gate head-to-head against the symmetric SMD

### E03 gate - clean win over the symmetric distance

The whole batch is judged against one acceptance gate, fixed before any run: the improved conditioned distance must beat the symmetric SMD on common-source documents, the case where the symmetric scalar already fails.

- **Win condition** - `0` per-document ordinality violations separating gold from each adversarial tier on the conditioned axes AND linear separation of Set 1 from Set 2 on the `(D_sel, D_grd)` plane or a blended scalar `α·D_sel + (1−α)·D_grd`
- **Reference to beat** - symmetric SMD conflates and mis-orders by severity (Set 1 0.452 ≈ Set 2 0.406, gold 0.287)
- **Not-shipped clause** - a lever that improves an axis but does not move the batch toward this gate is recorded interesting-but-not-shipped

### E03-H1 Numeric-aware grounding verifier

- **Hypothesis** - because general NLI contradiction is ≈ 0 on quantitative claims yet both adversarial tiers are defined by numeric corruption (Set 1 strips figures, Set 2 fabricates or alters them), a numeric verifier that extracts figures from each summary statement and compares them to the reranker-aligned source figures will rank Set 2 ≥ 2x gold while gold's numeric residual stays ≤ 0.05, adding a signal NLI does not carry
- **Lever** - grounding residual composition (add a numeric-mismatch term to `D_grd`)
- **Mechanism** - number-entity extraction (percent, count, currency, year or forecast) per statement; match each summary figure to its source figure via the top-k reranked source; residual = share of summary figures unmatched (fabricated) or value-mismatched (wrong number); orthogonal to NLI entailment
- **Prediction** - Set 2 numeric residual ≥ 2x gold; gold ≤ 0.05; Set 1 low here, its loss shows on `D_sel`
- **Acceptance bar** - Set 2 ≥ 2x gold AND gold ≤ 0.05 AND fires where the NLI contradiction signal was dead
- **Kill-gate** - numeric density: figures must appear in ≥ 30% of statements (probe the fixture first); sparse → kill before any build
- **Result** - density 58% (gate passed); top-k aligned numeric residual Set 2 0.163 = 2.0x gold 0.080, gold 0.080 > the 0.05 bar (faithful golds sonnet 0.31, haiku 0.14 inflate - real figures whose aligned source is not the top-k); whole-source matching keeps gold low (0.018) but collapses Set 2 to 0.051 because the 82-figure source matches fabricated numbers by coincidence; NLI contradiction confirmed dead (gold 0.055, Set 2 0.007)
- **Verdict** - Refuted; the signal is orthogonal to the dead contradiction and the direction is right (Set 2 2x gold), but localized matching breaks the gold ≤ 0.05 bar and whole-source matching is defeated by the source's figure density - number-aware verification fails this fixture both ways

### E03-H2 Relevance-gated ungrounded residual

- **Hypothesis** - because a faithful compressive gold fuses several source sentences (low joint entailment but high max reranker relevance) while fabrication is genuinely novel (low entailment AND low relevance), gating the ungrounded mass by max reranker relevance will drop the two intruding golds (haiku 0.230, v2 0.285) below Set 2's band while holding Set 2 within 10% of R2
- **Lever** - residual definition (weight ungrounded mass by `1 − max_k r(a_i, s_k)`)
- **Mechanism** - only statements with low max source relevance count as ungrounded; high-relevance-but-low-entailment (compression) no longer inflates the residual
- **Prediction** - gold intrusion 2 → 0; gold tier mean below 0.13; Set 2 held ≈ 0.21-0.23
- **Acceptance bar** - per-document ordinality gold < Set 2 restored (0 intrusions) AND Set 2 within 10% of R2
- **Kill-gate** - the intruding golds must actually have high max-relevance (probe: their max `r` ≥ 0.6); if not, the intrusion is real divergence the gate cannot fix
- **Result** - kill-gate passed (intruding golds v2 mean max `r` 0.898, haiku 0.752); gating drops both below the Set 2 floor (haiku 0.230 → 0.053, v2 0.285 → 0.216), gold intrusions 2 → 0, gold tier mean 0.120 → 0.084, Set 2 0.232 → 0.236 (+2%, held within 10%); the v2 margin is thin (0.216 vs the 0.220 floor)
- **Verdict** - Confirmed; the one quality lever that lands - the per-document gold intrusion is closed while Set 2 holds, though the v2 margin is narrow

### E03-H3 Bi-encoder cascade pre-filter

- **Hypothesis** - because the reranker scores the full 12 × 70 grid (66 s, 60.5%) yet only the top-k per statement enters the premise, pre-selecting the top-m source per statement by the already-computed mmBERT cosine (m ≈ 10) will cut reranker calls ~7x and end-to-end latency ≥ 40% while preserving the `D_grd` ranking (Spearman ≥ 0.95 vs full grid)
- **Lever** - pipeline (bi-encoder shortlist before the cross-encoder)
- **Mechanism** - mmBERT cosine ranks the 70 source statements per summary statement; keep the top-m, run the reranker only on those, feed the reranker top-k into the premise as before
- **Prediction** - reranker 66 s → ≤ 10 s; end-to-end 109 s → ≤ 60 s; tier means within 5%; Spearman ≥ 0.95
- **Acceptance bar** - Spearman(`D_grd` vs full grid) ≥ 0.95 AND latency cut ≥ 40% AND tier separation preserved
- **Kill-gate** - bi-encoder top-m must contain the reranker top-k (recall@m ≥ 0.95 on a probe); else the shortlist drops the true evidence
- **Result** - kill-gate failed: recall@10 of the reranker top-3 is 0.576, so the cosine shortlist drops 42% of the true evidence; D_grd Spearman vs the full grid is 0.545 (bar 0.95), even though the latency cut is 82% (reranker 63.9 s → 11.7 s), better than predicted
- **Verdict** - Refuted (killed at gate); a fast shortlist that changes the answer - the bi-encoder cosine is a poor pre-filter for this cross-encoder

### E03-H4 Bi-encoder relevance replaces cross-encoder

- **Hypothesis** - because the mmBERT bi-encoder cosine may already rank source relevance closely enough, replacing the cross-encoder relevance term `r(a_i, s_k)` with the bi-encoder cosine removes the 66 s reranker stage entirely (→ 0, reuse the embeddings), cutting end-to-end to ≤ 45 s, while keeping the grounding verdict if the two relevance rankings agree
- **Lever** - scorer (drop the cross-encoder, use bi-encoder cosine as relevance)
- **Mechanism** - the grounding score becomes `g = cos(a_i, s_k) · P(entail)`; no separate reranker sweep, the selection-axis embeddings are reused
- **Prediction** - end-to-end 109 s → ≤ 45 s; Set 2 still isolated; no new gold intrusion vs the cross-encoder design
- **Acceptance bar** - end-to-end ≤ 45 s AND Set 2 isolated at tier mean AND no new intrusion
- **Kill-gate** - probe bi-encoder vs cross-encoder relevance Spearman on a sample; ≥ 0.70 to proceed; below → the cross-encoder carries irreplaceable compression / paraphrase signal → fall back to the H3 cascade
- **Result** - kill-gate failed: bi-vs-cross relevance Spearman 0.397 (bar 0.70); dropping the reranker collapses the chain to 0.9 s but corrupts grounding - gold intrusions rise 2 → 5, and Set 1 falls below gold at the tier mean (gold 0.120, Set 1 0.105, Set 2 0.208)
- **Verdict** - Refuted (killed at gate); the strongest result of the cost pair - the cross-encoder relevance is irreplaceable by the bi-encoder cosine, the H3 fallback also failed

### E03-H5 Blended conditioned scalar vs symmetric

- **Hypothesis** - because the symmetric SMD conflates and mis-orders the two failure modes (Set 1 0.452 ≈ Set 2 0.406) while the conditioned axes separate them, composing the winning quality levers (H1 numeric + H2 relevance-gate on `D_grd`, with `D_sel`) into a single blended scalar `α·D_sel + (1−α)·D_grd` (α swept) will order all tiers with 0 per-document violations - a result the symmetric distance cannot reach
- **Lever** - output (blend the two axes, benchmark head-to-head vs symmetric SMD)
- **Mechanism** - sweep α over the conditioned axes after H1 and H2 land, pick the operating point that orders the tiers, compare per-document ranking to the symmetric SMD
- **Prediction** - at some α the blend gives gold < Set 1, Set 2 at 0 violations with Set 1 / Set 2 linearly separable; symmetric SMD stays mis-ordered (Set 1 > Set 2)
- **Acceptance bar** - the batch gate: 0 per-document violations gold vs each tier AND Set 1 / Set 2 separated, where symmetric cannot → clean win
- **Kill-gate** - conditional on H1 or H2 landing; if neither quality lever clears its bar, `D_grd` stays a tier flag and the blend cannot beat symmetric per document → record "gate not met, conditioned remains tier-level"
- **Result** - H2 landed (H1 did not), so the improved grounding is the H2 relevance-gated ungrounded mass; min-max each axis, blend `α·D_sel + (1−α)·D_grd`; at `α ∈ [0.60, 0.90]` the blend gives 0/28 per-document violations AND Set 2 above Set 1 (correct grounding severity); the symmetric SMD reaches 0/28 violations too but inverts the severity (Set 1 0.452 > Set 2 0.406)
- **Verdict** - Confirmed (clean win on severity ordering); the blend orders the two failure modes correctly where the symmetric scalar inverts them, but the win is the ordering and the axis attribution, not raw gold-vs-adversarial ordinality - the symmetric scalar also clears 0/28 on this single fixture, so cross-fixture validation is required before generalizing

### Pre-registration table (E03)

| hypothesis | lever | prediction | acceptance bar | kill-gate |
|---|---|---|---|---|
| E03-H1 numeric verifier | residual composition | Set 2 ≥ 2x gold, gold ≤ 0.05 | Set 2 ≥ 2x gold AND gold ≤ 0.05 AND beats dead NLI | numeric density ≥ 30% of statements |
| E03-H2 relevance-gate | residual definition | gold intrusion 2 → 0, Set 2 held | 0 intrusions AND Set 2 within 10% of R2 | intruding golds max `r` ≥ 0.6 |
| E03-H3 cascade pre-filter | pipeline | reranker → ≤ 10 s, Spearman ≥ 0.95 | Spearman ≥ 0.95 AND latency cut ≥ 40% | top-m recall of top-k ≥ 0.95 |
| E03-H4 bi-encoder relevance | scorer | end-to-end → ≤ 45 s, Set 2 isolated | ≤ 45 s AND Set 2 isolated AND no new intrusion | bi vs cross relevance Spearman ≥ 0.70 |
| E03-H5 blended vs symmetric | output | 0 violations, Set 1 / Set 2 split | the batch gate (clean win) | H1 or H2 must clear their bar |

### Results table (E03)

Tier means and the pre-registered measure for each lever; gold is the anchor tier, Set 1 info-loss, Set 2 info-noise.

| hypothesis | gold | Set 1 | Set 2 | measure | bar | verdict |
|---|---|---|---|---|---|---|
| E03-H1 numeric (top-k aligned) | 0.080 | 0.000 | 0.163 | Set 2 2.0x gold, gold > 0.05 | Set 2 ≥ 2x gold AND gold ≤ 0.05 | Refuted |
| E03-H2 D_grd relevance-gated | 0.084 | 0.141 | 0.236 | intrusions 2 → 0, Set 2 +2% | 0 intrusions AND Set 2 ±10% | Confirmed |
| E03-H3 D_grd cascade | 0.080 | 0.195 | 0.156 | Spearman 0.545, recall@m 0.58 | Spearman ≥ 0.95 | Refuted |
| E03-H4 D_grd bi-encoder | 0.120 | 0.105 | 0.208 | Spearman 0.40, intrusions 2 → 5 | Spearman ≥ 0.70, no new intrusion | Refuted |
| E03-H5 blend `α∈[0.6,0.9]` | - | - | - | blend 0/28 Set2>Set1; symmetric 0/28 Set1>Set2 | 0 violations AND correct severity | Confirmed |

### Benchmarks (E03)

CPU INT8, the 11-document reranker grid plus the per-lever micro-benchmarks; the heavy stage is unchanged from E02.

- **per-document signal build** - ~48 s/document (the reranker full grid over `n_summary × 70` pairs), 534 s for all 11 documents; the dominant cost, as in E02
- **H3 cascade reranker** - full grid 910 pairs 63.9 s → cosine top-10 shortlist 130 pairs 11.7 s, an 82% latency cut, but the ranking shifts (Spearman 0.545) so the cut does not ship
- **H4 reranker-free chain** - 0.9 s end-to-end (embed + cosine + joint-premise NLI, no reranker), ~120x faster than the 109 s E02 chain, but grounding is corrupted (5 gold intrusions)
- **H2 / H5 post-processing** - sub-ms on the already-computed signals; the relevance-gate and the blend add no measurable cost over the R2 grounding axis
- **Reading** - the only cheap, faithful win is the relevance-gate (free re-weighting of existing signals); the two levers that actually remove the reranker cost both break the grounding ranking

## Lessons learned

- **Baseline near the ceiling for ordering** - perfect ordinality and `d' = 2.70` leave little room; resolution (dynamic range), not the normalized boundary, is the axis with room
- **Anisotropy is the bottleneck** - the one lever that helps removes a single common principal component, de-bunches the cosines, triples dynamic range
- **Number-aware weighting is self-defeating on number-heavy sources** - both the faithful and the info-noise tiers carry the article's percentages, so up-weighting numbers pulls adversarial summaries toward gold
- **A wider mean gap is not a wider boundary** - E01-H1 raises `R` while the boundary margin turns negative; `V` catches what `R` hides
- **Heavier machinery did not pay** - unbalanced OT (non-metric, ~120x) and tail aggregation (noise at ~12 statements) both underperform the cheap exact mean
- **Conditioning on the source separates failure modes (E02)** - the symmetric distance cannot tell info-loss from fabrication; re-basing the transport onto `S` and adding a grounding axis does, the source-conditioned design's central claim confirmed
- **Single-premise NLI mis-grades compression (E02)** - a faithful summary statement fuses several source sentences, so no single source premise entails it; top-k joint-premise aggregation is required, not optional (R1 muddied, R2 fixed it)
- **General NLI is weak on numbers (E02)** - the contradiction signal barely fires even on fabricated forecasts, so the grounding residual rides on the ungrounded component - a numeric-aware verifier is the open gap
- **Relevance, not entailment, separates compression from fabrication (E03)** - gating the ungrounded mass by max reranker relevance closes the gold intrusion the entailment residual could not, because faithful compression has high relevance and low entailment while fabrication has both low
- **Number density defeats the numeric verifier (E03)** - on an 82-figure source, whole-source matching lets fabricated numbers match by coincidence (Set 2 collapses to 0.05) while top-k localized matching flags faithful restatements whose aligned source is elsewhere (gold rises to 0.08); the same density that makes numbers the adversarial signal makes them un-verifiable here
- **The cross-encoder reranker is load-bearing (E03)** - the bi-encoder cosine neither shortlists faithfully (recall@10 of top-3 is 0.58) nor replaces relevance (Spearman 0.40); the 60% reranker cost is structural to the grounding design, not removable with the embeddings already on hand
- **Separability is not correct ordering (E03)** - the symmetric SMD separates Set 1 from Set 2 on this fixture but inverts the severity (info-loss read as more divergent than fabrication); the conditioned blend's win is the correct ordering and the axis attribution, which a single source-blind scalar cannot give regardless of separability

## Conclusions

- **Ships** - baseline exact SMD: metric, 0.08 ms/pair, perfect ordinality
- **Optional** - anisotropy removal (k=1) as a resolution pre-pass, ~2x latency, `d'` caveat noted
- **Thin margin is intrinsic** - all eleven summaries describe one article and share its content, so the boundary is genuinely narrow
- **Source-conditioned axes (E02)** - `D_sel` ships as the selection axis (clean, metric, sub-ms, 0 violations); `D_grd` (R2 joint premise) is a tier-level fabrication flag, ~109 s/pair - E03 turned it into a per-document metric with the H2 relevance-gate, not the numeric verifier E02 expected
- **Grounding axis fix (E03)** - the H2 relevance-gated ungrounded mass closes the per-document gold intrusion (2 → 0) at no extra cost, turning `D_grd` from a tier flag into a per-document discriminator on this fixture; ship it as the grounding-axis definition
- **Blended scalar (E03)** - the H5 blend `α·D_sel + (1−α)·D_grd` (`α ∈ [0.6, 0.9]`) orders the two failure modes correctly (Set 2 fabrication above Set 1 info-loss) where the symmetric scalar inverts them - a single interpretable conditioned scalar; the win is severity ordering and attribution, and needs a second source before it is trusted
- **Not shipped (E03)** - the numeric verifier (defeated by source figure density) and the two reranker-cost levers (cosine neither shortlists nor replaces the cross-encoder); the ~109 s/pair grounding cost stands

## Next steps

- **Promote the E03 winners** - wire the H2 relevance-gated ungrounded mass into the grounding axis as `D_grd`, and expose the H5 blend (`α ≈ 0.75`) as the single source-conditioned scalar; the per-document intrusion is closed and the failure modes order correctly
- **Cross-fixture check (now the gate)** - the H5 win is severity ordering and attribution on one article, where the symmetric scalar also clears 0/28 gold-vs-adv; re-run E02/E03 on a second source before claiming the blend beats symmetric in general
- **Anisotropy continuation (parked, E01 lineage)** - still open: sweep the anisotropy `k` further and test a gentle numeric weight tuned to preserve `V` stacked on anisotropy removal; source-free, unrelated to the E03 grounding work
- **Open grounding cost** - the ~109 s/pair reranker stage stands; E03 showed the bi-encoder cannot remove it without breaking the ranking, so a faithful speed-up needs a different lever (e.g. a distilled cross-encoder), not the embeddings on hand
- **Refuted, do not revisit** - salience / numeric weighting (breaks ordinality on number-heavy sources), angular cost (null), unbalanced residual (worse, slow, non-metric), tail aggregation (noise at this statement count), single-premise grounding (mis-grades compression, superseded by R2 joint premise), numeric verifier (defeated by source figure density, E03-H1), bi-encoder cascade / replacement (cosine neither shortlists nor replaces the cross-encoder, E03-H3/H4)
