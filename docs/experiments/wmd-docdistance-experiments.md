# WMD Document-Distance Experiments - tier contrast and source conditioning

**Canonical Experiments Document**

Experiments log for the mmBERT Statement Mover's Distance from `../solution/wmd-docdistance-solution-sota.md`. Batch E01 ran five pre-registered levers to widen the source-free gold/adversarial gap in [`notebooks/experiments/E01-kj-wmd-contrast-hypotheses.ipynb`](../../notebooks/experiments/E01-kj-wmd-contrast-hypotheses.ipynb): one promoted (E01-H3 anisotropy removal), four refuted. Batch E02 builds and tests the source-conditioned two-axis distance (selection `D_sel` + grounding `D_grd`) from `../solution/wmd-source-conditioned-docdistance-solution-sota.md` in [`notebooks/experiments/E02-kj-source-conditioned-grounding.ipynb`](../../notebooks/experiments/E02-kj-source-conditioned-grounding.ipynb): selection axis confirmed, grounding axis confirmed at the tier level once aggregated to a joint premise. Batch E03 ran five source-conditioned improvement hypotheses in [`notebooks/experiments/E03-kj-source-conditioned-improvements.ipynb`](../../notebooks/experiments/E03-kj-source-conditioned-improvements.ipynb): the relevance-gated residual (H11) and the blended scalar (H14) confirmed, the numeric verifier (H10) and both reranker-cost levers (H12 cascade, H13 replacement) refuted - the cross-encoder reranker is load-bearing and the gate is met on correct failure-mode ordering, not on raw per-document ordinality where the symmetric scalar already passes this single fixture. Batch E04 ran five source-conditioned performance hypotheses on the GPU fp16 chain: anisotropy removal on the conditioned selection axis (H15) widens dynamic range 3.39x at `0` violations - the one win, shipped as the default resolution pre-pass (opt-out via `anisotropy=False`); coverage-temperature sharpening (H16), a distilled-reranker replacement (H17), a cross-encoder cascade (H18) and the composite (H19) are refuted - the cross-encoder reranker stays load-bearing and its ~109 s/pair cost stands, with the H18 cascade the closest faithful speed-up (Spearman 0.98, 2.3x, but recall@15 0.87 short of the gate). Batch E05 ran five CPU-speed hypotheses - three relevance-scorer hunts (a smaller multilingual cross-encoder, a late-interaction ColBERT-style scorer, a learned-sparse SPLADE-style scorer) and two structural grid cuts (length-bucketed batching, source-statement clustering) - against a `>= 50%` CPU latency cut at a preserved verdict; all five refuted, the v2-m3 cross-encoder reranker is reconfirmed irreplaceable (no smaller scorer recalls its top-3) and the one practical win is a `42.5%` free CPU cut from length-bucketing (E06-H25, bit-identical scores), 7 points short of the bar. Batch E06 reopens the two directions E05 left untested - a *trained* multilingual late-interaction ColBERT and a *trained* multilingual learned-sparse scorer (the E05 probes ran on an untrained mmBERT backbone) - and promotes length-bucketing to a reserved CPU-speed candidate, scored against a looser gate than E05: any lift over the shipped chain in either a performance metric or CPU latency, not the `>= 50%` cut. Batch E12 pre-registers eleven post-1.1.5 improvement hypotheses from a repo-only research fanout (calibration transfer, reporting fidelity, winner decomposition, runtime memoisation); unrun at registration.

- **Branch / artefacts** - baseline `notebooks/04-kj-wmd-document-distance.ipynb`; E01 execution [`notebooks/experiments/E01-kj-wmd-contrast-hypotheses.ipynb`](../../notebooks/experiments/E01-kj-wmd-contrast-hypotheses.ipynb); design `../solution/wmd-docdistance-solution-sota.md`
- **Data** - `data/interim/exec-summaries/ibm-ai-adoption/` (one source article, eleven summaries)

> [!NOTE]
> **Which side to trust.** Every number in this log is a hand transcription from the notebook or
> `reports/*.json` named in its section. Where the two disagree, **the committed artefact governs**
> - the prose was written against earlier runs and the batches were re-executed afterwards. The
> headline, verdict-bearing and user-facing figures were reconciled against the artefacts on
> 2026-08-12; the per-hypothesis long tail has not been, so treat any unreconciled figure as
> provisional and read it off the artefact. Note two renumberings when you check: the E01 notebook
> uses `H1, H1b, H2..H5` where this log uses `E01-H1..E01-H6`, and the E04 report keys are
> `E04-H1..H5` where this log uses `E04-H15..E04-H19`.

## Authors

- `@kj` Konrad Jelen

## Problem overview

Eleven executive summaries of one IBM AI-adoption article, three quality tiers, scored against the reference gold (`exec-summary-gold-opus-4-8`).

- **Tiers** - 7 gold (faithful, shared rules), 2 adv1 (information loss - numbers stripped), 2 adv2 (information noise - bloat, fabricated forecasts)
- **Size** - summaries segment to ~12 statements, the source article to 70; clouds small, exact OT is cheap
- **Baseline** - perfect ordinality (`0 / 24` violations), boundary margin `+4.58` closeness points, contrast ratio `1.68x`
- **Core difficulty** - mmBERT embeddings are anisotropic; cosines bunch at 0.7-0.9, so the `√(2 − 2cos)` cost matrix is compressed and tiers sit close
- **Not tested** - generalisation beyond one article and one degradation design; a controlled probe, not a benchmark

## Executive summary

Two distances ship from this work. **Solution 1 - symmetric SMD** answers "how far apart are two documents" with one source-blind, metric scalar (E01); it orders every tier on this fixture at 0.08 ms/pair but cannot say *why* two documents differ. **Solution 2 - source-conditioned d(A,B|S)** re-bases the comparison on the shared source and splits the distance into `D_sel` (selection - does it cover the same content) and `D_grd` (grounding - is what it says supported), separating the two adversarial failure modes the symmetric scalar conflates (E02/E03). The converged design runs in `notebooks/05-kj-source-conditioned-distance.ipynb` (CPU INT8 or GPU) and the shipped library is validated end-to-end in `notebooks/09-kj-docdistance-api-e2e.ipynb`.

Five levers, one promoted (E01-H3 anisotropy removal), five variants refuted. The baseline exact SMD already orders every tier without error (`d' = 4.60`, `V = 0/24`) at 0.08 ms/pair, and no lever manufactures separation the embedding geometry does not already support.

| hypothesis | lever | mechanism | predicted | result | verdict |
|---|---|---|---|---|---|
| E01-H1 | weights | salience-weighted transport | margin up, `R` up | `V = 1`, margin negative, `R` 1.29 | Refuted |
| E01-H2 | weights | numeric-density fallback | margin up, `V = 0` | `V = 1`, margin negative, `R` 1.31 | Refuted |
| E01-H3 | embedding geometry | anisotropy removal (all-but-the-top) | dynamic range ≥ 1.5x | `DR` 3.2x, margin up, `V = 0` | **Promoted** |
| E01-H4 | cost function | angular distance `arccos` | margin up, metric kept | `d'` flat (2.72), margin down | Refuted (null) |
| E01-H5 | OT formulation | unbalanced residual | widest margin (≥ +0.040) | worse than baseline, ~120x slower | Refuted |
| E01-H6 | aggregation | tail-aware plan statistic | margin up | `V = 3`, margin sharply negative | Refuted |
| E02-H7 | selection axis | coverage-profile OT over S | Set 1 D_sel > gold | gold 0.023, Set 1 0.060, 0 viol | Confirmed |
| E02-H8 | grounding axis | reranker × NLI, joint premise | Set 2 D_grd > Set 1, gold | R2 Set 2 0.232 vs Set 1 0.130, 2 gold intrude | Partially confirmed |
| E02-H9 | two-axis output | (D_sel, D_grd) plane | splits Set 1 / Set 2 | symmetric 0.452 ≈ 0.406, 2D splits | Confirmed |
| E03-H10 | grounding residual | numeric-aware verifier | Set 2 ≥ 2x gold, gold ≤ 0.05 | Set 2 2.0x gold, gold 0.080 > 0.05 | Refuted |
| E03-H11 | residual definition | relevance-gated ungrounded mass | gold intrusion 2 → 0 | intrusions 2 → 0, Set 2 held +2% | Confirmed |
| E03-H12 | pipeline | bi-encoder cascade pre-filter | reranker 66s → ≤10s, Spearman ≥ 0.95 | recall@m 0.58, Spearman 0.55, cut 82% | Refuted |
| E03-H13 | scorer | bi-encoder relevance replaces cross-encoder | end-to-end 109s → ≤45s | 0.9s but Spearman 0.40, intrusions 2 → 5 | Refuted |
| E03-H14 | output | blended scalar vs symmetric | 0 violations, clean win | blend Set2>Set1 at α∈[0.6,0.9], symmetric inverts | Confirmed |
| E04-H15 | embedding geometry | anisotropy on the conditioned axes | `D_sel` DR ≥ 1.5x, `V` 0 | DR 0.0370 → 0.1253 = 3.39x at best k=2, `V` 0 | **Confirmed** |
| E04-H16 | coverage temperature | sharpen the soft source assignment | contrast ≥ +20%, `V` 0 | lower `τ` drops contrast 1.47x, `V` 0→2 | Refuted |
| E04-H17 | reranker model | distilled cross-encoder replacement | reranker ≥ 3x, Spearman ≥ 0.95 | recall@3 0.55, Spearman 0.70, 2.6x | Refuted (gate) |
| E04-H18 | pipeline | small cross-encoder cascade pre-filter | `v2-m3` ≥ 2x, recall@m ≥ 0.95 | Spearman 0.98, 2.3x, recall@15 0.87 | Refuted (gate) |
| E04-H19 | composition | stack the landed DR + speed winners | DR up + latency down, guardrails held | no speed lever; DR 3.39x only | Refuted (gate) |
| E05-H20 | reranker model | smaller multilingual cross-encoder | recall@3 ≥ 0.90, Spearman ≥ 0.95 | recorded against a superseded run - see the E05-H20 note | Refuted (disputed) |
| E05-H21 | scorer architecture | late-interaction (ColBERT) MaxSim | recall@3 ≥ 0.80 (beat cosine) | MaxSim 0.47 (cosine 0.39), Spearman 0.08 | Refuted (gate) |
| E05-H22 | scorer architecture | learned-sparse (SPLADE) pre-filter | recall@15 ≥ 0.80 | untrained proxy 0.34 < cosine 0.61 | Killed (proxy) |
| E05-H23 | tokenization | length-bucketed batching | cut ≥ 50%, scores identical | 6.38s→6.28s = 1.7% cut on a 70-pair probe, rho 1.000 | Refuted |
| E05-H24 | source cardinality | cluster source 70 → k medoids | recall ≥ 0.95 at ≥ 50% cut | k=45 36% recall 0.94; 50%-cut breaks fidelity | Refuted |
| E06-H25 | tokenization | length-bucketing, reserved CPU candidate | CPU cut ≥ 40% at identical scores | 42.5% CPU cut (426.6s→245.2s over 6860 pairs), score ρ 0.99996, D_grd ρ 1.0, intrusions unchanged 6→6 | Ships (reserved) |
| E06-H26 | scorer architecture | trained multilingual ColBERT (MaxSim) | recall@3 ≥ 0.90 / recall@m ≥ 0.95 | recall@3 0.4456 **below** the untrained proxy 0.47, ρ 0.309, m@0.95 = 39 | Killed at gate |
| E06-H27 | scorer architecture | trained multilingual learned-sparse (bge-m3) | recall@m ≥ 0.95 at m<35 AND ρ ≥ 0.95 | recall@3 0.5612, ρ +0.770, m@0.95 = 33 (m bar met, ρ bar missed) | Refuted |
| E12-H59 | verdict calibration | structural threshold cross-article sweep | 0.90 mis-sides wergeland rung-2 | 0/29 thresholds safe on both articles, band empty by 0.0019 | Confirmed |
| E12-H60 | verdict calibration | no-counterpart cost valley histogram | 3-10% matched falsely cross 0.6 | 0/61 matched statements above 0.6, corpus max 0.5774 | Killed-at-gate |
| E12-H61 | reporting-diff | moved = displacement AND inversion | 0/6 false moved on insertion | false moved 4/6 → 0/6, 0/13675 true moves lost | Promoted |
| E12-H62 | order axis | OPW−SMD residual under k deletions | gap > 2x paraphrase p95 by k=2 | median gap 0.134 at k=2, 13.4x the bar, 280/280 above | Confirmed |
| E12-H63 | verdict calibration | 0.77 band LOO + INT8-drift fragility | LOO edge moves ≥ 1 pt | LOO edges +2.30 / −1.64 pts, INT8-fp drift 40.6% of a margin | Confirmed |
| E12-H64 | embedding geometry | all-but-the-top: centering vs projection | centering keeps ≥ 60% of gain | centering 103.7% of the gain, projection −4.5% | Confirmed |
| E12-H65 | embedding geometry | CMR pool-size crossover | d' crosses baseline at pool 4-6 | no crossover, d' 5.02 → 7.16 monotone, min pool 8 docs | Confirmed |
| E12-H66 | weights | pre-L2 pooled-norm transport mass | d' +0.2-0.8, margin held | d' −0.011, margin +0.012 pts, null | Refuted (null) |
| E12-H67 | runtime | per-document embed memo in --pairs | 110 → 11 encodes, byte-identical | 110 → 11 encodes, 9.94x, byte-identical | Promoted |
| E12-H68 | runtime | (doc, source) grounding-grid memo | grids 110 → 11, ≥ 5x sweep | 110 → 11 grids, 10.03x, byte-identical | Promoted |
| E12-H69 | grounding cascade | relevance gate: mechanism vs soft form | LEAN - form incidental | Set 2 max-r 4/4 above 0.6 beside gold 74/77, no separating gate | Killed-at-gate |

E02 is a separate goal from E01 - not widening the symmetric gap but splitting the distance into selection and grounding axes when both documents share a source. The source-conditioned distance separates the two adversarial failure modes the symmetric scalar conflates; the selection axis is clean and the grounding axis works at the tier level once aggregated.

**Baseline performance** (notebook 04, SMD against the reference gold)

| measure | value |
|---|---|
| mean gold → ref | 0.236 |
| mean adversarial → ref | 0.397 |
| reference contrast ratio `R` | 1.68x |
| all-pairs contrast (gold-adv / gold-gold) | 1.15x |
| boundary margin `M` | +4.58 closeness pts |
| dynamic range `DR` (std of → ref) | 0.0854 |
| separation `d'` | 4.60 |
| ordinality violations `V` | 0 / 24 |
| gold closeness band | 80.3-85.9% |
| adversarial closeness band | 68.5-75.7% |

## Methodology and metrics

Each lever rebuilds the distance to the reference gold over all eleven summaries, recomputes the metrics, and compares to the baseline row.

- **Boundary margin** - `min(gold closeness) − max(adversarial closeness)`, closeness points on the 0-1 scale; comparable across methods, baseline `+4.58`, must stay `> 0`
- **Contrast ratio `R`** - `mean(adv → ref) / mean(gold → ref)`, scale-free, baseline `1.68x`
- **Dynamic range `DR`** - std of the ten `→ ref` distances, resolution proxy, baseline `0.0854`
- **Separation `d'`** - `(mean adv − mean gold) / pooled std`, scale-free effect size, baseline `4.60`
- **Ordinality violations `V`** - count of (gold, adversarial) pairs ranked wrong, hard guardrail, must stay `0 / 24`
- **Metric guardrail** - a hypothesis claiming to be a metric must keep the triangle inequality; non-metric variants are flagged as discriminative scores

## KPIs - what we optimize and why

The metrics above, grouped by the decision each one drives, so every lever is judged against the KPI it targets and never at the cost of a guardrail. Four families - correctness (must hold), resolution (the axis with room), cost (deployment) and generalization (trust).

**Correctness guardrails** - hard constraints, never traded for a gain on another KPI

- **Ordinality violations `V`** - count of (gold, adversarial) pairs ranked wrong, current `0/24`; a distance that mis-ranks a faithful document against a degraded one is wrong at the most basic level, so this stays `0`
- **Gold intrusions** - faithful golds landing in the adversarial grounding band, recorded `2 → 0` at the E03-H11 gate and re-derived at E12-H69 on the current INT8 models as `6` with the shipped soft gate (severity inverted, Set1>Set2) against `0` ungated (Set2>Set1), so the H11 headline does not reproduce; the per-document grounding-quality measure, a faithful document scored as fabrication is a false alarm
- **Severity order** - whether Set 2 (fabrication) sits above Set 1 (info-loss), the correct grounding severity; the conditioned distance exists to name which failure mode diverged, so an inverted severity breaks its central claim
- **Metric property** - the triangle inequality on the selection axis; clustering, retrieval and fixed thresholds rely on it, so a non-metric variant is a discriminative score, not a distance

**Resolution - the axis with room** - E01 showed ordering is near-ceiling, so dynamic range is where the gains are; the E04 dynamic-range levers target this family

- **Dynamic range `DR`** - std of the `→ ref` distances, symmetric baseline `0.0854`, conditioned `D_sel` ~`0.037`; bunched distances cannot be thresholded or finely ranked, so wider spread makes the number actionable
- **Separation `d'`** - `(mean adv − mean gold) / pooled std`, baseline `4.60`; the scale-free gold-vs-adversarial effect size, resolution normalized
- **Boundary margin `M`** - `min(gold) − max(adversarial)` in closeness points, baseline `+4.58`; the safety gap, a thin margin is a fragile threshold (the E03-H11 v2 margin is thin at `0.216` vs the `0.220` floor)
- **Contrast ratio `R`** - `mean(adv) / mean(gold)`, baseline `1.68x`; scale-free gold → adversarial spread

**Cost - deployment** - decides whether each axis is always-on or on-demand; the E04 speed levers target this family

- **Latency** - symmetric `0.08` ms/pair, source-conditioned ~`109` s/pair on CPU INT8 with the reranker full grid at ~`60%`, ~`63x` faster on GPU fp16; the grounding cost decides whether `D_grd` runs on every pair or only on demand
- **Footprint** - the grounding cross-encoders are ~`1-2` GB FP and ~`300-570` MB INT8 each; `D_sel` alone (mmBERT) is the cheap, CPU / edge-deployable half
- **Batch throughput** (added at E12) - wall time per `--pairs` row at a stated duplicate-document rate, beside the per-pair latency; byte-identity of the JSONL stream against the uncached run is recorded with it - the acceptance currency for the runtime memoisation levers; measured at E12 (in-notebook, not yet shipped): 55-row `distance-semantic --pairs` `11.032 s → 1.110 s` (`9.94x` at a 90% duplicate-document rate, `20.2` ms/row memo-on, 9/9 arms byte-identical, zero-hit overhead `−0.63%`, E12-H67); 55-pair wrt-source sweep `2671.72 s → 266.32 s` (`10.03x` at 90% repeated (doc, source) slots, 55/55 payloads byte-identical, zero-hit overhead `+0.22%`, E12-H68)

**Reporting fidelity** (added at E12) - the annotation layer (`changed`, `moved`, `no_counterpart`) is scored separately from the distance layer: false-flag rate and true-flag retention on labelled fixtures, at bit-identical default-path `smd` / `order_gap` - the acceptance currency for reporting-diff and verdict-calibration levers; measured at E12: `moved` false-flag rate on the insertion fixture `4/6` shipped → `0/6` conjunction at `13675/13675` true flags retained (E12-H61, not yet shipped); `no_counterpart` false-flag rate `0/61` matched at 0.6 and `0/579` on the ibm pool, flag inert (E12-H60); structural verdict false-flag under order-preserving deletions - `structure_closeness` `0.8902` pooled at k=4 and `0.8219` tail k=3, below 0.90 (E12-H62)

**Generalization - trust** - the standing gate, parked for a separate fixture

- **Cross-fixture replication** - every number here is one article and one degradation design, so a KPI that holds only on this fixture is not yet trustworthy; the severity win must replicate on a second source before the blend is trusted as a single scalar (scored later, not in E04)

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
- **Result** - `V = 1`, margin `−0.23`, `d'` 2.47, `R` 1.292; numeric fallback E01-H2 same failure (`V = 1`, margin `−0.16`, `R` 1.309)
- **Verdict** - Refuted; up-weighting numbers pulls the number-retaining adv2 tier into the gold band and breaks ordering, fallback confirms the mechanism not IDF noise

### E01-H2 Numeric-density fallback weighting

Run as E01-H1's fallback variant and recorded in its Result line and the E01 tables; declared here so the ledger parses the id - no new content.

- **Hypothesis** - because numeric-density weighting alone removes the IDF noise from E01-H1's salience weights, the fallback will lift the boundary margin while holding `V = 0`
- **Lever** - transport weights (numeric-density only, no IDF)
- **Result** - `V = 1`, margin `−0.16`, `R` 1.309 (E01 results table)
- **Verdict** - Refuted; same failure mode as E01-H1, confirming the mechanism (number up-weighting) not IDF noise

### E01-H3 Embedding anisotropy removal

- **Hypothesis** - because mmBERT embeddings are anisotropic and statement cosines are compressed, subtracting the dominant principal component will raise distance dynamic range ≥ 1.5x while preserving `0/24` ordinality violations
- **Lever** - embedding geometry
- **Mechanism** - mean-center the statement embeddings, subtract the top 1-3 principal components (all-but-the-top), re-L2-normalize, then cost
- **Prediction** - cost matrix de-compresses, `DR ≥ 1.5x`, margin widens
- **Acceptance bar** - `DR ≥ 1.5x` baseline and `V = 0`, swept `k ∈ {1,2,3}`
- **Result** (k=1) - `DR` 0.2286 = 2.7x baseline, margin `+18.48` (up from +4.58), `V = 0`, `d'` **rises** to 7.17, `R` falls 1.680 → 1.482, latency ~1.4x
- **Verdict** - Promoted; clears the `DR ≥ 1.5x` bar and widens the margin at `V = 0`, the only lever to do so; caveat - it spreads the gold band too, so `d'` drops (resolution, not a sharper boundary)

### E01-H4 Sharpened ground cost - angular distance

- **Hypothesis** - because `arccos` expands the high-cosine region where statements bunch, the angular ground cost will widen the boundary margin while keeping the metric property and `0/24` violations
- **Lever** - ground cost function
- **Mechanism** - replace `√(2 − 2cos)` with angular distance `arccos(cos) / π`, a true metric that expands the high-cosine region
- **Prediction** - more spread among near-duplicate statements, margin up, metric preserved
- **Acceptance bar** - margin up, `V = 0`, triangle inequality intact
- **Result** - `V = 0`, `d'` 4.58 (baseline 4.60), margin `+2.08` (down from +4.58), `R` 1.687, `DR` 0.0276, metric kept
- **Verdict** - Refuted (null); `arccos` is near-affine to `√(2 − 2cos)` at these cosines, so the ranking barely moves - a valid metric, a no-op for separation

### E01-H5 Unbalanced / partial transport residual

- **Hypothesis** - because balanced OT force-matches every statement and hides omissions and additions, unbalanced OT with a folded residual will widen the margin most (`≥ +0.040`) while holding `0/24` violations
- **Lever** - OT formulation
- **Mechanism** - unbalanced OT (marginal-relaxation `reg_m`) with the unmatched residual folded into the score (`+ √2 · residual`)
- **Prediction** - residual loads the adversarial tiers hardest, the widest margin of the five (`≥ +0.040`)
- **Acceptance bar** - margin up, `V = 0`, sweep `reg_m`
- **Result** (reg_m=0.5, the committed run) - `V = 0`, margin `+8.57` (**above** baseline +4.58), `d'` 5.19, `R` 1.564, `DR` 0.1363, non-metric, 7.64 ms/pair (~109x exact SMD). The earlier `reg_m=2.0` figures this section carried appear in no committed artefact
- **Verdict** - Refuted; the boldest prediction lands worse than baseline, drops the metric property, costs two orders of magnitude more latency; both tiers share enough content that the residual does not concentrate

### E01-H6 Tail-aware aggregation

- **Hypothesis** - because a few badly-matched statements are averaged away by the mean, a p90 tail of matched cost will sharpen tier separation while holding `0/24` violations
- **Lever** - aggregation of the transport plan
- **Mechanism** - report the cost-weighted p90 of matched cost instead of the mean, surfacing the few badly-matched statements the mean averages away
- **Prediction** - the tail separates the tiers more sharply
- **Acceptance bar** - margin up, `V = 0`
- **Result** - `V = 2`, margin `−2.30`, `d'` 2.43 (baseline 4.60), the worst performer
- **Verdict** - Refuted; at ~12 statements the p90 tail is dominated by one or two noisy alignments and reorders the tiers - the mean is the right aggregator

### Results table (E01)

| hypothesis | V | margin (clos pts) | d' | R | DR | metric | ms/pair | verdict |
|---|---|---|---|---|---|---|---|---|
| baseline (SMD mean) | 0/24 | +4.58 | 4.60 | 1.680 | 0.0854 | yes | 0.07 | reference |
| E01-H1 salience (IDF×num) | 0/24 | +4.14 | 4.51 | 1.661 | 0.0842 | yes | 0.07 | Refuted |
| E01-H2 numeric proxy | 0/24 | +4.58 | 4.66 | 1.690 | 0.0860 | yes | 0.07 | Refuted |
| E01-H3 anisotropy (k=1) | 0/24 | +18.48 | 7.17 | 1.482 | 0.2286 | yes | 0.10 | **Promoted** |
| E01-H4 angular cost | 0/24 | +2.08 | 4.58 | 1.687 | 0.0276 | yes | 0.06 | Refuted (null) |
| E01-H5 unbalanced (reg_m 0.5) | 0/24 | +8.57 | 5.19 | 1.564 | 0.1363 | no | 7.64 | Refuted |
| E01-H6 tail (p90) | 2/24 | −2.30 | 2.43 | 1.337 | 0.0772 | no | 0.05 | Refuted |

### Benchmarks (E01)

Latency per document-pair, exact baseline SMD = 1x, over the ten non-reference summaries. The transport solve is `ot.emd2` on numpy and runs on the CPU; the RTX 5000 Ada named elsewhere in this log covers the embedding pass only.

- **baseline exact SMD** - 0.08 ms/pair, the reference
- **E01-H1 / E01-H2 weighting** - 0.08 ms/pair, same solver, weights are free
- **E01-H4 angular** - 0.10 ms/pair, only the cost matrix changes
- **E01-H3 anisotropy** - 0.17 ms/pair (~2x), one extra SVD on the pooled embeddings
- **E01-H6 tail** - 0.06 ms/pair, cheapest, reuses the balanced plan
- **E01-H5 unbalanced** - 9.4 ms/pair (~120x), the majorization-minimization solver dominates

## E02 - experiment batch 2: source-conditioned grounding axis

A different question from E01 - not widening the symmetric gap, but splitting the distance into two axes when both documents derive from one source `S`. Tests the design in [`../solution/wmd-source-conditioned-docdistance-solution-sota.md`](../solution/wmd-source-conditioned-docdistance-solution-sota.md), executed in [`notebooks/experiments/E02-kj-source-conditioned-grounding.ipynb`](../../notebooks/experiments/E02-kj-source-conditioned-grounding.ipynb). Two axes - selection `D_sel` (coverage-profile OT over `S`, already shipped) and grounding `D_grd` (reranker × NLI residual, the deferred build). Pipeline adds `bge-reranker-v2-m3` and `mdeberta-mnli-xnli`, both OpenVINO INT8 on CPU.

- **Anchor** - `gold` (Opus 3-sweep); every summary scored `d(anchor, X | S)` on both axes plus the symmetric SMD baseline
- **Two rounds of grounding** - R1 single-premise (SummaC max over source), R2 top-k joint premise (`k = 3`, the design's aggregation)
- **Source** - 70 statements; the grounding sweep scores every (summary statement × source statement) pair through both cross-encoders

### E02-H7 Selection axis separates information loss

- **Hypothesis** - because info-loss strips source figures, `D_sel` will rank Set 1 above gold while gold stays clustered (selection axis carries omission)
- **Result** - D_sel gold 0.023, Set 1 0.060 (2.6x), Set 2 0.073; every adversarial above every gold, `0` ordinality violations on the selection axis
- **Verdict** - Confirmed; D_sel cleanly separates both adversarial tiers from gold, info-loss included

### E02-H8 Grounding axis isolates fabrication (aggregation required)

- **Hypothesis** - because info-noise fabricates unsupported claims, `D_grd` will rank Set 2 above gold and above Set 1, once grounding is aggregated over evidence (R2) not single-premise (R1)
- **Result R1** (single-premise) - gold 0.097, Set 1 0.206, Set 2 0.233; faithful info-loss almost level with fabrication, axis muddied
- **Result R2** (joint premise) - gold 0.120, Set 1 0.130, Set 2 0.232; aggregation drops info-loss to gold level while holding fabrication (1.8x over gold), but two gold summaries (haiku 0.230, v2 0.285) intrude into Set 2's range
- **Verdict** - Partially confirmed; R2 isolates Set 2 at the tier mean and beats R1, but the axis is a tier-level fabrication flag, not a clean per-document discriminator; contradiction mass is near-zero across tiers (numeric-entailment weakness), so the residual rides on the ungrounded component

### E02-H9 Two axes separate what the symmetric scalar conflates

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

Five levers to turn the E02 grounding axis from a tier-level flag into a per-document metric and to cut its cost, all over the same `data/interim/exec-summaries/ibm-ai-adoption` fixture, executed in [`notebooks/experiments/E03-kj-source-conditioned-improvements.ipynb`](../../notebooks/experiments/E03-kj-source-conditioned-improvements.ipynb). Three target quality (the grounding weaknesses E02 exposed), two target performance (the reranker bottleneck). Two confirmed (H11, H14), three refuted (H10, H12, H13).

- **Headline** - the relevance-gate (H11) closes the per-document gold intrusion (2 → 0) and the blend (H14) then orders the two failure modes correctly (Set 2 fabrication above Set 1 info-loss) where the symmetric scalar inverts them; number-aware verification (H10) and both reranker-cost levers (H12, H13) do not survive this fixture
- **Load-bearing reranker** - neither a bi-encoder cosine pre-filter (H12, recall@10 of top-3 only 0.58) nor a cosine replacement (H13, relevance Spearman 0.40) preserves the grounding ranking; the 60% reranker cost is structural to this design

- **Aim** - improve the quality and performance of the source-conditioned distance `d(A,B|S)`
- **Quality targets** - the four E02 grounding weaknesses: gold intrusion (2 golds in Set 2's band), dead contradiction signal, per-document noise, number blindness
- **Performance target** - the 109 s/pair cost, dominated by the reranker full grid (66 s, 60.5%)
- **Pairing** - E03-H12 (conservative cascade) and E03-H13 (aggressive replacement) attack the same reranker cost; the H13 kill-gate routes to H12 on failure
- **Capstone** - E03-H14 composes the winning quality levers and tests the batch gate head-to-head against the symmetric SMD

### E03 gate - clean win over the symmetric distance

The whole batch is judged against one acceptance gate, fixed before any run: the improved conditioned distance must beat the symmetric SMD on common-source documents, the case where the symmetric scalar already fails.

- **Win condition** - `0` per-document ordinality violations separating gold from each adversarial tier on the conditioned axes AND linear separation of Set 1 from Set 2 on the `(D_sel, D_grd)` plane or a blended scalar `α·D_sel + (1−α)·D_grd`
- **Reference to beat** - symmetric SMD conflates and mis-orders by severity (Set 1 0.452 ≈ Set 2 0.406, gold 0.287)
- **Not-shipped clause** - a lever that improves an axis but does not move the batch toward this gate is recorded interesting-but-not-shipped

### E03-H10 Numeric-aware grounding verifier

- **Hypothesis** - because general NLI contradiction is ≈ 0 on quantitative claims yet both adversarial tiers are defined by numeric corruption (Set 1 strips figures, Set 2 fabricates or alters them), a numeric verifier that extracts figures from each summary statement and compares them to the reranker-aligned source figures will rank Set 2 ≥ 2x gold while gold's numeric residual stays ≤ 0.05, adding a signal NLI does not carry
- **Lever** - grounding residual composition (add a numeric-mismatch term to `D_grd`)
- **Mechanism** - number-entity extraction (percent, count, currency, year or forecast) per statement; match each summary figure to its source figure via the top-k reranked source; residual = share of summary figures unmatched (fabricated) or value-mismatched (wrong number); orthogonal to NLI entailment
- **Prediction** - Set 2 numeric residual ≥ 2x gold; gold ≤ 0.05; Set 1 low here, its loss shows on `D_sel`
- **Acceptance bar** - Set 2 ≥ 2x gold AND gold ≤ 0.05 AND fires where the NLI contradiction signal was dead
- **Kill-gate** - numeric density: figures must appear in ≥ 30% of statements (probe the fixture first); sparse → kill before any build
- **Result** - density 58% (gate passed); top-k aligned numeric residual Set 2 0.163 = 2.0x gold 0.080, gold 0.080 > the 0.05 bar (faithful golds sonnet 0.31, haiku 0.14 inflate - real figures whose aligned source is not the top-k); whole-source matching keeps gold low (0.018) but collapses Set 2 to 0.051 because the 82-figure source matches fabricated numbers by coincidence; NLI contradiction confirmed dead (gold 0.055, Set 2 0.007)
- **Verdict** - Refuted; the signal is orthogonal to the dead contradiction and the direction is right (Set 2 2x gold), but localized matching breaks the gold ≤ 0.05 bar and whole-source matching is defeated by the source's figure density - number-aware verification fails this fixture both ways

> [!WARNING]
> **E03-H11's headline is contradicted by the later batches.** This section records
> `intrusions 2 -> 0` and `Set 2 > Set 1`. The shipped chain as measured in the later runs reads
> **6 gold intrusions** and severity **`Set1>Set2`** - `reports/E06-trained-scorers-cpu-metrics.json`
> (`shipped_intrusions: 6`, `shipped_severity: "Set1>Set2"`) and
> `reports/E04-source-conditioned-performance-metrics.json` agree. Every E04/E05/E06 gate that
> names "0 gold intrusions AND Set 2 above Set 1" as the reference to hold is therefore scored
> against a reference that is itself failing. The Confirmed verdict below is left as recorded and
> is **not** re-adjudicated: resolving it needs `D_grd` re-derived on the current embeddings.

### E03-H11 Relevance-gated ungrounded residual

- **Hypothesis** - because a faithful compressive gold fuses several source sentences (low joint entailment but high max reranker relevance) while fabrication is genuinely novel (low entailment AND low relevance), gating the ungrounded mass by max reranker relevance will drop the two intruding golds (haiku 0.230, v2 0.285) below Set 2's band while holding Set 2 within 10% of R2
- **Lever** - residual definition (weight ungrounded mass by `1 − max_k r(a_i, s_k)`)
- **Mechanism** - only statements with low max source relevance count as ungrounded; high-relevance-but-low-entailment (compression) no longer inflates the residual
- **Prediction** - gold intrusion 2 → 0; gold tier mean below 0.13; Set 2 held ≈ 0.21-0.23
- **Acceptance bar** - per-document ordinality gold < Set 2 restored (0 intrusions) AND Set 2 within 10% of R2
- **Kill-gate** - the intruding golds must actually have high max-relevance (probe: their max `r` ≥ 0.6); if not, the intrusion is real divergence the gate cannot fix
- **Result** - kill-gate passed (intruding golds v2 mean max `r` 0.898, haiku 0.752); gating drops both below the Set 2 floor (haiku 0.230 → 0.053, v2 0.285 → 0.216), gold intrusions 2 → 0, gold tier mean 0.120 → 0.084, Set 2 0.232 → 0.236 (+2%, held within 10%); the v2 margin is thin (0.216 vs the 0.220 floor)
- **Verdict** - Confirmed; the one quality lever that lands - the per-document gold intrusion is closed while Set 2 holds, though the v2 margin is narrow

### E03-H12 Bi-encoder cascade pre-filter

- **Hypothesis** - because the reranker scores the full 12 × 70 grid (66 s, 60.5%) yet only the top-k per statement enters the premise, pre-selecting the top-m source per statement by the already-computed mmBERT cosine (m ≈ 10) will cut reranker calls ~7x and end-to-end latency ≥ 40% while preserving the `D_grd` ranking (Spearman ≥ 0.95 vs full grid)
- **Lever** - pipeline (bi-encoder shortlist before the cross-encoder)
- **Mechanism** - mmBERT cosine ranks the 70 source statements per summary statement; keep the top-m, run the reranker only on those, feed the reranker top-k into the premise as before
- **Prediction** - reranker 66 s → ≤ 10 s; end-to-end 109 s → ≤ 60 s; tier means within 5%; Spearman ≥ 0.95
- **Acceptance bar** - Spearman(`D_grd` vs full grid) ≥ 0.95 AND latency cut ≥ 40% AND tier separation preserved
- **Kill-gate** - bi-encoder top-m must contain the reranker top-k (recall@m ≥ 0.95 on a probe); else the shortlist drops the true evidence
- **Result** - kill-gate failed: recall@10 of the reranker top-3 is 0.576, so the cosine shortlist drops 42% of the true evidence; D_grd Spearman vs the full grid is 0.545 (bar 0.95), even though the latency cut is 82% (reranker 63.9 s → 11.7 s), better than predicted
- **Verdict** - Refuted (killed at gate); a fast shortlist that changes the answer - the bi-encoder cosine is a poor pre-filter for this cross-encoder

### E03-H13 Bi-encoder relevance replaces cross-encoder

- **Hypothesis** - because the mmBERT bi-encoder cosine may already rank source relevance closely enough, replacing the cross-encoder relevance term `r(a_i, s_k)` with the bi-encoder cosine removes the 66 s reranker stage entirely (→ 0, reuse the embeddings), cutting end-to-end to ≤ 45 s, while keeping the grounding verdict if the two relevance rankings agree
- **Lever** - scorer (drop the cross-encoder, use bi-encoder cosine as relevance)
- **Mechanism** - the grounding score becomes `g = cos(a_i, s_k) · P(entail)`; no separate reranker sweep, the selection-axis embeddings are reused
- **Prediction** - end-to-end 109 s → ≤ 45 s; Set 2 still isolated; no new gold intrusion vs the cross-encoder design
- **Acceptance bar** - end-to-end ≤ 45 s AND Set 2 isolated at tier mean AND no new intrusion
- **Kill-gate** - probe bi-encoder vs cross-encoder relevance Spearman on a sample; ≥ 0.70 to proceed; below → the cross-encoder carries irreplaceable compression / paraphrase signal → fall back to the H12 cascade
- **Result** - kill-gate failed: bi-vs-cross relevance Spearman 0.397 (bar 0.70); dropping the reranker collapses the chain to 0.9 s but corrupts grounding - gold intrusions rise 2 → 5, and Set 1 falls below gold at the tier mean (gold 0.120, Set 1 0.105, Set 2 0.208)
- **Verdict** - Refuted (killed at gate); the strongest result of the cost pair - the cross-encoder relevance is irreplaceable by the bi-encoder cosine, the H12 fallback also failed

### E03-H14 Blended conditioned scalar vs symmetric

- **Hypothesis** - because the symmetric SMD conflates and mis-orders the two failure modes (Set 1 0.452 ≈ Set 2 0.406) while the conditioned axes separate them, composing the winning quality levers (H10 numeric + H11 relevance-gate on `D_grd`, with `D_sel`) into a single blended scalar `α·D_sel + (1−α)·D_grd` (α swept) will order all tiers with 0 per-document violations - a result the symmetric distance cannot reach
- **Lever** - output (blend the two axes, benchmark head-to-head vs symmetric SMD)
- **Mechanism** - sweep α over the conditioned axes after H10 and H11 land, pick the operating point that orders the tiers, compare per-document ranking to the symmetric SMD
- **Prediction** - at some α the blend gives gold < Set 1, Set 2 at 0 violations with Set 1 / Set 2 linearly separable; symmetric SMD stays mis-ordered (Set 1 > Set 2)
- **Acceptance bar** - the batch gate: 0 per-document violations gold vs each tier AND Set 1 / Set 2 separated, where symmetric cannot → clean win
- **Kill-gate** - conditional on H10 or H11 landing; if neither quality lever clears its bar, `D_grd` stays a tier flag and the blend cannot beat symmetric per document → record "gate not met, conditioned remains tier-level"
- **Result** - H11 landed (H10 did not), so the improved grounding is the H11 relevance-gated ungrounded mass; min-max each axis, blend `α·D_sel + (1−α)·D_grd`; at `α ∈ [0.60, 0.90]` the blend gives 0/28 per-document violations AND Set 2 above Set 1 (correct grounding severity); the symmetric SMD reaches 0/28 violations too but inverts the severity (Set 1 0.452 > Set 2 0.406)
- **Verdict** - Confirmed (clean win on severity ordering); the blend orders the two failure modes correctly where the symmetric scalar inverts them, but the win is the ordering and the axis attribution, not raw gold-vs-adversarial ordinality - the symmetric scalar also clears 0/28 on this single fixture, so cross-fixture validation is required before generalizing

### Pre-registration table (E03)

| hypothesis | lever | prediction | acceptance bar | kill-gate |
|---|---|---|---|---|
| E03-H10 numeric verifier | residual composition | Set 2 ≥ 2x gold, gold ≤ 0.05 | Set 2 ≥ 2x gold AND gold ≤ 0.05 AND beats dead NLI | numeric density ≥ 30% of statements |
| E03-H11 relevance-gate | residual definition | gold intrusion 2 → 0, Set 2 held | 0 intrusions AND Set 2 within 10% of R2 | intruding golds max `r` ≥ 0.6 |
| E03-H12 cascade pre-filter | pipeline | reranker → ≤ 10 s, Spearman ≥ 0.95 | Spearman ≥ 0.95 AND latency cut ≥ 40% | top-m recall of top-k ≥ 0.95 |
| E03-H13 bi-encoder relevance | scorer | end-to-end → ≤ 45 s, Set 2 isolated | ≤ 45 s AND Set 2 isolated AND no new intrusion | bi vs cross relevance Spearman ≥ 0.70 |
| E03-H14 blended vs symmetric | output | 0 violations, Set 1 / Set 2 split | the batch gate (clean win) | H10 or H11 must clear their bar |

### Results table (E03)

Tier means and the pre-registered measure for each lever; gold is the anchor tier, Set 1 info-loss, Set 2 info-noise.

| hypothesis | gold | Set 1 | Set 2 | measure | bar | verdict |
|---|---|---|---|---|---|---|
| E03-H10 numeric (top-k aligned) | 0.080 | 0.000 | 0.163 | Set 2 2.0x gold, gold > 0.05 | Set 2 ≥ 2x gold AND gold ≤ 0.05 | Refuted |
| E03-H11 D_grd relevance-gated | 0.084 | 0.141 | 0.236 | intrusions 2 → 0, Set 2 +2% | 0 intrusions AND Set 2 ±10% | Confirmed |
| E03-H12 D_grd cascade | 0.080 | 0.195 | 0.156 | Spearman 0.545, recall@m 0.58 | Spearman ≥ 0.95 | Refuted |
| E03-H13 D_grd bi-encoder | 0.120 | 0.105 | 0.208 | Spearman 0.40, intrusions 2 → 5 | Spearman ≥ 0.70, no new intrusion | Refuted |
| E03-H14 blend `α∈[0.6,0.9]` | - | - | - | blend 0/28 Set2>Set1; symmetric 0/28 Set1>Set2 | 0 violations AND correct severity | Confirmed |

### Benchmarks (E03)

CPU INT8, the 11-document reranker grid plus the per-lever micro-benchmarks; the heavy stage is unchanged from E02.

- **per-document signal build** - ~48 s/document (the reranker full grid over `n_summary × 70` pairs), 534 s for all 11 documents; the dominant cost, as in E02
- **H12 cascade reranker** - full grid 910 pairs 63.9 s → cosine top-10 shortlist 130 pairs 11.7 s, an 82% latency cut, but the ranking shifts (Spearman 0.545) so the cut does not ship
- **H13 reranker-free chain** - 0.9 s end-to-end (embed + cosine + joint-premise NLI, no reranker), ~120x faster than the 109 s E02 chain, but grounding is corrupted (5 gold intrusions)
- **H11 / H14 post-processing** - sub-ms on the already-computed signals; the relevance-gate and the blend add no measurable cost over the R2 grounding axis
- **GPU vs CPU INT8 (grounding chain)** - the reranker full grid runs ~63x faster on the RTX 5000 Ada fp16 (107.5 s CPU INT8 → 1.72 s), and the whole 11-document signal build drops from 534 s to ~16 s; the tier verdicts (`D_sel` 0 violations, `D_grd` 0 gold intrusions, blend Set2>Set1) are identical on both devices despite different absolute fp16-vs-INT8 values (nb05)
- **Reading** - the only cheap, faithful win is the relevance-gate (free re-weighting of existing signals); the two levers that actually remove the reranker cost both break the grounding ranking

## E04 - experiment batch 4: source-conditioned performance

Five levers to make the shipped source-conditioned distance faster and higher-resolution without losing the correctness E02/E03 secured, all over the same `data/interim/exec-summaries/ibm-ai-adoption` fixture, executed on the GPU fp16 chain in [`notebooks/experiments/E04-kj-source-conditioned-performance.ipynb`](../../notebooks/experiments/E04-kj-source-conditioned-performance.ipynb). Two target resolution (the dynamic-range axis E01 left as the one with room, never applied to the conditioned distance), two target the reranker cost (the structural 60% the E03 bi-encoder levers could not remove), one composite capstone stacks the winners. One confirmed (H15), four refuted.

- **Headline** - anisotropy removal on the conditioned `D_sel` (H15) widens dynamic range 3.39x at `0` violations, the one win; coverage temperature (H16) trades contrast for range the wrong way, a distilled reranker (H17) drops half the evidence, a cross-encoder cascade (H18) is the near-miss that clears fidelity and speed but fails the recall gate, and with no faithful speed lever the composite (H19) gains resolution only
- **Aim** - improve the conditioned distance on the resolution and cost KPIs without regressing a correctness guardrail
- **Resolution levers (H15, H16)** - act on the embedding axes (`D_sel` and the geometric residual), where anisotropy and the coverage temperature live; the reranker `D_grd` is text-based and untouched by them
- **Speed levers (H17, H18)** - act only on the reranker, the load-bearing cross-encoder; both keep a cross-encoder, since E03 proved the bi-encoder cosine neither shortlists nor replaces it
- **Pairing** - H17 (aggressive replacement) and H18 (conservative cascade) attack the same reranker cost; the H17 kill-gate routes to H18 on failure, as E03-H13 routed to H17
- **Capstone** - E04-H19 composes the landed resolution and speed winners and re-checks every KPI head-to-head against the E03 operating point and the symmetric SMD

### E04 gate - faster and sharper at no correctness cost

The batch is judged against one acceptance gate, fixed before any run: a lever ships only if it improves its target KPI while holding every correctness guardrail.

- **Win condition** - the shipped conditioned design ends with higher dynamic range and/or lower latency AND holds `0` `D_sel` ordinality violations, `0` gold intrusions on `D_grd`, and Set 2 above Set 1 severity
- **Reference to hold** - the E03 operating point (`D_sel` `0/24`, H16-gated `D_grd` `0` intrusions, blend Set2>Set1 at `α ≈ 0.75`)
- **Not-shipped clause** - a lever that moves its KPI but breaks a guardrail is recorded interesting-but-not-shipped
- **Out of scope** - cross-fixture generalization is parked for a separate second-source fixture; E04 is scored on the existing fixture only

### E04-H15 Anisotropy removal on the conditioned axes

- **Hypothesis** - because `D_sel` and the geometric residual ride the same anisotropic mmBERT cosines that compressed the symmetric distance (E01-H3 widened DR 0.0854 → 0.2286, 2.7x), projecting out the shared direction over the pooled {A, B, S} statements before the coverage profile will widen the conditioned `D_sel` dynamic range ≥ 1.5x while holding `0` selection violations - and here the common direction is estimated from a real corpus (the 70-statement source pooled with A and B), the recommended use, not the single-pair case the library docstring warns against
- **Lever** - embedding geometry on the conditioned axes (`compute_source_conditioned(anisotropy=True)`, `all_but_the_top` already in `src/docdistance/distance.py`, never run on the conditioned distance)
- **Mechanism** - mean-center the pooled {A, B, S} statements, subtract the top-`k` principal components, re-L2-normalize, then build `coverage_profile` and the residual on the de-bunched cosines; `k` swept {1, 2, 3}
- **Prediction** - `D_sel` DR ≥ 1.5x its raw ~`0.019`, gold → adversarial contrast widens, `0` violations; the reranker `D_grd` is unchanged, anisotropy cannot reach a text-based score
- **Acceptance bar** - DR(`D_sel`) ≥ 1.5x baseline AND `0` selection-axis violations AND `D_grd` gold intrusions unchanged
- **Kill-gate** - the pooled {A, B, S} statement cosines must be anisotropic (mean off-diagonal cosine ≥ 0.6 on a probe); already-spread → removal is a no-op → kill
- **Result** - kill-gate passed (pooled mean off-diagonal cosine 0.736); all-but-the-top widens `D_sel` DR at the sweep's best `k` (`best_k` 2, DR 0.0370 → 0.1253 = 3.39x) at `0` violations; far past the 1.5x bar; contrast slips 4.34 → 1.91, `D_grd` unchanged (text-blind, by construction)
- **Verdict** - Confirmed; clears DR ≥ 1.5x at `0` violations by a wide margin, the conditioned analogue of E01-H3 and a larger gain (3.39x vs 2.7x) because the conditioned coverage profiles start more bunched; caveat - it spreads the within-tier band too, so contrast does not improve (resolution, not a sharper boundary); ship as the default resolution pre-pass on the conditioned path (opt-out via `anisotropy=False`)

### E04-H16 Coverage-temperature sharpening

- **Hypothesis** - because the coverage softmax at `temperature = 0.1` spreads each statement's mass across many source statements and blurs the profile, lowering the temperature concentrates coverage onto the true source statement, widening the `D_sel` dynamic range and the gold → adversarial contrast while holding `0` violations
- **Lever** - the `coverage_profile` temperature (`src/docdistance/distance.py`, default `0.1`)
- **Mechanism** - a sharper `softmax(−cost / τ)` drives each statement's assignment toward one-hot on its nearest source statement, so covered and dropped content separate more in the profile before the selection OT
- **Prediction** - at some `τ < 0.1` the `D_sel` Set/gold contrast rises ≥ 20% and DR ≥ 1.3x its `τ = 0.1` value at `0` violations; too-low `τ` collapses to argmax and loses the soft signal
- **Acceptance bar** - contrast up ≥ 20% AND DR up AND `0` violations, `τ` swept {0.2, 0.1, 0.05, 0.02, 0.01}
- **Kill-gate** - the `τ` sweep must show a rising-then-saturating contrast curve; flat across `τ` (the soft assignment is already near-hard) → kill
- **Composability** - orthogonal to H15 (geometry vs softmax sharpness); both feed the H19 composite
- **Result** - the prediction inverts: lowering `τ` raises DR monotonically (0.0193 at 0.1 → 0.0487 at 0.01) but contrast FALLS (2.31x → 1.47x) and ordinality breaks (`V` 1 at τ=0.02, `V` 2 at τ=0.01); raising τ to 0.2 lifts contrast only to 2.46x (1.07x, below the +20% bar) while DR drops; no `τ` clears contrast ≥ 1.2x with DR up at `0` violations
- **Verdict** - Refuted; the sharper softmax does spread the distances but within-tier, lowering the gold/adversarial contrast and eventually breaking ordering - the opposite direction to the prediction; temperature trades contrast for range, it does not buy both

### E04-H17 Distilled cross-encoder reranker - replacement

- **Hypothesis** - because the cross-encoder is load-bearing (E03 killed both bi-encoder shortcuts) yet the `bge-reranker-v2-m3` full grid is 60% of the 109 s/pair cost, replacing it with a smaller distilled cross-encoder that keeps cross-attention (e.g. `bge-reranker-base` or a MiniLM cross-encoder) will cut the reranker stage ≥ 3x while preserving the `D_grd` ranking (Spearman ≥ 0.95 vs `v2-m3`)
- **Lever** - the reranker model (a smaller cross-encoder, not a bi-encoder - the distinction from the refuted E03-H12/H18)
- **Mechanism** - the same grounding chain with a cheaper cross-encoder backbone scoring the grid; the joint-premise aggregation and the H16 relevance-gate are unchanged, `D_grd` recomputed
- **Prediction** - reranker 66 s → ≤ 22 s CPU INT8, Spearman ≥ 0.95, tier means within 5%, `0` new gold intrusions
- **Acceptance bar** - Spearman(`D_grd` vs `v2-m3`) ≥ 0.95 AND reranker latency cut ≥ 3x AND `0` new gold intrusions
- **Kill-gate** - the small reranker top-3 must recall the `v2-m3` top-3 (recall@3 ≥ 0.90 on a probe); below → it changes the evidence like the cosine did → route to the H18 cascade
- **Result** - kill-gate failed: `bge-reranker-base` top-3 recalls only 0.547 of the `v2-m3` top-3 (bar 0.90); `D_grd` Spearman 0.70 (bar 0.95), reranker cut 2.64x (1.74 s → 0.66 s/document, below the 3x bar), and `1` new gold intrusion; every bar missed
- **Verdict** - Refuted (killed at gate); the distilled same-family reranker drops nearly half the true top-3 evidence and changes the grounding ranking - the cross-encoder is load-bearing even against its own smaller sibling, reinforcing the E03 finding

### E04-H18 Small cross-encoder cascade pre-filter

- **Hypothesis** - because the E03-H12 cascade failed only because the bi-encoder cosine was a poor pre-filter (recall@10 of the top-3 was 0.58), using a small cross-encoder to shortlist the top-`m` source per statement before `v2-m3` reranks only those `m` will cut `v2-m3` calls ≥ 2x while preserving the `D_grd` ranking (Spearman ≥ 0.95), because a cross-encoder pre-filter recalls the evidence the cosine dropped
- **Lever** - pipeline (a small cross-encoder shortlist before the `v2-m3` cross-encoder, replacing the refuted bi-encoder shortlist)
- **Mechanism** - the small cross-encoder scores the `n_summary × 70` grid cheaply and keeps the top-`m` per statement; `v2-m3` reranks only those `m` and its top-3 feed the joint premise as before
- **Prediction** - `v2-m3` calls 70 → `m` ~ 15 per statement, reranker stage ≥ 2x faster, Spearman ≥ 0.95, `0` new intrusions
- **Acceptance bar** - Spearman ≥ 0.95 AND `v2-m3` latency cut ≥ 2x AND `0` new intrusions
- **Kill-gate** - the small cross-encoder top-`m` must recall the `v2-m3` top-3 (recall@`m` ≥ 0.95, the bar the cosine failed at 0.58); below → the cascade drops true evidence → kill
- **Result** - the near-miss: MiniLM top-15 → `v2-m3` clears the fidelity bar (`D_grd` Spearman 0.976 ≥ 0.95) and the speed bar (2.28x, `v2-m3` calls 70 → 15 = 4.67x), but the kill-gate fails (recall@15 of the `v2-m3` top-3 is 0.867, bar 0.95) and the 13% dropped evidence adds `1` gold intrusion
- **Verdict** - Refuted (killed at gate); the cross-encoder cascade is far closer than the E03 bi-encoder (Spearman 0.98 vs 0.55) and would ship on fidelity and speed alone, but the dropped top-3 evidence breaks a correctness guardrail - the most promising open speed direction (a larger `m` or a better-matched pre-filter)

### E04-H19 Composite conditioned distance

- **Hypothesis** - because the resolution levers (H15, H16) act on the embedding geometry and the speed levers (H17, H18) act on the reranker, orthogonal parts of the chain, stacking the landed winners into the shipped conditioned scalar will raise resolution and cut cost at once with no correctness regression on this fixture
- **Lever** - composition (stack the landed levers, benchmark the composite scalar against the E03 operating point and the symmetric SMD)
- **Mechanism** - apply the landed resolution lever(s) to `D_sel` and the residual, run `D_grd` through the landed faster reranker, blend `α·D_sel + (1−α)·D_grd` at the E03 operating point `α ≈ 0.75`, recompute every KPI
- **Prediction** - composite DR ≥ the best single-lever DR, end-to-end latency ≤ the best single speed lever, all correctness guardrails held, the severity win over symmetric preserved
- **Acceptance bar** - net DR up AND end-to-end latency down AND `0` `D_sel` violations AND `0` gold intrusions AND Set 2 above Set 1, all at once, no guardrail regressed
- **Kill-gate** - at least one resolution lever (H15 or H16) AND at least one speed lever (H17 or H18) must clear its bar; if a class is empty the composite reduces to the E03 design and the missing axis is recorded "no E04 gain"
- **Cross-fixture** - parked; the second-source replication is scored later on a separate fixture, not in this batch
- **Result** - kill-gate failed: no speed lever landed (H17 and H18 both refuted), so the composite reduces to the H15 resolution lever on the baseline reranker - `D_sel` DR 3.39x, `0` violations, `0` gold intrusions, blend Set2>Set1 (vs symmetric Set1>Set2), every guardrail held and severity correct, but `latency cut 1.0x`, no speed gain
- **Verdict** - Refuted (killed at gate); the resolution half holds every guardrail and fixes the severity, but with no faithful speed lever the composite gains resolution only, not the combined faster-and-sharper win - the reranker cost stands

### Pre-registration table (E04)

| hypothesis | lever | prediction | acceptance bar | kill-gate |
|---|---|---|---|---|
| E04-H15 anisotropy on conditioned | embedding geometry | `D_sel` DR ≥ 1.5x, `V` 0 | DR ≥ 1.5x AND `V` 0 AND `D_grd` intrusions unchanged | pooled cosine anisotropy ≥ 0.6 |
| E04-H16 coverage temperature | coverage `τ` | contrast ≥ +20%, DR up, `V` 0 | contrast ≥ +20% AND DR up AND `V` 0 | rising-then-saturating contrast curve over `τ` |
| E04-H17 distilled reranker (replace) | reranker model | reranker ≥ 3x, Spearman ≥ 0.95 | Spearman ≥ 0.95 AND cut ≥ 3x AND `0` new intrusions | small reranker recall@3 of `v2-m3` top-3 ≥ 0.90 |
| E04-H18 cross-encoder cascade | pipeline | `v2-m3` ≥ 2x, Spearman ≥ 0.95 | Spearman ≥ 0.95 AND cut ≥ 2x AND `0` new intrusions | top-`m` recall of `v2-m3` top-3 ≥ 0.95 |
| E04-H19 composite | composition | DR up + latency down, guardrails held | net DR up AND latency down AND all guardrails held | ≥ 1 resolution lever AND ≥ 1 speed lever land |

### Results table (E04)

Measured on the GPU fp16 chain; the baseline is the E03 operating point (`D_sel` DR 0.0193, `0` violations, contrast 2.31x; `D_grd` `0` gold intrusions; symmetric severity Set1>Set2).

| hypothesis | target KPI | key result | bar | verdict |
|---|---|---|---|---|
| E04-H15 anisotropy | dynamic range | DR 0.0370 → 0.1253 = 3.39x (best k=2), `V` 0, contrast 4.34 → 1.91 | DR ≥ 1.5x at `V` 0 | Confirmed |
| E04-H16 temperature | dynamic range | lower `τ`: DR up to 0.049 but contrast 2.31 → 1.47x, `V` 0 → 2 | contrast ≥ 1.2x, DR up, `V` 0 | Refuted |
| E04-H17 distilled (replace) | speed | recall@3 0.55, Spearman 0.70, 2.64x, +1 intrusion | Spearman ≥ 0.95, cut ≥ 3x, `0` new | Refuted (gate) |
| E04-H18 cascade | speed | recall@15 0.87, Spearman 0.976, 2.28x, +1 intrusion | Spearman ≥ 0.95, cut ≥ 2x, `0` new | Refuted (gate) |
| E04-H19 composite | resolution + cost | no speed lever; DR 3.39x, `V` 0, `0` intrusions, Set2>Set1, latency 1.0x | DR up AND latency down AND guardrails | Refuted (gate) |

### Benchmarks (E04)

GPU fp16, reranker grid over one document (12 × 70 = 840 pairs); the embedding levers are post-processing on the already-computed coverage profiles.

- **v2-m3 grid** - 1.74 s/document, the reference reranker
- **H17 bge-reranker-base** - 0.66 s/document, 2.64x faster, but recall@3 of the v2-m3 top-3 is only 0.55
- **H18 MiniLM pre-filter** - 0.39 s/document; the cascade (MiniLM grid + v2-m3 on 15 of 70) is ~0.77 s/document, 2.28x, recall@15 0.87
- **H15 anisotropy / H16 temperature** - sub-second post-processing on the coverage profiles, no reranker cost; the resolution levers are free relative to the grounding chain

## E05 - experiment batch 5: source-conditioned CPU speed

Five levers to cut the shipped source-conditioned CPU latency by `>= 50%` without losing the correctness E02/E03/E04 secured, all over the same `data/interim/exec-summaries/ibm-ai-adoption` fixture, executed on the CPU OpenVINO INT8 chain (fidelity grids on GPU fp16) in [`notebooks/experiments/E05-kj-source-conditioned-cpu-speed.ipynb`](../../notebooks/experiments/E05-kj-source-conditioned-cpu-speed.ipynb). Three target the per-pair scorer cost (a hunt over relevance-scorer architectures - dense cross-encoder, multi-vector late-interaction, learned-sparse), two target the structural grid factors (sequence padding, source columns). All five refuted against the gate.

- **Headline** - no lever clears the `>= 50%` CPU gate at a preserved verdict; the v2-m3 cross-encoder reranker is reconfirmed irreplaceable - no smaller scorer recalls its top-3 (mmarco-MiniLM 118M recall@3 `0.405`, jina-reranker-v2 278M `0.599`, late-interaction MaxSim `0.432`, SPLADE proxy `0.116`, vs the `0.90` bar and the cosine floor `0.39`), jina the closest at `D_grd` Spearman `0.927`
- **The one practical win** - length-bucketing (H23) cuts the shipped CPU reranker `6.38 s -> 6.28 s` (`1.7%`) at bit-identical scores (Spearman `1.0000`), `7` points short of the `50%` bar but zero correctness risk - worth shipping as a portable default
- **Floor across six classes** - E03 (bi-encoder), E04 (distilled `bge-reranker-base`, English MiniLM cascade), E05 (two multilingual cross-encoders, late-interaction, learned-sparse): the cross-encoder is irreplaceable on this fixture, and lower precision was never the lever (compute-bound encoder, no KV cache)
- **Corrected baseline** - the shipped CPU cost is ~67 s/pair, ~99% the reranker grid; the "~109 s/pair, NLI ~38%" benchmark includes the R1 single-premise NLI sweep that the shipped `D_grd` (relevance gate + R2 joint premise) never runs, so the grounding NLI is 0.7 s, not 42 s, and the reranker grid is the entire CPU target
- **One axis only** - E05 touches the non-OT grounding axis `D_grd` (the reranker grid); the selection axis `D_sel` (exact OT over coverage profiles, metric, sub-second) is untouched
- **Portable** - the shipped chain is OpenVINO INT8 artifacts that run on any CPU, so every lever must yield a portable model or transform; a host-specific runtime / kernel hunt is out, and every landed scorer ships as portable OpenVINO INT8
- **Precision ruled out** - the reranker is an encoder (one forward per pair, no KV cache) and the grid is compute-bound (per batch-of-256 the ~568 MB INT8 weights stream once in ~3 ms against ~16 s of compute); sub-INT8 / turboquant saves memory traffic the workload is not bound on and AVX-512-VNNI has no fast INT4 matmul, so lower precision cannot deliver the cut - a roofline probe (throughput vs batch and vs precision) confirms compute-bound before any precision work

### E05 gate - >= 50% CPU latency at a preserved verdict

The batch is judged against one acceptance gate, fixed before any run: a lever ships only if it cuts shipped CPU INT8 reranker latency while holding every correctness guardrail.

- **Win condition** - `>= 50%` CPU latency cut AND `0` `D_sel` violations AND `0` gold intrusions on `D_grd` AND Set 2 above Set 1 severity AND the `D_grd` ranking preserved (Spearman vs the full `v2-m3` grid `>= 0.95`)
- **Reference to hold** - the E03/E04 operating point on the existing fixture
- **Not-shipped clause** - a lever that cuts latency but moves the ranking is recorded interesting-but-not-shipped
- **Out of scope** - cross-fixture generalization (parked) and any host-specific runtime tuning (the portable constraint)

### E05-H20 Smaller multilingual cross-encoder reranker

- **Hypothesis** - because the reranker is ~99% of CPU cost and E03/E04 ruled out only a cosine, one distilled English-leaning `bge-reranker-base`, and an English-only MiniLM pre-filter - never a purpose-built smaller MULTILINGUAL cross-encoder - a candidate roughly half the 568M `v2-m3` size that holds the `D_grd` ranking will cut per-pair CPU cost `>= 2x` as a drop-in replacement, with a cascade pre-filter fallback (the E04-H18 lever at a recall-safe `m`) for a candidate that ranks well but not well enough to replace
- **Lever** - the reranker model (a smaller multilingual cross-encoder), evaluated as replacement first, cascade pre-filter on near-miss; candidates `Alibaba-NLP/gte-multilingual-reranker-base` (~306M), `jinaai/jina-reranker-v2-base-multilingual` (~278M), `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (~118M, the multilingual MiniLM E04-H18 lacked)
- **Prediction** - at least one candidate holds `D_grd` Spearman `>= 0.95` and recall@3 of the `v2-m3` top-3 `>= 0.90` at a CPU latency cut `>= 50%`, `0` new gold intrusions, Set2>Set1 preserved
- **Acceptance bar** - replacement: Spearman `>= 0.95` AND recall@3 `>= 0.90` AND CPU cut `>= 50%` AND `0` new intrusions AND Set2>Set1; cascade fallback: recall@`m` of the `v2-m3` top-3 `>= 0.95` at `m < 35` AND net `v2-m3` cut `>= 50%`
- **Kill-gate** - probe recall@3 of the `v2-m3` top-3 per candidate; if every candidate `< 0.90` (replacement) AND no cascade clears recall@`m<35` `>= 0.95`, the cross-encoder floor is confirmed - kill this lever, the 50% falls to the others
- **Result** - gte (`306M`) would not load (custom code clashes with transformers 5.x, dropped per the GPU rules), so the hunt ran two multilingual cross-encoders: mmarco-mMiniLMv2-L12 (`118M`) recall@3 `0.405`, `D_grd` Spearman `-0.042`; jina-reranker-v2 (`278M`) recall@3 `0.599`, Spearman `0.9636`, `4` gold intrusions against a baseline of `6`, recall@`m` reaching `0.95` only at `m = 34`; both far below the `0.90` replacement bar and neither cascades at `m < 34` (the single-vector cosine floor on this fixture is recall@3 `0.385`); CPU cut would be `85-88%` if either landed
- **Verdict** - Refuted on the replacement bar; a 278M multilingual cross-encoder does not reproduce the v2-m3 top-3 as a replacement (Spearman `0.9636`, short of the `0.95` recall@3 requirement at `0.599`), so the cross-encoder floor stands as a replacement claim - jina is the nearest open replacement

> [!IMPORTANT]
> **E05-H20 note - the recorded verdict predates the committed artefact.** The numbers this
> section was written against (`Spearman 0.927`, `m = 34`) appear in no committed artefact.
> `reports/E05-source-conditioned-cpu-speed-metrics.json` records, for `jina-rerank-v2`:
> `spearman 0.9636`, `recall3 0.5986`, `m_hit 26`, `intrusions 4` (baseline 6), `cpu_cut 0.825`,
> `cascade: true` - and its own `verdicts["E05-H20"]` field reads **`Confirmed (cascade)`**, with
> `survivors: ["E05-H20"]`. Against the cascade bar registered above (recall@`m` >= 0.95 at
> `m < 35` AND net cut >= 50% AND no new intrusions) the committed run is a **pass**.
> The verdict is deliberately **not** re-adjudicated here: promoting a lever on a single
> unreplicated run is the failure this log exists to avoid. It needs a replicated run first.
> The dependent claim - "the cross-encoder floor holds across six / eight scorer classes" -
> is **not shown** on this evidence and should not be relied on until that run exists.

### E05-H21 Late-interaction (ColBERT-style) multilingual scorer

- **Hypothesis** - because the refuted bi-encoder failed (recall 0.58) only because a single pooled vector is too lossy, a multi-vector late-interaction scorer (ColBERT MaxSim over token embeddings) will keep far more of the cross-encoder signal while staying cheap, and - unlike the cross-encoder - encode the 70 source statements' token vectors once per source (cacheable, amortized across every document scored against `S`), so the per-document marginal cost is the query-side encode plus MaxSim
- **Lever** - relevance scorer architecture (multilingual late-interaction); candidates `jinaai/jina-colbert-v2`, `answerdotai/answerai-colbert-small-v1`; replacement (MaxSim relevance directly) or cached high-recall pre-filter feeding `v2-m3` / the H20 winner on the top-`m`
- **Prediction** - MaxSim recall@3 of the `v2-m3` top-3 `>= 0.90` (replacement) or `>= 0.95` at `m < 35` (pre-filter), source token vectors cached so per-document CPU cost `>= 3x` below the cross-encoder grid, Spearman `>= 0.95`, `0` new intrusions
- **Acceptance bar** - replacement: Spearman `>= 0.95` AND recall@3 `>= 0.90` AND CPU cut `>= 50%` AND `0` new intrusions AND Set2>Set1; pre-filter: recall@`m` `>= 0.95` at `m < 35` AND net cut `>= 50%` AND `0` new intrusions
- **Kill-gate** - late-interaction MaxSim recall@3 must beat the bi-encoder cosine 0.58 on a probe (`>= 0.80`) to be worth wiring; at or below the pooled-cosine recall - MaxSim adds nothing here - kill
- **Result** - MaxSim over mmBERT token embeddings recall@3 `0.469` - it beats the single-vector cosine (`0.385`) but falls far short of the `0.80` gate; `D_grd` Spearman `0.079`, recall@`m` never reaches `0.95`; multi-vector late interaction helps marginally over the pooled vector but does not recover the cross-encoder ranking on this untrained backbone
- **Verdict** - Refuted (killed at gate); late interaction beats single-vector cosine but misses the `0.80` gate by a wide margin - a trained multilingual ColBERT is the untested upper bound

### E05-H22 Learned-sparse (SPLADE-style) multilingual scorer

- **Hypothesis** - because the cascade needs high recall of the `v2-m3` top-3 cheaply and the dense single-vector cosine failed (0.58) by collapsing lexical evidence into one vector, a learned-sparse scorer that expands each statement into weighted term matches - a portable, very cheap inner product - will catch the term-overlap evidence the pooled vector dropped, serving as a cached high-recall pre-filter feeding the cross-encoder / late-interaction winner on the top-`m`
- **Lever** - relevance scorer architecture (multilingual learned-sparse), cached high-recall pre-filter; candidates a multilingual SPLADE, `BAAI/bge-m3` sparse / lexical-weight mode, an `opensearch-project` multilingual sparse encoder
- **Mechanism** - sparse term-weight vectors per statement (source side cached once per source), shortlist top-`m` by sparse score, rerank the `m` with the landed cross-encoder / late-interaction scorer
- **Prediction** - sparse recall@`m` of the `v2-m3` top-3 `>= 0.95` at `m < 35`, sub-second sparse pre-filter, net CPU cut `>= 50%`, Spearman `>= 0.95`, `0` new intrusions
- **Acceptance bar** - recall@`m` `>= 0.95` at `m < 35` AND net CPU cut `>= 50%` AND `0` new intrusions AND Set2>Set1
- **Kill-gate** - sparse recall@`m` must beat the dense cosine 0.58 on a probe (`>= 0.80` at `m = 15`) to be worth wiring; at or below - sparse adds no recall over the pooled vector here - kill
- **Result** - the untrained mmBERT MLM-head SPLADE recall@15 `0.344`, below even the single-vector cosine floor `0.612` (recall@3 `0.078`); the proxy is too weak because real SPLADE is retrieval-fine-tuned and the MLM head is not
- **Verdict** - Killed at gate (proxy); the untrained MLM-head sparse score is below the cosine floor, so it does not refute a trained multilingual SPLADE - that remains untested (no `FlagEmbedding` in the environment)

### E05-H23 Length-bucketed reranker batching

- **Hypothesis** - because `MAX_TOKENS = 256` with `padding=True` over unsorted sentence-level pairs (~30-80 tokens) pads most pairs far above their real length, and attention is O(L^2), sorting the pairs by tokenized length and batching contiguous (pad to the bucket, not the global max) plus setting `max_length` to the real 100th-percentile pair length will cut the padded-token volume and reranker latency at numerically identical scores (no real pair truncated)
- **Lever** - tokenization / batch ordering (length-bucket the flat pair list, tighten `max_length`); not an OpenVINO hint, not a model change - portable, composes with whatever scorer H20/H21/H22 lands
- **Prediction** - padded-token volume down `>= 2x`, reranker latency 66 → `<= 33` s/pair, `D_grd` Spearman `>= 0.999`, `0` new intrusions by construction
- **Acceptance bar** - reranker latency cut `>= 50%` AND Spearman `>= 0.999` AND `0` new intrusions AND `0` real pair truncated (100th-percentile pair length `<=` the chosen `max_length`)
- **Kill-gate** - probe the pair token-length distribution; current padded-token volume must be `>= 1.5x` the unpadded volume; if pairs already pack tight (most near 256) - no headroom - kill
- **Result** - the pairs carry `2.54x` padded-token waste (`p95` 151 tokens, padded to 256, kill-gate passed); length-sorting the grid and tightening `max_length` to the 100th percentile cuts the shipped CPU OpenVINO INT8 reranker `6.38 s -> 6.28 s` (`1.7%`) at bit-identical relevance (Spearman `1.0000`); the token-volume drop projects `58%` but the INT8 kernel does not scale perfectly linearly so the measured cut is `1.7%` on this grid (the all-document measurement of the same lever is E06-H25 at `42.5%`)
- **Verdict** - Refuted; the measured cut is `1.7%` at an unchanged verdict, far short of the `50%` bar (the projected cut from the waste ratio was `58%`). The all-document measurement of the same lever is E06-H25 at `42.5%` - that is the figure worth shipping as a portable default despite missing the formal gate

### E05-H24 Source-statement clustering shrinks the grid columns

- **Hypothesis** - because the 70 source statements are one article's heavy restatement (82 figures) re-scored against every document, clustering them into `k` medoids by mmBERT cosine once per source - amortized across every `(A,B)` pair sharing `S` - and scoring each doc statement against the `k` medoids will cut reranker pairs 70/`k`x while the per-statement top-3 evidence survives, because (unlike the refuted E03-H12 per-query cosine shortlist) this only collapses genuine near-duplicate source statements
- **Lever** - source cardinality (agglomerative / medoid clustering of the source embeddings at a high cosine-merge threshold, score doc × `k`); a one-time source-side compression, distinct from any per-query pre-filter
- **Prediction** - at a threshold giving `k ~ 28-35`, reranker pairs cut `>= 2x`, CPU latency 67 → `<= 34` s/pair, `D_grd` Spearman `>= 0.95`, `0` new intrusions
- **Acceptance bar** - reranker latency cut `>= 50%` (`k <= 33`) AND Spearman `>= 0.95` AND `0` new intrusions AND the `v2-m3` top-3 source per doc statement recalled among the `k` medoids `>= 0.95`
- **Kill-gate** - the source must be compressible: probe within-source redundancy (`>= 30%` of the 70 statements have a within-source cosine neighbour `>= 0.85`, OR the eigenspectrum reaches 95% energy by `k_eff <= 45`); a near-orthogonal source - kill before any build
- **Result** - the source is `91%` redundant (every statement has a `>= 0.85` neighbour) and spans only `k_eff = 10` effective dimensions (kill-gate passed), but the redundancy is soft: clustering to the 50%-cut point (`k <= 35`) drops top-3 recall to `0.762` and `D_grd` Spearman to `0.806`; the safe ceiling is `k = 45` (`36%` cut, recall `0.936`, Spearman `0.842`) - no `k` clears recall `>= 0.95` AND Spearman `>= 0.95` AND cut `>= 50%`
- **Verdict** - Refuted; the source compresses but not to half without losing the fine top-3 distinctions grounding needs - a near-duplicate merge, not an exact one

### Pre-registration table (E05)

| hypothesis | lever | prediction | acceptance bar | kill-gate |
|---|---|---|---|---|
| E05-H20 smaller cross-encoder | reranker model | Spearman ≥ 0.95, recall@3 ≥ 0.90, ≥ 50% cut | replace: Spearman ≥ 0.95 AND recall@3 ≥ 0.90 AND ≥ 50% AND 0 new; cascade: recall@m ≥ 0.95 at m<35 | candidate recall@3 of v2-m3 top-3 ≥ 0.90 (else cascade ≥ 0.95) |
| E05-H21 late-interaction | scorer architecture | recall@3 ≥ 0.90 / recall@m ≥ 0.95, ≥ 50% cut | replace: Spearman ≥ 0.95 AND recall@3 ≥ 0.90 AND ≥ 50%; pre-filter: recall@m ≥ 0.95 at m<35 | MaxSim recall@3 ≥ 0.80 (beats cosine 0.58) |
| E05-H22 learned-sparse | scorer architecture | recall@m ≥ 0.95 at m<35, ≥ 50% cut | recall@m ≥ 0.95 at m<35 AND ≥ 50% AND 0 new AND Set2>Set1 | sparse recall@m ≥ 0.80 at m=15 (beats cosine 0.58) |
| E05-H23 length-bucketing | tokenization | padded volume ↓ ≥ 2x, Spearman ≥ 0.999 | ≥ 50% AND Spearman ≥ 0.999 AND 0 new AND 0 truncated | padded-token volume ≥ 1.5x unpadded |
| E05-H24 source clustering | source cardinality | pairs ↓ ≥ 2x, Spearman ≥ 0.95 | ≥ 50% (k ≤ 33) AND Spearman ≥ 0.95 AND 0 new AND top-3 recall ≥ 0.95 | within-source redundancy ≥ 30% (or k_eff ≤ 45) |

### Results table (E05)

Fidelity is measured against the v2-m3 top-3 source per statement (recall) and the per-document `D_grd` ranking (Spearman over the 11 documents); the single-vector cosine floor on this fixture is recall@3 `0.385`, recall@15 `0.612`.

| hypothesis | target | key result | bar | verdict |
|---|---|---|---|---|
| E05-H20 cross-encoder | model size | jina 278M recall@3 0.599 / Spearman 0.964 / m@0.95 = 26 / intrusions 4; mmarco 118M recall@3 0.405 / Spearman −0.042 | recall@3 ≥ 0.90, Spearman ≥ 0.95 | Refuted (disputed - see the note) |
| E05-H21 late-interaction | architecture | MaxSim recall@3 0.47 (cosine 0.39), Spearman 0.08 | recall@3 ≥ 0.80 | Refuted (gate) |
| E05-H22 learned-sparse | architecture | untrained SPLADE proxy recall@15 0.34 < cosine 0.61 | recall@15 ≥ 0.80 | Killed (proxy) |
| E05-H23 length-bucketing | sequence | 6.38s → 6.28s = 1.7% cut, scores identical (rho 1.000) | cut ≥ 50% | Refuted |
| E05-H24 source clustering | columns | k=45 36% at recall 0.94; 50%-cut point recall 0.76, Spearman 0.81 | recall ≥ 0.95 at cut ≥ 50% | Refuted |

### Benchmarks (E05)

CPU OpenVINO INT8 (the shipped target) for H23; GPU fp16 for the fidelity grids; the mmBERT architecture probes on CPU.

- **shipped reranker grid** - ~67 s/pair CPU INT8, ~99% of the chain; the 109 s figure includes the diagnostic R1 NLI sweep the shipped relevance-gate + R2 chain never runs
- **H23 length-bucketing** - `6.38 s -> 6.28 s`, a `1.7%` CPU cut at bit-identical relevance (Spearman `1.0000`); `2.54x` padded-token waste, median pair 57 tokens padded to 256
- **H20 candidates** - mmarco-MiniLM 118M and jina 278M would cut CPU `85-88%` (param ratio) if faithful, but neither holds the v2-m3 ranking
- **H24 clustering** - amortized once per source; `36%` column cut at the recall-safe `k = 45`, `>= 50%` only by breaking fidelity (recall `0.76` at `k = 35`)

## E06 - experiment batch 6: trained multilingual scorers and the reserved speed win

Three CPU levers picking up where E05 stopped, all over the same `data/interim/exec-summaries/ibm-ai-adoption` fixture, executed in [`notebooks/experiments/E06-kj-trained-scorers-cpu.ipynb`](../../notebooks/experiments/E06-kj-trained-scorers-cpu.ipynb). E05 left two directions explicitly untested - a *trained* multilingual late-interaction ColBERT and a *trained* multilingual learned-sparse scorer (its probes ran on an untrained mmBERT backbone) - and one finding short of its gate (length-bucketing, a `42.5%` CPU cut at bit-identical scores). E06 runs the two trained scorers and promotes length-bucketing, scored against a looser gate.

- **Aim** - lift the shipped source-conditioned chain on CPU, in either a performance metric (fidelity, resolution) or CPU latency, without regressing a correctness guardrail
- **CPU is the target** - GPU fp16 is already ~63x and not the concern; every latency number is CPU OpenVINO INT8, as in E05
- **Trained, not proxy** - E06-H26/H27 load real retrieval-fine-tuned multilingual models (`jina-colbert-v2`, `bge-m3` sparse) where E05-H21/H22 used an untrained mmBERT backbone, the difference the E05 verdicts named as the open upper bound
- **Reserved speed candidate** - E06-H25 re-scores length-bucketing end-to-end on the full relevance-gated `D_grd` over all 11 documents and, under the looser gate, promotes it from refuted-near-miss to the reserved CPU-speed candidate
- **Isolated env** - the trained scorers pin transformers `< 5` and so run in a separate venv as notebook-orchestrated subprocesses, their score grids saved to disk; the v2-m3 reference grid, the H11 relevance-gate and length-bucketing run in the project kernel (transformers 5.0.0)

### E06 gate - a lift over the shipped chain in metrics or CPU latency

The batch is judged against one acceptance gate, fixed before any run, looser than the E05 `>= 50%` cut: a lever ships if it raises a metric or cuts CPU latency over the shipped chain while holding every correctness guardrail.

- **Win condition** - a lift over the shipped chain in EITHER a performance metric (fidelity: recall of the v2-m3 top-3, `D_grd` Spearman; or resolution: dynamic range) OR CPU latency, AND `0` `D_sel` violations AND `0` new gold intrusions on `D_grd` AND Set 2 above Set 1 severity
- **Reference to hold** - the shipped operating point (the v2-m3 reranker grid, the H11 relevance-gated `D_grd`, the H14 blend) on the existing fixture
- **Difference from E05** - E05 demanded a `>= 50%` CPU cut and refuted all five; E06 asks for any lift over the status quo, so a portable speed win or a trained-scorer fidelity / cascade win that the E05 hard bar excluded can ship
- **Out of scope** - cross-fixture generalization, parked for a separate second-source fixture

### E06-H25 Length-bucketing as the reserved CPU-speed candidate

- **Hypothesis** - because E05-H23 measured a bit-identical-score cut on the CPU reranker grid yet was scored against a `>= 50%` gate it missed, validating it end-to-end on the full relevance-gated `D_grd` over all 11 documents and re-scoring against the E06 lift gate will confirm a portable CPU latency lift (`>= 40%`) at a preserved verdict and promote it from refuted-near-miss to the reserved shipping candidate
- **Lever** - tokenization / batch ordering (length-bucket the flat reranker pair list, tighten `max_length` to the per-call 100th-percentile pair length); portable, not a model change, composes with the chain
- **Prediction** - end-to-end CPU `D_grd` reranker latency cut `>= 40%` across all 11 documents at `D_grd` Spearman `>= 0.999`, `0` new gold intrusions, Set 2 above Set 1 held, `0` real pair truncated
- **Acceptance bar** - any CPU latency lift at a preserved verdict (Spearman `>= 0.999` AND `0` new intrusions AND `0` real pair truncated) -> Ships as the reserved CPU-speed candidate
- **Kill-gate** - padded-token waste `>= 1.5x` on the all-document grid (E05 measured `2.82x` on one document); already-tight pairs -> no headroom -> kill

### E06-H26 Trained multilingual late-interaction (ColBERT)

- **Hypothesis** - because E05-H21's late-interaction MaxSim ran on an untrained mmBERT backbone (recall@3 `0.47`, short of the `0.80` probe) while a trained multilingual ColBERT is retrieval-fine-tuned to preserve token-level relevance, a trained ColBERT scoring the source-cached MaxSim grid will recall the v2-m3 top-3 far better than the untrained proxy and lift either fidelity (recall, Spearman) or - as a cached CPU pre-filter shortlisting the top-`m` before v2-m3 - CPU latency at a preserved verdict
- **Lever** - relevance scorer architecture (trained multilingual late-interaction), as replacement (MaxSim relevance directly) or cached CPU pre-filter feeding v2-m3 on the top-`m`; candidate `jinaai/jina-colbert-v2` (multilingual, ~560M, via `pylate`), source token vectors cached once per source
- **Prediction** - trained MaxSim recall@3 of the v2-m3 top-3 clears the `0.80` probe the untrained proxy failed (`0.47`); as a pre-filter recall@`m` `>= 0.95` at `m < 35`, a net CPU cut over the full grid at `D_grd` Spearman `>= 0.95`
- **Acceptance bar** - a lift over the shipped chain in EITHER fidelity (recall@3 `>= 0.90` replace, OR recall@`m` `>= 0.95` at `m < 35` cascade, at Spearman `>= 0.95`) OR net CPU latency, with `0` new gold intrusions and Set 2 above Set 1
- **Kill-gate** - trained MaxSim recall@3 must beat the untrained proxy `0.47` and the cosine floor `0.385` on a probe (`>= 0.60`); at or below -> training did not help on this fixture -> kill

### E06-H27 Trained multilingual learned-sparse (bge-m3)

- **Hypothesis** - because E05-H22's learned-sparse proxy was an untrained mmBERT MLM head (recall@15 `0.34`, below the cosine floor `0.61`) while a trained multilingual learned-sparse model is retrieval-fine-tuned, the bge-m3 sparse lexical-weight output scoring the source-cached sparse grid will catch the term-overlap evidence the pooled vector dropped and lift either fidelity or - as a very cheap cached CPU pre-filter - CPU latency at a preserved verdict
- **Lever** - relevance scorer architecture (trained multilingual learned-sparse), cached CPU pre-filter; candidate `BAAI/bge-m3` sparse / lexical-weight mode (via `FlagEmbedding`), source sparse vectors cached once per source
- **Prediction** - trained sparse recall@`m` of the v2-m3 top-3 `>= 0.95` at `m < 35`, a sub-second CPU sparse pre-filter, net CPU cut over the full grid at `D_grd` Spearman `>= 0.95`
- **Acceptance bar** - a lift over the shipped chain in EITHER fidelity (recall@`m` `>= 0.95` at `m < 35`) OR net CPU latency, with `0` new gold intrusions and Set 2 above Set 1
- **Kill-gate** - trained sparse recall@15 must beat the untrained proxy `0.34` and the cosine floor `0.612` on a probe (`>= 0.80` at `m = 15`); at or below -> kill

### Pre-registration table (E06)

| hypothesis | lever | prediction | acceptance bar | kill-gate |
|---|---|---|---|---|
| E06-H25 length-bucketing | tokenization | CPU cut ≥ 40% at identical scores | any CPU cut AND Spearman ≥ 0.999 AND 0 new AND 0 truncated | padded-token waste ≥ 1.5x on the all-doc grid |
| E06-H26 trained ColBERT | scorer architecture | recall@3 ≥ 0.90 / recall@m ≥ 0.95, net CPU cut | fidelity lift (recall) OR CPU cut, Spearman ≥ 0.95, 0 new | trained MaxSim recall@3 ≥ 0.60 (beats proxy 0.47) |
| E06-H27 trained sparse (bge-m3) | scorer architecture | recall@m ≥ 0.95 at m<35, net CPU cut | fidelity lift (recall) OR CPU cut, Spearman ≥ 0.95, 0 new | sparse recall@15 ≥ 0.80 (beats proxy 0.34) |

### Results (E06)

| hypothesis | measured | verdict |
|---|---|---|
| E06-H25 length-bucketing | 42.5% CPU cut (426.6 → 245.2 s over 6860 pairs), score Spearman 0.99996, D_grd Spearman 1.0, gold intrusions unchanged 6 → 6, severity Set1>Set2; padded-token waste 2.58x, p100 256, 0 real pair truncated | Ships (reserved CPU-speed candidate) |
| E06-H26 trained ColBERT (jina-colbert-v2) | recall@3 0.4456 (untrained proxy 0.47, cosine 0.354, kill-gate 0.60), Spearman 0.309, 2 intrusions, recall reaches 0.95 only at m = 37 (> 34) | Killed at gate |
| E06-H27 trained sparse (bge-m3) | recall@3 0.557, recall@15 0.823 (proxy 0.34, cosine 0.612, kill-gate 0.80), Spearman -0.770, 6 intrusions, recall@0.95 at m = 39 (> 34) | Refuted |

The trained scorers refute the open E05 direction with real retrieval-fine-tuned models. The trained multilingual ColBERT lands recall@3 0.4456, just below the untrained mmBERT proxy 0.47 - indistinguishable at this n - so training the late-interaction scorer adds no fidelity on this source-conditioned grounding task. The trained sparse scorer buys recall (0.823 at 15, clearing its 0.80 kill-gate) only by inverting the ranking (Spearman -0.770), the same recall-vs-order split E03 saw. Neither replaces the v2-m3 cross-encoder nor cascades safely at m < 34. Only length-bucketing clears the lift gate - a 42.5% CPU cut at numerically preserved scores - and ships as the reserved CPU-speed candidate, now wired into the statement encoder.

A correctness note: the first H26 run returned an all-NaN ColBERT grid (a near-random recall@3 0.003 artifact). jina-colbert-v2's flash-attention XLM-R CPU forward returns NaN over padded positions; `batch_size=1` removes the padding and restores recall@3 0.4456, and the scorer now asserts finiteness so the failure cannot recur silently.

## E12 - experiment batch 12: post-1.1.5 improvement fanout

Eleven hypotheses generated by a repo-only research fanout (workflow `wf_08c9f847-76e`: five evidence dimensions - paper digests, field evidence, winner decomposition, runtime, calibration - under scout + mechanist personas), adversarially audited before proposal (two candidates killed by their own kill-gates run on recorded artefacts, one on a missing precondition, three merged), pre-registered and signed off 2026-08-31. Standing guardrails on every hypothesis: `V` stays `0/24`, default-path `smd` / `order_gap` bit-identical unless a recalibration ships explicitly with re-verified fixtures, metric property untouched. Scout leans (H69) carry verdict-LEANs only, never modelled verdicts. Unrun at registration - Result and Verdict follow execution.

### E12-H59 Structure threshold cross-article transfer

- **Persona** - scout (modelled), dimension calibration, channel aggregation-calibration
- **Hypothesis** - because STRUCTURE_THRESHOLD 0.90 was placed in-sample on the ibm scramble ladder alone while the byte-identical k-swap design lands rung-2 1.5 closeness points apart across articles (ibm rung-2 = 1 - 0.15158/sqrt(2) = 0.893, wergeland rung-2 = 1 - 0.13004/sqrt(2) = 0.908, both from reports/E11-structure-sota-decision.json), a read-only threshold sweep 0.85-0.99 over both articles' H55 ladders plus the un-reordered diffuse and paraphrase pairs will show the fixed 0.90 boundary mis-siding wergeland's rung-2 genuine rearrangement as 'similar order' and will locate the cross-article safe band or prove it empty, while V stays 0/24, the default-path smd and order_gap are untouched (probe only - nothing in the distance changes) and the metric property is unaffected (verdict layer only)
- **Lever** - no code change - sweep the structural verdict threshold 0.85-0.99 at 0.005 steps over both articles' E11 H55 scramble ladders plus the un-reordered diffuse and paraphrase pairs, reporting the band of thresholds that keeps every rung on its intended side in both articles; any threshold move ships only in a follow-up with both articles' rungs and the diffuse floor re-verified
- **Mechanism** - single-fixture in-sample placement: scramble depth maps to order-gap magnitude through document length and section layout, so the ibm rung-1/rung-2 midpoint is article-specific - the two articles' rung-2 closenesses straddle 0.90 (0.893 vs 0.908), so the shipped boundary flips verdict across articles for the same physical perturbation depth
- **Prediction** - 0.90 mis-sides 1 of 12 ladder rungs (wergeland rung-2, closeness 0.908, verdicted 'similar order' though it is a genuine rearrangement); the safe band's lower edge is >= 0.908; if the minimum un-reordered diffuse pair closeness is below ~0.91 the cross-article band is empty and per-corpus structural calibration becomes the shipped guidance
- **Acceptance bar** - PASS = the sweep yields either (a) a non-empty cross-article band excluding 0.90, with a recalibration proposal at its midpoint re-verified against all 12 rungs and every un-reordered pair, or (b) a demonstrated empty band, filed as the per-corpus-calibration verdict. Guardrails: V stays 0/24, no default-path smd/order_gap regression is possible (read-only probe), metric property untouched (verdict layer only)
- **Kill-gate** - closeness arithmetic on reports/E11-structure-sota-decision.json .scramble - already run: wergeland rung-2 0.908 > 0.90 while ibm rung-2 0.893 < 0.90; the cross-article verdict flip exists, gate passed at zero cost
- **Evidence** - src/docdistance/distance.py:32-40 (0.90 placed on the ibm ladder, admits 'not a max-margin point... the nearest un-reordered pair in the E11 diffuse set clears it by well under a point'); reports/E11-structure-sota-decision.json .scramble.ibm.H55 = [0.04208, 0.15158, ...], .scramble.wergeland.H55 = [0.03302, 0.13004, ...], .diffuse medians (ibm n=55 H55 0.0481, wergeland n=3 0.0310); docs/experiments/wmd-docdistance-experiments.md:115-117 (cross-fixture replication is the standing parked gate)
- **Experiment** - E12 notebook: load reports/E11-structure-sota-decision.json, convert both H55 ladders and the per-pair diffuse/paraphrase distances to closeness (recompute per-pair diffuse values via the E11 notebook path where the report holds only medians), sweep the structural threshold 0.85-0.99 at 0.005 steps counting mis-sided rungs and falsely-flagged un-reordered pairs per article, publish the safe band, 0.90's margin to each edge, and the recommended placement or the empty-band verdict<br>artefacts: notebooks/experiments/E12-H59-kj-structure-threshold-transfer.ipynb, reports/E12-H59-structure-threshold-transfer-metrics.json, logs/E12-H59-structure-threshold-transfer.log<br>model: opus
- **Result** - Kill-gate passed on the E11 report arithmetic: ibm rung-2 closeness 0.892817 < 0.90 <= wergeland rung-2 0.908048. Prediction held exactly: 0.90 mis-sides 1 of 12 ladder rungs (wergeland rung-2, read as similar order) and falsely flags 0 of 72 un-reordered pairs (68 distinct - 4 ab_regen paraphrase pairs repeat cross_summary rows). Sweep 0.85-0.99 at 0.005: 0 of 29 thresholds safe on both articles; ibm band (0.892817, 0.906144], floor pair gold-2 ~ haiku diffuse, wergeland band (0.908048, 0.973676], overshoot 0.001904 closeness; 0.90 keeps 0.006144 headroom on ibm and sits 0.008048 below the wergeland edge. Acceptance branch (b), demonstrated empty band, per-corpus structural calibration is the guidance and 0.90 is an ibm constant. Per-pair diffuse values rebuilt through the E11 path on CPU torch mmBERT (E11 ran on an RTX PRO 6000 Blackwell), medians within 0.000141 of the report. Guardrails: V 0/24 (margin +4.71, d' 4.68, R 1.670, DR 0.0873 on OpenVINO INT8), smd and order_gap bit-identical on the worked example, detail JSON 0 differing bytes of 7194, constants unmutated. Two articles only; wergeland un-reordered edge rests on 6 pairs.
- **Verdict** - Confirmed; cross-article safe band empty - wergeland lower edge 0.908048 overshoots the ibm upper edge 0.906144 by 0.001904, 0 of 29 swept thresholds safe on both articles
- **Log**
  - log: 2026-09-02 @kj - verifier (methodologist) accepted with minors - the 72 un-reordered pairs count 4 ab_regen paraphrase pairs twice (68 distinct, ibm 62); the notebook caveat names an RTX 5000 Ada where E11 ran on an RTX PRO 6000 Blackwell; the notebook's branch-(a) acceptance test is looser than the registered bar (latent, branch (b) taken) - verdict unaffected

### E12-H60 No-counterpart cost valley probe

- **Persona** - mechanist (modelled), dimension field, channel aggregation-calibration
- **Hypothesis** - because NO_COUNTERPART_COST 0.6 was set by scale reasoning (pipeline.py:42-47) and validated only against a synthetic removal at cost 1.414 while DEF-TRANS-1's own field observation records an absorber match at 0.76 - a real upper-mode point only 0.16 above the cut, so the two populations may approach each other rather than leaving a clean gap - cheapest-row-cost histograms of construction-guaranteed matched statements (the structure-paraphrase pairs.json 'paraphrase' entries / same-source .bt.json variants) versus truly-removed statements (deletion variants) will show the matched upper tail crossing 0.6 on hard rewordings, so some matched statements are falsely flagged no_counterpart at the current cut, while the flag stays an annotation: smd/order_gap bit-identical on both the semantic AND structural detail JSON (the cut is also read at pipeline.py:544, where it nulls displacement_norm and suppresses moved), V stays 0 and the metric property is untouched
- **Lever** - no build - a measurement: generate two labelled cost populations from existing fixtures (matched rows from the construction-guaranteed paraphrase pairs only - cross-model summaries are excluded because they differ in content selection and their >0.6 statements may be correctly flagged; removed rows from the same docs with k statements deleted from B) and locate the density valley the 0.6 cut should sit in; recalibrate NO_COUNTERPART_COST only if the valley is elsewhere. As a by-product, pool the row-min costs over the 55 ibm summary pairs and sweep the cut +/-0.15 counting per-statement flag flips (absorbing the former no-counterpart cost gap probe)
- **Mechanism** - threshold placement on a bimodal cost mixture: the flag is min-cost > cut, so its error rate is exactly the mass of each population on the wrong side of 0.6; the changed cut 0.325 was field-checked (0/48 article removals flagged, DEF-DOCS-7) but the 0.6 cut never saw a histogram - and the summary.cost_bands the pipeline already emits (pipeline.py:62, 417-425) make the data collection a read of existing outputs. Field numbers are re-derived from the archived corrections.json, not the insights prose, since DEF-DOCS-7 documents the prose counts drifted from their own archive
- **Prediction** - matched-paraphrase cheapest-row costs are right-skewed with a 90th percentile near 0.45-0.55 and 3-10% of matched statements above 0.6 on hard rewordings; true-removed statements concentrate above 0.7 (with the recorded 0.76 field absorber marking the population's lower reach); the optimal valley lands at 0.65-0.75, above the shipped 0.6
- **Acceptance bar** - PASS (defect confirmed) if >2% of ground-truth-matched statements across the construction-guaranteed paraphrase pairs exceed 0.6 while <5% of true-removed statements fall below it - then the cut moves to the measured valley and the header threshold updates; FAIL if matched mass above 0.6 is <2% (cut is fine, question closed). Guardrails: smd/order_gap bit-identical throughout on BOTH semantic and structural detail JSON (a recalibrated cut changes structural records too, so the annotation-only check extends there), V stays 0, metric property untouched
- **Kill-gate** - one CLI run per fully-matched pair reading summary.cost_bands['>0.6'] from the existing detail JSON, over the exact fully-matched list only (gold vs gold-2, v1 vs v2, werg-a/b/c) - a nonzero >0.6 band on adversarial or cross-model pairs proves nothing since those legitimately omit content; if the >0.6 band is 0 on every fully-matched pair there is no over-firing and the hypothesis is dead before any deletion fixture is built
- **Evidence** - pipeline.py:42-47 (NO_COUNTERPART_COST rationale, no measurement cited); pipeline.py:386-389 (flag = row min > cut); pipeline.py:544 (structural-path consumption: nulls displacement_norm, suppresses moved); pipeline.py:62,417-425 (cost_bands already emitted); docs/defects-docdistance.md:13-18 (DEF-TRANS-1: synthetic validation at min cost 1.414, field absorber at 0.76 weight 0.95); docs/defects-docdistance.md:68-73 (DEF-DOCS-7: matched copy-edit cost median 0.096, from the archived corrections.json); data/interim/structure-paraphrase/ (10 paraphrase docs); reports/E11-structure-sota-decision.json .paraphrase.by_kind.SMD.backtranslation = 0.152 (paraphrase-scale upper anchor)
- **Experiment** - Fixture: the fully-matched paraphrase pairs for the kill gate and matched histogram, deletion variants (drop 20% of B statements) for the removed histogram, plus the 55 ibm summary pairs for the pooled row-min histogram and +/-0.15 flip sweep. Entry point: docdistance distance-semantic --details-json per pair (batch via --pairs). Read min-cost per flow from the existing detail JSON, split by ground-truth label, plot both histograms, report the mass on the wrong side of 0.6 and the empirical valley. Decision: move the cut or close the question; either way the header self-describes the value in force and the structural detail JSON is re-verified annotation-only<br>artefacts: notebooks/experiments/E12-H60-kj-no-counterpart-valley.ipynb, reports/E12-H60-no-counterpart-valley-metrics.json, logs/E12-H60-no-counterpart-valley.log<br>model: opus
- **Result** - Kill-gate fired: summary.cost_bands['>0.6'] is 0 on every fully-matched pair (gold vs gold-2 0/11, v1 vs v2 0/12, werg-a/b/c vs their back-translations 0/11, 0/13, 0/14), 0 of 61 statements above the cut, matched cheapest-row cost median 0.1631, p90 0.2496, max 0.3477, headroom 0.2523; the CLI file path reads 0 as well. Prediction falsified in its own direction (p90 0.2496 against 0.45-0.55) and no deletion fixture was built, so the valley is unmeasured and the bar not evaluated. By-product over the 55 ibm pairs (579 statements): max 0.5774, p99 0.5707, sweep flags 0 at 0.60 through 0.75, 12 (2.07%) at 0.55, 27 (4.66%) at 0.50, 76 (13.13%) at 0.45 - the cut sits 0.023 above the largest cost this corpus produces, the flag is inert rather than over-firing. Guardrails: V 0/24 (separation +0.0666), smd and order_gap np.array_equal and both detail JSONs byte-identical under cuts patched to 0.45, 0.70, 0.75 (0.45 fires 6 semantic and 7 structural flags), triangle 0/495. NO_COUNTERPART_COST stays 0.6. Field anchors re-derived from docs/defects-docdistance.md because corrections.json is absent.
- **Verdict** - Killed-at-gate; 0 of 61 matched statements above 0.6 on all 5 fully-matched pairs, matched max 0.3477 against the 0.6 cut
- **Log**
  - log: 2026-09-02 @kj - verifier accepted with minors - matched p95 0.2572 exists only in the metrics JSON; the matched statistics are the 61 gate statements, not the 124-statement matched population the record names for the histogram (the run stopped at the gate as registered); notebook re-saved after the logged execution, markdown only - verdict unaffected

### E12-H61 Moved flag as displacement-and-inversion conjunction

- **Persona** - mechanist (modelled), dimension field, channel reporting-diff
- **Hypothesis** - because the two displacement signals fail in complementary regimes - raw rank displacement saturates under length change (DEF-TRANS-2) but is exactly 0 under any order-preserving insertion drift, while displacement_norm is length-stable under shrinkage but drifts by ~1/n per inserted statement, exceeding the fixed 0.05 band whenever n < ~19, the regime of every calibration summary (3-11 lines) - a single leading insertion into a 6-statement document will make the shipped norm-only flag (pipeline.py:551-553) fire moved on 4 of 6 untouched statements versus the intended 0, and redefining moved as (displacement != 0) AND (|displacement_norm| > 0.05) will drop that to 0 while keeping the worked-example genuine swap firing (ranks +-1, norm 0.167) and keeping the shrinkage regime quiet, while detail smd/order_gap and the plan stay bit-identical, V stays 0 and the metric property is untouched (the flag is an annotation)
- **Lever** - change the moved definition from a single-signal threshold to the conjunction of the two signals already computed per record (both in the record at pipeline.py:542,551-553; no new solve), AND update the header definitions block (pipeline.py:488-489 docstring, thresholds/definitions emission) plus docs/example-map-interpretations.md, whose moved table documents the current definition and would go stale
- **Mechanism** - a move in the ordinary sense requires both a rank change and a position change, so the conjunction is the correct estimator and each signal's defect regime is suppressed by the other: rank displacement is 0 under monotone alignments (distance.py:249-257, kind=stable) exactly where norm over-fires on insertion, and norm stays under band on shrinkage exactly where raw ranks saturated (DEF-TRANS-2: 1,077/1,086 fired)
- **Prediction** - on the 6-statement worked-example doc-a with one statement inserted at the top of B: shipped flag fires moved on 4 of 6 untouched statements (norms 6/42, 5/42, 4/42, 3/42 all > 0.05); conjunction flag fires 0 of 6; on the unmodified worked example the swap pair still fires 2; direction: false-moved count drops 4 -> 0 with zero true-moved loss
- **Acceptance bar** - PASS if the conjunction gives 0 false moved on the insertion fixture, keeps both swap statements flagged on the worked example, keeps the reversal control at 12/12, and stays quiet on all kept statements of a locally rebuilt 12->8 shrinkage synthetic (the DEF-TRANS-2 evidence shape - the original 1,086-statement archive pair is external to the repo and is not used); FAIL if any true swap/relocation in the structure-fixture reorder pool loses its flag. Guardrails: detail smd/order_gap bit-identical, plan untouched, V stays 0, metric property untouched
- **Kill-gate** - pure arithmetic plus one alignment check before any code change: the arithmetic (n-i)/(n(n+1)) > 0.05 for n=6, i<=3 is already confirmed; then verify order_alignment on the insertion fixture returns the monotone shift [1,2,3,4,5,6] - the death condition is ANY non-monotone alignment, whatever the cause (the primary risk is the LP spreading the absorber's demand across rows and flipping a row argmax, not near-duplicate tie-breaking): a non-monotone align means nonzero rank displacements and the conjunction would not suppress the false fires
- **Evidence** - pipeline.py:48-53 (MOVED_TOLERANCE rationale covers shrinkage jitter only, not insertion drift); pipeline.py:542,551-553 (displacement carried but unused by moved; moved = norm-only); distance.py:249-257 (rank displacement 0 under monotone alignments, kind=stable); docs/defects-docdistance.md:19-25 (DEF-TRANS-2: raw-index flag saturated 1,077/1,086; fix went norm-only); docs/example-map-interpretations.md:100-116 (6-statement fixture, swap ground truth)
- **Experiment** - Fixture: doc-a/doc-b of docs/example-map-interpretations.md plus a B-variant with one novel statement inserted at position 0, a locally built 12->8 shrinkage pair, and the processed structure-fixture reorder pool as the true-positive control. Entry point: structural_distance_with_details before and after the one-line flag change plus the doc/header updates. What varies: only the moved definition (norm-only vs conjunction). Score: false-moved count on insertion, true-moved retention on swaps/reorders, quietness on shrinkage; assert detail smd, order_gap and plan-derived fields byte-identical across the two definitions<br>artefacts: notebooks/experiments/E12-H61-kj-moved-conjunction.ipynb, reports/E12-H61-moved-conjunction-metrics.json, logs/E12-H61-moved-conjunction.log<br>model: opus
- **Result** - Kill-gate passed: order_alignment on the insertion fixture is the monotone shift [1,2,3,4,5,6], structure_displacement all 0, band arithmetic predicts 4 fires. Sanity check: reversal control 12/12 under the shipped definition (orthogonal synthetic and real text). Prediction held exactly: shipped flag fires moved on 4 of 6 untouched statements at norms 0.1429, 0.1190, 0.0952, 0.0714 (6/42..3/42), conjunction 0 of 6; worked-example swap pair A2/A3 2/2 under both; both 12→8 shrinkage fixtures 0/8 moved on kept statements; reorder pool 1680 permutations (1540 k>0 plus 140 reversal-anchor k=-1) with 0 of 13675 relocated statements losing their flag. Guardrails: details dict byte-identical on all 6 fixtures once the three flag-derived fields are removed, smd 0.171318 and order_gap 0.218512 exact-equal, plan-derived columns np.array_equal, V 0/24, triangle 0/990; variant compiled in-notebook from the shipped source with 3 changed lines, src/ untouched. Inherits the raw-rank regime: a length change that makes the alignment non-monotone reduces the conjunction to shipped behaviour. Shipping owes the pipeline header definitions block and the moved table of docs/example-map-interpretations.md.
- **Verdict** - Promoted; false moved on the insertion fixture 4/6 → 0/6 with 0 of 13675 true relocations lost across 1680 reorder-pool permutations
- **Log**
  - log: 2026-09-02 @kj - verifier accepted with minors - the reorder pool's 1680 permutations include 140 reversal-anchor entries (k = -1), not all k > 0; the sweep time 2.87 s is in the metrics JSON only; 'plan untouched' is measured by proxy on the plan-derived columns - verdict unaffected

### E12-H62 Order-gap length residual under deletions

- **Persona** - mechanist (modelled), dimension winners, channel order axis
- **Hypothesis** - because with n_a != n_b every coupling must split mass 1/N vs 1/M - the exact EMD does so optimally while the Sinkhorn OPW plan pays extra entropic diffuseness plus a distorted IDM/prior diagonal (the sqrt(1/N^2+1/M^2) rescale at distance.py:162 no longer aligns i/N with j/M) - and the E11 content-invariance evidence (paraphrase gap 0.0017, 0.5% of full scramble) comes exclusively from equal-length per-statement back-translations, order-preserving deletion pairs (A vs A with k statements removed, order kept) will read order_gap above the paraphrase p95 0.005 and rising with k, meaning the OPW-minus-SMD subtraction does not cancel content cleanly under length mismatch and structure_closeness silently carries a length-mismatch component, while the content axis is untouched so V stays 0 by construction, no default-path smd/order_gap changes (measurement only), and any follow-up correction stays labelled a score (H55 is already non-metric: triangle 15%, reports/E11-structure-sota-decision.json .triangle.H55)
- **Lever** - no code change - a measurement of the shipped opw_gap (distance.py:206-225) on length-mismatched but order-preserving pairs, a regime absent from every E10/E11 gate
- **Mechanism** - SMD and the OPW cost respond differently to forced mass-splitting: the exact EMD splits optimally while the row-update-terminated Sinkhorn OPW plan pays entropic diffuseness plus prior-diagonal distortion, so the difference no longer isolates order - it absorbs the shape mismatch, and the shipped clean-cancellation claim (distance.py:216) was never tested off the equal-length diagonal
- **Prediction** - gap grows monotonically in k, exceeding the paraphrase p95 0.005 by k=2 on 12-statement documents; the 0.01-0.05 magnitude at k=4 is an expectation, not a bar (asserted without derivation - the binding criterion is the 2x-p95 monotone test). The healthy null is gap <= 0.005 for all k <= 4, which certifies the shipped read for the deletion-heavy conversion-pipeline use case
- **Acceptance bar** - pre-registered dual verdict: 'not clean' confirmed if median deletion-pair gap > 2x paraphrase p95 at k >= 2 with a monotone trend (then a length-correction hypothesis is warranted, labelled a score); 'clean' confirmed if p95 holds through k = 4. Guardrails: content axis untouched so V stays 0 by construction, no default-path smd or order_gap change (measurement only), metric property claims unchanged (H55 already registered non-metric)
- **Kill-gate** - one call: opw_gap(gold, gold minus its last statement) on the structure-paraphrase fixture - requires one fixture re-encode first (no embeddings are stored in the repo), then milliseconds of solves; if it reads < 0.005 the contamination is likely negligible and the sweep is deprioritized
- **Evidence** - src/docdistance/distance.py:159-173 (_opw_plan: mid rescale at 162, row-update-terminated Sinkhorn in the docstring at 154-157), :206-225 (opw_gap, clean-cancellation claim at 216); docs/experiments/wmd-structure-distance-experiments.md:39 (paraphrase p95 0.005), :41 (E11 decision - per-statement back-translation, equal-length by construction, reword 0.0017 vs reorder 0.44); reports/E11-structure-sota-decision.json (paraphrase/scramble/shift/triangle keys); data/interim/structure-paraphrase/*.bt.json (the fixture to reuse)
- **Experiment** - Fixture: the E11 statements (exec-summaries gold plus the Wergeland article set), embeddings re-encoded once. Entry point: opw_gap and smd from src/docdistance/distance.py, no CLI. Varies: k in {1,2,3,4} deletions at head, tail, and random interior positions (order preserved), each against the intact document; report the gap distribution per k against the paraphrase distribution and one genuine reorder read (0.44 scale anchor). Directly tests the README's conversion-pipeline use case - pipelines that drop content without reordering it - where a dirty subtraction would misreport structural change<br>artefacts: notebooks/experiments/E12-H62-kj-order-gap-length-residual.ipynb, reports/E12-H62-order-gap-length-residual-metrics.json, logs/E12-H62-order-gap-length-residual.log<br>model: opus
- **Result** - Branch not clean. Kill-gate: opw_gap(gold, gold minus last statement) 0.114415, 22.9x the 0.005 deprioritization line. CPU OpenVINO INT8 re-encode reproduces the E11 anchors within 4% (paraphrase median 0.00116 vs 0.00112; full reversal 0.43015 vs the swap-40 scramble 0.41611, a scale anchor not the same statistic). Over 280 deletion pairs on 10 base documents the median gap is 0.07165 / 0.13404 / 0.12149 / 0.15528 at k=1..4, 13.4x the 0.010 not-clean bar at k=2, every pair above it (min 0.01753, max 0.28098), residual exceeding the content smd in 218/280 pairs (median gap/smd 1.677 vs 0.00695 on paraphrase). Trend clause read as Spearman rho(k, gap) 0.3973 at p 5.06e-12 (positive in 9/10 bases); pooled medians are not strictly monotone (k=3 sits 9% below k=2), only the tail family is; k=4 lands 3x above the unbarred 0.01-0.05 expectation; 12 of 280 interior rows repeat a deletion set. Practical effect: structure_closeness 0.9493 / 0.9052 / 0.9141 / 0.8902 pooled at k=1..4 and 0.8219 for tail deletions at k=3, below the 0.90 STRUCTURE_THRESHOLD, so an order-preserving content drop reads as a structural rearrangement. Guardrails: V 0/24 (margin +5.46), default-path smd 0.22343884174015136 and order_gap 0.1343654387715936 byte-identical, source sha256 unchanged.
- **Verdict** - Confirmed; branch not clean - median deletion-pair order_gap 0.13404 at k=2, 13.4x the 0.010 bar, 280 of 280 pairs above it
- **Log**
  - log: 2026-09-02 @kj - verifier accepted with minors - the 'monotone trend' clause was operationalised as Spearman rho > 0 at p < 0.05 plus a k=1 → k=2 median rise (under strict median monotonicity only the tail family passes); 12 of 280 interior rows repeat a deletion set; the 0.43015 vs 0.41611 comparison is a scale anchor, not the same statistic - verdict unaffected

### E12-H63 Semantic threshold band fragility

- **Persona** - scout (modelled), dimension calibration, channel aggregation-calibration
- **Hypothesis** - because DEFAULT_THRESHOLD 0.77 sits in an in-sample band [0.750, 0.797] whose edges are n=11 sample extremes each set by a single document (haiku defines min gold, adv2-b defines max adversarial) and the shipped OpenVINO INT8 backend drifts at emb_cos 0.9939 versus fp (reports/02-mmbert-quantization-metrics.json part_b_cpu_openvino_int8 - the shipped-artifact figure, 4x the part_a scheme-comparison 0.9984) - a perturbation on the same 10^-3 to 10^-2 closeness scale as the band's own 2.0/2.7-point margins - leave-one-out resampling plus an INT8-vs-fp backend swap on the live ibm corpus will show the verdict-safe band is single-document-fragile and backend-sensitive, while V stays 0/24 on the default INT8 path and is re-verified on both backends, no shipped number changes (probe only), and the metric property is untouched (verdict layer only)
- **Lever** - no code change - recompute the 11 summary-vs-reference closenesses on the shipped INT8 path and once on the fp backbone, then publish per-document margin to 0.77, leave-one-out band edges, backend drift per document, and the verdict-flip curve for thresholds 0.70-0.85
- **Mechanism** - order statistics of n=11: the band edges are sample extremes, the highest-variance statistic the fixture can produce, so removing one document moves an edge by construction; independently, INT8 quantization at emb_cos 0.9939 propagates to closeness shifts at the scale of the 0.020 lower margin of 0.77. Additionally: if notebook 09 cell 10's band was placed on the archived opus-4-8 corpus (summaries/@archive/, never-look rule) while the live corpus is the opus-4-6 generation, the corpus swap is itself a named fragility finding, reported explicitly rather than silently mixing generations
- **Prediction** - band half-width stays <= 2.7 closeness points; leave-one-out dropping haiku or adv2-b moves the respective edge by >= 1 closeness point; at least one document's INT8-vs-fp closeness drift exceeds 20% of its margin to 0.77; the deliverable is a fragility card (per-backend safe band with named edge documents), not a threshold change
- **Acceptance bar** - PASS = the per-document margin table, LOO edge table, backend-drift column and threshold-flip curve are published and the safe band restated with both caveats in the distance.py provenance comment's terms. Guardrails: no threshold change ships from this probe; V re-verified 0/24 on both backends and a backend showing V > 0 is named an explicit regression, never averaged away; metric property untouched (verdict layer only); default-path smd unchanged
- **Kill-gate** - re-read the recorded band before any run: if the 11 closeness margins to 0.77 were all above 5 points and backend emb_cos were 1.0 the fragility claim dies; the recorded values (min gold 0.797, max adv 0.750 - margins 2.7 and 2.0 points, distance.py:27-29) and shipped emb_cos 0.9939 already pass the gate
- **Evidence** - src/docdistance/distance.py:26-30 ('heuristic, calibrate per corpus... In-sample: the same 11 documents set it and are scored by it', band from notebook 09 cell 10 on OpenVINO INT8); reports/02-mmbert-quantization-metrics.json part_b_cpu_openvino_int8.emb_cos = 0.9939 (shipped backend; part_a 0.9984 is scheme comparison only); docs/experiments/wmd-docdistance-experiments.md:115-117 (single-fixture trust is the named weakest joint); data/interim/exec-summaries/ibm-ai-adoption/summaries/ (live opus-4-6 corpus, gold anchor exec-summary-gold-opus-4-6.md)
- **Experiment** - E12 notebook on the live ibm fixture (data/interim/exec-summaries/ibm-ai-adoption/summaries/*.md, reference exec-summary-gold-opus-4-6.md): score the 11 documents through the shipped distance-semantic path on INT8 and once on the fp backbone; tabulate closeness, verdict and margin to 0.77 per document; recompute band edges under all 11 leave-one-out subsets; sweep the threshold 0.70-0.85 at 0.005 counting verdict flips; if the recorded band traces to the archived opus-4-8 generation, report the corpus swap as a finding; ship the sensitivity card into the calibration docs<br>artefacts: notebooks/experiments/E12-H63-kj-semantic-threshold-band.ipynb, reports/E12-H63-semantic-threshold-band-metrics.json, logs/E12-H63-semantic-threshold-band.log<br>model: opus
- **Result** - Kill-gate passed on the recorded margins (haiku 2.7, adv2-b 2.0 pts, emb_cos 0.9939). Shipped INT8 path gives the band [0.7474, 0.7944], 4.71 closeness points, half-width 2.35 (fp 2.30); leave-one-out over the 10 scored documents moves the upper edge +2.30 pts when haiku is dropped and the lower edge -1.64 pts when adv2-b is dropped, width range 4.71-7.01 pts, anchor swap onto gold-2 gives [0.747, 0.813]; the fp backbone shifts both edges up (+0.92 lower, +0.82 upper) and adv2-b drifts -0.920 pts, 40.6% of its 2.26-pt margin, with 3 of 10 documents above 20% and 0 verdict flips at 0.77; zero tier errors over 0.750-0.790 on INT8 and 0.760-0.800 on fp, so 0.760-0.790 is safe on both. All four prediction clauses held. Provenance: the distance.py comment's 79.7 / 75.0 lands 0.26 pts above both recomputed edges and traces to notebook 09 cell 10's dd.embed path at 1.1.2, not the shipped semantic_distance path, which is unchanged at 1.1.5 (gold-2 closeness bit-identical); no corpus swap - the live opus-4-6 files are the recorded ones. Guardrails: V 0/24 on both backends, smd np.array_equal shipped vs probe, order_gap bit-identical, triangle 0/495 per backend, DEFAULT_THRESHOLD 0.77 unchanged. Card lives in the notebook and metrics JSON; docs/ and the provenance comment await a follow-up.
- **Verdict** - Confirmed; band 4.71 pts wide, dropping haiku moves the upper edge +2.30 pts and dropping adv2-b the lower edge -1.64 pts, INT8-to-fp drift consumes 40.6% of adv2-b's margin, recorded band 0.26 pts off the shipped path
- **Log**
  - log: 2026-09-02 @kj - one repair round - the first return attributed the 0.26-pt provenance gap to version drift; the verifier showed notebook 09 cell 13 reproduces gold-2 bit-identically at 1.1.2, so the gap is cell 10's dd.embed path, and the result was restated. Residual minors: leave-one-out covers 10 subsets plus an anchor swap, not 11; the calibration-docs shipment is a follow-up

### E12-H64 Anisotropy decomposition - centering vs projection

- **Persona** - mechanist (modelled), dimension winners, channel embedding geometry
- **Hypothesis** - because mmBERT anisotropy may concentrate in the nonzero common mean that all_but_the_top subtracts before any SVD (distance.py:104 centers, :107 projects, :108 renormalizes - three separable operations shipped as one lever), running only mean-centering plus re-L2-normalization (no PC projection) will recover >= 60% of the E01-H3 margin gain (+4.58 -> +18.48, i.e. reach >= +12.9 closeness pts) on the 11-summary fixture - making the named lever (top-1 PC removal) the vehicle and centering the cause, or confirming projection as the cause if it recovers < 30% - while V stays 0/24 in every scored arm, shipped defaults (anisotropy=False symmetric, k=1 conditioned) are untouched, no smd regression is possible (measurement only) and the metric property holds (cost stays sqrt(2-2cos) on unit vectors)
- **Lever** - decompose the shipped all_but_the_top(k=1) into three arms - (a) center+renorm only, (b) center, project out PC1, add the mean back, renorm (the clean projection-only isolate: the naive 'project uncentered' arm is confounded because the uncentered top singular vector aligns with the mean direction and partially removes it), (c) the full shipped composition - scoring each on the E01 panel and additionally reporting cos(uncentered-PC1, mean direction) for the record
- **Mechanism** - Mu & Viswanath's own decomposition: the common mean vector and the top PCs are distinct anisotropy carriers; on L2-normalized mean-pooled sentence embeddings the mean offset alone compresses cosines, so subtracting it may already de-bunch the cost matrix - the SVD projection would then be redundant apparatus riding on the centering, and dropping it would halve the lever's 0.17 ms/pair overhead
- **Prediction** - centering-only margin lands in [+12, +18] and DR >= 2x the 0.0854 baseline (direction: most of the gain survives without projection); the clean projection-only arm lands near baseline
- **Acceptance bar** - attribution stands if exactly one arm reaches >= 60% of the full-composition margin gain at V = 0/24 and metric property intact; both-arms-partial is recorded as 'effect real, mechanism composite'. Guardrails: V = 0 in every scored arm or the arm is discarded; shipped defaults untouched; no default-path smd/order_gap change (measurement only)
- **Kill-gate** - one numpy line on the re-encoded fixture (statement embeddings are unit-norm, so the probe is simply ||pooled mean|| < 0.15): if the pooled mean's norm is below 0.15 the common mean carries no meaningful anisotropy, centering cannot be the cause, and the centering arm dies before the panel run; the gate includes one fixture re-encode since no embeddings are stored on disk (seconds on CPU)
- **Evidence** - src/docdistance/distance.py:94-115 (all_but_the_top: center at 104, SVD 105, project 107, renorm 108); docs/experiments/wmd-docdistance-experiments.md:150-158 (E01-H3 record, margin +4.58 -> +18.48, d' 7.17, DR 0.2286), :197 (results table row), :209 (0.17 ms/pair SVD cost the centering arm would drop)
- **Experiment** - Fixture: the E01 11-summary set under data/interim/exec-summaries/ibm-ai-adoption, re-encoded with the shipped OpenVINO INT8 mmBERT encoder. Entry point: a variant of all_but_the_top with the projection step made optional (and the mean-restore variant for arm b), scored through smd/closeness over all pairs against the anchor. Varies: arm in {center+renorm, center-project-restore-renorm, full}; report margin, d', R, DR, V and cos(uncentered-PC1, mean) next to the E01 table. If centering wins, the shipped lever drops the SVD and the docstring's mechanism claim is corrected<br>artefacts: notebooks/experiments/E12-H64-kj-centering-vs-projection.ipynb, reports/E12-H64-centering-vs-projection-metrics.json, logs/E12-H64-centering-vs-projection.log<br>model: opus
- **Result** - Kill-gate passed: pooled mean norm of the 98 statement embeddings 0.9539 against the 0.15 death line. Sanity check on the shipped OpenVINO INT8 re-encode: naive row margin +4.71 / d' 4.68 / R 1.670 / DR 0.0873 / V 0/24 against the registered torch bf16 row +4.58 / 4.60 / 1.680 / 0.0854, max deviation 2.80% (executor-declared 10% band); full composition +19.41 / 7.31 / 1.475 / 0.2253 against +18.48 / 7.17 / 1.482 / 0.2286, 5.06% on margin. Arm a (center+renorm) margin +19.95 pts, gain +15.24 = 103.7% of the full composition's +14.71 (109.7% of the registered +13.90), DR 0.2712 = 3.11x baseline, d' 7.38; arm b (center, remove PC1, restore mean, renorm) +4.04 pts, recovery -4.5%, DR 0.0588; exactly one arm clears 60% at V 0/24 in all four geometries, attribution stands. The registered point band [+12, +18] did not hold (above it), the DR and direction clauses did. Mechanism: cos(uncentered PC1, mean direction) 0.999999, first uncentered singular value 9.44 against 1.51 second and 1.52 centered - removing the top component of an uncentered matrix is the centering. Guardrails: raw embeddings byte-identical after every arm, default-path smd/order_gap bit-identical (0x7c4a90c77548ce3f, 0x0264b939ae08c13f), unit norms to 2.4e-07, cost = sqrt(2-2cos) to 5.2e-08, zero triangle slack over 98^3 triples. Centering pushes adv1-a 1.44892 and adv2-a 1.42491 past sqrt(2) where closeness clips to 0 (the shipped composition already does so for adv2-a 1.41773); the margin-setting adversary stays below the ceiling. One fixture, one anchor, k=1.
- **Verdict** - Confirmed; center+renorm recovers 103.7% of the full composition's +14.71 pt margin gain at V 0/24 while the clean projection isolate recovers -4.5%
- **Log**
  - log: 2026-09-02 @kj - one repair round - ten arm a / arm b key numbers were not verbatim and the return was internally inconsistent; re-read from the notebook outputs, verdict unchanged. Residual minors: the 10% sanity tolerance is executor-declared; three saturation distances are quoted from the metrics JSON only; the [+12, +18] point band was missed on the high side

### E12-H65 CMR pool-size crossover

- **Persona** - mechanist (modelled), dimension winners, channel embedding geometry
- **Hypothesis** - because the removed top-1 direction's identity shifts from topic-contaminated and unstably estimated (small pool: E09-H51 bootstrap stability 0.77-0.86, below the 0.9 bar) to a stably-estimated corpus frequency direction (11-doc pool, the E01-H3 regime where d' rose 4.60 -> 7.17), sweeping the all_but_the_top estimation pool over p in {2,3,4,6,8,11} on the E01 fixture with >= 5 random subsets per pool size (error bars on any crossover claim) while always scoring the same pairs will show d' tracking the fitted direction's cosine to the 11-doc reference (Spearman >= 0.8) - either crossing the 4.60 baseline mid-sweep, or (pre-registered alternative) rising monotonically without ever dipping below baseline, which still confirms the direction-quality mechanism but pushes the E01/E09 discrepancy onto a fixture term, since the E09-H53 'hurt' endpoint (d' 2.81 -> 2.28, +1 violation) was measured on the E09 structure-fixture regime, not this fixture - while the full-pool arm reproduces E01-H3 (V = 0/24, margin ~+18.5) or the harness is declared wrong, shipped defaults (anisotropy=False single-pair) stay untouched and the metric property is unaffected (unit vectors in, sqrt(2-2cos) cost out)
- **Lever** - estimation-pool composition of all_but_the_top (which documents feed the pooled SVD), holding the scored pairs, k=1, and everything downstream fixed; supersedes nothing - E09-H51 swept depth D, E09-H50 swept per-pair vs offline, neither swept pool size on the E01 contrast fixture. Note: pool=2 is degenerate by construction - the subset IS the scored pair, so that arm is the E09-style per-pair CMR regime and the curve's left endpoint is interpreted as such
- **Mechanism** - PCA direction estimation from ~24 statements is unstable and topic-contaminated while ~130 statements average topic out and isolate the frequency direction; the gain then tracks direction quality, not the removal operation - unifying the E01 promotion and the E09 refutation as one curve
- **Prediction** - d'-vs-pool-size crosses the 4.60 baseline between pool sizes 4 and 6 (subject to the >= 5-subset error bars); direction cosine to the 11-doc reference reads < 0.9 at pool=2 and > 0.98 at pool=8+; V returns to 0/24 only above the crossover; the monotone-rising no-dip outcome is pre-registered as mechanism-confirmed-with-fixture-term
- **Acceptance bar** - confirmed if the d' curve crosses baseline within the sweep (or rises monotonically per the pre-registered alternative) AND direction-cosine correlates with d' gain at Spearman >= 0.8 across pool sizes; the deliverable is a measured minimum-pool guardrail replacing the qualitative docstring warning. Guardrails: full-pool arm must reproduce E01-H3 (V = 0/24, margin ~+18.5) or the harness is wrong; shipped defaults untouched; no default-path smd/order_gap change (measurement only); metric property unaffected
- **Kill-gate** - cosine between the 2-doc pooled top-1 direction and the 11-doc top-1 direction, over a handful of document pairs on re-encoded fixture embeddings - if uniformly > 0.95, pool composition cannot explain the E01/E09 discrepancy and the hypothesis dies before any sweep
- **Evidence** - docs/experiments/wmd-docdistance-experiments.md:157 (E01-H3: 11-summary pool, d' 7.17); docs/experiments/wmd-structure-distance-experiments.md:37 (E09-H53: 2-doc CMR, DR x3.25 but d' 2.81 -> 2.28, +1 violation - measured on the E09 structure-fixture regime), :600 (E09-H51 bootstrap stability 0.77/0.86/0.83, best D = 0), :611-618 (E09-H53 record); src/docdistance/distance.py:413-415 (anisotropy off by default with the corpus-estimation warning this sweep quantifies), :94-101 (all_but_the_top pools whatever it is handed)
- **Experiment** - Fixture: the E01 11-summary set. Entry point: all_but_the_top with the pool varied - for each p in {2,3,4,6,8,11}, >= 5 random document subsets (always containing the two scored documents), direction fit on the subset, ALL fixture pairs scored on the panel (margin, d', DR, V) plus the fitted direction's cosine to the 11-doc reference. One afternoon of CPU compute; output is the crossover curve with error bars and a concrete minimum-pool number for the anisotropy=True guidance in compute_distance and the batch --pairs path<br>artefacts: notebooks/experiments/E12-H65-kj-cmr-pool-crossover.ipynb, reports/E12-H65-cmr-pool-crossover-metrics.json, logs/E12-H65-cmr-pool-crossover.log<br>model: opus
- **Result** - Pre-registered alternative branch. Kill-gate passed: |cos| between 2-doc and 11-doc top-1 directions over all 55 pairs max 0.9369, median 0.8834, min 0.0965, 0/55 above 0.95. Sanity check on torch CPU fp32: naive row +4.61 / 4.613 / 1.679 / 0.0853 / V 0/24 against E01 +4.58 / 4.60 / 1.680 / 0.0854, full pool +18.52 / 7.161 / 1.481 / 0.2284 against E01-H3 +18.48 / 7.17 / 1.482 / 0.2286. d' rises strictly monotonically on the pool-size means 5.020 → 5.908 → 5.950 → 6.357 → 6.683 → 7.161 for p = 2, 3, 4, 6, 8, 11 with no mean below the 4.613 baseline (1 of 48 draws dips to 4.566 at p=3, sd 0.647); direction cosine to the 11-doc reference 0.8664 at p=2, 0.9285, 0.9652, 0.9834 at p=8; Spearman(cosine, d' gain) 1.000 across the six pool sizes (0.884 across all 48 draws) against the 0.80 bar - both quantities are pool-monotone, so the across-sizes figure does not separate direction quality from pool size on its own (within-size rho 0.21-0.55, n=8). V 0/24 at every pool size including p=2, so the E09-H53 hurt endpoint does not reproduce and the E01/E09 discrepancy is a fixture term. Deliverable: minimum estimation pool 8 documents, ~71 pooled statements, the first size whose mean cosine reaches 0.98 (per-pair worst 0.9045) while retaining 93.3% of the full-pool d'. Guardrails: fit/apply split bit-identical to the shipped all_but_the_top, smd and opw_gap np.array_equal after the sweep, unit norms 5.96e-08, cost identity 1.19e-07, triangle 0/4000 on the full-pool transform.
- **Verdict** - Confirmed; monotone-rising branch - d' 5.020 → 7.161 over pool sizes 2 → 11 with no mean below the 4.613 baseline, Spearman(cosine, d' gain) 1.000 across pool sizes
- **Log**
  - log: 2026-09-02 @kj - verifier accepted with minors - the across-pool-sizes Spearman 1.000 is confounded by pool size (within-size rho 0.21-0.55, n = 8), the bar applied as registered; the no-dip clause holds on the pool-size means, one draw of 48 dips; the 90%-of-full-pool clause of the minimum-pool rule is executor-chosen - verdict unaffected

### E12-H66 Norm-mass transport marginals

- **Persona** - scout (modelled), dimension digests, channel weights-marginals
- **Hypothesis** - because the pre-L2 pooled norm of a mean-pooled mmBERT statement embedding carries informativeness signal that L2 normalization discards (the statement-level analogue of Word Rotator's Distance, the steadiest improver over vanilla WMD in the Smirnov & Yamshchikov survey - and docdistance adopted WRD's cosine-cost half but not its mass half), setting each statement's transport mass proportional to its raw pre-normalization pooled norm (renormalized to a probability vector) will raise separation d' by 0.2-0.8 above the 4.60 naive baseline while holding the boundary margin at or above +4.58 closeness pts, while V stays 0/24, only the semantic solve's uniform marginals are touched (distance.py:56-58, where _ab feeds ot.emd2 - NOT distance.py:168, which is the OPW Sinkhorn's uniform a,b; the order axis stays uniform unless separately registered) and the result remains a true Wasserstein metric between the induced weighted measures (masses are per-document functions of the document's own embeddings, so d(A,A)=0 and symmetry hold by construction; d(A,A)=0 verified numerically on all 11 fixtures)
- **Lever** - replace the uniform semantic marginals with masses proportional to the pooled-embedding norms that encoders.py divides out at line 260. Registered divergence from WRD's single-space setup: masses come from the RAW pre-L2 pooled vectors while the cost geometry is post-anisotropy where enabled - the shipped all_but_the_top (distance.py:94-115) re-L2-normalizes, so post-anisotropy norms are identically 1 and cannot carry mass; this raw-mass/whitened-cost split is part of the record. Ground cost sqrt(2-2cos) and the exact EMD solve untouched
- **Mechanism** - WRD (Yokoi et al. 2020, via the Moving Other Way digest): cosine ground cost plus norm-scaled mass is the survey's most reliable WMD improvement because embedding norm tracks a unit's discriminative content. This is a different mechanism from the refuted E01-H1/H2 (IDF x numeric salience - hand-picked feature up-weighting that broke ordering): here the encoder's own geometry supplies the weight, no corpus statistics, no feature engineering
- **Prediction** - small consistent gain matching the survey's word-level finding: d' up 0.2-0.8 over 4.60, margin unchanged to +1 pt, DR roughly flat; latency unchanged (weights are free, per the E01-H1 latency row at 0.07 ms/pair)
- **Acceptance bar** - PASS iff V = 0/24 AND boundary margin >= +4.58 AND d' >= 4.60 AND default-path smd on the identity pair stays 0 AND the order_gap default path is unchanged (weights feed only the semantic solve); any V > 0 or margin drop is FAIL regardless of d'. Metric guardrail: d(A,A)=0 and symmetry by construction, triangle inequality holds between the induced measures - verified numerically on all 11 fixtures. The corr(norm, token count) figure is reported in the result table regardless of pass, so a 0.7-0.8-correlated weight cannot ship unflagged
- **Kill-gate** - one encode pass over the 11-summary fixture, zero OT solves: if the coefficient of variation of the pre-L2 pooled norms is < 0.03 the mass vector is effectively uniform and the lever is dead; if corr(norm, token count) > 0.8 (tightened from 0.9 - mean-pooled contextual norms are known to track length) the weight is statement length in disguise and the candidate is killed as an unregistered length-weighting scheme
- **Evidence** - src/docdistance/encoders.py:258-260 (pooled norm divided out and discarded); src/docdistance/distance.py:48-58 (_uniform marginals; _ab into ot.emd2 at 56-58 is the semantic call site; distance.py:168 is the OPW solve, not this lever's target); references/papers/[paper digest] Moving Other Way - Exploring Word Mover Distance Extensions.md:25,31 (WRD: mass scaled by embedding norm, the steadiest improver), :42 (the digest names frequency-aware weighting as the open axis); docs/experiments/wmd-docdistance-experiments.md:131-148 (E01-H1/H2 refutations this supersedes in territory but not in mechanism)
- **Experiment** - One toggle over the shipped default on the 11-summary ibm-ai-adoption fixture. Kill gate runs first as a single encode pass. Then: return pre-L2 norms from encode() (or recompute pooled norms in a probe cell), build w_i = norm_i / sum(norms) per document, pass as marginals to ot.emd2 in place of _uniform at the distance.py:56-58 site. Score the full panel (margin, R, DR, d', V, latency) against the naive baseline row; run the identity-pair and triangle-inequality checks from tests/test_distance.py. If the semantic axis passes, a second registered variant may feed the same masses to the OPW order axis; unregistered, the order axis stays uniform<br>artefacts: notebooks/experiments/E12-H66-kj-norm-mass-marginals.ipynb, reports/E12-H66-norm-mass-marginals-metrics.json, logs/E12-H66-norm-mass-marginals.log<br>model: opus
- **Result** - Kill-gate passed on the pooled reading: pre-L2 norm CV 0.0446 (death below 0.03), corr(norm, token count) Pearson -0.1597, Spearman -0.1033 (death above 0.8), one encode pass over 98 statements. Sanity check on OpenVINO INT8: naive row +4.71 / 4.68 / 1.670 / 0.0873 / V 0/24 against the registered +4.58 / 4.60 / 1.680 / 0.0854, max deviation 2.80% inside the 5% band declared before scoring. Norm-mass marginals move nothing: d' 4.684 → 4.673 (delta -0.011 against a predicted +0.2 to +0.8), margin +4.708 → +4.720 pts, R 1.6697 → 1.6679, DR 0.08728 → 0.08720, largest per-document distance change 0.00148, ranking unchanged, latency 0.877x on a shared CPU (not a measurement). The bar's absolute clauses read met (V 0/24, margin >= +4.58, d' >= 4.60, identity smd 0.0, order_gap bit-identical) but the co-measured uniform baseline already clears them, so the bar does not discriminate and the verdict follows the delta. Cause: the mass vector is a within-document object and the per-document norm CV runs 0.0003-0.0378 (median 0.0198), giving mass vectors 0.0001-0.0141 in total variation from uniform, heaviest statement 1.084x its uniform share, with two adversarial documents at 1 and 3 statements uniform by construction; mmBERT norms mean 30.17, sd 1.35. The premise survives - norm is not length in disguise - the magnitude does not. Guardrails: smd and order_gap np.array_equal before and after, identity 0.0 on 11 fixtures, symmetry 2.2e-16, triangle 0/495, src/ untouched. Second stage (OPW masses) not run, as conditioned.
- **Verdict** - Refuted (null); d' 4.684 → 4.673 (delta -0.011) against a predicted +0.2 to +0.8, margin delta +0.012 pts - bar met but non-discriminating
- **Log**
  - log: 2026-09-02 @kj - verifier accepted with minors - the label follows the null rule over the primer's toggle rule because the bar's absolute thresholds are the E01 baseline row's own numbers; the 5% sanity band and the null band (|delta d'| < 0.05, |delta margin| < 0.1 pts) are executor-declared; notebook re-saved after execution, markdown only - verdict unaffected

### E12-H67 Batch embed memoisation for --pairs sweeps

- **Persona** - scout (modelled), dimension runtime, channel runtime
- **Hypothesis** - because a --pairs batch re-segments and re-encodes every document on every row while realistic sweeps repeat documents across rows (110 embed calls for the 55-row all-pairs sweep of the 11 summaries; one source recurring in every row of a 1-source-x-N file) and encoding dominates pair cost (~30.8 ms/statement versus 0.08 ms exact SMD, docs/solution/wmd-docdistance-solution-sota.md:130,172), a per-DocDistance memo keyed by (resolved path, mtime, segmentation kwargs INCLUDING the fragment-filter kwargs, since drop rules change the embedded set) caching the RAW pre-anisotropy (statements, embeddings, n_dropped) tuple will cut embed calls 110 -> 11 and batch wall time >= 3x on the fixture sweep versus the shipped per-row build, while every emitted JSONL byte and per-pair details file is identical to the uncached run - which makes the V = 0 guardrail, default-path smd/order_gap preservation and the metric property hold trivially, since no number changes anywhere (pure caching)
- **Lever** - an embedding memo inside DocDistance.embed_statements (pipeline.py:667-705) scoped to the instance, so the CLI batch loop (_run_batch, cli.py:193-208) hits it automatically; one-shot convenience functions construct a fresh instance and are untouched. The cache boundary is pinned pre-anisotropy: all_but_the_top operates jointly over both documents of a pair (distance.py:100), so caching anything post-anisotropy would silently change scores and fail the set-diff. The memo restores n_dropped on every hit (embed_statements writes self._last_n_dropped, and _run_batch's extra() merges per-pair drop counts into each JSONL line - a hit skipping that write breaks byte-identity on filtered batches). Raw-text inputs bypass the cache - the bypass is load-bearing for the mtime key's correctness; the cache engages only when the input resolves to an existing file. Scope: the memo does not reach distance-wrt-source, whose reranker/NLI work is per-pair
- **Mechanism** - amortisation of the dominant cost: --pairs today amortises only the model load (cli.py:318); each row still pays full SaT segmentation plus the OpenVINO INT8 forward (encoders.py:246-267), a deterministic function of the file content, so duplicate documents make that work pure recomputation - the Rel-WMD/Rel-RWMD discipline of caching the expensive reusable part, since the LP at 0.08 ms/pair is not the bottleneck
- **Prediction** - on the 55-row all-pairs sweep: embed calls drop 110 -> 11 (90% skipped), encode-component wall time drops ~5-10x, total batch time approaches the LP + JSON-IO floor; on a disjoint-pairs file the overhead is one dict lookup per pair, < 1% regression; JSONL byte-identical throughout
- **Acceptance bar** - PASS: byte-identical JSONL and per-pair --details-json files versus the uncached run on the same TSV (the empty set-diff IS the guardrail proof - V stays 0, smd/order_gap regressions impossible by construction, metric untouched) AND >= 3x wall-time reduction on the 55-row fixture sweep AND < 2% regression on a worst-case disjoint-pairs TSV (zero hits). FAIL: any output byte differs, or the disjoint-pairs overhead exceeds 2%
- **Kill-gate** - two gates before building: (a) duplicate-path count on the intended sweep TSVs (cut -f1,2 | tr '\t' '\n' | sort | uniq -c) - if deployment sweeps are disjoint pairs (unique/total >= 0.9) the hit rate is ~0 and the memo is dead; the fixture all-pairs sweep passes by construction (11 unique in 110 slots). (b) stage-split timing of one distance-semantic call (two timestamps around existing stages, no build): if segment+encode of a single document is < 30% of total pair latency the achievable amortization is capped below 1.3x and the candidate dies before any cache is written
- **Evidence** - cli.py:193-208 (_run_batch calls dist(a,b) fresh per row); cli.py:318, 352-357 (--pairs amortises only the model load); pipeline.py:667-705 (embed_statements, self._last_n_dropped), 743-747 (semantic_distance re-embeds per call), 629-640 (DocDistance holds no artefact cache); encoders.py:246-267 (full tokenize + INT8 forward per call); distance.py:100 (anisotropy is pair-joint - fixes the cache boundary); docs/solution/wmd-docdistance-solution-sota.md:130,172 (encode-dominated pair cost); references/papers/[paper digest] Speeding up Word Mover's Distance and its variants via properties of distances between embeddings.md:13,27,38 (cache-the-reusable-part discipline)
- **Experiment** - Fixture: the 11-summary set as an all-pairs TSV (55 rows) plus a 1-source-x-N TSV (N in {2,5,11}) and a disjoint-pairs control built from data/external documents. Entry point: docdistance distance-semantic --pairs twice - shipped build vs memo - after the two kill gates. Varies: memo on/off, sweep shape, markdown segmentation and fragment filters on/off (kwargs are part of the key; filtered runs exercise the n_dropped restoration). Measured: wall time per batch, instrumented embed-call count, byte-diff of JSONL streams and detail files. Repeat for distance-structural to confirm order_gap lines are byte-identical; update the CPU card's latency row<br>artefacts: notebooks/experiments/E12-H67-kj-pairs-embed-memo.ipynb, reports/E12-H67-pairs-embed-memo-metrics.json, logs/E12-H67-pairs-embed-memo.log<br>model: opus
- **Result** - Kill-gates passed before any build: (a) worst intended-sweep unique/total 0.750 (1-source-x2), all-pairs 0.100, both under the 0.9 death line; (b) segment+encode of one document 107.9 ms median against 198.7 ms pair latency = 54.3% share, above the 30% death line. Cross-check on the E01 row: V 0/24, margin +4.708, d' 4.684, R 1.6697, DR 0.08728 within 2.80% of the registered values. Memo on the 55-row all-pairs sweep: real segment-and-encode calls 110 → 11 (99 hits), median batch wall time 11.032 s → 1.110 s = 9.94x (10.03x min-of-3) against the 3x bar; structural 9.47x, filtered 9.84x, markdown 9.95x, details 9.55x; 1-source-x-N 1.81x / 3.24x / 5.18x at N = 2 / 5 / 11; end-to-end CliRunner including model load 1.98x (one pair of runs). Disjoint-pairs control (110 generated documents from the fixture's 30 paragraphs, zero hits) 17.856 s → 17.745 s, -0.63% against the 2% ceiling, key overhead 17.5 ms over 110 slots. Byte-identity: 9 of 9 arms produce sha256-equal JSONL, 55 per-pair details files equal on all 3 repeats, distance-semantic --pairs CLI sha256 equal under both arms; 55 smd and 55 order_gap values np.array_equal; memo-served embeddings np.array_equal to fresh; identity 0.0, symmetry 2.2e-16, triangle 0/495; 5 memo properties unit-tested (mtime invalidation, kwargs keys, _last_n_dropped restored on hits, raw-text bypass). 56 timed runs at 1-minute load 5.97-7.97, 0 ceiling breaches. Cache unbounded and unmeasured, needs a size bound to ship; does not reach distance-wrt-source.
- **Verdict** - Promoted; 9.94x median batch wall time on the 55-row all-pairs sweep (11.032 s → 1.110 s, bar >= 3x), encodes 110 → 11, 9/9 arms byte-identical, disjoint control -0.63%
- **Log**
  - log: 2026-09-02 @kj - verifier accepted with minors - the result key embed_calls_on 11 means real embeds (the metrics JSON's real_embeds_on; its embed_calls_on is 110); the disjoint control is a generated corpus, not data/external; the CliRunner arithmetic implies a ~9.4 s model load, not 13.5 s; kill-gate (b)'s two timing loops disagree (198.7 ms per pair vs 2 x 107.9 ms) - verdict unaffected

### E12-H68 Shared-source grounding-grid memoisation

- **Persona** - scout (modelled), dimension runtime, channel runtime
- **Hypothesis** - because the grounding grid R(doc, S) and entailment vector are pure functions of (doc, source) (_grounding at pipeline.py:719-729 takes exactly doc_sents, src_sents) and the reranker full grid is ~99% of the ~67 s/pair CPU INT8 cost (docs/solution/wmd-source-conditioned-docdistance-solution-sota.md:86), adding --pairs batch support to distance-wrt-source (which today offers only positional a, b with a single --source, cli.py:593-684) with a per-(doc, source) memo over (statements, embeddings, n_dropped, R, entail) will cut an 11-candidate all-pairs sweep's grounding work from 110 grid evaluations to 11 and end-to-end wall time >= 5x versus a warm-instance sequential loop over the same DocDistance (the fair baseline - model-load amortisation versus 55 cold CLI invocations is measured and reported separately, never folded into the claimed win), while every pair's JSON is byte-identical to the one-shot outputs and the symmetric default path's smd/order_gap are untouched (grounding mode never enters them - V stays 0 and the metric property is preserved, since no scoring formula changes anywhere)
- **Lever** - two coupled additions: (1) --pairs TSV support on distance-wrt-source mirroring cli.py:352-357/486-491, with --source fixed for the batch; (2) a per-DocDistance memo over (doc identity, source identity) for the embed_statements output and the _grounding output, so pipeline.py:851-861 (which re-embeds a, b, source and re-runs both grids per call) reuses each document's grid across the sweep. The memo restores the per-document n_dropped counts recorded at pipeline.py:854-859, or filtered-run outputs diverge
- **Mechanism** - amortisation of pair-independent work: compute_source_conditioned consumes reranker_a/entail_a and reranker_b/entail_b that are functions of one document against the source only (pipeline.py:862-871); in an N-candidate round-robin every document's grid is recomputed N-1 times, so deduplicating converts a quadratic-in-N reranker bill (n_doc x n_src cross-encoder pairs per grid, encoders.py:331-336) into a linear one; the source embedding and each NLI vector amortise the same way
- **Prediction** - 55-pair all-pairs sweep of the 11 summaries against one source: grounding grids 110 -> 11, end-to-end 6-8x versus the warm sequential loop (~61 min of pairwise work dropping toward 11 grids + 12 embeds + 55 cheap transport/aggregation steps); per-pair JSON identical to one-shot outputs
- **Acceptance bar** - PASS: each of the 55 batch JSON lines byte-identical to the corresponding one-shot distance-wrt-source --json output, AND >= 5x wall-time reduction versus the warm-instance sequential baseline, AND a semantic/structural fixture rerun confirming the default path's smd/order_gap byte-identical (V stays 0, metric property preserved - the grounding mode never touches them). FAIL: any per-pair number differs, or end-to-end gain < 3x. Pre-merge sanity check (not a build gate): two instrumented runs on (A,B|S) and (A,C|S) diffing the two R(A,S) grids - nondeterminism cannot occur by construction (same single score_grid call per (doc, source) in both paths), but the check documents it
- **Kill-gate** - count repeated (doc, source) combinations in the intended sweep specifications before any build: if production sweeps are one pair per source (no candidate fanout) the hit rate is 0 and only the model-load amortisation of --pairs survives (~seconds against a 67 s pair - dead)
- **Evidence** - cli.py:593-684 (distance-wrt-source has no --pairs; batch exists only for distance-semantic at cli.py:352-357 and distance-structural at cli.py:486-491); pipeline.py:719-729 (_grounding pure in doc + source statements); pipeline.py:851-871 (per-call re-embedding and grid recomputation; n_dropped recording at 854-859); encoders.py:331-336 (score_grid builds the full cross-encoder pair list); docs/solution/wmd-source-conditioned-docdistance-solution-sota.md:86 (~67 s/pair CPU INT8, ~99% reranker grid)
- **Experiment** - Fixture: the 11 summaries plus their common source from the E02-E04 conditioned experiments. Baseline: a warm-instance sequential loop of 55 distance_wrt_source calls on one DocDistance (cold-CLI 55-invocation timing reported separately as the load-amortisation line). Then the batch mode with the (doc, source) memo. Varies: memo on/off, sweep shape (all-pairs fanout vs one-pair-per-source control), fragment filters on/off (exercises n_dropped restoration). Measured: end-to-end wall time, instrumented grounding-grid count, byte-diff of all 55 JSONs against one-shot outputs, and the semantic/structural default-path byte-identity rerun<br>artefacts: notebooks/experiments/E12-H68-kj-grounding-grid-memo.ipynb, reports/E12-H68-grounding-grid-memo-metrics.json, logs/E12-H68-grounding-grid-memo.log<br>model: opus
- **Result** - Kill-gate passed on the campaign's registered sweep specifications (no production specification exists, so the gate cannot fail on the headline row): the 55-pair fixture occupies 110 (doc, source) slots over 11 distinct, 90.0% repeated; intended shapes 25.0-90.0%; the two zero-repeat controls read 0%. Sanity check: R(A,S) and entail(A,S) np.array_equal across (A,B|S) and (A,C|S), max |diff| 0.0. Memo on the 55-pair all-pairs sweep of the 11 ibm summaries against one source: grounding grids 110 → 11 (99 hits), document embeds 165 → 12, wall time 2671.72 s → 266.32 s = 10.03x against the 5x bar, above the predicted 6-8x because the baseline measured 44.5 min against the prediction's ~61; median per pair 50.327 s → 0.112 s. Anchor-fanout shape 1.99x (baseline derived from arm A per-pair times); fragment-filter arm 3.02x; no-fanout control 0 hits, +0.22%, key overhead 6.5 ms. Byte-identity: 73 of 73 pair payloads across four arms equal to the shipped in-process distance_wrt_source output, 10 of 10 pairs re-scored by the shipped distance-wrt-source --json in fresh subprocesses stdout-identical covering all 11 documents; V 0/24 on the D_sel anchor panel from both arms, 55 d_sel np.array_equal; default-path rerun 55 smd and 55 order_gap np.array_equal with identity 0.0, symmetry 2.2e-16, triangle 0/495 identical off and on, D_sel triangle 0/495. Load amortisation reported separately: 55 cold CLI invocations extrapolate to 4295 s against 2698 s warm, 1.59x from the load alone, memo 9.21x on top. Single interleaved off/on pass, 3 of 11 between-sweep load readings above 8.0 fell on the memo-on side. Cache unbounded, arrays returned by reference; --pairs on distance-wrt-source still to build.
- **Verdict** - Promoted; 10.03x wall time on the 55-pair wrt-source sweep (2671.72 s → 266.32 s, bar >= 5x), grids 110 → 11, 55/55 payloads byte-identical
- **Log**
  - log: 2026-09-02 @kj - one repair round - run 3 (10.06x) returned five numbers absent from its outputs and a result JSON differing from the persisted one; the repair re-executed the notebook end to end (run 4, 59 min) and the recorded numbers are run 4's (10.03x). Residual minors: the kill-gate counts the campaign's own fixtures and cannot fail on the headline row; the --pairs half is unbuilt, so the 55-line clause was scored on in-process payloads (10 of 55 through the shipped CLI); a single interleaved pass with 3 of 11 load readings above 8.0, all on the memo-on side

### E12-H69 Relevance gate form (research lean)

- **Persona** - scout (lean), dimension winners, channel grounding cascade
- **Hypothesis** - because the E03-H11 kill-gate probe recorded the two intruding golds at max reranker relevance 0.898 and 0.752 - far above the 0.6 probe bar - while fabrication is defined by low relevance, the evidence leans toward the H11 win being the reranker's relevance separation itself rather than the soft (1 - max r) functional form: a hard gate (statements with max r >= tau excluded from the ungrounded mass) would likely reproduce intrusions 2 -> 0, making the soft weighting the vehicle - testable through the grounding cascade channel once D_grd is re-derived on the current embeddings, the missing piece the log's own WARNING names, since the shipped chain now reads 6 gold intrusions against H11's recorded 2 -> 0, the persisted E03 notebook execution postdates the doc record on different embeddings (its stored run shows zero intruders pre-gate and 4 post-gate, contradicting both the doc's 2 -> 0 and E06's shipped 6), and per-statement max-r was never persisted anywhere - so any decomposition scored today would score against a failing reference, and as a lean this claims no result
- **Lever** - residual functional form: hard threshold gate versus the shipped soft (1 - max_k r(a_i, s_k)) weight in grounding_residual, swept over tau - runnable only after the D_grd re-derivation, inside which the gold-vs-Set2 per-statement max-r overlap measurement comes free (the standalone phase-1 notebook read is deleted: cell 16 prints only per-intruder means, the stored run had zero intruders so it printed nothing, and docs[l]['maxR'] was a runtime variable, never persisted)
- **Mechanism** - if gold and fabricated statements are separable by max source relevance alone (bimodal max-r), any monotone gate closes the intrusion and the specific soft form is incidental; if they overlap, the soft weighting's partial credit is load-bearing - the discriminating measurement is the overlap of the per-statement max-r distributions of gold vs Set 2, which has never been persisted or reported
- **Prediction** - leans toward 'gate mechanism real, functional form incidental'; no specific tau prediction is registered until the re-derived max-r distributions exist (the premise numbers 0.898/0.752 and the 0.6 bar come from a run whose reference is now known-failing and may not survive re-derivation) - only the form-agreement bar stands; a secondary lean is that the re-derivation itself resolves part of the 6-intrusion contradiction, since every E04+ gate inherited a failing reference
- **Acceptance bar** - as a lean this claims no result; the eventual modelled test passes if hard and soft gates agree on intrusion count within 1 and Set 2 tier mean within 10% (form incidental) or diverge by more (soft form load-bearing). Guardrails: selection axis and V = 0 untouched (D_sel is not modified), symmetric-path smd/order_gap unchanged, and D_grd remains a score, not a metric, as shipped
- **Kill-gate** - no cheap standalone gate exists (the per-statement max-r artefact gap is itself part of the evidence); the gate lives inside the D_grd re-derivation: if Set 2's re-derived max-r mass sits above 0.6 alongside the golds', no threshold separates them, the hard-gate variant dies and the soft weighting is confirmed load-bearing by a set-diff alone
- **Evidence** - docs/experiments/wmd-docdistance-experiments.md:300-308 (WARNING: shipped chain reads 6 gold intrusions and Set1>Set2 per reports/E06-trained-scorers-cpu-metrics.json shipped_intrusions, contradicting the H11 headline; 'resolving it needs D_grd re-derived on the current embeddings'), :310-319 (H11 record: the (1 - max r) lever, kill-gate max r 0.898/0.752, thin v2 margin 0.216 vs 0.220); notebooks/experiments/E03-kj-source-conditioned-improvements.ipynb (stored run: zero intruders pre-gate, 4 post-gate - a three-way inconsistency with the doc's 2 -> 0 and E06's 6, direct proof the premise numbers may not survive re-derivation; executed after the doc record on different embeddings); reports/ listing (no E03 metrics JSON - the per-statement r artefact gap)
- **Experiment** - Single phase, gated on the re-derivation the WARNING already demands: re-run the reranker chain on the current embeddings over the ibm-ai-adoption fixture (12 docs x 70 source statements, minutes on the CPU INT8 chain), persist per-statement max-r by tier this time, report the gold-vs-Set2 overlap, then score three arms - no gate (R2), shipped soft gate, hard gate swept over tau - on intrusions, tier means, and Set1/Set2 severity ordering. The healthy outcome either way repairs the failing reference every E04/E05/E06 gate currently leans on<br>artefacts: notebooks/experiments/E12-H69-kj-relevance-gate-form.ipynb, reports/E12-H69-relevance-gate-form-metrics.json, logs/E12-H69-relevance-gate-form.log<br>model: opus
- **Result** - Registered as a lean; its single-phase experiment was run and the lean's premise is refuted by its own kill-gate. Re-derived per-statement max reranker relevance on the current INT8 chain (70 source statements; gold 77, Set 1 17, Set 2 4 statements, grid 339.7 s on first build, cached at data/interim/E12-H69-grounding-signals.npz): Set 2 share >= 0.6 is 1.000 (4/4, mean 0.996, min 0.993) alongside gold 0.961 (74/77, mean 0.942, median 0.995), Set 1 0.529; gold-vs-Set-2 overlap coefficient 0.818, Mann-Whitney AUC 0.429, best single-threshold balanced accuracy 0.714 at max r 0.9925 - no threshold separates fabrication from faithful compression, the hard-gate variant dies. Set-diff: ungated R2 reads 0 gold intrusions with correct Set2>Set1 severity (tier means 0.2137 / 0.2716 / 0.6319), the shipped soft gate reads 6 intrusions with inverted Set1>Set2 (0.043 / 0.291 / 0.0029, reproducing reports/E06-trained-scorers-cpu-metrics.json exactly), the hard gate at tau* 0.9984 reads 0 intrusions and Set2>Set1 (0.6376) only because it excludes almost nothing; 0 of 28 tau points meet both agreement clauses (intrusion delta 6 against 1, Set 2 mean delta 21772% against 10%). The soft form is load-bearing in the wrong direction: the reranker saturates near 1.0 on fabricated statements reusing source vocabulary, and the E03-H11 headline 2 → 0 is 0 → 6 on the current models; the E03 notebook's 0 → 4 scored a two-component signature, not the shipped scalar. Set 2 rests on n = 4 statements from 2 documents. Guardrails: V 0/24, smd and order_gap np.array_equal over the 10 anchor pairs against a pre-arm snapshot, D_sel untouched, D_grd still a score.
- **Verdict** - Killed-at-gate; Set 2 max-r share >= 0.6 is 1.000 (4/4, min 0.993) alongside gold 0.961 (74/77) - lean premise refuted, the soft form is load-bearing and inverts severity (6 intrusions, Set1>Set2) where the ungated axis reads 0 intrusions and Set2>Set1
- **Log**
  - log: 2026-09-02 @kj - verifier accepted with minors - registered as a lean, the single-phase experiment was executed and the verdict is that test's; the result's acceptance field reads not-met where the dual-branch bar resolved to divergence (superseded by the kill); reranker_grid_seconds 339.7 is the first build, the logged run restored from cache; gold_intrusions counts the anchor among the 7 golds in the tau sweep rows

### Pre-registration table (E12)

| hypothesis | lever | prediction | acceptance bar | kill-gate |
|---|---|---|---|---|
| E12-H59 structure threshold transfer | verdict calibration | 0.90 mis-sides wergeland rung-2 (0.908); safe band >= 0.908 or empty | non-empty cross-article band + recalibration re-verified, or empty-band -> per-corpus guidance | already passed - flip confirmed on recorded E11 JSON |
| E12-H60 no-counterpart valley | verdict calibration | 3-10% matched statements falsely cross 0.6; valley at 0.65-0.75 | >2% matched above 0.6 AND <5% removed below it -> move cut; <2% -> closed | cost_bands['>0.6'] on fully-matched pairs' existing detail JSON |
| E12-H61 moved conjunction | reporting-diff | insertion fixture: shipped flags 4/6 untouched as moved, conjunction 0/6 | 0 false moved on insertion, swaps kept, reversal 12/12, deletions quiet | arithmetic confirmed; verify order_alignment on insertion |
| E12-H62 order-gap length residual | order axis | deletion gap > 2x paraphrase p95 by k=2, monotone in k | dual verdict: not-clean (monotone > 2x p95) or clean (p95 holds) - both pre-registered | one opw_gap(gold, gold-minus-last) call after re-encode |
| E12-H63 semantic threshold band | verdict calibration | LOO moves a band edge >= 1 pt; INT8 drift > 20% of some margin | margin/LOO/drift tables published, band restated with caveats | recorded margins already thin - re-read before run |
| E12-H64 centering vs projection | embedding geometry | centering-only keeps >= 60% of +18.48 margin; projection-only near baseline | exactly one arm >= 60% of full gain at V=0 -> attribution; both partial -> composite | pooled-mean norm >= 0.15 (one numpy line) |
| E12-H65 cmr pool crossover | embedding geometry | d' crosses 4.60 baseline at pool 4-6; direction cosine tracks gain | crossover inside sweep AND Spearman(cosine, gain) >= 0.8 | 2-doc vs 11-doc top-1 direction cosine; > 0.95 everywhere kills |
| E12-H66 norm-mass marginals | weights-marginals | d' +0.2-0.8 over 4.60, margin held, latency free | V=0 AND margin >= +4.58 AND d' >= 4.60 AND identity smd 0 | pre-L2 pooled-norm CV < 0.03 kills (one encode pass) |
| E12-H67 pairs-embed-memo | runtime | all-pairs: 110 -> 11 encodes, ~5-10x encode wall-time | byte-identical JSONL + details vs uncached AND >= 3x on all-pairs | duplicate-path rate on intended TSVs |
| E12-H68 shared-source grounding memo | runtime | grids 110 -> 11, >= 5x end-to-end wrt-source sweep | byte-identical per-pair JSON AND >= 5x vs warm sequential | repeated (doc, source) count in sweep specs |
| E12-H69 relevance gate form | grounding cascade | LEAN: gate mechanism real, soft form incidental | lean only; modelled test later: hard vs soft agree within 1 intrusion / 10% tier mean | none cheap - lives inside the D_grd re-derivation |

Dropped at audit or synthesis, recorded: transport-map refinement directives (no injected-edit manifest exists - precondition absent), single-solve structural details (own gate: encode dominance caps the saving below 1%), wergeland summary corpus (density gate: 1 numeric token vs ~83 in the ibm article), single-lp align reuse / no-counterpart cost gap / batch shared-document cache (merged into E12-H60 and E12-H67).

### Results table (E12)

Executed 2026-09-01/02 as workflow `wf_08ea9581-df7` - one opus executor per hypothesis (H59-H66 and H69 in parallel, H67 then H68 alone for the timed arms), a methodologist verifier per hypothesis with one repair round (H63, H64, H68 repaired, verdicts unchanged), synthesis into the rows below. Tally: 5 Confirmed, 3 Promoted, 1 Refuted (null), 2 Killed-at-gate; V 0/24 and default-path smd / order_gap bit-identical on every run; nothing shipped changed.

| hypothesis | target KPI | key result | bar | verdict |
|---|---|---|---|---|
| E12-H59 structure threshold transfer | calibration (structural verdict) | 0.90 mis-sides 1/12 rungs, 0/72 un-reordered flagged; ibm band (0.8928, 0.9061], wergeland (0.9080, 0.9737], disjoint by 0.0019; 0/29 grid points cross-safe | non-empty band excluding 0.90, or demonstrated empty band | Confirmed (branch b, empty band) |
| E12-H60 no-counterpart valley | reporting fidelity | cost_bands['>0.6'] 0 on all 5 fully-matched pairs (0/61), matched max 0.3477; ibm pool max 0.5774, 0 flagged at 0.60-0.75 | >2% matched above 0.6 to move the cut | Killed-at-gate |
| E12-H61 moved conjunction | reporting fidelity | insertion 4/6 → 0/6 false moved, swap 2/2, reversal 12/12, shrinkage 0/8, 0/13675 relocations lost | 0 false moved, no true move lost | Promoted |
| E12-H62 order-gap length residual | order axis | median gap 0.07165 / 0.13404 / 0.12149 / 0.15528 at k=1..4, 280/280 above 0.010, structure_closeness 0.8902 pooled at k=4 | not-clean: > 2x paraphrase p95 by k=2, positive trend | Confirmed (not clean) |
| E12-H63 semantic threshold band | calibration (semantic verdict) | INT8 band [0.7474, 0.7944], fp [0.7566, 0.8027]; LOO haiku +2.30 pts, adv2-b −1.64 pts; drift 40.6% of a margin, 0 flips; safe on both 0.760-0.790 | LOO edge move ≥ 1 pt, drift > 20% of a margin, card published | Confirmed |
| E12-H64 centering vs projection | resolution | arm a +19.95 pts (103.7% of +14.71), DR 0.2712; arm b +4.04 pts (−4.5%); cos(PC1, mean) 0.999999 | exactly one arm ≥ 60% of the gain at V 0 | Confirmed |
| E12-H65 cmr pool crossover | resolution | d' 5.020 → 7.161 over pools 2 → 11, no mean below 4.613; cosine 0.8664 at 2, 0.9834 at 8; Spearman 1.000; min pool 8 docs | crossover, or monotone rise (pre-registered alternative), Spearman ≥ 0.8 | Confirmed (monotone branch) |
| E12-H66 norm-mass marginals | resolution | d' 4.684 → 4.673, margin +4.708 → +4.720 pts, max distance change 0.00148; within-document norm CV 0.0198 | V 0, margin ≥ +4.58, d' ≥ 4.60 (met by the baseline itself) | Refuted (null) |
| E12-H67 pairs-embed-memo | batch throughput | 55-row sweep 11.032 s → 1.110 s (9.94x), encodes 110 → 11, 9/9 arms byte-identical, disjoint control −0.63% | byte-identical AND ≥ 3x | Promoted |
| E12-H68 grounding-grid memo | batch throughput | 55-pair wrt-source sweep 2671.72 s → 266.32 s (10.03x), grids 110 → 11, 73/73 payloads byte-identical, no-fanout +0.22% | byte-identical AND ≥ 5x | Promoted |
| E12-H69 relevance gate form | correctness (gold intrusions) | Set 2 max-r 4/4 above 0.6 beside gold 74/77 (overlap 0.818); ungated 0 intrusions Set2>Set1, shipped soft gate 6 intrusions Set1>Set2 | hard and soft gates agree within 1 intrusion / 10% tier mean | Killed-at-gate |

### Calibration readings (E12)

The verdict layer after the batch - no constant moved, every reading is the shipped value's measured standing.

- **STRUCTURE_THRESHOLD 0.90** - an ibm constant: ibm safe band (0.892817, 0.906144], wergeland (0.908048, 0.973676], cross-article band empty by 0.001904; 0.90 keeps 0.006144 of headroom on ibm and mis-sides wergeland rung-2 (E12-H59). Structural closeness also falls below 0.90 under order-preserving deletions - 0.8902 pooled at k=4, 0.8219 for tail deletions at k=3 - so the structural verdict fires on content drops (E12-H62). Guidance: per-corpus structural calibration; a third corpus before any cross-article proposal
- **DEFAULT_THRESHOLD 0.77** - shipped-path band [0.7474, 0.7944] on INT8, [0.7566, 0.8027] on fp, safe on both over 0.760-0.790; the edges are single documents (haiku +2.30 pts, adv2-b −1.64 pts under leave-one-out) and INT8-to-fp drift consumes up to 40.6% of a margin with 0 verdict flips; the distance.py provenance comment's 79.7 / 75.0 came from notebook 09 cell 10's dd.embed path and sits 0.26 pts above the shipped-path band (E12-H63)
- **NO_COUNTERPART_COST 0.6** - inert on matched content: 0/61 fully-matched statements and 0/579 ibm-pool statements cross it, the corpus maximum 0.5774 sits 0.023 below the cut; the removed-population valley is unmeasured (E12-H60)
- **MOVED_TOLERANCE 0.05 (norm-only)** - over-fires on insertion in the 3-11 statement regime, 4/6 on the fixture; the conjunction with raw displacement reads 0/6 at no true-move loss (E12-H61, not yet shipped)
- **Gold intrusions** - the shipped soft relevance gate reads 6 on the current INT8 models with inverted severity (Set1>Set2) where the ungated axis reads 0 with Set2>Set1; the E03-H11 headline 2 → 0 does not reproduce and per-statement max-r is now persisted at data/interim/E12-H69-grounding-signals.npz (E12-H69)

## Lessons learned

- **Baseline near the ceiling for ordering** - perfect ordinality and `d' = 4.60` leave little room; resolution (dynamic range), not the normalized boundary, is the axis with room
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
- **Anisotropy carries to the conditioned axis (E04)** - the E01-H3 lever, applied to the conditioned `D_sel` over the pooled {A,B,S} corpus, widens dynamic range 3.39x at `0` violations, a larger gain than on the symmetric distance (2.7x) because the conditioned coverage profiles start more bunched; like E01-H3 it is resolution not a sharper boundary (contrast unchanged)
- **Coverage temperature trades contrast for range (E04)** - sharpening the soft source assignment spreads the distances but within-tier, so a lower `τ` raises DR while lowering the gold/adversarial contrast and breaking ordinality; the predicted contrast gain is the wrong direction
- **The cross-encoder is load-bearing against its own family (E04)** - a distilled same-family `bge-reranker-base` recalls only 0.55 of the `v2-m3` top-3 and changes the ranking (Spearman 0.70); even keeping cross-attention, a smaller reranker is not a faithful replacement
- **A cross-encoder cascade is the closest faithful speed-up (E04)** - a tiny MiniLM pre-filter clears the fidelity (Spearman 0.98) and speed (2.3x) bars where the E03 bi-encoder could not (0.55), but at `m = 15` still drops 13% of the top-3 evidence and adds a gold intrusion; the reranker cost stands, the cascade is the open direction
- **The cross-encoder floor holds across six candidate classes (E05)** - two multilingual cross-encoders (mmarco-MiniLM 118M recall@3 0.405, jina 278M 0.599), late-interaction MaxSim (0.47) and a learned-sparse proxy (0.08) all fail to recall the `v2-m3` top-3 (bar 0.90), jina the closest at `D_grd` Spearman 0.927; with E03 (bi-encoder) and E04 (distilled, English MiniLM) that is six scorer classes - the cross-encoder is irreplaceable on this fixture
- **Padding, not precision, is the free CPU lever (E05)** - the reranker pairs carry 2.33x padded-token waste; length-bucketing cuts the CPU INT8 reranker 42.5% at bit-identical scores over the full 6,860-pair grid (E06-H25), a real free win short of the 50% bar E05 scored it against; the grid is compute-bound (an encoder, no KV cache), so lower precision is not the lever
- **A redundant source is not a compressible one (E05)** - 91% of the 70 source statements have a `>= 0.85` neighbour (`k_eff` 10), but the redundancy is soft: clustering to half (k=35) drops top-3 recall to 0.76 and breaks the grounding ranking - near-duplicate sources are interchangeable for retrieval but not collapsible without losing the fine distinctions `D_grd` needs; index-recall understates fidelity here, so `D_grd` Spearman is the truer measure
- **Training does not lift the late-interaction floor (E06)** - a trained multilingual ColBERT (jina-colbert-v2) MaxSim recall@3 `0.4456` lands just below the untrained mmBERT proxy `0.47`; retrieval fine-tuning adds no fidelity on this source-conditioned grounding task, so the E05 "untrained backbone" caveat is closed - the floor is the architecture, not the training
- **Learned-sparse buys recall by inverting order (E06)** - trained bge-m3 sparse recall@15 `0.823` finally beats the cosine floor `0.612`, but its `D_grd` Spearman is `-0.770`; lexical term-weight scoring catches the overlap the pooled vector dropped yet orders the documents backwards, so no safe cascade (`m@0.95 = 39 > 34`) - the recall-vs-order split of E03, again
- **The cross-encoder floor holds against trained scorers too (E06)** - with E03 (bi-encoder), E04 (distilled, English MiniLM) and E05 (multilingual cross-encoders, untrained late-interaction / sparse) already ruled out, E06 adds trained multilingual late-interaction and trained learned-sparse; eight scorer classes, none matches the v2-m3 ranking - the grounding cost is structural, not a tuning gap
- **A flash-attention scorer NaNs on CPU padding (E06)** - jina-colbert-v2's flash-attn XLM-R CPU forward returns NaN over padded positions, so a default padded batch silently corrupts the MaxSim grid (an all-zero grid masked by `nan_to_num`, a near-random recall@3 `0.003`); encoding one sequence at a time (`batch_size=1`) removes the padding and fixes it - a methodological caveat for any CPU late-interaction encode

## Conclusions

- **Ships** - baseline exact SMD: metric, 0.08 ms/pair, perfect ordinality
- **Optional** - anisotropy removal (k=1) as a resolution pre-pass, ~2x latency, `d'` caveat noted
- **Thin margin is intrinsic** - all eleven summaries describe one article and share its content, so the boundary is genuinely narrow
- **Source-conditioned axes (E02)** - `D_sel` ships as the selection axis (clean, metric, sub-ms, 0 violations); `D_grd` (R2 joint premise) is a tier-level fabrication flag, ~109 s/pair - E03 turned it into a per-document metric with the H11 relevance-gate, not the numeric verifier E02 expected
- **Grounding axis fix (E03)** - the H11 relevance-gated ungrounded mass closes the per-document gold intrusion (2 → 0) at no extra cost, turning `D_grd` from a tier flag into a per-document discriminator on this fixture; ship it as the grounding-axis definition
- **Blended scalar (E03)** - the H14 blend `α·D_sel + (1−α)·D_grd` (`α ∈ [0.6, 0.9]`) orders the two failure modes correctly (Set 2 fabrication above Set 1 info-loss) where the symmetric scalar inverts them - a single interpretable conditioned scalar; the win is severity ordering and attribution, and needs a second source before it is trusted
- **Not shipped (E03)** - the numeric verifier (defeated by source figure density) and the two reranker-cost levers (cosine neither shortlists nor replaces the cross-encoder); the ~109 s/pair grounding cost stands
- **Resolution lever ships as the default (E04)** - anisotropy removal on the conditioned `D_sel` (`k = 1`, now the shipped default, opt-out via `anisotropy=False`) widens dynamic range 3.39x at `0` violations, with the E01-H3 caveat that it spreads the band rather than sharpening the boundary
- **Reranker cost still stands (E04)** - neither a distilled replacement (recall 0.55, Spearman 0.70) nor a cross-encoder cascade (recall 0.87 at `m = 15`, one new intrusion) clears the correctness guardrails; the ~109 s/pair grounding cost is unmoved, the cascade the most promising future lever
- **CPU speed: the floor holds, one free win (E05)** - no lever clears the `>= 50%` CPU cut at a preserved verdict; the v2-m3 cross-encoder is reconfirmed irreplaceable (no smaller cross-encoder, late-interaction or sparse scorer recalls its top-3), jina-reranker-v2 the closest (Spearman 0.9636 but recall@3 0.599, short of the 0.90 bar; see the E05-H20 note). The one shippable improvement is length-bucketing - a `42.5%` CPU cut at bit-identical scores, `7` points short of the bar but zero correctness risk; ship it as a portable default and keep v2-m3 as the reranker
- **Trained scorers do not move the grounding floor (E06)** - a trained multilingual ColBERT (recall@3 `0.4456`, the untrained proxy `0.47`) and a trained learned-sparse bge-m3 (recall@15 `0.823` but Spearman `-0.770`) both fail to replace v2-m3 or cascade safely at `m < 34`; the open E05 direction is closed, the cross-encoder grounding cost is structural across eight scorer classes
- **One CPU win ships (E06)** - length-bucketing, re-scored end-to-end over all 11 documents under the looser lift gate, cuts CPU INT8 reranker latency `42.5%` at score Spearman `0.99994` / `D_grd` Spearman `1.0` and `0` intrusion change; promoted from the E05 refuted-near-miss to the reserved CPU-speed candidate and wired into the shipped statement encoder (`OpenVINOEncoder` / `TorchEncoder`), where the same padded-batch waste lives

## Next steps

- **Promote the E03 winners** - wire the H11 relevance-gated ungrounded mass into the grounding axis as `D_grd`, and expose the H14 blend (`α ≈ 0.75`) as the single source-conditioned scalar; the per-document intrusion is closed and the failure modes order correctly
- **E04 resolution lever shipped** - the H15 anisotropy pre-pass (`k = 1`) is now the default on the conditioned `D_sel` (`anisotropy=False` to opt out), wired through the library (`compute_source_conditioned`, `distance_wrt_source`) and the CLI (`--no-anisotropy`); folded into the SOTA design doc alongside the symmetric E01-H3 anisotropy
- **E04 cascade near-miss, closed by E05** - the E04-H18 MiniLM→`v2-m3` cascade was the open direction (recall@15 0.87); E05 tested a larger `m` and a multilingual pre-filter (mmarco-MiniLM) and they fail too - jina-reranker-v2 only reaches recall@m 0.95 at `m = 34` (half the source), so no cascade clears the gate with a real saving
- **Cross-fixture check (parked, separate fixture)** - the H14 win is severity ordering and attribution on one article, where the symmetric scalar also clears 0/28 gold-vs-adv; a second-source fixture is prepared separately later, then E02/E03/E04 re-run on it before claiming the blend beats symmetric in general; E04 itself is scored on the existing fixture meanwhile
- **Anisotropy continuation (parked, E01 lineage)** - still open: sweep the anisotropy `k` further and test a gentle numeric weight tuned to preserve `V` stacked on anisotropy removal; source-free, distinct from E04-H15, which applies anisotropy removal to the conditioned `D_sel`, not the symmetric distance
- **E06 closes the trained-scorer thread** - no trained multilingual late-interaction or learned-sparse model clears the cross-encoder floor; the only open grounding-speed lever is the E04-H18 cross-encoder cascade on a faithful pre-filter, not a cheaper scorer
- **Length-bucketing wired into the statement encoder (E06-H25)** - the lever now batches `OpenVINOEncoder` / `TorchEncoder` by token length; the `42.5%` is reranker-grid-measured, the statement-encoder CPU cut is benchmarked separately in [`notebooks/10-kj-encoder-length-bucketing.ipynb`](../../notebooks/10-kj-encoder-length-bucketing.ipynb)
- **Open grounding cost** - the reranker grid stands as the ~99% bottleneck (~67 s/pair shipped CPU; the ~109 s figure includes the diagnostic R1 NLI sweep the shipped `D_grd` skips); E03/E04/E05 ruled out six scorer classes (bi-encoder, distilled and multilingual cross-encoders, late-interaction, learned-sparse) - none recalls the v2-m3 top-3; the structural wins are length-bucketing (42.5%, shippable) and source clustering (36% safe ceiling), neither alone reaching 50%
- **Ship the E05 length-bucketing win** - length-sort the reranker pair list and tighten `max_length` to the per-call 100th-percentile pair length; a `42.5%` CPU cut at bit-identical scores, portable, the one E05 lever worth wiring into the grounding chain when it lands in the library
- **E05 open replacement directions** - jina-reranker-v2 (278M) is the closest faithful cross-encoder (Spearman 0.9636 but recall@3 0.5986, short of the 0.90 bar); a mid-size multilingual cross-encoder and a *trained* multilingual ColBERT / SPLADE are untested (the E05 architecture probes ran on an untrained mmBERT backbone, and no `FlagEmbedding`/`pylate` in the environment) - these are the open scorer hunts, not the small models tried so far
- **Refuted, do not revisit** - salience / numeric weighting (breaks ordinality on number-heavy sources), angular cost (null), unbalanced residual (worse, slow, non-metric), tail aggregation (noise at this statement count), single-premise grounding (mis-grades compression, superseded by R2 joint premise), numeric verifier (defeated by source figure density, E03-H10), bi-encoder cascade / replacement (cosine neither shortlists nor replaces the cross-encoder, E03-H12/H13), smaller / distilled / multilingual cross-encoder replacement (E04-H17, E05-H20 - none recalls the v2-m3 top-3), late-interaction MaxSim and learned-sparse over an untrained backbone (E05-H21/H22 - below the gate; a *trained* ColBERT/SPLADE is still open), source-statement clustering past ~36% (E05-H24 - breaks the grounding ranking)
