# WMD Document Distance with mmBERT (SOTA)

## Abstract

An embedding-grounded, metric document distance for pipelines that cannot read model logits. Each document is segmented into statements, embedded with mmBERT, isotropy-corrected (all-but-the-top), and compared by exact optimal transport - the Statement Mover's Distance (SMD). It adapts Word Mover's Distance[<sup>ref1</sup>](#ref1) (Kusner et al. 2015, digest in `../references/papers/from-word-embeddings-to-document-distances.md`) by swapping the embedded unit (statement, not word) and the encoder (mmBERT, not word2vec), keeping the transport skeleton verbatim. SMD is a true metric, ranks the executive-summary quality tiers with zero ordering errors (batch E01, [`experiments/wmd-docdistance-experiments.md`](../experiments/wmd-docdistance-experiments.md)), and runs CPU-INT8 end-to-end (~5.5 s for two A4 pages on one core). This is the conclusion doc; the experiments log is its evidence.

## Problem

Comparing two documents for shared content is hard when whole-document embeddings are too coarse and model logits are unavailable.

- **Whole-document cosine is too coarse** - two summaries that carry the same claims in different places, in a different order, or with content added or dropped collapse to one bunched cosine score
- **No logits in frontier pipelines** - agentic conversion and extraction run through frontier models that do not expose token-level logits, so KL divergence cannot measure how far an output has drifted from its source
- **A metric is required** - the distance must satisfy the triangle inequality to compose and threshold reliably, and stay interpretable enough to localize what changed

## Solution

Lift the comparison to statements and optimal transport: segment each document, embed and isotropy-correct the statements, and take the exact transport cost between the two clouds.

- **Distance interpretation** - transport cost between the two statement clouds; `0` identical, larger = more divergent content, read as `closeness = 1 − SMD/√2` (1 identical, 0 unrelated)
- **Why and how to use** - flag content drift between two documents when model logits are unavailable; two texts in → one metric distance, a similar/not verdict, and the statement alignment showing what moved, dropped, or was added
- **Deterministic and reproducible** - the same two texts give the same distance every run; the CPU-INT8 production path is bit-exact (segmentation, embedding, the SVD and exact OT are all deterministic), the only non-determinism being the natural floating-point reduction-order variance on GPU - unlike a sampling LLM judge
- **Headline result** - perfect tier ordinality (`0/24` violations), anisotropy removal widens the dynamic range 2.7x, and a 2 × A4 pair compares in ~5.5 s on one CPU core

## Pipeline

Two documents in → one closeness score and a similar/not verdict; the transport plan is the statement-to-statement alignment.

- **Segment** - split each document into statements with `sat-3l-sm`[<sup>ref4</sup>](#ref4) (nb 01); statements are the transport unit, so the comparison is position-invariant. Since 1.1.5 an optional markdown-aware mode makes each table row, heading, list item and fenced-code line one statement (markup stripped for embedding), and fragment filters (`min_words`, `skip_table_fragments`) drop noise fragments before encoding - filtering renormalizes the uniform marginals, so the SMD legitimately changes. After 1.1.5 the option `input_format` (`auto`, `plain`, `markdown`, `html`) names the reader. `auto`, the default, chooses it per document from the file extension: `.md` and `.markdown` → the markdown reader; `.html` and `.htm` → the HTML reader, which writes the block structure of an HTML document as the lines the markdown reader reads; every other file and raw text → the plain reader. The content is not read for the choice. The plain reader is the prose segmenter of 1.1.5 on the whole text, and with `input_format="plain"` every 1.1.5 number is bit-identical except `moved` and `n_moved` of the structural detail file (55 ibm pairs and 3 grounded results, 0 differences outside those two keys). The default therefore changes the numbers of `.md` and `.html` files against 1.1.5 (decision of 2026-10-10, a registered exception to the bit-identical rule); the measured size of the change on the ibm corpus is under [Limitations](#limitations). E16-H81, which tested a reader chosen from the content as the default, was killed at its gate because the scored summaries hold no marker statement; the extension rule is a product decision and not its verdict
- **Embed** - mmBERT[<sup>ref5</sup>](#ref5), mean-pooled and L2-normalized per statement; quantized encoder (OpenVINO INT8 on CPU, torchao FP8 on GPU)
- **Anisotropy removal** - subtract the dominant principal component from the pooled statement embeddings and re-L2-normalize (all-but-the-top); de-bunches the mmBERT cosines so the cost matrix is not compressed - a standard step, not optional
- **Ground cost** - Euclidean on L2-normalized embeddings `= √(2 − 2cos)`, metric-safe cosine; `1 − cos` and squared-Euclidean are rejected - they rank like cosine but void the triangle inequality
- **Distance** - exact optimal transport between the two clouds, uniform weights `1/n`, solved with POT[<sup>ref6</sup>](#ref6) `ot.emd`; this scalar is the SMD
- **Lower bounds** - WCD (centroid distance, exactly the whole-document-cosine baseline) and RWMD (one-sided relaxation, greedy nearest-statement alignment) nest as `WCD ≤ RWMD ≤ SMD`; cheap prefetch-and-prune floors
- **Verdict** - closeness `1 − SMD/√2` (1 identical, 0 orthogonal) plus a similar/not call against a closeness threshold (shipped default `0.77`, recalibrated in 1.1.4; originally set from the gold cluster's own diameter)
- **Diff flags** - the detail output names the drift per statement: `changed` when the cheapest match over the statement's full cost row exceeds `(1 − threshold)·√2`, `no_counterpart` (removed content) when it exceeds `0.6`, `identical` on normalized-text equality; a target-side view gives each B statement its cheapest incoming source, so added content reads directly. From the release after 1.1.5 a one-to-one matching of the statements labels each A statement `kept` or `removed` and each B statement `kept` or `added` (`match_status`; a pair needs a ground cost below `MATCH_CUTOFF` 0.325, E13-H71); the labels annotate the map and change neither the plan nor the SMD ([API reference](../api-reference.md#matching-and-the-matched-order-gap))

## Mechanism

The distance is optimal transport between the two statement clouds; the ground cost must be a true metric, the encoder's anisotropy is corrected first so the cost matrix is not compressed, and the transport is solved exactly.

**Why not cosine similarity.** Cosine is a similarity in `[−1, 1]`, not a distance - it peaks at identity, has no zero floor, and obeys no triangle inequality, so it cannot serve as a transport ground cost. Optimal transport needs a genuine metric: non-negative, zero only for identical points, triangle-respecting - and WMD inherits the metric property only when its ground cost is one (Kusner et al. 2015). The cosine-derived costs `1 − cos` and `2 − 2cos` rank pairs identically to cosine but break the triangle inequality, so they would void that guarantee. The Euclidean norm of the difference of L2-normalized vectors does not, and for unit vectors it collapses to a function of cosine:

$$
\lVert \mathbf{x} - \mathbf{y} \rVert_2^{2} = \lVert \mathbf{x} \rVert_2^{2} + \lVert \mathbf{y} \rVert_2^{2} - 2\,\mathbf{x}^{\top}\mathbf{y} = 2 - 2\cos(\mathbf{x}, \mathbf{y})
$$

So the ground cost is the chord length between the two unit vectors - a real metric that ranks exactly like cosine, only as a distance:

$$
c(\mathbf{x}_i, \mathbf{x}_j) = \lVert \mathbf{x}_i - \mathbf{x}_j \rVert_2 = \sqrt{2 - 2\cos(\mathbf{x}_i, \mathbf{x}_j)}
$$

The chord distance spans `[0, 2]` in general - `cos = −1` (antipodal) gives the maximum `2`. mmBERT statement embeddings are non-negatively correlated, though - they occupy a narrow cone, `cos ∈ [0, 1]` - so in practice the cost lives in `[0, √2]`, orthogonal statements (`cos = 0`) sitting at the ceiling `√2`. That practical ceiling is the closeness normalizer: `closeness = 1 − SMD/√2` maps orthogonal clouds to 0 and identical clouds to 1.

**The distance.** With that cost, the Statement Mover's Distance is the minimum-cost plan that moves one cloud's uniform mass onto the other's:

$$
\mathrm{SMD}(A, B) = \min_{T \ge 0} \sum_{i=1}^{n_A} \sum_{j=1}^{n_B} T_{ij}\, c(\mathbf{a}_i, \mathbf{b}_j)
\quad \text{s.t.} \quad
\sum_{j} T_{ij} = \tfrac{1}{n_A}, \;\; \sum_{i} T_{ij} = \tfrac{1}{n_B}
$$

**The solver - exact EMD, not Sinkhorn.** This is a linear program, solved exactly with the network-simplex (POT `ot.emd` with `log=True`), which returns the sparse transport plan and, in its log, the true optimal cost. The entropic-regularized alternative, Sinkhorn[<sup>ref3</sup>](#ref3) (Cuturi 2013), trades exactness for speed by adding an entropy term `H(T)` with blur `ε`:

$$
\mathrm{SMD}_{\varepsilon}(A, B) = \min_{T \ge 0} \sum_{i, j} T_{ij}\, c(\mathbf{a}_i, \mathbf{b}_j) \;-\; \varepsilon\, H(T), \qquad H(T) = -\sum_{i,j} T_{ij}\big(\log T_{ij} - 1\big)
$$

Sinkhorn only pays off when the clouds are large; at statement scale (~12 points) exact EMD runs at ~0.08 ms/pair, faster than Sinkhorn and free of the entropic bias, so Sinkhorn is dropped.

**Anisotropy.** mmBERT statement embeddings are anisotropic - a common mean and a few dominant principal directions sway every vector, so pairwise cosines bunch at 0.7-0.9 and the cost matrix is compressed. The paper reads the dominant directions of word embeddings as word frequency, not meaning. On the symmetric panels measured here the gain of the pre-pass comes from subtracting the common mean, and removing the first direction alone does not add to it (readings in the bullets below). All-but-the-top postprocessing[<sup>ref2</sup>](#ref2) (Mu & Viswanath, ICLR 2018; digest `../references/papers/all-but-the-top-simple-and-effective-postprocessing-for-word-representations.md`) subtracts the pooled mean and projects away the top D directions (`D ≈ d/100`), then re-L2-normalizes:

$$
\tilde{\mathbf{v}} = \mathbf{v} - \boldsymbol{\mu}, \qquad
\mathbf{v}' = \tilde{\mathbf{v}} - \sum_{i=1}^{D} (\mathbf{u}_i^{\top} \mathbf{v})\, \mathbf{u}_i
$$

- **Effect** - de-bunched cosines, dynamic range up 2.7x (DR 0.0854 → 0.2286) at zero ordinality violations (E01)
- **Metric preserved** - re-normalized vectors keep Euclidean = metric-safe cosine, so the distance stays a metric
- **Pool of the estimate** - every scoring call and every CLI command estimates the shared direction from the statements of the documents of one call: the two documents of a pair on the symmetric path, also for each row of a `--pairs` run, and A, B and the source on the source-conditioned path. No command pools a batch or a corpus; the E01 reading above pooled the 11 ibm summaries in the notebook, and the library function `all_but_the_top` accepts any number of documents. A document pair (~two dozen vectors) is too small to estimate the direction reliably, so the pre-pass is off by default on the symmetric path and a pairwise comparison uses raw embeddings, on which the tier ordering is already perfect (`0/24`). The minimum estimation pool is about 71 pooled statements (`ANISOTROPY_MIN_POOL`, E12-H65); a symmetric call with the pre-pass on and a smaller pool logs one warning that names its count and is scored all the same
- **Which step carries the gain, symmetric panels** - the centering: centering and re-normalizing without the projection recovers 103.7% of the margin gain of the full pre-pass on the ibm panel (E12-H64) and 102.8% on the Wergeland pool (E14-H75); the projection isolate (first centered component removed, mean restored) reads -4.5% and -3.6%
- **Source-conditioned path, ibm fixture** - centering without the projection reads 0.5471x the `DR(D_sel)` of the shipped pre-pass and 1 of 28 selection-axis violations against 0 of 28 (E14-H76), so the shipped pre-pass stays on that path; the difference is traced to 22 one-word markup statements that the plain reader returns from that source (`-` 13 times, `###` 6, `##` 2, `Overview` 1), and with them removed centering alone reads 0.9165x at 0 of 28; this is one source and one reader setting, not a reading of the source-conditioned path in general

**A kernel pre-pass was measured and does not ship (E14-H77, Refuted).** The probe replaced the linear pre-pass by the empirical map of an RBF kernel `k(x, y) = exp(−γ(2 − 2cos))`: the Gram matrix of the pooled statements is double-centered and eigendecomposed, the top coordinate is dropped and the rows are re-L2-normalized. The bandwidth took five values fixed before the run, `γ = m / (median squared chordal distance of the pool)` with `m` = 0.25, 1, 4, 16 and 64. A bandwidth whose statement costs rank like the costs of the linear pre-pass, Spearman at or above 0.995 on each of the 11 gate pools (the ibm symmetric pool and 10 conditioned triples), counts as a copy of it and is not scored. Readings from `reports/E14-H77-kernel-pca-prepass-metrics.json`:

| `m` | lowest cost Spearman over the 11 gate pools | copy of the linear pre-pass | stability of the top component: ibm symmetric pool / 10 triples / Wergeland pool | conditioned contrast (violations of 28) | ibm symmetric margin in points (`V` of 24) | Wergeland margin in points (`V` of 12) |
|---|---|---|---|---|---|---|
| 0.25 | 0.9816 | no | 0.972 / 0.997 / 0.588 | 2.489 (0) | +19.04 (0) | +53.90 (0) |
| 1 | 0.9091 | no | 0.939 / 0.992 to 0.993 / 0.559 | 2.494 (0) | +16.25 (0) | +50.26 (0) |
| 4 | 0.5087 | no | 0.513 / 0.956 to 0.973 / 0.286 | 2.336 (0) | +6.11 (0) | +33.00 (0) |
| 16 | 0.0712 | no | 0.183 / 0.955 to 0.977 / 0.098 | 1.913 (4 of 28) | +0.00 (0) | +6.78 (0) |
| 64 | −0.0664 | no | 0.122 / 0.956 to 0.977 / 0.099 | 0.269 (22 of 28) | −9.54 (12) | −1.60 (2) |
| shipped linear pre-pass | - | - | 0.979 / 0.998 / 0.631 | 2.413 (0) | +19.41 (0) | +54.74 (0) |

- **Not a copy at any bandwidth** - no bandwidth reaches 0.995 on every gate pool, so all five were scored on the three panels
- **Component stability** - the absolute cosine between the top component of the full pool and the top component of a resample of its statements, mean of 50 resamples with replacement. On the ibm symmetric pool (98 statements) and the Wergeland pool (76 statements) it falls as the kernel narrows; on the 10 triples it reads 0.95 or above at every bandwidth
- **Bar** - conditioned contrast at least 1.10 times the shipped 2.413 (2.655) at 0 selection-axis violations, and both symmetric margins at most 1 point below the shipped margins. 0 of 5 bandwidths meet every clause. The nearest, `m` = 0.25, meets the two margin clauses and reads a contrast of 1.031 times the shipped one
- **Mechanism readings** - at `m` = 0.25 the kernel is close to an affine function of the cosine (R² 0.9927) and its dropped coordinate agrees with the linear one at 0.9998; narrower kernels put most statement pairs at nearly the same distance (R² 0.3723, 0.0464 and 0.0356 at `m` = 4, 16 and 64), and the margins fall with them
- **Decision** - no kernel pre-pass ships; `all_but_the_top` stays the only pre-pass

## Performance

Eleven executive summaries of one IBM AI-adoption article, three tiers (7 gold, 2 info-loss, 2 info-noise), scored against the reference gold; full evidence in [batch E01](../experiments/wmd-docdistance-experiments.md).

| measure | SMD + anisotropy removal (shipped) | SMD, no postprocessing |
|---|---|---|
| ordinality violations `V` | 0 / 24 | 0 / 24 |
| dynamic range `DR` | 0.2286 (2.7x) | 0.0854 |
| boundary margin (closeness pts) | +18.48 | +4.58 |
| separation `d'` | 7.17 | 4.60 |
| reference contrast `R` | - | 1.680 |
| metric | yes | yes |

- **Perfect ordering** - every gold summary is closer to the anchor than every adversarial one, zero crossings, with or without postprocessing
- **Resolution gain ships** - of five levers tested in E01, anisotropy removal is the only one that widens dynamic range and the boundary margin without breaking the order, so it is standard
- **No `d'` tradeoff** - the isotropy step was expected to spread the gold band and cost effect size; measured, the separation `d'` **rises** (4.60 → 7.17) alongside the 2.7x resolution gain, at unchanged perfect ordering. The trade this section previously reported did not occur
- **Closeness bands** (no postprocessing) - gold 80.3-85.9%, adversarial 68.5-75.7%; the two 3-sweep golds nearest
- **What it responds to** - SMD tracks shared statement content; the info-noise tier (kept the source's numbers) lands slightly closer than the info-loss tier (stripped them), so numeric retention correlates with closeness

## Setup

Preparation for the single-core benchmark (run in `notebooks/04-kj-wmd-document-distance.ipynb`, final section).

- **Hardware** - AMD Ryzen Threadripper PRO 7975WX (32 cores / 64 threads); measured on **one core** (OpenVINO `INFERENCE_NUM_THREADS=1`, torch `set_num_threads(1)`)
- **Models** - `sat-3l-sm` segmenter (FP32, CPU) and `mmBERT-base` encoder as OpenVINO INT8 (CPU) - the deployable CPU stack
- **Workload** - two A4-page text blocks (~554 words each, segmenting to ~21 statements each, ~42 total) - the "compare two documents" unit
- **Pipeline timed** - segment → embed → anisotropy removal → exact SMD, the full end-to-end

## Methods of measurement

How each figure is taken; single-stream throughout, so a number is the per-core unit.

- **Single-core** - thread counts pinned to 1, so each figure is one core; multiply by vCPUs for parallel throughput
- **Encoder latency** - one statement, dynamic padding, 50 reps after 5 warmups, mean ms/sentence; tokens/s from the real token count
- **Encoder amortized** - a 55-statement batch (one document), 5 reps, ms/sentence = batch time / count; on a single core batching does not parallelize, so amortized is slightly above single-statement latency (padding to the batch's longest statement)
- **Exact SMD** - 100 reps at 55×55 statements, mean ms/pair
- **End-to-end** - one timed pass per stage over the 2×A4 workload, after a segmentation warmup
- **Caveat** - segmentation is FP32 `sat-3l-sm` on CPU; an INT8 OpenVINO SAT would cut its share

## Throughput and footprint

Single core, full pipeline, AMD Ryzen Threadripper PRO 7975WX.

| stage | single core |
|---|---|
| segment (SAT FP32 CPU) | 2553 ms / 2 docs |
| embed (mmBERT INT8 CPU) | 2998 ms / 2 docs (~30.8 ms/sentence, 32 sent/s/core, ~1300 tok/s) |
| anisotropy (SVD) | 2.7 ms |
| exact SMD | 0.40 ms/pair at 55×55 (0.08 ms at 12×12) |
| **end-to-end (2 × A4)** | **5.55 s** |

- **Per-core unit** - ~0.2 document-pairs/s/core; embedding is 54% of the time, segmentation 46%, the distance negligible
- **AWS Lambda sizing** - ~1 vCPU per 1769 MB, so a ~2 GB function ≈ 1 core ≈ 0.2 pairs/s; scale ~linearly with vCPUs (a 6-vCPU / 10 GB function ≈ ~1.2 pairs/s)
- **GPU contrast** - the encoder reaches ~2588 sentences/s on GPU FP8 (~80× one CPU core, nb03), collapsing the embed stage; the single CPU core is the serverless worst-case floor
- **Footprint** - INT8 encoder IR ~310 MB, SAT ~few hundred MB, so the stack fits a ~1-2 GB Lambda; dependencies `wtpsplit` (SAT), `transformers` / `openvino` (mmBERT), `ot` (POT)

## Calibration of the semantic verdict

The verdict compares `closeness` with `DEFAULT_THRESHOLD`: `similar` at or above it, `not similar` below it. The shipped value `0.77` is in-sample: the 11 ibm summaries that set it are the documents it is scored on. E12-H63 measured how far the value can move before a tier is mis-sided. The card below reads the 10 non-anchor summaries against the gold anchor through the shipped `semantic_distance` path (`reports/E12-H63-semantic-threshold-band-metrics.json`).

| reading | OpenVINO INT8 (shipped) | fp backend |
|---|---|---|
| band that separates the tiers | [0.7474, 0.7944] | [0.7566, 0.8027] |
| lower edge, the highest adversarial document | adv2-b | adv2-b |
| upper edge, the lowest gold document | haiku | haiku |
| width | 4.71 points | 4.61 points |
| thresholds with 0 tier errors, swept in steps of 0.005 | 0.750 to 0.790 | 0.760 to 0.800 |

- **Safe on both backends** - 0.760 to 0.790. `0.77` lies inside that range, and `V` is 0/24 on both backends
- **Each edge is one document** - leave-one-out over the 10 scored documents: without haiku the upper edge moves +2.30 points, to 0.8174; without adv2-b the lower edge moves −1.64 points, to 0.7309. The other 8 subsets leave both edges unchanged. The band width ranges from 4.71 to 7.01 points
- **Backend drift** - from INT8 to fp the lower edge shifts +0.92 points and the upper edge +0.82. Per document the largest share is adv2-b: its INT8 closeness lies 0.920 points below its fp closeness, 40.6% of its 2.26-point margin to `0.77`. 3 of 10 documents drift by more than 20% of their margin and 0 verdicts flip at `0.77`. A threshold calibrated on one backend is not automatically valid on the other
- **Anchor** - with gold-2 as the anchor in place of gold the INT8 band is [0.747, 0.813]
- **Provenance of the earlier band** - the band [0.750, 0.797] quoted with the threshold up to 1.1.5 came from the `dd.embed` path of notebook 09 cell 10, not from `semantic_distance`, and lies 0.26 points above the shipped-path band at both edges
- **On another corpus** - score pairs with known labels on the backend that will serve, take the band between the highest closeness of a pair that must read `not similar` and the lowest closeness of a pair that must read `similar`, and set the threshold inside it. The structural threshold has its own procedure in the [structure document](wmd-structure-distance-sota.md#calibration-of-the-structural-verdict)

## Limitations

- **Boundary margin** - +4.58 closeness points at the gold/adversarial boundary on the raw axis (+18.48 after anisotropy removal); the ordering is clean and the margin is modest, on documents built to be hard to separate
- **Intrinsic to the fixture** - all eleven summaries describe one article and share its content, so the margin is narrow by construction, not a defect of the distance
- **Single source, single degradation design** - a controlled probe, not a benchmark; cross-fixture validation on a second article is pending
- **`d'` raised by the isotropy step** - anisotropy removal widens the margin and the dynamic range and raises the effect-size `d'` (4.60 → 7.17); ordering is unaffected
- **Selection vs grounding not separated** - a symmetric distance cannot tell same-source-different-picks from off-source fabrication; the source-conditioned variant `d(A,B|S)` in [`wmd-source-conditioned-docdistance-solution-sota.md`](wmd-source-conditioned-docdistance-solution-sota.md) adds that axis
- **Readings of this document and the reader of a file** - every reading here that comes from a fixture file passed to the package was measured with the plain reader, except the readings of the default call that this bullet quotes; a reading on statements that an experiment loaded from the structure fixture (`data/processed/structure-fixture/statements.json`) or segmented itself has no reader of the package behind it. After 1.1.5 the default reader of a `.md` file is the markdown reader. On the 55 ibm pairs the default call against `input_format="plain"`: SMD differs in 27 pairs by at most 0.0322, closeness by at most 0.0228, the statement count of a document by at most 1, and no verdict at `0.77` changes (`reports/E13-campaign/checks/default-reader/diff.py`); the tier band is the same, 0.7474 to 0.7944 (the `auto` arm of E16-H81, which equals the default call on these files in 55 of 55 distances). The 11 summaries hold a title line and prose only and no marker statement (0 of 98 statements), so these 55 pairs are not a bound for a `.md` file with list, heading or table lines: the two source articles, ibm against Wergeland, which are in none of the 55 pairs, read SMD 0.6416 with the plain reader and 0.5255 with the default reader, on 70 and 52 statements against 48 and 51 (`outside_record` of the E16-H81 metrics file, outside its record). `--input-format plain` restores the 1.1.5 reading of a file
- **In-sample verdict threshold** - `0.77` separates the tiers of one corpus inside a band of 4.71 closeness points whose edges are one document each; see [Calibration of the semantic verdict](#calibration-of-the-semantic-verdict)
- **No capped ground cost** - a statement without a counterpart pays the ground cost to the statements of the other document, so the cost of one inserted statement depends on the host document: the SMD increment of an inserted statement is its mean cost to the `n` host statements divided by `n + 1`. A cap on the ground cost was registered as the remedy and was not built: E13-H72 was killed at its gate, because 172 of 574 cross-article insertions (30.0%) cost above 0.6 to their nearest host statement and the gate requires at least 50%; on inserted statements of 5 or more words the share is 5.2%
- **Nearest-source cost is valid on raw embeddings only** - the cost from a statement to its nearest source statement separates the statements of a source's own summaries from the statements of another article at AUC 0.9670 on raw embeddings and at 0.7889 after the anisotropy pre-pass (E13-H73: 98 own against 38 foreign statements, gate 0.90). The unsupported-statement count of the source-conditioned result is therefore read before the pre-pass

## FAQ

- **Why not KL divergence?** - KL between the model's output token distributions is the natural drift measure, but it needs token-level logits, which frontier models behind agentic pipelines do not expose; KL is also asymmetric (not a metric) and lives over tokens, not document content. SMD is the embedding-grounded, logit-free stand-in that needs only the two texts
- **Why not embeddings + cosine similarity?** - whole-document cosine is exactly the WCD lower bound here, the cheap floor SMD refines; it bunches every document into the anisotropic 0.7-0.9 band, discards statement-level structure (the same claims in different places read as far apart), and cosine is a similarity, not a metric - no triangle inequality, no zero floor
- **Why not a cross-encoder?** - a cross-encoder (e.g. `bge-reranker`) scores relevance between two texts but returns no metric, no zero point, and no alignment, and costs O(n·m) over statement pairs; it is the right tool for the source-conditioned grounding variant `d(A,B|S)`, not for a symmetric metric distance
- **Why not an LLM judge?** - no logits exposed, non-deterministic, slow and costly per call, and not a metric; SMD is deterministic, cheap (sub-ms transport, ~5.5 s/pair full CPU pipeline), and thresholdable
- **Why statements, not words (literal WMD)?** - statements are the unit the use case cares about (a claim moved, dropped, or added), give far fewer points so exact OT stays cheap, and align interpretably; word-level pooling is slower and noisier
- **Why not Sinkhorn (entropic OT)?** - at ~12-55 statements exact EMD is faster and unbiased; Sinkhorn's entropic blur only pays off on large clouds (see Mechanism)
- **Why PCA/SVD for anisotropy removal?** - the anisotropy is a few dominant directions of shared variance across all statement embeddings; PCA - computed by the SVD of the mean-centered matrix - is exactly what finds those top-variance directions, so projecting them out strips the common component the all-but-the-top paper identifies as frequency, not meaning; the SVD is exact, deterministic, and cheap at statement scale, and removing the top-D components (`D ≈ d/100`) needs no training

## Implementation

- **Notebooks** - `notebooks/01-kj-document-segmentation.ipynb` (segment), `notebooks/04-kj-wmd-document-distance.ipynb` (the distance, validated end-to-end)
- **Experiments** - `docs/experiments/wmd-docdistance-experiments.md` (batch E01: five contrast levers, one promoted, four refuted)
- **Functions** - `smd` (`ot.emd`), `wcd` (centroid norm), `rwmd` (one-sided relaxation), `closeness` (`1 − SMD/√2`), `all_but_the_top` (anisotropy), `partial_matching` (the one-to-one statement pairs behind `match_status`; an annotation, not a distance); pairwise verdict threshold defaults to closeness `0.77` (`--threshold` / `threshold=` to recalibrate per corpus). After 1.1.5 the exact solver raises `SolverNotOptimal` when it stops before an optimal coupling, with an iteration cap that grows with the problem size, where 1.1.5 returned the non-optimal value with a warning from POT
- **References** - WMD (Kusner et al. 2015) and all-but-the-top postprocessing (Mu & Viswanath, ICLR 2018); digests under `../references/papers/`
- **Library** - shipped as the `docdistance` package: `src/docdistance/distance.py` (pure-numpy OT core), `encoders.py` (SAT + mmBERT INT8 / torch backends, `split_markdown`, `html_to_markdown`), `pipeline.py` (`semantic_distance`, `structural_distance`, `source_conditioned_distance`, the reusable `DocDistance` class), and a `docdistance` CLI (`distance-semantic`, `distance-structural`, `distance-wrt-source`, `init`; `--pairs` batches many pairs through one model load and, after 1.1.5, encodes a document file that recurs across rows once - readings in the [CLI reference](../cli-reference.md#batch-mode))
- **Library validation** - `notebooks/09-kj-docdistance-api-e2e.ipynb` reproduces the `0/24` ordinality through the public API and confirms openvino-vs-torch backend agreement (Pearson 0.9991)

## Conclusions

A single metric distance with a bounded, interpretable readout, built for pipelines that cannot read model logits.

- **Range** - the ground cost is `[0, 2]` in theory (`cos = −1`, antipodal); for these non-negatively-correlated embeddings it spans `[0, √2]`, so `SMD ∈ [0, √2]` and `closeness = 1 − SMD/√2 ∈ [0, 1]` (1 identical, 0 unrelated)
- **Verdict** - similar when closeness clears the threshold (shipped default `0.77`); the transport plan is the statement alignment, so the number arrives with what changed (`changed`), was removed (`no_counterpart`), or was added (the target-side view)
- **Use scenario** - agentic document conversion and extraction through frontier models, where token-level logits are not exposed so KL divergence cannot be computed; SMD is the embedding-grounded stand-in that flags when a converted or extracted output has drifted from its source, and the alignment localizes the drift
- **Operating point** - exact, metric, deterministic, runs CPU-INT8 end-to-end; the transport is ~0.08 ms/pair at statement scale, so a document pair's cost is dominated by encoding, not the distance
- **Performance** - a 2 × A4 document pair compares in ~5.55 s end-to-end on one core of the Threadripper 7975WX (segmentation 46%, embedding 54%, transport negligible); ~0.2 pairs/s/core, so a ~2 GB AWS Lambda ≈ one core and throughput scales ~linearly with vCPUs

## Bibliography

- <span id="ref1">ref1 Kusner, Sun, Kolkin, Weinberger. *From Word Embeddings To Document Distances*. ICML 2015. Digest `../references/papers/from-word-embeddings-to-document-distances.md`</span>
- <span id="ref2">ref2 Mu, Viswanath. *All-but-the-Top: Simple and Effective Postprocessing for Word Representations*. ICLR 2018. Digest `../references/papers/all-but-the-top-simple-and-effective-postprocessing-for-word-representations.md`</span>
- <span id="ref3">ref3 Cuturi. *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*. NeurIPS 2013</span>
- <span id="ref4">ref4 Frohmann et al. *Segment Any Text* (`sat-3l-sm`, wtpsplit). Shipped INT8: `stellars/sat-3l-sm-openvino-int8`</span>
- <span id="ref5">ref5 `mmBERT-base` (`jhu-clsp/mmBERT-base`), a multilingual ModernBERT encoder. Shipped INT8: `stellars/mmBERT-base-openvino-int8`</span>
- <span id="ref6">ref6 Flamary et al. *POT: Python Optimal Transport*. JMLR 2021 (`ot.emd`)</span>
