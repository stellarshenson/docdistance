# SUMMAC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization - Digest

**Authors**: Philippe Laban (UC Berkeley), Tobias Schnabel (Microsoft), Paul N. Bennett (Microsoft), Marti A. Hearst (UC Berkeley)
**Venue**: Transactions of the ACL (TACL) 2022 - arXiv:2111.09525, 18 Nov 2021
**Source**: `[paper] SUMMAC - Re-Visiting NLI-based Models for Inconsistency Detection in Summarization.pdf`

## Overview

The paper rescues natural language inference (NLI) as a tool for detecting factual inconsistency in summaries. Prior work had written NLI off - out-of-the-box NLI models scored only 52% on binary inconsistency detection, barely above chance. The authors trace this failure to a single cause: a granularity mismatch. NLI models are trained on sentence pairs, but earlier methods fed them the whole document as premise and the whole summary as hypothesis. Run NLI at the sentence level instead - every document sentence against every summary sentence - and aggregate the resulting scores, and NLI becomes state of the art. Their `SummaC_Conv` model reaches 74.4% balanced accuracy on a new six-dataset benchmark, 5 points above the best non-NLI method.

## The Problem

Modern summarizers are fluent but frequently unfaithful - some state-of-the-art models generate inconsistent summaries in over 70% of cases. Inconsistency detection asks whether every claim in a summary is supported by the source document, regardless of whether the claim is true in the world (an accurate but unsupported addition still counts as inconsistent). NLI is the natural fit - entailment is exactly "is this hypothesis supported by this premise" - yet early attempts failed. The paper's Figure 1 shows why: an inconsistent summary, scored whole-document-as-premise, gets `P(entail) = 0.91` (the model says "supported"), because the long premise dilutes the one bad sentence. Scored sentence-by-sentence, the offending summary sentence `S3` is entailed by no document sentence (max entailment ≈ 0.04) and the inconsistency surfaces.

## Key Facts

- The fix is granularity, not a better NLI model - split both texts into sentences and score all pairs
- `SummaC_ZS` (zero-shot) and `SummaC_Conv` (one trained convolution layer) are the two model variants, both built on the same NLI pair matrix
- `SummaC_Conv` scores 74.4% balanced accuracy overall, `SummaC_ZS` 72.1%, vs 69.4% for QuestEval, the best method not using NLI
- Evaluated on the new SUMMAC Benchmark - the six largest inconsistency datasets standardized to one binary task
- Balanced accuracy is the primary metric because class balance ranges from 6% to 91% positive across datasets
- Throughput ≈ 430 documents/minute on one GPU - roughly 10x faster than question-answer-generation (QAG) methods
- The best NLI backbone is a BERT-Large model trained on MNLI + VitaminC; NLI progress transfers directly to SummaC gains
- Models and datasets are public (`github.com/tingofurro/summac`)

## How SummaC Works

The first step, common to both models, builds an **NLI pair matrix**. The document is split into `M` sentence premises `D1..DM`, the summary into `N` sentence hypotheses `S1..SN`, and every `(Di, Sj)` pair is run through an off-the-shelf NLI model. The matrix `X` is `M × N` and holds the entailment probability `Eij` of each pair (the contradiction and neutral channels are explored separately). The matrix reads as a weighted bipartite graph - each summary sentence is supported, or not, by each document sentence.

`SummaC_ZS` reduces the matrix with two operators and no trained parameters. For each summary sentence (column) it takes the **max** entailment over all document sentences - the single strongest piece of support - then takes the **mean** over summary sentences for a final scalar. Appendix B confirms max-then-mean as the best operator pair. The model is directly interpretable: a low score traces back to the specific summary sentence no document sentence supports.

`SummaC_Conv` replaces the brittle max with the whole distribution. Each column of the matrix is binned into a fixed `H`-bin histogram (`H = 50`, bin width 0.02), and a learned 1-D convolution of kernel size `H` compiles each histogram into a per-sentence score, averaged for the final score. The convolution is trained end-to-end on a 10,000-pair subsample of the FactCC synthetic data (cross-entropy, Adam, batch 32, learning rate 1e-2). Looking at the full distribution rather than the extremum makes it robust to NLI noise, which is the source of its gain over `SummaC_ZS`.

## The SUMMAC Benchmark

The authors standardize the six largest summary-consistency datasets - CoGenSumm, XSumFaith, Polytope, FactCC, SummEval, FRANK - into one binary `(document, summary, label)` classification task with fixed validation/test splits. Documents come from CNN/DM and XSum news. Because positive-class rates vary so widely (6% to 91%), the primary metric is **balanced accuracy** (the average of true-positive rate and true-negative rate, so majority-class voting scores 50%), with per-dataset thresholds tuned on the validation split and ROC-AUC as a secondary metric.

## Results

`SummaC_Conv` (74.4%) and `SummaC_ZS` (72.1%) take the top two overall balanced-accuracy places, ahead of QuestEval (69.4%), DAE (64.2%), FactCC-CLS (62.8%), MNLI-doc (61.3%) and NER-Overlap (56.8%). The same ordering holds on ROC-AUC (`SummaC_Conv` 77.8%, `SummaC_ZS` 74.3%). Both SummaC models improve significantly over prior work overall - `SummaC_ZS` at `p = 0.05`, `SummaC_Conv` at `p = 0.01` under bootstrap testing with Bonferroni correction. SummaC is strong across all six datasets, where competitors spike on one and collapse on another (FactCC-CLS is top on FactCC but near-worst on FRANK and XSumFaith).

Three ablations matter:

- **NLI backbone** (Table 3) - SNLI is worst (image-caption domain, far from news); MNLI and VitaminC are both near-best, and MNLI + VitaminC jointly is the default model; Transformer Large beats base by ~1.3 points on average
- **NLI category** (Table 4) - entailment alone is already strong; adding the contradiction channel gives small boosts for MNLI and ANLI backbones
- **Granularity** (Table 5) - finer is better; `(sentence, sentence)` and `(two-sentence, sentence)` win; the document granularity should be **coarser than or equal to** the summary granularity, because a multi-sentence summary claim is rarely entailed by a single document sentence (sentence fusion)

## Limitations and Future Work

`SummaC_Conv` is less interpretable than `SummaC_ZS` - the convolution makes it harder to trace a low score back to one offending sentence. The benchmark is entirely news-domain, leaving legal, scholarly and other domains open. The sentence-fusion case (a summary sentence that legitimately combines several source sentences) is the granularity setting's known weak spot. Proposed directions include combining multiple NLI models, combining multiple granularity levels via multi-hop reasoning, and feeding stronger detectors back into summarizer training.

## Relevance To This Project

SummaC is the design template for the grounding axis `D_grd` of the source-conditioned distance `d(A, B | S)` (see `docs/wmd-source-conditioned-docdistance-solution-sota.md`, where it is cited as the multi-premise NLI pattern).

- **Granularity justifies statement-level processing** - the project segments documents into atomic statements (`sat-3l-sm`) and runs statement-pair NLI rather than whole-document NLI, exactly the sentence-level insight that makes SummaC work; whole-document entailment would dilute the very inconsistencies `D_grd` must catch
- **The pair-matrix-then-aggregate shape is shared** - SummaC scores every `(source sentence, summary sentence)` pair and aggregates; `D_grd` scores a reranker grid over `(summary statement, source statement)` pairs and aggregates per summary statement, the same matrix-reduction idea
- **Localization echoes SummaC's max** - SummaC keeps, per summary sentence, the single strongest supporting document sentence; the project's `bge-reranker-v2-m3` cross-encoder localizes the top source statements per summary statement before the entailer grades, a learned stand-in for SummaC's column-max
- **Joint premise is the project's answer to sentence fusion** - SummaC §5.3.3 names sentence fusion as the granularity weak spot; the project concatenates the top-3 reranked source statements into one premise so a compressed summary statement can be entailed by the fused evidence, which moved information-loss scoring to gold level (a SummaC-motivated adaptation, not SummaC's own max/histogram aggregation)
- **Same NLI lineage** - SummaC's best backbone is MNLI + VitaminC; the project's entailer is `mdeberta-v3-base-mnli-xnli` (multilingual MNLI/XNLI), and its INT8 quantization is calibrated and validated against the public VitaminC set - direct descent from SummaC's NLI choice
- **Where the project diverges** - SummaC outputs a binary consistency classifier (threshold on a max-then-mean or convolution score); `D_grd` uses entailment as one factor in a continuous distance, multiplies it by reranker relevance, and adds a relevance gate `(1 - max_k r)` on the ungrounded mass to separate faithful compression from fabrication - signal not present in SummaC
- **Caveat carried over** - SummaC's finding that the contradiction channel adds only small gains matches the project's E03 result that the contradiction signal is near-dead on its fixture, so `D_grd` rides on the ungrounded (low-entailment) component rather than explicit contradiction
