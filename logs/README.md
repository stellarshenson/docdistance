# Logs

Background-job and notebook-execution logs for this project.

- `01-document-segmentation.log` - nbconvert execution log for `notebooks/01-kj-document-segmentation.ipynb` (SAT GPU segmentation of the source PDF into statements)
- `02-mmbert-quantization.log` - nbconvert execution log for `notebooks/model-quantization/Q01-kj-mmbert-quantization.ipynb` (mmBERT quantization: SmoothQuant study, OpenVINO CPU INT8, torchao GPU FP8/INT8 sweep)
- `Q02-deberta-quantization.log` - nbconvert execution log for `notebooks/model-quantization/Q02-kj-deberta-int8-smoothquant.ipynb` (mDeBERTa NLI INT8 SmoothQuant for the source-conditioned grounding axis: calibrate on source-vs-summary statement pairs, gate on distance fidelity)
- `03-mmbert-throughput-saturation.log` - nbconvert execution log for `notebooks/03-kj-mmbert-throughput-saturation.ipynb` (GPU batch-saturation throughput sweep)
- `04-wmd-document-distance.log` - nbconvert execution log for `notebooks/04-kj-wmd-document-distance.ipynb` (WMD/SMD distance on the executive-summary fixtures: segment, embed, score vs gold, separability, latency)
- `05-source-conditioned-distance.log` - nbconvert execution log for `notebooks/05-kj-source-conditioned-distance.ipynb` (source-conditioned distance `d(A,B|S)` design: D_sel + reranker×NLI D_grd with E03-H2 gate + E03-H5 blend, CPU INT8 vs GPU torch benchmark)
- `E01-wmd-contrast-hypotheses.log` - execution of the R1 tier-contrast hypothesis sweep (notebooks/experiments/E01)
- `09-docdistance-api-e2e.log` - nbconvert execution log for `notebooks/09-kj-docdistance-api-e2e.ipynb` (docdistance library validated end-to-end through the public API on the executive-summary fixtures)
- `E02-source-conditioned-grounding.log` - nbconvert execution of the E02 source-conditioned grounding experiment (reranker x NLI grounding axis over the fixtures, two rounds)
- `E03-source-conditioned-improvements.log` - nbconvert execution of the E03 batch (numeric verifier, relevance-gated residual, bi-encoder cascade/replacement, blended-scalar gate over the fixtures)
- `E04-source-conditioned-performance.log` - nbconvert execution of the E04 batch (anisotropy and coverage-temperature on the conditioned axes, distilled-reranker and cross-encoder-cascade speed levers, composite capstone; GPU fp16)
- `gpu-grounding-benchmark.log` - GPU (RTX 5000 Ada, torch fp16) timing of the grounding chain (mmBERT + FP reranker + FP NLI) vs the CPU INT8 reference
- `E05-source-conditioned-cpu-speed.log` - nbconvert execution of the E05 batch (CPU-speed scorer hunt: smaller multilingual cross-encoders, late-interaction, learned-sparse, length-bucketing, source clustering)
- `E06-scorer-env-setup.log` - isolated venv install for the E06 trained scorers (`pylate` + `FlagEmbedding`, CPU torch, transformers 5.3)
- `E06-scorer-probe.log` - API + cache probe of the trained scorers (`jina-colbert-v2` MaxSim, `bge-m3` sparse) in the isolated venv
- `E06-execution.log` - nbconvert execution of the E06 batch (trained multilingual ColBERT/sparse scorers and the reserved length-bucketing CPU-speed win)
- `11-structure-fixture.log` - nbconvert execution log for `notebooks/11-kj-structure-fixture.ipynb` (builds the reusable structure-distance fixture: segmented statements with block labels, byte-identical reorder pool, cross-summary pairs, section-swap perturbations)
- `E07-structure-distance.log` - nbconvert execution of the E07 batch (structure axis on the SMD transport plan `T`: tau-induced distance, bounded [0,1] normalizers, naive baseline, GW reorder-invariance control, section axis)
- `E08-structure-distance-metric.log` - nbconvert execution of the E08 batch (metric, wide-dynamic-range OT structure distance: position-augmented Wasserstein E08-H1, positional Fused GW E08-H2, optimal-assignment footrule E08-H3, offline-frozen anisotropy E08-H4, anti-monotone transport mass E08-H5; dynamic-range-ratio and triangle-violation comparison vs the E07 footrule/off-diagonal baselines)
- `12-structure-distance-e2e.log` - nbconvert execution log for `notebooks/12-kj-structure-distance-e2e.ipynb` (E08-H44 end-to-end showcase: two documents to semantic SMD and structural position-augmented Wasserstein at lambda=0.25, with the recovered structural mapping / movers table over three contrasting pairs)
