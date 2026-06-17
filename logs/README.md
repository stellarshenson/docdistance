# Logs

Background-job and notebook-execution logs for this project.

- `01-document-segmentation.log` - nbconvert execution log for `notebooks/01-kj-document-segmentation.ipynb` (SAT GPU segmentation of the source PDF into statements)
- `02-mmbert-quantization.log` - nbconvert execution log for `notebooks/02-kj-mmbert-quantization.ipynb` (mmBERT quantization: SmoothQuant study, OpenVINO CPU INT8, torchao GPU FP8/INT8 sweep)
- `03-mmbert-throughput-saturation.log` - nbconvert execution log for `notebooks/03-kj-mmbert-throughput-saturation.ipynb` (GPU batch-saturation throughput sweep)
- `04-wmd-document-distance.log` - nbconvert execution log for `notebooks/04-kj-wmd-document-distance.ipynb` (WMD/SMD distance on the executive-summary fixtures: segment, embed, score vs gold, separability, latency)
- `E01-wmd-contrast-hypotheses.log` - execution of the R1 tier-contrast hypothesis sweep (notebooks/experiments/E01)
