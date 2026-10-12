# AWS deployment reference

docdistance runs CPU-only and pulls its INT8 models from S3 at `init` time. `init` provisions a mode's models, writes `docdistance.json`, and the distance commands then load the models from the local mirror and make no request to the HuggingFace Hub. `init` reads the HuggingFace cache or the Hub for one item when the source holds none that loads: the tokenizer of the SaT segmenter, named in the `sat-3l-sm` bullet below.

## Models and footprint

The OpenVINO INT8 IRs sit at the bucket root (or under a prefix); `init --source` points at whichever holds them and mirrors the dirs a mode needs.

- **S3 layout** - one dir per model (`openvino_model.xml` + `.bin` + tokenizer; NLI also `config.json` for the entailment index); the dirs sit at the bucket root by default - `s3://your-bucket/<model>/`
- **Recommended grouping** - on a shared bucket, group the IRs under a `models/` prefix (`s3://your-bucket/models/<model>/`) and keep deployment config under a sibling `settings/`; point `--source` at the `models/` prefix
- **mmbert-openvino-int8** - 329 MB, the mmBERT statement encoder (both modes)
- **sat-3l-sm** - 816 MB, the SaT segmenter (both modes; `config.json` + `model.safetensors`, and the tokenizer files in a complete folder). `init --source` copies every file of the source folder into `<home>/models/sat-3l-sm/`. When the copied folder holds a `tokenizer.json`, `init` loads the tokenizer from the folder and reads its three special token ids (`cls`, `sep`, `pad`). When the folder holds no `tokenizer.json`, the load fails or an id is missing, `init` saves the tokenizer `facebookAI/xlm-roberta-base` beside the weights, over the tokenizer files of the folder: from the HuggingFace cache of the host when it is there, otherwise from the Hub; when neither has it, `init` stops with an error that names the tokenizer. A source folder that carries the tokenizer files that `init` saved needs no Hub. To make a source folder complete, copy the tokenizer files that `init` saved in `<home>/models/sat-3l-sm/` (`tokenizer.json` and `tokenizer_config.json` with transformers 5.0.0) into the source folder `sat-3l-sm`. The files `tokenizer.json` and `tokenizer_config.json` as the Hub repository `facebookAI/xlm-roberta-base` publishes them load in that folder without the special token ids (transformers 5.0.0), and a `tokenizer.json` that an interrupted write left truncated does not load: `init` replaces both. The segmenter loads the weights and the tokenizer from `<home>/models/sat-3l-sm/`; 1.1.5 requested the tokenizer from the Hub at the first load
- **reranker-openvino-int8** - 563 MB, bge-reranker-v2-m3 cross-encoder (wmd-wrt-source only)
- **nli-openvino-int8** - 321 MB, mDeBERTa MNLI/XNLI head (wmd-wrt-source only)
- **Mode totals** - `wmd` ≈ 1.15 GB (mmbert + sat); `wmd-wrt-source` ≈ 2.03 GB (+ reranker + nli)

## Provisioning from S3

- **CLI** - `docdistance init <mode> --source s3://your-bucket` (or `.../models` if grouped); add `--aws-profile NAME` locally, omit it in Lambda to use the execution-role credential chain; `--aws-endpoint-url URL` targets an S3-compatible store
- **Python** - `docdistance.init(mode, source="s3://your-bucket", home="/tmp/docdistance")`
- **Resolution order** - per model: the source named by `--source` (an S3 prefix or a local dir, whichever was given), then HuggingFace; the source served is recorded per model in `docdistance.json`
- **Any prefix** - `--source` accepts the bucket root or any `s3://bucket/prefix`; mirror the dirs to your own bucket with `make sync_models_up` (or `aws s3 sync`)
- **Extra** - the S3 path needs botocore: `pip install 'docdistance[s3]'` (or bake it into the image)
- **Readiness** - `init` writes `docdistance.json` to `$DOCDISTANCE_HOME` (else the current folder); a mode never init'd exits 1 with `run: docdistance init <mode>`

## Lambda

- **Writable home** - the task root is read-only; set `DOCDISTANCE_HOME=/tmp/docdistance` so `init` writes the model mirror + `docdistance.json` to `/tmp`
- **Ephemeral storage** - size `/tmp` above the mode total: ≥ ~1.5 GB for `wmd`, ≥ ~2.5 GB for `wmd-wrt-source`; Lambda `/tmp` defaults to 512 MB, configurable to 10 GB
- **Memory and vCPU** - Lambda allocates ~1 vCPU per 1769 MB, up to 6 vCPU at 10240 MB; the OpenVINO INT8 path is CPU-bound, so more memory buys more cores and a faster reranker grid
- **Offline** - each model loader holds the HuggingFace Hub library in offline mode while its model loads (`offline=True`, the default) and puts the earlier mode back afterwards, so a load reads the init mirror or the HuggingFace cache and makes no Hub calls; the loaders do not set the `HF_HUB_OFFLINE` environment variable
- **IAM** - the execution role needs `s3:GetObject` + `s3:ListBucket` on `arn:aws:s3:::your-bucket` and `arn:aws:s3:::your-bucket/*`

## Deployment shapes

- **Models baked into the image** - copy the model dirs into the container and `init <mode> --source /var/task/models` (local resolution); no S3 at runtime, fastest cold start, the image must hold ~2 GB (Lambda container limit 10 GB)
- **S3 at cold start** - `init <mode> --source s3://...` into `/tmp` on a container's first invocation; one ~2 GB download amortized across warm invocations
- **EFS mount** - mirror the models onto an EFS access point and `init <mode> --source /mnt/models` (local); shared across functions, no per-container download

## Latency

Measured single-core CPU OpenVINO INT8 on a workstation; per-Lambda figures extrapolate by vCPU and are not yet measured on Lambda.

- **wmd (symmetric)** - sub-millisecond per pair after load; the production default for similarity, dedup and drift detection
- **wmd-wrt-source (grounding)** - the reranker x NLI grid dominates at ~67 s/pair CPU (length-bucketed); size the function timeout accordingly (Lambda max 15 min) and prefer batch / async invocation over synchronous request paths
- **Portable** - the same INT8 IRs run on any x86-64 (AVX2 / AVX-512-VNNI) or ARM CPU, so one set of artifacts serves Lambda, Fargate and a laptop
