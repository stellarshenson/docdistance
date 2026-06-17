# mmBERT Quantization Solution

How the mmBERT statement encoder is quantized for the document-distance pipeline, and why the answer is two different artifacts - one for CPU, one for GPU. The work lives in `notebooks/02-kj-mmbert-quantization.ipynb` and the shipped CPU model is [`stellars/mmBERT-base-openvino-int8`](https://huggingface.co/stellars/mmBERT-base-openvino-int8).

## The problem

The pipeline embeds statements with `jhu-clsp/mmBERT-base` (multilingual ModernBERT, 307M parameters, 768-dim) and compares documents by optimal transport over those embeddings. A quantized encoder cuts memory and latency, but only earns its place if it preserves the relative similarities transport depends on. So every quantization choice is judged by one number: the correlation of its pairwise similarities against the FP32 reference. The goal had two halves - good CPU performance and excellent GPU performance - and those turned out to need different toolchains.

## Two targets, two artifacts

The single most important finding: there is no one quantized model that serves both targets well.

- **CPU** - OpenVINO INT8. OpenVINO is an Intel/CPU runtime; on an NVIDIA box it gives the CPU win and zero GPU acceleration
- **GPU** - torchao FP8, fused by `torch.compile`. This is the only path that turns the high-fidelity quantization into real NVIDIA-GPU speed

Both start from the same SmoothQuant insight but compile to different kernels.

## Performance at a glance

All configurations on one scale, normalized to CPU FP (full precision) as the 1.0x baseline.

| config | ms/sentence | sentences/sec | speedup |
|---|---|---|---|
| CPU FP (base, full precision) | 30.6 | 33 | 1.0x |
| CPU OpenVINO INT8 | 21.4 | 47 | 1.4x |
| GPU bf16 eager (raw base) | 0.84 | 1196 | 37x |
| GPU bf16 compiled | 0.44 | 2281 | 70x |
| **GPU FP8 + compile** | **0.39** | **2588** | **79x** |

Read top to bottom: CPU INT8 buys 1.4x over CPU FP; moving to the GPU at all is a ~37x jump; compiling doubles that to ~70x; FP8 tensor cores take it to ~79x over the CPU baseline (and ~55x over the shipped CPU INT8 model). Fidelity stays near-identical across rows (GPU 0.999, CPU 0.98 vs FP32).

The cross-device multiple is directional, not a like-for-like benchmark: GPU rows are throughput at batch 128 / seq 128 (tensor cores saturated), CPU rows are per-sentence latency at small batch. A batched CPU run would lift CPU sentences/sec somewhat, but the order of magnitude holds.

## CPU solution - OpenVINO INT8 (SmoothQuant)

The CPU artifact is a real OpenVINO INT8 IR produced by `openvino.convert_model` (optimum-intel does not yet support the ModernBERT architecture for export, so the graph is traced directly) followed by `nncf.quantize` with SmoothQuant.

- **Scheme** - per-channel symmetric INT8 weights, per-tensor static INT8 activations, SmoothQuant alpha 0.6
- **Fidelity** - Pearson 0.956 on in-domain statements, 0.979 on a public generic sentence set, vs FP32
- **Performance** - 1.4x faster (29.1 → 20.6 ms/sentence, CPU), 1.98x smaller (615 → 310 MB)
- **Shipped** - [`stellars/mmBERT-base-openvino-int8`](https://huggingface.co/stellars/mmBERT-base-openvino-int8)

OpenVINO's runtime quantizes activations per-tensor and static, which is what caps CPU fidelity at ~0.96 - a coarser scheme than the simulation's ceiling, but a clean, dependency-light CPU deployable.

## GPU solution - torchao FP8 + torch.compile

On the RTX 5000 Ada (sm_89, FP8 tensor cores), the sweep measured fidelity against FP32 and real throughput at a batch-128, sequence-128 workload.

| scheme | sent/s | vs bf16-eager | vs bf16-compiled | fidelity |
|---|---|---|---|---|
| bf16 eager | 1218 | 1.00x | - | - |
| bf16 compiled | 2260 | 1.85x | 1.00x | - |
| int8 weight-only | 33 | 0.03x | 0.01x | 0.999 |
| int8 dynamic W8A8 | 2025 | 1.66x | 0.90x | 0.996 |
| **FP8 dynamic e4m3** | **2599** | **2.13x** | **1.15x** | **0.999** |

- **Winner** - FP8 dynamic e4m3 with `torch.compile`: 2.13x over the bf16 baseline, 1.15x over even compiled bf16, near-lossless 0.999 agreement
- **Recipe, not a file** - the GPU model is `quantize_(model, Float8DynamicActivationFloat8WeightConfig())` applied to the FP checkpoint at load, then `torch.compile`; torchao re-quantizes cheaply, avoiding fragile quantized-tensor serialization

## Inference

Recommended operating point per target, measured on the shipped INT8 IR (CPU, 32-core Threadripper 7975WX) and the FP8 recipe (GPU, RTX 5000 Ada). Choose by whether single-request latency or aggregate throughput matters. CPU numbers are also given per core (aggregate / 32) so the operating point is hardware-independent.

| target | config | batch | sent/s | sent/s/core | ms/sentence |
|---|---|---|---|---|---|
| CPU latency | LATENCY hint, 1 stream | 1 | 52 | 1.6 | 19.3 |
| CPU latency | LATENCY hint, 1 stream | 64 | 137 | 4.3 | 7.3 |
| CPU throughput | THROUGHPUT hint, 16 streams | 1 | 131 | 4.1 | 7.6 |
| CPU throughput | THROUGHPUT hint, 16 streams | 64 | 195 | 6.1 | 5.1 |
| GPU throughput | FP8 + torch.compile | 32 | ~3180 | - | 0.31 |

- **CPU optimum** - OpenVINO THROUGHPUT hint (16 streams across the 32 cores), batch 64 → 195 sent/s, **6.1 sent/s/core**
- **Per-core ceiling** - each core tops out at ~6 sent/s under the THROUGHPUT hint; single-stream LATENCY caps at ~4.3 sent/s/core, so multi-stream is what turns 1 good core into 32 used cores
- **CPU latency mode** - LATENCY hint, batch 1 → 52 sent/s (1.6 sent/s/core), 19.3 ms/sentence; use when one request must return fast
- **CPU saturation** - both hints flatten by batch 32-64; batching past 64 adds little
- **GPU optimum** - FP8 + `torch.compile`, batch ~32 → ~3180 sent/s; batching past the knee only costs memory
- **Cross-target** - the whole 32-core CPU at its optimum (195 sent/s) is ~16x slower than a single GPU at its optimum (~3180 sent/s)

- **Rule** - CPU serving: THROUGHPUT hint + batch ≥ 32 (expect ~6 sent/s/core); CPU single request: LATENCY hint + batch 1; GPU: compile + batch ~32, FP8 over bf16 for a ~10% edge

## Key technical findings

The story is in why the numbers land where they do.

- **Per-token beats per-tensor** - the simulation showed naive INT8 jumps from 0.929 (per-tensor) to 0.998 (per-token) activation quantization. Per-token gives each token its own scale, so one outlier token cannot coarsen the rest. Once activations are per-token, SmoothQuant adds only ~0.001 - the per-token scaling already tames the outliers SmoothQuant targets
- **OpenVINO cannot express per-token on NVIDIA** - its INT8 runtime is per-tensor static, which is why the deployed CPU fidelity (0.956) sits below the simulated ceiling (0.999). The gap is the runtime's scheme, not the model
- **torchao needs compilation to be fast** - in eager mode every quant scheme is slower than bf16 (INT8 dynamic was 14x slower) because the quant is not fused into the tensor-core GEMM. `torch.compile` fuses it; without compile there is no GPU speedup
- **Compilation needs a C compiler** - Inductor/Triton build the fused kernels with a host C compiler, which the environment lacked. Installing the conda-forge toolchain and pointing `CC`/`CXX` at it unblocked the entire GPU speedup path
- **Compiling bf16 alone is already a 1.85x win** - part of "GPU performance" is simply compiling the model; FP8 then adds a further 1.15x on top
- **INT8 weight-only is a trap here** - 33 sent/s (0.03x); its kernel path dequantizes weights to bf16 with no GEMM benefit. Avoid it for this model

## Recommendations

- **CPU / GPU-free serving** - use the OpenVINO INT8 model from the Hub; 0.96 fidelity, 1.4x, 310 MB
- **GPU serving** - apply the FP8 torchao recipe and `torch.compile` (ensure `CC`/`CXX` point at a C compiler); 0.999 fidelity, 2.13x throughput
- **Closing the CPU gap** - per-token-dynamic INT8 fidelity (~0.99) on CPU would need a runtime that supports it (ONNX Runtime dynamic, torch.ao), not OpenVINO default - a future option if CPU fidelity must rise

## Caveats

- The torchao FP8/INT8 GPU win is measured at a realistic batch and sequence; at tiny batch or very short sequences the GEMMs are too small for tensor-core gains to show
- FP8 requires Ada or newer tensor cores; the recipe falls back to INT8 dynamic on older GPUs
- The OpenVINO INT8 latency is CPU; OpenVINO does not run INT8 on NVIDIA GPUs
- Installing `optimum-intel[openvino]` downgraded transformers 5.10 to 5.0 environment-wide; mmBERT still loads, but note the pin if other work depends on 5.10
