# API reference

The `docdistance` public API: a one-shot function per distance, a reusable pipeline for many pairs, three
result objects, and a four-command CLI. Everything below is exported from the top-level `docdistance`
package; the SOTA docs carry the mechanics.

## Library - high-level

The entry points most callers use. Inputs are raw text or a path to a text/markdown file (auto-detected);
a leading markdown `# ` title line in a file is stripped so the title is not counted as a statement.

| Symbol | Signature | Returns | Notes |
| --- | --- | --- | --- |
| `init` | `(mode="wmd", *, source=None, backend="openvino", aws_profile=None, aws_endpoint_url=None, aws_region=None, home=None)` | `dict` | provision a mode's models from local / S3 / HuggingFace; writes `docdistance.json`, returns a per-model source summary |
| `semantic_distance` | `(a, b, *, backend="openvino", anisotropy=False, threshold=0.725, offline=True, device=None)` | `DistanceResult` | symmetric SMD; loads models then scores in one call |
| `structural_distance` | `(a, b, *, backend="openvino", anisotropy=False, threshold=0.725, offline=True, device=None)` | `StructuralResult` | symmetric OPW order-gap; loads models then scores in one call |
| `source_conditioned_distance` | `(a, b, source, *, backend="openvino", anisotropy=True, offline=True, device=None)` | `SourceConditionedResult` | `d(A, B | S)`; selection axis + reranker x NLI grounding residuals |
| `DocDistance` | `DocDistance(backend="openvino", offline=True, device=None)` | pipeline | construct once, models load lazily on first use, then score many pairs |
| `DocDistance.semantic_distance` | `(a, b, *, anisotropy=False, threshold=0.725)` | `DistanceResult` | symmetric distance on the loaded models |
| `DocDistance.semantic_distance_with_details` | `(a, b, *, anisotropy=False, threshold=0.725)` | `(DistanceResult, dict)` | the distance plus the content alignment details, one encode pass |
| `DocDistance.structural_distance` | `(a, b, *, anisotropy=False, threshold=0.725)` | `StructuralResult` | structural OPW order-gap distance on the loaded models |
| `DocDistance.structural_distance_with_details` | `(a, b, *, anisotropy=False, threshold=0.725)` | `(StructuralResult, dict)` | the structural distance plus the order details, one encode pass |
| `DocDistance.distance_wrt_source` | `(a, b, source, *, anisotropy=True)` | `SourceConditionedResult` | source-conditioned distance, runs the reranker x NLI grounding |
| `DocDistance.distance_wrt_source_with_map` | `(a, b, source, *, anisotropy=True, top_k=3)` | `(SourceConditionedResult, dict)` | the conditioned result plus the statement → source map, one encode pass |
| `DocDistance.embed` | `(doc)` | `ndarray [n, dim]` | segment then embed into L2-normalized statement vectors |

- **init** - run once per mode before scoring; `"wmd"` pulls mmBERT + SaT, `"wmd-wrt-source"` adds the bge-reranker-v2-m3 + mDeBERTa NLI; `source` is an `s3://` prefix, a local dir, or `None` for the Hub; an un-init'd mode raises `NotInitializedError`
- **backend** - `"openvino"` (CPU INT8, default) or `"torch"`; pass `device="cuda"` with `backend="torch"` for GPU
- **offline** - `True` loads from the local cache only; run `docdistance.init(mode)` once to populate it
- **anisotropy** - all-but-the-top postprocessing; off for a bare pair, on by default for the conditioned axis (E04-H15)
- **threshold** - closeness cutoff for the similar / not-similar verdict, default `0.725`

## Library - low-level

Pure functions over already-embedded statement clouds `X`, `Y` (L2-normalized `ndarray [n, dim]`); no model
loading. Use these when you hold the embeddings already.

| Symbol | Signature | Returns | Notes |
| --- | --- | --- | --- |
| `smd` | `(X, Y)` | `float` | the distance: exact Statement Mover's Distance via the network-simplex LP |
| `transport_plan` | `(X, Y)` | `ndarray [n_X, n_Y]` | the exact OT coupling behind `smd`: `T[i,j]` = mass moved from `X[i]` to `Y[j]`, marginals `1/n_X` / `1/n_Y` |
| `wcd` | `(X, Y)` | `float` | lower bound: mean-pooled cloud distance (whole-doc cosine) |
| `rwmd` | `(X, Y)` | `float` | lower bound: one-sided relaxation, greedy nearest-statement |
| `closeness` | `(d)` | `float` | map a distance to 0..1 similarity, `1 − d/√2` |
| `opw_plan` | `(X, Y)` | `ndarray [n_X, n_Y]` | soft order-preserving Sinkhorn coupling, the dense plan H55 computes then discards (use `transport_plan` for crisp pin-pointing) |
| `opw_cost` | `(X, Y)` | `float` | order-preserving transport cost `(opw_plan · cost).sum()` |
| `opw_gap` | `(X, Y)` | `float` | H55 order-gap: structural distance `max(0, opw_cost − smd)`, translation-invariant, `≥ 0` (a score, not a metric) |
| `order_alignment` | `(X, Y)` | `ndarray [n_X]` | per `X` statement, its aligned `Y` index - crisp exact-EMD argmax with a diagonal tie-break |
| `structure_displacement` | `(X, Y)` | `ndarray [n_X]` | rank shift from the crisp alignment; `0` = in place, nonzero = moved |
| `compute_distance` | `(X, Y, *, anisotropy=False, threshold=0.725)` | `DistanceResult` | assemble the full symmetric result from embeddings |
| `compute_source_conditioned` | `(X, Y, S, *, anisotropy=True, reranker_a=None, reranker_b=None, entail_a=None, entail_b=None)` | `SourceConditionedResult` | assemble the conditioned result; pass the grounding arrays for `grd_a`/`grd_b`/`d_grd` |
| `grounding_residual` | `(reranker, entail)` | `float` | E03-H11 relevance-gated ungrounded mass `mean_i (1 − entail_i)·(1 − max_j R[i,j])` |
| `grounding_blend` | `(d_sel, d_grd, *, alpha=0.75)` | `float` | E03-H14 two-axis blend `alpha·d_sel + (1 − alpha)·d_grd` |

- **bound chain** - `WCD ≤ RWMD ≤ SMD`, the two cheap bounds bracket the exact distance below
- **ground cost** - `√(2 − 2cos)` on L2-normalized embeddings, a metric, so SMD is a metric too
- **structure axis** - `opw_gap` is the H55 structural distance (order-only, content cancelled); `structure_closeness = 1 − opw_gap/√2` puts it on the same 0..1 scale as `closeness` (1 = same order)

## Result objects

All are dataclasses with a `to_dict()` method (the shape the CLI `--json` emits).

`DistanceResult`:

| Field | Type | Meaning |
| --- | --- | --- |
| `smd` | `float` | the distance |
| `wcd`, `rwmd` | `float` | the two lower bounds |
| `closeness` | `float` | `1 − smd/√2`, 0..1 |
| `threshold` | `float` | the verdict cutoff used |
| `verdict` | `str` | `"similar"` or `"not similar"` |
| `anisotropy` | `bool` | whether all-but-the-top was applied |
| `n_statements_a`, `n_statements_b` | `int` | statement counts |

`StructuralResult`:

| Field | Type | Meaning |
| --- | --- | --- |
| `smd` | `float` | the semantic distance |
| `order_gap` | `float` | H55 OPW structural distance, translation-invariant, `≥ 0` |
| `structure_closeness` | `float` | `1 − order_gap/√2`, 0..1 (1 = same order) |
| `threshold` | `float` | the verdict cutoff used |
| `verdict` | `str` | `"similar"` or `"not similar"` |
| `anisotropy` | `bool` | whether all-but-the-top was applied |
| `n_statements_a`, `n_statements_b` | `int` | statement counts |

`SourceConditionedResult`:

| Field | Type | Meaning |
| --- | --- | --- |
| `d_sel` | `float` | selection divergence: metric OT between the two coverage profiles over `S` |
| `residual_a`, `residual_b` | `float` | each document's geometric distance to the source (the metric stand-in) |
| `closeness_a`, `closeness_b` | `float` | the residuals mapped to 0..1 |
| `n_statements_a`, `n_statements_b`, `n_statements_source` | `int` | statement counts |
| `coverage_a`, `coverage_b` | `list[float]` | each document's coverage distribution over the source statements |
| `grd_a`, `grd_b` | `float \| None` | each document's reranker x NLI grounding residual (E03-H11); `None` on the metric-only path |
| `d_grd` | `float \| None` | grounding-axis separation `|grd_a − grd_b|`; `None` without grounding |

## CLI

`docdistance <command>`; human output is rich and coloured, `--json` is machine-readable, `--result-only` is
the bare scalar. Logs go to stderr, so stdout carries only the result.

| Command | Purpose | Key options |
| --- | --- | --- |
| `init [MODE]` | provision a mode's models from local / S3 / HuggingFace (the only command that fetches); writes `docdistance.json` | `--source`, `--backend`, `--aws-profile`, `--aws-endpoint-url`, `--region`, `--home` |
| `distance-semantic A B` | Statement Mover's Distance content closeness between two documents | `--backend`, `--gpu`, `--anisotropy`, `--threshold`, `--details-json`, `--json`, `--result-only` |
| `distance-structural A B` | structural OPW order-gap distance between two documents | `--backend`, `--gpu`, `--anisotropy`, `--threshold`, `--details-json`, `--json`, `--result-only` |
| `distance-wrt-source A B --source S` | source-conditioned `d(A, B | S)` | `--source/-s` (required), `--backend`, `--gpu`, `--source-map-json`, `--json`, `--result-only` |

## Examples

One-shot symmetric distance:

```python
import docdistance
from docdistance import semantic_distance

docdistance.init("wmd")                     # provision once (writes docdistance.json)
r = semantic_distance("report_v1.md", "report_v2.md")
print(r.closeness)   # 0..1 similarity, 1 - smd/sqrt(2)
print(r.verdict)     # "similar" | "not similar"
print(r.smd, r.wcd, r.rwmd)
```

Reusable pipeline - load once, score many pairs:

```python
from docdistance import DocDistance

dd = DocDistance(backend="openvino")        # construct once; models load lazily on first call
for a, b in pairs:
    print(dd.semantic_distance(a, b).closeness)  # no reload per call
```

Source-conditioned distance `d(A, B | S)`:

```python
import docdistance
from docdistance import source_conditioned_distance

docdistance.init("wmd-wrt-source")   # provision mmBERT + SaT + reranker + NLI once
r = source_conditioned_distance("summary_a.md", "summary_b.md", source="article.md")
print(r.d_sel)                       # how differently A and B select from the source
print(r.grd_a, r.grd_b)              # each summary's reranker x NLI grounding residual
```

Transport map - the interpretable statement-to-statement alignment behind the distance:

```python
from docdistance import DocDistance

dd = DocDistance()
result, tmap = dd.semantic_distance_with_details("report_v1.md", "report_v2.md")
print(result.smd)                                 # the distance
for flow in tmap["flows"]:                        # each statement of A
    best = flow["matches"][0]                     # the B statement it maps to (most mass)
    print(flow["text"], "->", best["target_text"], best["weight"], best["cost"])
```

The low-level `transport_plan(X, Y)` returns the raw `[n_X, n_Y]` coupling if you hold the embeddings
and want the matrix directly; `semantic_distance_with_details` is the text-aware wrapper that pairs it with statements.

Structural diff - what MOVED in order:

```python
from docdistance import DocDistance

dd = DocDistance()
result, details = dd.structural_distance_with_details("report_v1.md", "report_v2.md")

print(details["smd"])                  # semantic distance: how far the meaning drifted
print(details["order_gap"])            # H55 OPW structural distance, translation-invariant, >= 0
print(details["structure_closeness"])  # 1 - order_gap/sqrt(2): the 0..1 order readout (1 = same order)

for st in details["statements"]:       # one record per statement of A
    print(st["text"], "->", st["target_text"])
    print("  displacement", st["displacement"], "moved" if st["moved"] else "in place")
```

Each statement in the order details carries the structural reading. `displacement` is that statement's
rank shift under the crisp alignment - `0` means it stayed in place, nonzero means it moved; `moved`
fires on any nonzero displacement. The content reading lives in the companion `semantic_distance_with_details`
flows, where `cost` is the aligned-pair ground cost - `0` means identical meaning, higher means the content
drifted - and `changed` fires when the best-match `cost` clears `DIFF_CHANGED_COST` (the `(1 − threshold)·√2`
content cutoff). The axes are orthogonal: a statement can keep its meaning yet move (`cost ≈ 0`,
`displacement ≠ 0`), or hold its position yet change (`cost` high, `displacement = 0`). At the top level,
`smd` is the whole-document semantic distance and `order_gap` its structural counterpart, with
`structure_closeness = 1 − order_gap/√2` reading order on the same 0..1 scale as `closeness`. The same dict
is what the CLI writes with `distance-structural --details-json FILE`.

Low-level, on embeddings you already hold:

```python
from docdistance import DocDistance, smd, closeness

dd = DocDistance()
X, Y = dd.embed("a.md"), dd.embed("b.md")
d = smd(X, Y)
print(d, closeness(d))
```

CLI:

```bash
docdistance init wmd                                             # provision the symmetric models once
docdistance init wmd-wrt-source                                  # + reranker + NLI grounding models
docdistance distance-semantic a.md b.md                          # rich, coloured verdict
docdistance distance-semantic a.md b.md --json                   # machine-readable
docdistance distance-semantic a.md b.md --result-only            # bare SMD scalar
docdistance distance-structural a.md b.md                        # structural OPW order-gap
docdistance distance-wrt-source sum_a.md sum_b.md -s article.md  # source-conditioned
```
