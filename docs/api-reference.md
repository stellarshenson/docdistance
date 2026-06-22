# API reference

The `docdistance` public API: a one-shot function per distance, a reusable pipeline for many pairs, two
result objects, and a three-command CLI. Everything below is exported from the top-level `docdistance`
package; the SOTA docs carry the mechanics.

## Library - high-level

The entry points most callers use. Inputs are raw text or a path to a text/markdown file (auto-detected);
a leading markdown `# ` title line in a file is stripped so the title is not counted as a statement.

| Symbol | Signature | Returns | Notes |
| --- | --- | --- | --- |
| `document_distance` | `(a, b, *, backend="openvino", anisotropy=False, threshold=0.725, offline=True, device=None)` | `DistanceResult` | symmetric SMD; loads models then scores in one call |
| `source_conditioned_distance` | `(a, b, source, *, backend="openvino", anisotropy=True, offline=True, device=None)` | `SourceConditionedResult` | `d(A, B | S)`; selection divergence + each document's distance to `S` |
| `DocDistance` | `DocDistance(backend="openvino", offline=True, device=None)` | pipeline | construct once (models load here), then score many pairs |
| `DocDistance.distance` | `(a, b, *, anisotropy=False, threshold=0.725)` | `DistanceResult` | symmetric distance on the loaded models |
| `DocDistance.distance_with_map` | `(a, b, *, anisotropy=False, threshold=0.725)` | `(DistanceResult, dict)` | the distance plus the optimal-transport statement map, one encode pass |
| `DocDistance.distance_wrt_source` | `(a, b, source, *, anisotropy=True)` | `SourceConditionedResult` | source-conditioned distance on the loaded models |
| `DocDistance.embed` | `(doc)` | `ndarray [n, dim]` | segment then embed into L2-normalized statement vectors |

- **backend** - `"openvino"` (CPU INT8, default) or `"torch"`; pass `device="cuda"` with `backend="torch"` for GPU
- **offline** - `True` loads from the local cache only; run `docdistance install` once to populate it
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
| `compute_distance` | `(X, Y, *, anisotropy=False, threshold=0.725)` | `DistanceResult` | assemble the full symmetric result from embeddings |
| `compute_source_conditioned` | `(X, Y, S, *, anisotropy=True)` | `SourceConditionedResult` | assemble the conditioned result from embeddings |

- **bound chain** - `WCD ≤ RWMD ≤ SMD`, the two cheap bounds bracket the exact distance below
- **ground cost** - `√(2 − 2cos)` on L2-normalized embeddings, a metric, so SMD is a metric too

## Result objects

Both are dataclasses with a `to_dict()` method (the shape the CLI `--json` emits).

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

`SourceConditionedResult`:

| Field | Type | Meaning |
| --- | --- | --- |
| `d_sel` | `float` | selection divergence: metric OT between the two coverage profiles over `S` |
| `residual_a`, `residual_b` | `float` | each document's geometric distance to the source |
| `closeness_a`, `closeness_b` | `float` | the residuals mapped to 0..1 |
| `n_statements_a`, `n_statements_b`, `n_statements_source` | `int` | statement counts |
| `coverage_a`, `coverage_b` | `list[float]` | each document's coverage distribution over the source statements |

## CLI

`docdistance <command>`; human output is rich and coloured, `--json` is machine-readable, `--result-only` is
the bare scalar. Logs go to stderr, so stdout carries only the result.

| Command | Purpose | Key options |
| --- | --- | --- |
| `install` | download + cache the models (the only command that fetches) | `--backend openvino\|torch\|both` |
| `distance A B` | symmetric SMD between two documents | `--backend`, `--gpu`, `--anisotropy`, `--threshold`, `--transport-map-json`, `--json`, `--result-only` |
| `distance-wrt-source A B --source S` | source-conditioned `d(A, B | S)` | `--source/-s` (required), `--backend`, `--gpu`, `--json`, `--result-only` |

## Examples

One-shot symmetric distance:

```python
from docdistance import document_distance

r = document_distance("report_v1.md", "report_v2.md")
print(r.closeness)   # 0..1 similarity, 1 - smd/sqrt(2)
print(r.verdict)     # "similar" | "not similar"
print(r.smd, r.wcd, r.rwmd)
```

Reusable pipeline - load once, score many pairs:

```python
from docdistance import DocDistance

dd = DocDistance(backend="openvino")        # models load here
for a, b in pairs:
    print(dd.distance(a, b).closeness)      # no reload per call
```

Source-conditioned distance `d(A, B | S)`:

```python
from docdistance import source_conditioned_distance

r = source_conditioned_distance("summary_a.md", "summary_b.md", source="article.md")
print(r.d_sel)                       # how differently A and B select from the source
print(r.residual_a, r.residual_b)    # each summary's distance to the source
```

Transport map - the interpretable statement-to-statement alignment behind the distance:

```python
from docdistance import DocDistance

dd = DocDistance()
result, tmap = dd.distance_with_map("report_v1.md", "report_v2.md")
print(result.smd)                                 # the distance
for flow in tmap["flows"]:                        # each statement of A
    best = flow["matches"][0]                     # the B statement it maps to (most mass)
    print(flow["text"], "->", best["target_text"], best["weight"], best["cost"])
```

The low-level `transport_plan(X, Y)` returns the raw `[n_X, n_Y]` coupling if you hold the embeddings
and want the matrix directly; `distance_with_map` is the text-aware wrapper that pairs it with statements.

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
docdistance install                                              # cache the models once
docdistance distance a.md b.md                                   # rich, coloured verdict
docdistance distance a.md b.md --json                            # machine-readable
docdistance distance a.md b.md --result-only                     # bare SMD scalar
docdistance distance-wrt-source sum_a.md sum_b.md -s article.md  # source-conditioned
```
