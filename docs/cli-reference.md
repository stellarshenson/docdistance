# CLI reference

`docdistance <command>`: three commands - `install` (fetch models), `distance` (symmetric SMD), and
`distance-wrt-source` (source-conditioned `d(A,B|S)`). Human output is rich and coloured; `--json` is
machine-readable, `--result-only` is the bare scalar. Logs go to stderr, so stdout carries only the result.

- **Install** - `pip install docdistance`, then `docdistance install` once to cache the models
- **Arguments** - `A`, `B`, `S` are each a file path or raw text (auto-detected); a leading markdown `# ` title line in a file is stripped so the title is not a statement
- **Global** - `--version` prints the version; `--help` (and `<command> --help`) prints usage; exit code 1 on a missing model or an unsecured `--gpu`
- **Backends** - `--backend openvino` (CPU INT8, default) or `torch`; `--gpu` forces torch on CUDA and errors if GPU support is not installed
- **Offline** - after `docdistance install`, every distance call runs fully offline from the local cache

## install

Download and cache the models - the only command that fetches from the Hub.

| Flag | Default | Effect |
| --- | --- | --- |
| `--backend openvino\|torch\|both` | `both` | which encoder weights to fetch |
| `--verbose`, `-v` | off | DEBUG logging to stderr |

## distance

Symmetric Statement Mover's Distance between two documents - the exact metric, one scalar.

| Flag | Default | Effect |
| --- | --- | --- |
| `--backend openvino\|torch` | `openvino` | statement encoder backend |
| `--gpu` | off | force the torch backend on CUDA; errors if GPU support is not secured |
| `--anisotropy / --no-anisotropy` | `--no-anisotropy` | all-but-the-top anisotropy removal; needs a corpus, off for a bare pair |
| `--threshold FLOAT` | `0.725` | closeness cutoff for the similar / not-similar verdict |
| `--transport-map-json FILE` | off | also write the optimal-transport map (which B statements each A statement's mass flows to, with weights and match cost) to `FILE` |
| `--json` | off | machine-readable JSON to stdout |
| `--result-only` | off | bare SMD scalar to stdout, no clutter |
| `--verbose`, `-v` | off | DEBUG logging to stderr |

- **Default output** - a rich panel: SMD, closeness, verdict + threshold, the `WCD ≤ RWMD ≤ SMD` bounds, statement counts, anisotropy on/off
- **`--json`** - the result dict: `smd`, `wcd`, `rwmd`, `closeness`, `threshold`, `verdict`, `anisotropy`, `n_statements_a`, `n_statements_b`
- **`--transport-map-json`** - writes a separate JSON file (the result still prints as usual): `{smd, anisotropy, n_statements, flows}`, where `flows` is a per-A-statement list of `{index, text, matches}` and each match is `{target_index, target_text, weight, cost}` - the B statements that statement's transport mass flows to, `weight` the fraction of its mass (sums to 1 per statement), `cost` the ground distance of the match; the exact OT coupling behind the SMD, so a human or a machine can read which statement aligns to which
- **`--result-only`** - the bare SMD float, for scripts
- **Reading it** - closeness `1.0` identical, `0.0` unrelated; near 1 with verdict `similar` is a faithful match, falling toward 0 with `not similar` means the meaning changed; the transport map names *which* statement of B each statement of A maps to and how good the match is (low `cost`)

## distance-wrt-source

Source-conditioned distance `d(A, B | S)` - selection divergence plus each document's distance to the shared source `S`.

| Flag | Default | Effect |
| --- | --- | --- |
| `--source`, `-s` | required | the common source document |
| `--backend openvino\|torch` | `openvino` | statement encoder backend |
| `--gpu` | off | force the torch backend on CUDA; errors if GPU support is not secured |
| `--anisotropy / --no-anisotropy` | `--anisotropy` | anisotropy removal on the conditioned selection axis, on by default (E04-H15) |
| `--source-map-json FILE` | off | also write a statement → source alignment map (top-3 source statements each A/B statement covers, with weights) to `FILE` |
| `--json` | off | machine-readable JSON to stdout |
| `--result-only` | off | bare `D_sel,residual_a,residual_b` to stdout |
| `--verbose`, `-v` | off | DEBUG logging to stderr |

- **Default output** - a rich panel: `D_sel` (selection divergence), each document's residual distance to the source + closeness, statement counts (A / B / S)
- **`--json`** - the result dict: `d_sel`, `residual_a`, `residual_b`, `closeness_a`, `closeness_b`, `n_statements_a`, `n_statements_b`, `n_statements_source`, `coverage_a`, `coverage_b`
- **`--source-map-json`** - writes a separate JSON file (the result still prints as usual): `{top_k, anisotropy, n_statements, a, b}`, where `a` and `b` are per-statement lists of `{index, text, matches}` and each match is `{source_index, source_text, weight}` - the top-3 source statements that statement covers, weights summing toward 1
- **`--result-only`** - `D_sel,residual_a,residual_b`, comma-separated
- **Reading it** - both are distances, lower is closer; `d_sel` near 0 means A and B drew on the same source content, a high `residual` flags a document that drifted from `S` (dropped or unsupported content); the source map names *which* source statements each statement covers

## Examples

```bash
# one-time setup
pip install docdistance                        # from PyPI
docdistance install                            # cache both backends
docdistance install --backend openvino         # cache CPU INT8 only

# method 1 - symmetric distance
docdistance distance report_v1.md report_v2.md
docdistance distance "first text" "second text"            # raw text, not files
docdistance distance a.md b.md --json                      # machine-readable
docdistance distance a.md b.md --transport-map-json map.json   # + statement → statement map
docdistance distance a.md b.md --result-only               # bare SMD scalar
docdistance distance a.md b.md --threshold 0.8             # stricter verdict
docdistance distance a.md b.md --gpu                       # torch on CUDA

# method 2 - source-conditioned d(A,B|S)
docdistance distance-wrt-source sum_a.md sum_b.md --source article.md
docdistance distance-wrt-source a.md b.md -s s.md --json
docdistance distance-wrt-source a.md b.md -s s.md --source-map-json map.json   # + statement → source map
docdistance distance-wrt-source a.md b.md -s s.md --result-only   # D_sel,res_a,res_b
```
