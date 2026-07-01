# CLI reference

`docdistance <command>`: four commands - `init` (provision a mode's models), `distance-semantic` (symmetric content SMD),
`distance-structural` (symmetric order distance), and `distance-wrt-source` (source-conditioned `d(A,B|S)`). Human output is rich and coloured; `--json` is
machine-readable, `--result-only` is the bare scalar. Logs go to stderr, so stdout carries only the result.

- **Init** - `pip install docdistance`, then `docdistance init <mode>` once to provision that mode's models
- **Arguments** - `A`, `B`, `S` are each a file path or raw text (auto-detected); a leading markdown `# ` title line in a file is stripped so the title is not a statement
- **Global** - `--version` prints the version; `--help` (and `<command> --help`) prints usage; exit code 1 on a missing model, an un-init'd mode, or an unsecured `--gpu`
- **Backends** - `--backend openvino` (CPU INT8, default) or `torch`; `--gpu` forces torch on CUDA and errors if GPU support is not installed
- **Offline** - after `docdistance init <mode>`, every distance call for that mode runs fully offline from the local cache

## init

Provision a mode's models from local / S3 / HuggingFace and write `docdistance.json` (the readiness record). `MODE` is `wmd` (symmetric: mmBERT + SaT) or `wmd-wrt-source` (adds the bge-reranker-v2-m3 + mDeBERTa NLI grounding models). The only command that downloads.

| Flag | Default | Effect |
| --- | --- | --- |
| `MODE` (argument) | `wmd` | which distance mode to provision (`wmd` or `wmd-wrt-source`) |
| `--source URI` | HuggingFace | model source base: an `s3://bucket/prefix`, a local dir, or omit for the Hub |
| `--backend openvino\|torch` | `openvino` | which weights to fetch (OpenVINO INT8 or torch) |
| `--aws-profile NAME` | none | AWS named profile for an `s3://` source (omit in Lambda for the execution-role chain) |
| `--aws-endpoint-url URL` | none | custom S3 endpoint for an S3-compatible store |
| `--region NAME` | none | AWS region for an `s3://` source |
| `--home DIR` | `$DOCDISTANCE_HOME` or cwd | where to write `docdistance.json` + the model mirror |
| `--verbose`, `-v` | off | DEBUG logging to stderr |

- **Resolution** - per model: an `s3://` prefix, then a local dir, then the HuggingFace Hub (the always-available fallback); the source served is recorded per model in `docdistance.json`
- **Readiness** - a distance run whose mode is not init'd exits 1 with `mode '<mode>' is not initialized - run:  docdistance init <mode>`

## distance-semantic

Symmetric Statement Mover's Distance between two documents - the exact content metric, one scalar.

| Flag | Default | Effect |
| --- | --- | --- |
| `--backend openvino\|torch` | `openvino` | statement encoder backend |
| `--gpu` | off | force the torch backend on CUDA; errors if GPU support is not secured |
| `--anisotropy / --no-anisotropy` | `--no-anisotropy` | all-but-the-top anisotropy removal; needs a corpus, off for a bare pair |
| `--threshold FLOAT` | `0.725` | closeness cutoff for the similar / not-similar verdict |
| `--details-json FILE` | off | also write the content alignment (which B statements each A statement's mass flows to, with weights, match cost, and a per-flow changed flag) to `FILE` |
| `--json` | off | machine-readable JSON to stdout |
| `--result-only` | off | bare SMD scalar to stdout, no clutter |
| `--verbose`, `-v` | off | DEBUG logging to stderr |

- **Default output** - a rich panel: SMD, closeness, verdict + threshold, the `WCD ≤ RWMD ≤ SMD` bounds, statement counts, anisotropy on/off
- **`--json`** - the result dict: `smd`, `wcd`, `rwmd`, `closeness`, `threshold`, `verdict`, `anisotropy`, `n_statements_a`, `n_statements_b`
- **`--details-json`** - writes a separate JSON file (the result still prints as usual): `{smd, anisotropy, n_statements, flows}`, where `flows` is a per-A-statement list of `{index, text, matches, changed}` and each match is `{target_index, target_text, weight, cost}` - the B statements that statement's transport mass flows to, `weight` the fraction of its mass (sums to 1 per statement), `cost` the ground distance of the match (the semantic gap of the aligned pair), `changed` = the statement's best-match `cost` past the changed-cost cutoff; the exact OT coupling behind the SMD, so a human or a machine can read which statement aligns to which and what drifted in MEANING
- **`--result-only`** - the bare SMD float, for scripts
- **Reading it** - closeness `1.0` identical, `0.0` unrelated; near 1 with verdict `similar` is a faithful match, falling toward 0 with `not similar` means the meaning changed; the content alignment names *which* statement of B each statement of A maps to and how good the match is (low `cost`)

## distance-structural

Symmetric statement-order distance between two documents - the E11-H55 OPW order-gap, translation-invariant, with a `structure_closeness` readout on the SMD-closeness scale.

| Flag | Default | Effect |
| --- | --- | --- |
| `--backend openvino\|torch` | `openvino` | statement encoder backend |
| `--gpu` | off | force the torch backend on CUDA; errors if GPU support is not secured |
| `--anisotropy / --no-anisotropy` | `--no-anisotropy` | all-but-the-top anisotropy removal; needs a corpus, off for a bare pair |
| `--threshold FLOAT` | `0.725` | closeness cutoff for the similar / not-similar verdict |
| `--details-json FILE` | off | also write the order details (per statement: its aligned target and rank displacement) to `FILE` |
| `--json` | off | machine-readable JSON to stdout |
| `--result-only` | off | bare order_gap scalar to stdout, no clutter |
| `--verbose`, `-v` | off | DEBUG logging to stderr |

- **Default output** - a rich panel: order_gap, structure_closeness, verdict + threshold, statement counts, anisotropy on/off
- **`--json`** - the result dict: `smd`, `order_gap`, `structure_closeness`, `threshold`, `verdict`, `anisotropy`, `n_statements_a`, `n_statements_b`
- **`--details-json`** - writes a separate JSON file (the result still prints as usual): `{smd, order_gap, structure_closeness, anisotropy, n_statements, statements}`, where `order_gap` is the E11-H55 OPW structural distance (translation-invariant, `>= 0`) and `structure_closeness` = `1 - order_gap / sqrt(2)` the shipped SOTA 0..1 structure readout on the SMD-closeness scale (1 = same order); `statements` is a per-A-statement list of `{index, text, target_index, target_text, displacement, moved}` - `displacement` the rank shift of that statement (0 = in place), `moved` = `displacement != 0`; this pins down precisely what MOVED in order, statement by statement
- **`--result-only`** - the bare order_gap float, for scripts
- **Reading it** - structure_closeness `1.0` same order, `0.0` fully reordered; the order details name *which* statement of B each statement of A maps to and how far it MOVED (`displacement`)

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
| `--result-only` | off | bare `D_sel,grd_a,grd_b` to stdout |
| `--verbose`, `-v` | off | DEBUG logging to stderr |

- **Requires** - `docdistance init wmd-wrt-source` (pulls the reranker + NLI grounding models); un-init'd exits 1
- **Default output** - a rich panel: `D_sel` (selection divergence), each document's grounding residual `grd_a` / `grd_b` (E03-H11) and the `D_grd` separation, the geometric residual to source + closeness, statement counts (A / B / S)
- **`--json`** - the result dict: `d_sel`, `residual_a`, `residual_b`, `closeness_a`, `closeness_b`, `n_statements_a`, `n_statements_b`, `n_statements_source`, `coverage_a`, `coverage_b`, `grd_a`, `grd_b`, `d_grd`
- **`--source-map-json`** - writes a separate JSON file (the result still prints as usual): `{top_k, anisotropy, n_statements, a, b}`, where `a` and `b` are per-statement lists of `{index, text, matches}` and each match is `{source_index, source_text, weight}` - the top-3 source statements that statement covers, weights summing toward 1
- **`--result-only`** - `D_sel,grd_a,grd_b`, comma-separated
- **Reading it** - lower is closer on every axis; `d_sel` near 0 means A and B drew on the same source content, a high grounding residual `grd_a` / `grd_b` flags a document that drifted from `S` (dropped or unsupported content); the source map names *which* source statements each statement covers

## Examples

```bash
# one-time setup
pip install docdistance                        # from PyPI ('docdistance[s3]' for S3 sources)
docdistance init wmd                           # provision the symmetric-distance models
docdistance init wmd-wrt-source                # + the reranker + NLI grounding models
docdistance init wmd-wrt-source --source s3://your-bucket --aws-profile NAME
docdistance init wmd --source /path/to/models  # from a local mirror

# method 1 - semantic (content) distance
docdistance distance-semantic report_v1.md report_v2.md
docdistance distance-semantic "first text" "second text"          # raw text, not files
docdistance distance-semantic a.md b.md --json                    # machine-readable
docdistance distance-semantic a.md b.md --details-json content.json   # + statement → statement content map
docdistance distance-semantic a.md b.md --result-only             # bare SMD scalar
docdistance distance-semantic a.md b.md --threshold 0.8           # stricter verdict
docdistance distance-semantic a.md b.md --gpu                     # torch on CUDA

# method 2 - structural (order) distance
docdistance distance-structural report_v1.md report_v2.md
docdistance distance-structural a.md b.md --json                  # machine-readable
docdistance distance-structural a.md b.md --details-json order.json   # + per-statement order displacement
docdistance distance-structural a.md b.md --result-only           # bare order_gap scalar

# method 3 - source-conditioned d(A,B|S)
docdistance distance-wrt-source sum_a.md sum_b.md --source article.md
docdistance distance-wrt-source a.md b.md -s s.md --json
docdistance distance-wrt-source a.md b.md -s s.md --source-map-json map.json   # + statement → source map
docdistance distance-wrt-source a.md b.md -s s.md --result-only   # D_sel,grd_a,grd_b
```
