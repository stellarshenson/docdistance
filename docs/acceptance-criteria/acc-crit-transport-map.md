# Acceptance Criteria - Transport Map

Optional interpretable output exposing the exact optimal-transport coupling behind the symmetric Statement Mover's Distance - for each statement of A, the statements of B its transport mass flows to, with the mass fraction and the ground cost of each match. Surfaced as the CLI flag `--details-json FILE` and the Python method `DocDistance.semantic_distance_with_details`; the raw coupling is `transport_plan(X, Y)`. See [Structure Diff](acc-crit-structure-diff.md) for the structural (order) companion that pairs this content map with the OPW order-gap.

- [x] **Core plan** - `transport_plan(X, Y)` returns the exact `ot.emd` coupling, shape `[n_X, n_Y]`, pure numpy, no model load
  - log: 2026-06-22 implemented (v1.0.16)
- [x] **Marginals** - row `i` sums to the source marginal `1/n_X`, column `j` to `1/n_Y` (uniform statement masses)
  - log: 2026-06-22 implemented (v1.0.16)
- [x] **Realizes SMD** - `(transport_plan(X,Y) * cost_matrix(X,Y)).sum()` equals `smd(X, Y)` within `1e-6`
  - log: 2026-06-22 implemented (v1.0.16)
- [x] **Map builder** - `_build_semantic_details(sa, ea, sb, eb, *, anisotropy=False)` returns `{smd, anisotropy, n_statements, flows}`, pure, no model load
  - log: 2026-06-22 implemented (v1.0.16)
- [x] **Flows shape** - `flows` is one entry per A statement `{index, text, matches, changed}`, `len(flows) == n_a`
  - log: 2026-06-22 implemented (v1.0.16)
- [x] **Match shape** - each match is `{target_index, target_text, weight, cost}`
  - log: 2026-06-22 implemented (v1.0.16)
- [x] **Weight semantics** - `weight` = fraction of that A statement's transport mass landing on B[j] (row-normalized); per-statement weights sum to 1
  - log: 2026-06-22 implemented (v1.0.16)
- [x] **Cost semantics** - `cost` = ground distance `sqrt(2-2cos)` of the matched pair, the match quality (low = good)
  - log: 2026-06-22 implemented (v1.0.16)
- [x] **Match ordering** - matches per statement sorted by descending weight
  - log: 2026-06-22 implemented (v1.0.16)
- [x] **Changed flag** - `changed` = the A statement's best (lowest-`cost`) match has `cost` above the change cutoff (best-match cost > cutoff); `true` flags a statement whose aligned meaning drifted
  - log: 2026-07-01 added with the semantic/structural split (v1.1.2)
- [x] **Sparse** - only nonzero flows kept (raw mass below `1e-9` dropped); network-simplex sparsity keeps the map compact
  - log: 2026-06-22 implemented (v1.0.16)
- [x] **Interpretable** - statement texts inline (`text`, `target_text`), so a human reads the mapping directly and a machine consumes the JSON without the embedding arrays
  - log: 2026-06-22 implemented (v1.0.16)
- [x] **Anisotropy consistency** - the map applies all-but-the-top exactly as `compute_distance`, so it reflects the same geometry the distance is scored on; the `anisotropy` flag is recorded in the map
  - log: 2026-06-22 implemented (v1.0.16)
- [x] **map.smd matches** - the `smd` field of the written map equals the printed SMD of the same pair
  - log: 2026-06-22 verified on ibm-ai-adoption fixtures (v1.0.16)
- [x] **API method** - `DocDistance.semantic_distance_with_details(a, b, *, anisotropy=False, threshold=0.725)` returns `(DistanceResult, dict)` sharing one encode pass
  - log: 2026-06-22 implemented (v1.0.16)
- [x] **CLI flag** - `distance-semantic --details-json FILE` writes the map JSON to `FILE`
  - log: 2026-06-22 implemented (v1.0.16)
- [x] **CLI result preserved** - the distance result still prints to stdout as usual when the flag is set
  - log: 2026-06-22 implemented (v1.0.16)
- [x] **CLI note to stderr** - a confirmation line `transport map written: ... (A n -> B m statements)` goes to stderr, so stdout stays the result only
  - log: 2026-06-22 implemented (v1.0.16)
- [x] **Tests** - core (marginals + realizes-SMD), map builder (shape, weights sum to 1, descending, valid target, cost), anisotropy flag, identical-docs identity, CLI help lists the flag; 22/22 pass
  - log: 2026-06-22 implemented (v1.0.16)
- [x] **Docs** - README (CLI line + Transport map bullet), cli-reference (flag row + output shape + example), api-reference (method, `transport_plan`, example)
  - log: 2026-06-22 implemented (v1.0.16)
- [x] **Edge: unequal counts** - `n_a != n_b` handled; the coupling is rectangular and every A statement still maps somewhere
  - log: 2026-06-22 verified (12 vs 11 on fixtures) (v1.0.16)
- [x] **Edge: identical documents** - A == B gives the identity coupling: each statement maps to its twin at `weight` 1, `cost` 0
  - log: 2026-06-22 tested in `test_transport_map_identical_maps_each_to_itself` (v1.0.16)
- [ ] **Edge: empty document** - a document that segments to zero statements raises `ValueError` via `embed_statements`, same as the plain distance path
  - log: 2026-06-22 inherited from `embed_statements`, not separately asserted

## API

- CLI `docdistance distance-semantic A B --details-json FILE` -> writes `FILE`, still prints the result; stderr note `transport map written: FILE (A n -> B m statements)`
- Python `DocDistance.semantic_distance_with_details(a, b, *, anisotropy=False, threshold=0.725)` -> `(DistanceResult, map)`
- Python `transport_plan(X, Y)` -> `ndarray [n_X, n_Y]` (raw coupling, embeddings in hand)
- Map shape: `{smd: float, anisotropy: bool, n_statements: {a, b}, flows: [{index, text, matches: [{target_index, target_text, weight, cost}], changed}]}`
