# Acceptance Criteria - Structure Diff

Optional interpretable output that separates content drift from rearrangement between two documents - per statement, its positional displacement (order), plus the whole-document order-gap and its bounded `structure_closeness`. Surfaced as the CLI flag `--details-json FILE` and the Python method `DocDistance.structural_distance_with_details`; the structural core is `opw_gap` / `order_alignment` / `structure_displacement` in `distance.py`. The order-gap is the decided SOTA structure metric (E11-H55); see [Transport Map](acc-crit-transport-map.md) for the content-only companion.

- [x] **OPW plan** - `opw_plan(X, Y)` returns the log-stabilized order-preserving Sinkhorn coupling, shape `[n_X, n_Y]`, pure numpy, no model load
  - log: 2026-07-01 implemented (v1.1.2)
- [x] **Valid coupling** - the OPW plan is rounded onto the transport polytope `U(a, b)` (`_round_to_polytope`, Altschuler 2017) so both marginals hold exactly (row `1/n_X`, column `1/n_Y`); the fixed-iteration Sinkhorn alone meets only the row marginal
  - log: 2026-07-01 added after adversarial verify found the column marginal off up to 0.12 on n != m (v1.1.2)
- [x] **OPW cost** - `opw_cost(X, Y)` = `(opw_plan * cost_matrix).sum()`, the order-regularized transport cost
  - log: 2026-07-01 implemented (v1.1.2)
- [x] **Order gap >= 0** - `opw_gap(X, Y)` = `max(0, opw_cost - smd)`; because the plan is a valid coupling `opw_cost >= smd`, so the gap is genuinely non-negative and the `max(0, .)` clamp only absorbs LP rounding noise (worst pre-clamp `-2e-16`)
  - log: 2026-07-01 implemented; clamp rationale corrected from "Sinkhorn noise" to "LP noise, plan now valid" (v1.1.2)
- [x] **D computed once** - `opw_gap` builds `cost_matrix` once and reuses it across the OPW cost and the SMD (`ot.emd2`)
  - log: 2026-07-01 implemented (v1.1.2)
- [x] **Constants** - `OPW_LAMBDA1 = 50.0`, `OPW_LAMBDA2 = 0.1`, `OPW_SIGMA = 1.0` (Su & Hua / E10-H55 defaults)
  - log: 2026-07-01 implemented (v1.1.2)
- [x] **Crisp alignment** - `order_alignment(X, Y)` = the exact-EMD argmax per A statement, read off `_aligned_plan` (an `ot.emd` plan with an `eps`-scaled positional tie-break, `eps = 1e-6*(Dmax+1)`), NOT the entropic OPW plan
  - log: 2026-07-01 implemented (v1.1.2)
- [x] **Tie-break** - duplicate / near-duplicate statements make the ground cost degenerate; the positional tie-break resolves the LP's optimal-coupling freedom toward the in-place map, so unmoved statements get zero displacement (phantom-displacement rate 30/200 -> 0/200)
  - log: 2026-07-01 added after adversarial verify found phantom displacement on duplicates (v1.1.2)
- [x] **Displacement** - `structure_displacement(X, Y)` = rank transform of the alignment (`argsort(argsort(align)) - arange(n)`); `0` = in place, nonzero = moved
  - log: 2026-07-01 implemented (v1.1.2)
- [x] **Structure closeness** - `structure_closeness` = `1 - order_gap/sqrt(2)` = `closeness(order_gap)`, on the library's 0..1 scale; `1` = same order
  - log: 2026-07-01 implemented (v1.1.2)
- [x] **Structural detail builder** - `_build_structural_details(sa, ea, sb, eb, *, anisotropy=False)` returns `{smd, order_gap, structure_closeness, anisotropy, n_statements, statements}`, pure, no model load
  - log: 2026-07-01 implemented (v1.1.2)
- [x] **Statements shape** - `statements` is one entry per A statement in reading order, `len(statements) == n_a`
  - log: 2026-07-01 implemented (v1.1.2)
- [x] **Statement fields** - each is `{index, text, target_index, target_text, displacement, moved}`
  - log: 2026-07-01 implemented (v1.1.2)
- [x] **Displacement field** - `displacement` = the per-statement position shift; `moved` = `displacement != 0`; names what MOVED in order
  - log: 2026-07-01 implemented (v1.1.2)
- [x] **JSON-serializable** - the whole dict is `json.dumps`-able (native ints / floats / strings), no numpy types leak
  - log: 2026-07-01 implemented (v1.1.2)
- [x] **Anisotropy consistency** - the structural detail applies all-but-the-top exactly as `compute_distance`; the `anisotropy` flag is recorded in the dict
  - log: 2026-07-01 implemented (v1.1.2)
- [x] **Content-invariance** - a pure reorder leaves `smd` ~ unchanged while `order_gap` rises; a faithful reword reads `order_gap` ~ 0 (measured 0.5% of a full scramble)
  - log: 2026-07-01 asserted in `tests/test_structure.py` (v1.1.2)
- [x] **Scramble-monotone** - `order_gap` grows with displacement (an adjacent swap reads below a full reverse)
  - log: 2026-07-01 asserted in `tests/test_structure.py` (v1.1.2)
- [x] **Content-edit order isolation** - changing one statement's content leaves every `displacement` ~ 0, so a pure reword does not perturb the order projection (the content-localization signal lives in the [Transport Map](acc-crit-transport-map.md))
  - log: 2026-07-01 asserted in `tests/test_structure.py` (v1.1.2)
- [x] **API method** - `DocDistance.structural_distance_with_details(a, b, *, anisotropy=False, threshold=0.725)` returns `(StructuralResult, dict)` sharing one encode pass
  - log: 2026-07-01 implemented (v1.1.2)
- [x] **CLI flag** - `distance-structural --details-json FILE` writes the structural details JSON to `FILE`
  - log: 2026-07-01 implemented (v1.1.2)
- [x] **CLI result preserved** - the distance result still prints to stdout when the flag is set
  - log: 2026-07-01 implemented (v1.1.2)
- [x] **CLI note to stderr** - a confirmation line `details written: FILE (order_gap=.., structure_closeness=..)` goes to stderr, so stdout stays the result only
  - log: 2026-07-01 implemented (v1.1.2)
- [x] **Tests** - `tests/test_structure.py` battery (OPW marginals, `order_gap >= 0`, content-invariance, content-edit order isolation, displacement recovery, closeness monotonicity, structural details schema, CLI `--details-json`) plus four adversarial regressions; full suite 69 passed / 1 pre-existing skip
  - log: 2026-07-01 implemented (v1.1.2)
- [x] **Edge: identical documents** - A == B gives `order_gap` ~ 0, `structure_closeness` 1.0, all `displacement` 0
  - log: 2026-07-01 tested (v1.1.2)
- [x] **Edge: duplicate statements** - repeated / near-duplicate statements do not invent displacement (the tie-break); `moved` stays false where nothing moved
  - log: 2026-07-01 tested, regression `test_build_structural_details_duplicate_content_edit_does_not_leak_displacement` (v1.1.2)
- [x] **Edge: unequal counts (aggregate)** - `n_a != n_b` handled; `order_gap` / `structure_closeness` are well-defined (the OT handles unbalanced marginals)
  - log: 2026-07-01 tested (v1.1.2)
- [ ] **Edge: unequal counts (per-statement, real drop)** - when `n_a != n_b` because a statement is genuinely dropped, the alignment is non-bijective and `structure_displacement`'s rank transform is ill-defined; documented as a limitation, not solved (aggregate `order_gap` unaffected)
  - log: 2026-07-01 open limitation, no false passing test asserted
- [x] **Docs** - README (Structure distance section, Which-distance pointer, Documentation list), `docs/solution/wmd-structure-distance-sota.md` (Status / Functions / Limitations / FAQ), api-reference, cli-reference, this acc-crit
  - log: 2026-07-01 documentation sweep (v1.1.2)

## API

- CLI `docdistance distance-structural A B --details-json FILE` -> writes `FILE`, still prints the result; stderr note `details written: FILE (order_gap=.., structure_closeness=..)`
- Python `DocDistance.structural_distance_with_details(a, b, *, anisotropy=False, threshold=0.725)` -> `(StructuralResult, details)`
- Python low-level: `opw_plan(X, Y)` -> `ndarray [n_X, n_Y]`; `opw_cost(X, Y)` -> `float`; `opw_gap(X, Y)` -> `float`; `order_alignment(X, Y)` -> `ndarray[int]`; `structure_displacement(X, Y)` -> `ndarray[int]`
- Details shape: `{smd: float, order_gap: float, structure_closeness: float, anisotropy: bool, n_statements: {a, b}, statements: [{index, text, target_index, target_text, displacement, moved}]}`
