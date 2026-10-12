# Example - Semantic and Structural Details Interpretations

The split distance ships two interpretable commands that localize *what changed* between two documents, per statement: the semantic details (`distance-semantic --details-json` / `semantic_distance_with_details`) and the structural details (`distance-structural --details-json` / `structural_distance_with_details`). This is a worked example - two short documents with known edits, the exact CLI calls, the JSON they emit, and the reading protocol that recovers every edit from the JSON alone. Everything below is a real run of the `docdistance` 1.1.12 development build, which follows release 1.1.5: it adds the one-to-one matching fields and reads `moved` on matched statements. The values of the keys that the 1.1.2 run of this example showed are unchanged.

## The two documents

Document A and document B, six statements each. B is A with four deliberate edits - the ground truth we will try to recover from the JSON without looking back at this list.

Document A:

```
The API now supports pagination on all list endpoints.
Authentication tokens expire after 24 hours.
The dashboard loads in under two seconds on average.
Users can export reports as CSV or PDF.
The mobile app supports offline mode for cached data.
We fixed a memory leak in the background sync worker.
```

Document B:

```
The API now supports pagination on all list endpoints.
Dark mode is now available across the entire interface.
Users can export reports as CSV or PDF.
The dashboard loads in under two seconds on average.
Push notifications can now be scheduled by the user.
We fixed a memory leak in the background sync worker.
```

The edits A → B:

- **statement 1** - replaced (content change): tokens-expiry → dark-mode
- **statement 4** - replaced (content change): offline-mode → push-notifications
- **statements 2 and 3** - swapped (order change, content intact): dashboard ⇄ export
- **statements 0 and 5** - untouched

## Run it

```
docdistance distance-semantic doc-a.md doc-b.md --details-json semantic.json
docdistance distance-structural doc-a.md doc-b.md --details-json structural.json
```

Top-level read (both details carry `smd`; the structural details also carries the order fields): `smd` 0.171 → semantic closeness 0.879; `order_gap` 0.219 → `structure_closeness` 0.845. So the documents drifted moderately in content and shifted somewhat in order - the per-statement outputs say exactly where. The `--json` result of `distance-structural` also carries `matched_order_gap` 0.317 and `n_matched` 4, read in [the structural section](#the-structural-details---what-moved-in-order).

## The semantic details - what changed in meaning

`distance-semantic --details-json` returns the content alignment: for each A statement, a `flows` entry with the B statement(s) its mass flows to - `weight` (mass fraction, sums to 1 per statement), `cost` (ground distance of the match, low = good) - a per-flow `changed` flag (best-match `cost` over the 0.325 cutoff), and the result of the one-to-one matching, `match_index` and `match_status`. One entry verbatim:

```json
{
  "index": 1,
  "text": "Authentication tokens expire after 24 hours.",
  "matches": [
    {
      "target_index": 4,
      "target_text": "Push notifications can now be scheduled by the user.",
      "weight": 1.0,
      "cost": 0.5835
    }
  ],
  "section": "",
  "changed": true,
  "no_counterpart": false,
  "identical": false,
  "match_index": null,
  "match_status": "removed"
}
```

All six statements, compact - every statement flows fully (weight 1.00) to a single B statement:

| A → B | cost | changed | match_index | match_status | reading |
|---|---|---|---|---|---|
| A0 → B0 | 0.000 | - | 0 | kept | in place, identical |
| A1 → B4 | 0.584 | yes | null | removed | no faithful match - content replaced |
| A2 → B3 | 0.000 | - | 3 | kept | same content, moved (2 → 3) |
| A3 → B2 | 0.000 | - | 2 | kept | same content, moved (3 → 2) |
| A4 → B1 | 0.444 | yes | null | removed | no faithful match - content replaced |
| A5 → B5 | 0.000 | - | 5 | kept | in place, identical |

The `targets` array gives the B side. B1 (dark mode) and B4 (push notifications) read `match_status` `added` with `match_source_index` `null`, at `min_incoming_cost` 0.4444 and 0.4909; the other four read `kept`. The header `summary` counts `n_kept` 4, `n_removed` 2, `n_added` 2.

Reading rules:

- **changed = true** (`cost` > 0.325, the changed-cost cutoff) - the content of that statement drifted; `cost` is how far, on the ground scale √(2 − 2cos), 0 = identical meaning, √2 ≈ 1.414 = opposite
- **cost ≈ 0, target index = source index** - unchanged, in place
- **cost ≈ 0, target index ≠ source index** - same content, moved (the rank shift is read off the structural details)
- **cost high** - content changed; the best available match is still far. The named `target` is the leftover assignment, not a semantic claim - tokens did not "become" push notifications
- **match_status = kept** - the statement is in a pair of the one-to-one matching, and `match_index` names its B statement. A pair needs a ground cost below `match_cutoff` 0.325 (the constant `MATCH_CUTOFF`, stated in the `thresholds` block)
- **match_status = removed / added** - the statement is in no such pair: `removed` on a flow (A side), `added` on a `targets` record (B side). A replacement reads as one `removed` and one `added` statement, as A1 / B1 and A4 / B4 do here. `no_counterpart` stays `false` on A1 and A4, because their costs 0.584 and 0.444 are below its cutoff 0.6

The matching labels are an annotation: the plan, the `matches` and `smd` are the same with and without them. Measured on 900 edited document pairs with known edits, the labels kept / removed / added read a held-out macro-F1 of 0.9772 at the cutoff 0.325, and 0.9457 in the other direction of the split with the cutoff 0.3100 chosen there, against 0.3130 and 0.3266 for the 1.1.5 flags (E13-H71). The calibration of the cutoff and the measured limits are in the [API reference](api-reference.md#matching-and-the-matched-order-gap).

## The structural details - what moved in order

`distance-structural --details-json` returns, per A statement, the order projection. `target_index` / `target_text`, `cost` and `displacement` (rank shift) come from the exact-plan alignment. `matched_displacement` and the `moved` flag come from the one-to-one matching of the statements both documents hold. One entry verbatim:

```json
{
  "index": 2,
  "text": "The dashboard loads in under two seconds on average.",
  "section": "",
  "target_index": 3,
  "target_text": "The dashboard loads in under two seconds on average.",
  "target_section": "",
  "cost": 0.0,
  "semantic_target_index": 3,
  "semantic_target_cost": 0.0,
  "targets_differ": false,
  "displacement": 1,
  "matched_displacement": 1,
  "displacement_norm": 0.1667,
  "moved": true,
  "no_counterpart": false
}
```

All six statements, compact. The norm is `abs(matched_displacement / n_matched)` with `n_matched` 4:

| A | statement | displacement | matched_displacement | norm | moved |
|---|---|---|---|---|---|
| 0 | pagination | 0 | 0 | 0.00 | - |
| 1 | tokens (→ dark mode) | 3 | null | - | - |
| 2 | dashboard | +1 | +1 | 0.25 | yes |
| 3 | export | -1 | -1 | 0.25 | yes |
| 4 | offline (→ push notif.) | -3 | null | - | - |
| 5 | memory leak | 0 | 0 | 0.00 | - |

The definition, from the `definitions` block of the file:

```
moved = matched AND matched_displacement != 0 AND abs(matched_displacement / n_matched) > moved_tolerance AND NOT no_counterpart
```

- **matched** - the statement is in a pair of the one-to-one matching, pair ground cost below `match_cutoff` 0.325; statements 0, 2, 3 and 5 are matched here, so `n_matched` is 4
- **matched_displacement** - the rank of the matched target among the matched targets minus the rank of the statement among the matched statements (`+` later, `−` earlier); `null` for a statement without a match
- **norm** - `abs(matched_displacement / n_matched)`, compared with `moved_tolerance` 0.05
- **no_counterpart** - the exact-plan target costs more than 0.6; such a record reads `moved = false`

Reading rules:

- **moved = true** - the statement changed its rank among the matched statements, content intact
- **moved = false** - in place, or without a match (replaced or removed)

A replaced statement has no match: statements 1 and 4 read `matched_displacement` null and `moved = false`. Their `displacement` (3 and −3) is the leftover pairing of the exact plan (see the caveat below). A statement inserted or removed without a match changes no rank among the matched statements and flags nothing. On a document that repeats a statement this does not always hold: `moved` flagged a kept statement in 4 of 120 pure deletions of a 62-statement document with 9 repeated statements (E15-H80; bullet "Limit: repeated statements" of the [API reference](api-reference.md#matching-and-the-matched-order-gap)).

On a pair with equal statement counts and every statement matched, `moved` equals the conjunction `displacement != 0 AND abs(target_index/n_b - index/n_a) > moved_tolerance` of the exact-plan alignment, with one exception at the tolerance. A rank step of exactly 5% of the matched count (1 of 20 statements, 2 of 40) is not above `moved_tolerance` and reads `moved = false`, while `target_index/n_b - index/n_a` exceeds 0.05 at some positions through float rounding. Measured in E15-H80: equal on every one of 1680 permutations of documents with 11 to 14 statements; with two statements exchanged, equal on 183 of 190 pairs at 20 statements and 765 of 780 at 40.

**`matched_order_gap`.** The result of `distance-structural` (`--json`, and the panel rows `matched order gap` / `matched statements`) carries the order gap over the matched statements only:

```json
{
  "smd": 0.17131764193375904,
  "order_gap": 0.21851211798274506,
  "structure_closeness": 0.8454885996029661,
  "threshold": 0.9,
  "verdict": "not similar",
  "anisotropy": false,
  "n_statements_a": 6,
  "n_statements_b": 6,
  "matched_order_gap": 0.31681132314249016,
  "n_matched": 4
}
```

- **matched_order_gap** - `order_gap` of A0, A2, A3, A5 against B0, B2, B3, B5, the four statements both documents hold, each document in its own order; `null` when no statement is matched
- **Not bounded by order_gap** - here it reads 0.317 over 4 statements against 0.219 over 6
- **What it is for** - `order_gap` rises when statements are deleted and the order is kept, and `structure_closeness` can then fall below the threshold. `matched_order_gap` reads such a deletion at the identity floor of the shorter document: at or below 0.005 in 280 of 280 order-preserving deletions on documents of 11 to 14 statements, where `order_gap` reads above 0.005 in 280 of 280 (E13-H70)
- **The verdict does not read it** - `structure_closeness` and `verdict` come from `order_gap` only

The semantic `cost` is the coupling cost of the aligned pair, and the structural `displacement` is the rank shift of that same alignment.

## What an agent recovers

Reading the JSON alone, with no knowledge of the edits:

- statements 1 and 4 - `changed`, high `cost` and `match_status` `removed` in the semantic details, with B1 and B4 `added` on the target side → the two content edits (correct)
- statements 2 and 3 - `moved` with `displacement` ±1 in the structural details, `changed` false and `cost` 0 in the semantic details → the swap (correct)
- statements 0 and 5 - flat on both → untouched (correct)

Six of six edits recovered, each typed correctly as content-drift versus rearrangement. This is the design intent: an agent driving the CLI gets per-statement indicators that pin down what drifted in meaning and what moved in arrangement, not just a single scalar.

## Caveat - replaced statements

When a statement is genuinely replaced (not reworded in place), the balanced one-to-one alignment has no faithful target, so the transport LP pairs it with whatever B statement is left over - here A1 ⇄ B4 and A4 ⇄ B1 crossed. For such a row, trust the semantic details `changed` + `cost` and ignore the structural details `displacement` / `target`; they are the spurious leftover pairing, not a real move. `matched_displacement` and `moved` do not read that pairing: a replaced statement has no match below `match_cutoff`, so it reads `matched_displacement` null and `moved = false`. For content-preserved rows the `displacement` is reliable. This is the same effect documented as the n ≠ m displacement limitation: a replacement makes the local alignment non-faithful even when the statement counts match.

## See also

- [CLI reference](cli-reference.md) - the `distance-semantic` and `distance-structural` commands and their `--details-json` flag
- [API reference](api-reference.md) - `DocDistance.semantic_distance_with_details` and `structural_distance_with_details`
- [Structure distance SOTA](solution/wmd-structure-distance-sota.md) - the order-gap and `structure_closeness`
- [Acceptance criteria - structure diff](acceptance-criteria/acc-crit-structure-diff.md), [transport map](acceptance-criteria/acc-crit-transport-map.md)
- [Notebook 13](../notebooks/13-kj-structure-diff-library-e2e.ipynb) - the same two-axis story through the library on the curated fixture
