# Example - Semantic and Structural Details Interpretations

The split distance ships two interpretable commands that localize *what changed* between two documents, per statement: the semantic details (`distance-semantic --details-json` / `semantic_distance_with_details`) and the structural details (`distance-structural --details-json` / `structural_distance_with_details`). This is a worked example - two short documents with known edits, the exact CLI calls, the JSON they emit, and the reading protocol that recovers every edit from the JSON alone. Everything below is a real run of the shipped `docdistance` 1.1.2 CLI.

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

Top-level read (both details carry `smd`; the structural details also carries the order fields): `smd` 0.171 → semantic closeness 0.879; `order_gap` 0.219 → `structure_closeness` 0.845. So the documents drifted moderately in content and shifted somewhat in order - the per-statement outputs say exactly where.

## The semantic details - what changed in meaning

`distance-semantic --details-json` returns the content alignment: for each A statement, a `flows` entry with the B statement(s) its mass flows to - `weight` (mass fraction, sums to 1 per statement), `cost` (ground distance of the match, low = good) - and a per-flow `changed` flag (best-match `cost` over the 0.389 cutoff). One entry verbatim:

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
  "changed": true
}
```

All six statements, compact - every statement flows fully (weight 1.00) to a single B statement:

| A → B | cost | changed | reading |
|---|---|---|---|
| A0 → B0 | 0.000 | - | in place, identical |
| A1 → B4 | 0.584 | yes | no faithful match - content replaced |
| A2 → B3 | 0.000 | - | same content, moved (2 → 3) |
| A3 → B2 | 0.000 | - | same content, moved (3 → 2) |
| A4 → B1 | 0.444 | yes | no faithful match - content replaced |
| A5 → B5 | 0.000 | - | in place, identical |

Reading rules:

- **changed = true** (`cost` > 0.389, the changed-cost cutoff) - the content of that statement drifted; `cost` is how far, on the ground scale √(2 − 2cos), 0 = identical meaning, √2 ≈ 1.414 = opposite
- **cost ≈ 0, target index = source index** - unchanged, in place
- **cost ≈ 0, target index ≠ source index** - same content, moved (the rank shift is read off the structural details)
- **cost high** - content changed; the best available match is still far. The named `target` is the leftover assignment, not a semantic claim - tokens did not "become" push notifications

## The structural details - what moved in order

`distance-structural --details-json` returns, per A statement, the order projection - `target_index` / `target_text`, `displacement` (rank shift) and the `moved` flag; no `cost` or `changed` here. One entry verbatim:

```json
{
  "index": 1,
  "text": "Authentication tokens expire after 24 hours.",
  "target_index": 4,
  "target_text": "Push notifications can now be scheduled by the user.",
  "displacement": 3,
  "moved": true
}
```

All six statements, compact:

| A | statement | displacement | moved |
|---|---|---|---|
| 0 | pagination | 0 | - |
| 1 | tokens (→ dark mode) | 3 | yes |
| 2 | dashboard | +1 | yes |
| 3 | export | -1 | yes |
| 4 | offline (→ push notif.) | -3 | yes |
| 5 | memory leak | 0 | - |

Reading rules:

- **moved = true** - the statement was rearranged, content intact; `displacement` is its rank shift (`+` later, `−` earlier)
- **moved = false** - in place

Check `changed` on the semantic side first: a genuinely replaced statement also reads `moved = true` here (see the caveat below), so the semantic `changed` is the deciding flag. The two details share one alignment: the semantic `cost` is the coupling cost of the aligned pair, and the structural `displacement` is the rank shift of that same alignment.

## What an agent recovers

Reading the JSON alone, with no knowledge of the edits:

- statements 1 and 4 - `changed`, high `cost` in the semantic details → the two content edits (correct)
- statements 2 and 3 - `moved` with `displacement` ±1 in the structural details, `changed` false and `cost` 0 in the semantic details → the swap (correct)
- statements 0 and 5 - flat on both → untouched (correct)

Six of six edits recovered, each typed correctly as content-drift versus rearrangement. This is the design intent: an agent driving the CLI gets per-statement indicators that pin down what drifted in meaning and what moved in arrangement, not just a single scalar.

## Caveat - replaced statements

When a statement is genuinely replaced (not reworded in place), the balanced one-to-one alignment has no faithful target, so the transport LP pairs it with whatever B statement is left over - here A1 ⇄ B4 and A4 ⇄ B1 crossed. For such a row, trust the semantic details `changed` + `cost` and ignore the structural details `displacement` / `target`; they are the spurious leftover pairing, not a real move. For content-preserved rows the `displacement` is reliable. This is the same effect documented as the n ≠ m displacement limitation: a replacement makes the local alignment non-faithful even when the statement counts match.

## See also

- [CLI reference](cli-reference.md) - the `distance-semantic` and `distance-structural` commands and their `--details-json` flag
- [API reference](api-reference.md) - `DocDistance.semantic_distance_with_details` and `structural_distance_with_details`
- [Structure distance SOTA](solution/wmd-structure-distance-sota.md) - the order-gap and `structure_closeness`
- [Acceptance criteria - structure diff](acceptance-criteria/acc-crit-structure-diff.md), [transport map](acceptance-criteria/acc-crit-transport-map.md)
- [Notebook 13](../notebooks/13-kj-structure-diff-library-e2e.ipynb) - the same two-axis story through the library on the curated fixture
