# Corrections to apply to the published Medium article

13 text edits and 7 figures. Each edit gives the exact text to remove and the exact text to put in its place.

## What each number is

| Number | Unit | What it measures |
| --- | --- | --- |
| `0.67`, `0.86`, `79.7%`, `75.0%` | closeness, 0 to 1, shown as a percent | how close one document is to the reference; 1 means identical |
| `4.7` | percentage points of closeness | the gap between two closeness percentages; not a percentage of anything |
| `0.085`, `0.229` | transport distance, 0 to √2 | standard deviation of the eleven distances to the reference |
| `2.7` | a multiplier | how much the encoder correction widened that standard deviation |
| `4.60`, `7.17` | pooled standard deviations | how far the two groups sit apart, relative to the scatter inside each |
| `15%`, `0.5%` | a percentage of a count | triples that break the triangle inequality; share of a full scramble's reading |
| `154` | core-hours | CPU time for 100,000 pairs at 5.55 seconds each |

## 1. Section "Why one cosine score is not enough"

**REMOVE**:

```
all landed between 0.7 and 0.9. Best-to-worst spread: **0.057**. Any threshold I picked sat inside the noise.
```

**REPLACE WITH**:

```
each came back as a single number. I never scored that baseline against the tiers, so I cannot tell you how well it separated them. What I could see is that the number does not say where the two documents differ, and that is the question I needed answered.
```

**Source**: none - and that is the correction. The published `0.057` and the `0.085` first proposed as its replacement are both from the exact-SMD run (E01 cell 14 `DR=0.0854`; nb04 cell 18 spans 85.9% to 67.7%), not from a whole-document cosine sweep, which exists in no committed artefact. See `v2-sweep-report.md` finding 1

## 2. Section "Three ways to compare, from cheapest to best"

**REMOVE**:

```
It is also why the Word Centroid Distance scored all eleven summaries within 0.057 of each other. A lower bound does not have to separate the documents that sit above it, and here it did not.
```

**REPLACE WITH**:

```
It is also why a lower bound promises nothing about separation: it constrains the true distance from below and says nothing about the gap between two documents that both sit above it. How the Word Centroid Distance ranks these eleven, and how far apart it puts them, I never measured, so the case for exact transport rests on the lower-bound property and the statement-level alignment, not on beating the centroid.
```

**Source**: no measurement supports the removed claim, and none supports a ranking claim either - the `ranked these eleven correctly` wording was cut after the paragraph sweep for resting on a single unreplicated run

## 3. Section "One detail that decides whether the numbers hold up"

**REMOVE**:

```
It also has fixed end points: since embedding cosines sit between 0 and 1, the distance sits between 0 and `√2`
```

**REPLACE WITH**:

```
It also has fixed end points: while the encoder keeps cosines non-negative - this one does on these documents, though nothing guarantees it - the distance sits between 0 and `√2`
```

## 4. Section "The embedding model makes everything look alike"

**REMOVE**:

```
It widened my spread from **0.057 to 0.180**, roughly three times the room to place a threshold.
```

**REPLACE WITH**:

```
It widened that standard deviation from **0.085 to 0.229**, 2.7 times the room to place a threshold.
```

**Source**: E01 cell 14 `DR=0.0854`, cell 21 `DR=0.2286 (2.7x)`

## 5. Section "Did it change the words, or only the order?"

**REMOVE**:

```
The order number breaks the triangle inequality about **4.5%** of the time
```

**REPLACE WITH**:

```
The order number breaks the triangle inequality on about **15%** of the triples I tested
```

**Source**: `reports/E11-structure-sota-decision.json`, `triangle.H55 = 0.15`

## 6. Section "What the numbers say", first paragraph

**REMOVE**:

```
The closest faithful summary and the furthest degraded one are **0.92 points** apart on a 0-to-100 scale. The ordering is perfect and the gap is narrow, and I would rather say that than bury it.
```

**REPLACE WITH**:

```
The weakest faithful summary scored **79.7%** closeness against the reference; the strongest degraded one **75.0%**. Nothing landed in between, so the cutoff goes in that gap - 4.7 percentage points wide, and the library defaults to 77%. The ordering is perfect and the gap is narrow, and I would rather say that than bury it.
```

**Source**: notebook 09 cell 10 - min gold 79.7%, max adversarial 75.0%, 0 of 24 ordinality violations

> [!IMPORTANT]
> The repo copy reads `(79.4 against 74.7)`. Those endpoints are from an unreplicated run. Publish `79.7` and `75.0`, and fix the repo copy too.

## 7. Section "What the numbers say", third paragraph

**REMOVE**:

```
**What the encoder correction cost.** Removing the shared directions tripled the spread, which was the point, but it also pushed the seven faithful summaries further apart from each other. Take the distance between the faithful group and the degraded group, and divide it by how scattered the two groups are internally: that ratio fell from **2.70 to 2.34**. I chose the wider spread over the tighter grouping. The opposite choice is defensible.
```

**REPLACE WITH**:

```
**What the encoder correction cost.** I expected a trade: more spread bought with looser grouping inside the faithful set. Divide the gap between the two groups by how scattered each group is internally and you get their separation in standard deviations. It **rose from 4.60 to 7.17** - the correction widened the spread and tightened the separation at the same time, so the trade never appeared.
```

**Source**: E01 cell 14 `d'=4.60`, cell 21 `d'=7.17`

## 8. Section "What it costs"

**REMOVE**:

```
costs about 139 core-hours. That is four and a half hours on one 32-core machine
```

**REPLACE WITH**:

```
costs about 154 core-hours. That is just under five hours on one 32-core machine
```

**Source**: 100,000 x 5.55 s = 154.2 core-hours; 154.2 / 32 = 4.8 hours

## 9. Section "Where it falls short", first bullet

**REMOVE**:

```
- **Thin margin** - 0.92 points between the groups, on documents built to be hard to separate
```

**REPLACE WITH**:

```
- **Thin margin** - the cutoff sits in a window 4.7 percentage points wide, from 75.0% to 79.7% closeness, on documents built to be hard to separate
```

## 10. Section "Where it falls short", third bullet

**REMOVE**:

```
- **The structural number is a score** - 4.5% triangle violations, so never chain it
```

**REPLACE WITH**:

```
- **The structural number is a score** - it breaks the triangle inequality on 15% of triples, so never chain it
```

## 11. Section "Where it falls short", new fourth bullet

Nothing to remove. Add this directly below the bullet from item 10.

**ADD**:

```
- **The cheap baseline was never scored against this one** - the case for exact transport rests on the lower-bound property and the statement-level alignment, not on a measured win over the centroid distance. That comparison is still owed
```

## 12. Section "References", opening paragraph

**REMOVE**:

```
So are the measurements - 24 of 24, 0.92 points, 0.5% against a full scramble
```

**REPLACE WITH**:

```
So are the measurements - 24 of 24, a cutoff window 4.7 percentage points wide, 0.5% against a full scramble
```

## 13. Section "References", the embedding geometry entry

**REMOVE**:

```
This is where 0.057 -> 0.180 came from
```

**REPLACE WITH**:

```
This is where 0.085 -> 0.229 came from
```

## Figures to replace

Re-render each from the corrected SVG in `docs/medium/images/` and replace the uploaded copy. The other ten figures and all three covers stay as they are.

| Figure | Caption on Medium | Remove → replace |
| --- | --- | --- |
| `03-cosine-collapse.svg` | Eleven summaries, one narrow band | `0.057 wide - best to worst` → `0.085 spread - std across 11` |
| `08-one-sided-error.svg` | One-sided error | footer claim that the centroid failed to separate the eleven → removed |
| `09-anisotropy.svg` | Correcting the encoder | `0.057`/`0.180` → `0.085`/`0.229`; badge `3.2x` → `2.7x`; drawn fan 64 → 54 degrees |
| `11-what-the-numbers-say.svg` | What the numbers say | `0.057 → 0.180` → `0.085 → 0.229`; `3.2x` → `2.7x`; `+0.92` → `+4.7`; third panel `↓ the one number that got worse  2.70 → 2.34` → `↑ the number I expected to get worse  4.60 → 7.17` |
| `12-where-the-time-goes.svg` | Where the time goes | `139 core-hours` → `154 core-hours` |
| `13-reading-the-output.svg` | Reading the output | `139 core hours` → `154 core hours` |
| `17-where-it-falls-short.svg` | Where it falls short | `+0.92 pts` → `+4.7 pts`; `4.5%` → `15%`; both appear twice |


## Additional corrections found after this list was written

A per-paragraph sweep of the article (see `v2-sweep-report.md`) found nine further errors live on Medium that are not in the 13 items above, including a reversed pruning rule, `7 x 4 = 24`, an INT8 speedup overstated as "several times" against a measured 1.4x, and a display equation that compares a document with itself. Apply those from the sweep report, or take the corrected text wholesale from `docdistance-measuring-document-drift-v2.md`.
