# Concept draft - docdistance article deck (v2, explanatory rebuild)

15 SVGs. Every number is pinned to `../docdistance-measuring-document-drift.md`. No figure may invent or re-round a number.

**Audience** - data scientists who know cosine similarity and have never touched optimal transport. The deck must TEACH the base concept in each figure, not label it. A reader who skims only the images should still come away understanding what a transport plan is, what a lower bound is, and why the triangle inequality matters.

**Hard rule on language** - no unexplained jargon on any canvas. If a technical term must appear, its plain-English meaning appears beside it in the same figure. Banned as bare labels: "fused metric", "ordinality violation", "anisotropy", "entropic bias", "permutation-invariant". Each may appear ONLY paired with its plain gloss.

**Theme** - bg `#0a1a24`, cyan `#0096d1`, light cyan `#5cc8e0`, dim cyan `#1a5a6e`, orange `#da8230`, light orange `#d4a04a`. Dark mode on every file.

**Role convention** - cyan = the shipped mechanism / the right answer. orange = the rejected alternative, the failure, or the problem state. light orange = a caution or an accepted cost.

---

```svg-infographics
file: 01-four-situations.svg
format: doc-grid (2x2)
content: four quadrants, one per situation, each a tiny pictogram plus a one-line caption.
  (1) thick document shrinking to one page, magnifier with a question mark over it
  (2) engineering report becoming a glossy brochure, one claim in the brochure traced back
      to the report, one claim tracing back to nothing (dangling line, orange)
  (3) two near-identical pages side by side, a few passages linked, two unlinked
  (4) one source document fanning out into many small variants, one flagged orange
constraint: quadrant 4 is visually the largest or most emphasised - it is the one that
  decides the design.
facts: "two hundred of these and no expert"; personalised campaigns for forty segments;
  a hundred thousand pairs cannot be checked by hand.
teach: all four are the SAME question - how far apart, and which part matches which.
```

```svg-infographics
file: 02-the-logit-gap.svg
format: doc-flow
content: source document -> frontier model (drawn as a sealed box) -> output document.
  A probe labelled "token probabilities" tries to enter the sealed box and is blocked.
  Consequence strip: "KL divergence needs those numbers. You cannot have them."
  Below, the two documents remain available as plain text, in cyan - that is the whole input.
teach: the constraint is not that KL is bad. It is that KL is unavailable. Say so on canvas.
facts: frontier models behind an API do not expose token-level logits.
```

```svg-infographics
file: 03-why-divergence-fails.svg
format: doc-stats (two panels)
content: left panel - two sets of statements that say the same thing in different words;
  zero identical items, so the overlap region is empty. Big orange readout: "overlap = 0
  -> distance = infinity". right panel - the same two sets drawn as points in space,
  clearly close together. Caption: "these are paraphrases. A divergence cannot see that."
teach: a divergence only measures where two distributions overlap; if nothing is identical
  there is nothing to measure, no matter how close the meanings are.
facts: paraphrase is the normal case, not the edge case.
```

```svg-infographics
file: 04-earth-mover-intuition.svg
format: doc-flow
content: the sand-pile picture, drawn literally. Document A as a set of small piles at
  points in space; document B as piles at other points. Curved flow arrows moving sand
  from A's piles to B's, arrow thickness = amount moved, arrow length = cost.
  Side note: "short move = cheap. long move = expensive. the distance is the cheapest
  total plan."
teach: this is the one figure that must make Wasserstein click for a first-time reader.
  No formula on this canvas. Physical intuition only.
```

```svg-infographics
file: 05-the-ladder.svg
format: doc-flow (three stacked rungs)
content: a literal ladder, three rungs, cheapest at the bottom.
  rung 1 "average each document to one point, compare" = your whole-document cosine
  rung 2 "give each statement its nearest match, ignore conflicts"
  rung 3 "the cheapest globally consistent plan" = the real distance
  Each rung labelled with its short name in small type: WCD, RWMD, SMD.
  A vertical bracket on the right: "each is a lower bound on the next".
teach: the naive baseline is not a rival to optimal transport, it is the bottom of the
  same ladder. Show the nesting as a physical climb.
```

```svg-infographics
file: 06-one-sided-error.svg
format: doc-stats
content: a single horizontal axis "true distance". A cyan marker at the true value.
  Two orange markers to its LEFT (rung 1, rung 2), and a shaded forbidden zone to the
  RIGHT labelled "a lower bound can never land here".
  Two verdict chips below: cyan "sound filter - if the cheap number already exceeds your
  threshold, the true one does too" / orange "unsound verdict - it can call two documents
  closer than they are".
teach: THE key idea of the ladder section. A lower bound errs in one direction only.
  That single fact makes it safe as a filter and unsafe as an answer.
facts: rung one crushed eleven summaries into a 0.057 spread.
```

```svg-infographics
file: 07-anisotropy.svg
format: doc-stats (before / after)
content: left - embedding vectors drawn as a narrow cone, all pointing nearly the same way,
  with a caption "every vector shares a few dominant directions, so everything looks
  similar to everything". Spread readout 0.057. right - the same vectors after the shared
  directions are subtracted, fanned wider. Spread readout 0.180. Gain badge "3.2x more
  room to place a threshold".
constraint: the word "anisotropic" may appear ONLY with the plain gloss beside it.
facts: Mu and Viswanath, ICLR 2018; the shared directions encode word frequency, not
  meaning; the correction needs a batch, so isolated pairs go without it.
```

```svg-infographics
file: 08-changed-or-moved.svg
format: doc-grid (2x2)
content: a 2x2 of the four real cases, each drawn as a small before/after pair of
  statement stacks. x-axis "did the words change?", y-axis "did the order change?".
  quadrants: unchanged / reworded / rearranged / both. Under each, which of the two
  numbers moves.
teach: two independent questions need two independent numbers. Show that rewording moves
  one number and leaves the other flat, and vice versa.
```

```svg-infographics
file: 09-the-version-i-got-wrong.svg
format: doc-stats
content: one test, two bars. The test: a document round-tripped through German, so every
  sentence reworded and the order untouched. A structural measure should read this as
  nothing. Bar 1 (orange, the version shipped first) 73.5%. Bar 2 (cyan, the replacement)
  0.5%. A reference line at "a fully scrambled document = 100%".
  Caution strip in light orange: "the replacement is not a true distance - it breaks the
  triangle rule 4.5% of the time, so never chain it".
teach: a number can be mathematically well-behaved and still answer the wrong question.
  Show the honest trade, both sides.
```

```svg-infographics
file: 10-what-the-numbers-say.svg
format: doc-stats (two rows of tiles)
content: row 1 meaning axis - ordering errors 0 of 24; spread 0.057 -> 0.180 (3.2x);
  safety gap between good and bad summaries +0.92. row 2 arrangement axis -
  reword reads 0.5% vs 73.5%; translation reads 0.001 vs 0.417; agreement with a
  hand-built ranking 1.00.
  One tile in light orange, not hidden: separation d' fell 2.70 -> 2.34, an accepted cost.
constraint: the regression tile and the 4.5% caveat must be as visible as the wins.
```

```svg-infographics
file: 11-where-the-time-goes.svg
format: doc-stats
content: one proportional stacked bar for a single CPU core on two A4 pages.
  cutting into statements 2.55 s, embedding 3.0 s, the optimal transport 0.4 ms.
  Call out the transport sliver explicitly: "the clever part is 0.007% of the runtime".
  Below, the scale arithmetic: 0.2 pairs/second/core -> 100,000 pairs = 139 core-hours
  = about 4.5 hours on one 32-core machine.
teach: the expensive part is the encoder, not the mathematics. Pay for what you think
  you are paying for.
```

```svg-infographics
file: 12-reading-the-output.svg
format: doc-grid (2 panels)
content: left - a transport-map row: statement text, what it matched, weight 1.0,
  cost 0.2237, changed = false. Annotated in plain words: "one clean match, low cost,
  this statement survived". right - a row whose weight splits 0.4 / 0.35 / 0.25 across
  three targets, annotated "no single counterpart - look here". Bottom strip: the order
  readout, displacement 0, moved = false.
teach: how to actually read the two outputs, in plain language, with the annotation doing
  the teaching rather than the field names.
```

```svg-infographics
file: 13-drift-to-rules.svg
format: doc-flow (loop)
content: a closed loop. source document + generated output -> transport map -> two typed
  findings: (a) orange "a claim was dropped or invented - mass split three ways at high
  cost", (b) light orange "a claim only moved - matched cleanly, twelve positions away".
  Each finding feeds a rule card. The rule cards accumulate into a ruleset, which feeds
  back into the generating agent.
  Honest footer in dim type: "not benchmarked - the mechanism, not a measured gain".
teach: a single score cannot say which claim or by how much, so it cannot become a rule.
  A plan can. Show the two failure types producing two different rules.
```

```svg-infographics
file: 14-where-it-falls-short.svg
format: doc-grid (cards)
content: one card per limitation, each stated plainly, no softening.
  thin margin +0.92 on one fixture; one fixture only, not a benchmark; the structural
  number breaks the triangle rule 4.5% of the time so never chain it; the encoder
  correction needs a batch; it cannot separate "drew on a different part of the source"
  from "invented it"; it is not a judge.
constraint: entirely light orange / orange. This figure is not allowed to look like a win.
```

```svg-infographics
file: 15-when-to-use-it.svg
format: doc-flow (decision tree)
content: root "can you read the model's token probabilities?" -> yes: use KL divergence
  (orange, the simpler answer when it is available). no: -> "how far did the meaning
  move?" use the semantic distance -> "did anything move position?" add the structural
  number -> "do you hold the shared source?" the source-conditioned variant, tagged
  experimental (light orange, validated on a single fixture, seconds not milliseconds).
teach: end the reader on a decision they can actually make, including the branch where
  the answer is "do not use this".
```
