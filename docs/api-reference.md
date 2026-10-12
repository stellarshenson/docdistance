# API reference

The `docdistance` public API: a one-shot function per distance, a reusable pipeline for many pairs, three
result objects, and a four-command CLI. Everything below is exported from the top-level `docdistance`
package unless its entry names a module; the SOTA docs carry the mechanics.

## Library - high-level

The entry points most callers use. Inputs are raw text or a path to a text/markdown file (auto-detected);
every markdown `# ` title line in a file is stripped so the title is not counted as a statement. `input_format`
names the reader of the documents: `auto` (the default, by file extension), plain text, markdown or HTML ([Input format](#input-format)).

| Symbol | Signature | Returns | Notes |
| --- | --- | --- | --- |
| `init` | `(mode="wmd", *, source=None, backend="openvino", aws_profile=None, aws_endpoint_url=None, aws_region=None, home=None)` | `dict` | provision a mode's models from local / S3 / HuggingFace; writes `docdistance.json`, returns a per-model source summary |
| `semantic_distance` | `(a, b, *, backend="openvino", anisotropy=False, threshold=0.77, offline=True, device=None)` | `DistanceResult` | symmetric SMD; loads models then scores in one call |
| `structural_distance` | `(a, b, *, backend="openvino", anisotropy=False, threshold=0.90, offline=True, device=None)` | `StructuralResult` | symmetric OPW order-gap; loads models then scores in one call |
| `source_conditioned_distance` | `(a, b, source, *, backend="openvino", anisotropy=True, offline=True, device=None)` | `SourceConditionedResult` | `d(A, B | S)`; selection axis + reranker x NLI grounding residuals |
| `DocDistance` | `DocDistance(backend="openvino", offline=True, device=None, memo_size=0)` | pipeline | construct once, models load lazily on first use, then score many pairs; `memo_size` above 0 switches on the two batch memos (below) |
| `DocDistance.semantic_distance` | `(a, b, *, anisotropy=False, threshold=0.77)` | `DistanceResult` | symmetric distance on the loaded models |
| `DocDistance.semantic_distance_with_details` | `(a, b, *, anisotropy=False, threshold=0.77)` | `(DistanceResult, dict)` | the distance plus the content alignment details, one encode pass |
| `DocDistance.structural_distance` | `(a, b, *, anisotropy=False, threshold=0.90)` | `StructuralResult` | structural OPW order-gap distance on the loaded models |
| `DocDistance.structural_distance_with_details` | `(a, b, *, anisotropy=False, threshold=0.90)` | `(StructuralResult, dict)` | the structural distance plus the order details, one encode pass |
| `DocDistance.distance_wrt_source` | `(a, b, source, *, anisotropy=True)` | `SourceConditionedResult` | source-conditioned distance, runs the reranker x NLI grounding |
| `DocDistance.distance_wrt_source_with_map` | `(a, b, source, *, anisotropy=True, top_k=3)` | `(SourceConditionedResult, dict)` | the conditioned result plus the statement → source map, one encode pass |
| `DocDistance.embed` | `(doc)` | `ndarray [n, dim]` | segment then embed into L2-normalized statement vectors |

- **init** - run once per mode before scoring; `"wmd"` pulls mmBERT + SaT, `"wmd-wrt-source"` adds the bge-reranker-v2-m3 + mDeBERTa NLI; `source` is an `s3://` prefix, a local dir, or `None` for the Hub; an un-init'd mode raises `NotInitializedError`
- **backend** - `"openvino"` (CPU INT8, default) or `"torch"`; pass `device="cuda"` with `backend="torch"` for GPU
- **offline** - `True` loads from the local cache only; run `docdistance.init(mode)` once to populate it
- **anisotropy** - all-but-the-top postprocessing; off for a bare pair, on by default for the conditioned axis (E04-H15). The estimate needs about 71 pooled statements (`ANISOTROPY_MIN_POOL`, E12-H65); the symmetric scoring calls of `DocDistance` log one warning that names the count when the pool of the two documents is smaller, and score it all the same. The source-conditioned calls log no such warning
- **threshold** - closeness cutoff for the similar / not-similar verdict; default `0.77` on the semantic
  entry points, `0.90` on the structural ones (the order-gap scale is compressed, so the semantic cutoff never fires there). Both are calibrated on one corpus: [semantic card](solution/wmd-docdistance-solution-sota.md#calibration-of-the-semantic-verdict), [structural procedure](solution/wmd-structure-distance-sota.md#calibration-of-the-structural-verdict)
- **memo_size** - added in the release after 1.1.5 (written "after 1.1.5" below), default `0` (off: every call reads and encodes its documents, as in 1.1.5). Above 0 a `DocDistance` keeps two memos of at most that many entries each and evicts the least recently used entry beyond it. The first holds the statements, the embeddings and the drop count of a document file, keyed by resolved path, modification time, size and the three segmentation arguments (`input_format` as requested, so `"auto"` and the reader it chooses for a file are two entries) and the reader the extension of the name as given chooses (a symbolic link `notes.txt` to `a.md` is a second entry, read by the reader of its own name), so a file that recurs across calls is segmented and encoded once. The second holds the relevance grid and the entailment vector of a (document file, source file) pair for the source-conditioned calls. A document passed as raw text enters neither memo. The memos change no returned value while every rewrite of a file changes its modification time or its size. The CLI `--pairs` batches pass `MEMO_SIZE` (64); measured readings are in the [CLI reference](cli-reference.md#batch-mode)
- **segmentation arguments** - the three one-shot functions and every `DocDistance` method of the table also take the keyword arguments `markdown=False`, `input_format="auto"`, `min_words=0` and `skip_table_fragments=False`. `input_format` is `"auto"`, `"plain"`, `"markdown"` or `"html"` and was added after 1.1.5; `markdown=True` equals `input_format="markdown"`. `markdown=True` beside `"plain"` or `"html"`, and a value outside the four, raise `ValueError` before a model loads. Readers, extension rule and conversion: [Input format](#input-format)
- **marker statements** - when the plain reader ran, also under `input_format="auto"`, a document whose statements include bare markdown markers (`-`, `###`: no letter and no digit) logs one warning per document that names their count and the options that remove them (`markdown=True`, `min_words`). The statements are scored as they are. The ibm source article reads 21 of 70 such statements with the plain reader and 0 of 48 with `markdown=True` (DEF-SEGM-12)

### Input format

Added after 1.1.5. `input_format` names the reader of every document of a call. The default is `"auto"`, which chooses the reader of each document from its file extension.

| Value | Reader |
| --- | --- |
| `"auto"` | the default: one of the three below, chosen per document from the file extension |
| `"plain"` | the whole text goes to the prose segmenter (SaT); the reader of every document up to 1.1.5 |
| `"markdown"` | `split_markdown`: one statement per heading, table row, list item and fenced-code line, prose paragraphs through SaT; equals `markdown=True` |
| `"html"` | `html_to_markdown`, then the markdown reader |

Extension rule of `"auto"`:

| Document | Reader |
| --- | --- |
| a file with the extension `.md` or `.markdown` | `"markdown"` |
| a file with the extension `.html` or `.htm` | `"html"` |
| a file with another extension, a file without extension | `"plain"` |
| raw text | `"plain"` |

`html_to_markdown` is in `docdistance.encoders` and is not exported from the top-level package, like `split_markdown`.

| Symbol | Signature | Returns | Notes |
| --- | --- | --- | --- |
| `html_to_markdown` | `(text)` | `str` | the block structure of an HTML text as the markdown lines `split_markdown` reads; standard library parser, no dependency |

- **Extension** - the part of the file name from its last dot, compared without letter case: `REPORT.MD` reads as `"markdown"`, `notes.md.txt` as `"plain"`. A name whose only dots stand at its start (`.md`) has no extension. The document argument is a `str` or a `Path`. The content is not read for the choice, and raw text reads as `"plain"` whatever it holds, also when it ends in `.md`
- **Explicit value** - `"plain"`, `"markdown"` or `"html"` is the reader of every document of the call, whatever the extension of each file is. `markdown=True` equals `"markdown"`; beside `"plain"` or `"html"` it raises `ValueError`. `"auto"` is the default and cannot be told from an omitted argument, so `markdown=True` beside it is `markdown=True` alone
- **Numbers against 1.1.5** - releases up to 1.1.5 read every document with the plain reader. With the default `"auto"` the statements and every number for `.md`, `.markdown`, `.html` and `.htm` files can differ from 1.1.5; `input_format="plain"` restores the 1.1.5 reading. Files with another extension and raw text read as before
- **Files** - the title lines of a file are removed whatever its reader is. For an HTML file they are removed after the conversion: the `h1` is no statement, like the `# ` line of a markdown file, and a `# ` line inside a `pre` block stays. Raw text keeps both
- **Detail calls** - the section scan of the `*_with_details` / `*_with_map` methods reads the converted text, so the `section` fields of an HTML document are the heading chains of its `h1` to `h6`
- **Memo and log** - the memo key holds `input_format` as requested and the reader that the extension of the file name chooses (see `memo_size` above). One DEBUG line per document read names the reader that ran and, for a file under `"auto"`, that its extension chose it

Conversion of `html_to_markdown`:

| HTML | Written as |
| --- | --- |
| `h1` to `h6` | `#` repeated by the level, a space, the text on one line |
| `li` | a marker and the text on one line: `- ` in a `ul`; in an `ol` the number of the item and a full stop (`1. `, `2. `, ...), counted from 1 in each `ol`, a nested one as well. The `start` attribute is not read, and an item without text takes its number and writes no line. A nested item is indented by two spaces per level, and the text of an item ends where a nested list starts |
| `tr` | `\| cell \| cell \|` from its `td` and `th` texts; a `\|` inside a cell becomes a space; no separator row |
| `pre` | a fenced block with the lines of its text |
| `p`, `div`, `blockquote`, `section`, `article`, `dt`, `dd`, `figcaption`, `caption`, `header`, `footer`, `main`, `nav`, `aside`, `address`, `form`, `figure`, `details`, `summary`, and text outside any block | a paragraph on one line, its whitespace collapsed to single spaces, followed by a blank line |
| `br`, `hr` | a line break inside the block, the only one a paragraph holds; the end of the paragraph |
| `script`, `style`, `template`, `noscript`, `svg`, `title` (also outside a `head`), comments | dropped with their content |
| every other tag (`a`, `em`, `strong`, `span`, `code`, `sup`, `img`, `head`, ...) | the text inside it only; an `img` leaves nothing. `head` is not dropped, so a document that omits `</head>` and `<body>` keeps its text |

Character references are decoded. A byte order mark (U+FEFF) is removed. Whitespace inside a heading, a list item and a cell collapses to single spaces. A `p` inside an `li` is text of that item, and every block inside a cell is text of that cell. Malformed HTML raises nothing: an unclosed block ends at the next block or at the end of the text. A line break inside a paragraph of the HTML source is whitespace: the prose segmenter cuts at every line break, and a paragraph written on one line keeps the statements of an HTML document independent of how its source text is wrapped. Four limits: a paragraph line that starts with a markdown marker is read as that structure; inline markup that a markdown document keeps as characters (`**bold**`) has no counterpart in the converted text; text outside any block is one paragraph up to the next block, so a text without tags is converted to one line. The fourth: a `pre` block whose text holds a line that starts with three backticks or three tildes is not read as one fenced block, because the markdown reader opens or closes a fence at every such line. With two such lines the lines between them are read as markdown structure (a `# ` line as a heading); with one such line the rest of the document is read as code lines.

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
| `partial_matching` | `(X, Y)` | `ndarray [n_pairs, 2]` | one-to-one statement pairs `(i, j)`, ascending in `i`; a pair needs a ground cost below `MATCH_CUTOFF`; an annotation, not a distance |
| `matched_order_gap` | `(X, Y)` | `float \| None` | `opw_gap` over the statements paired by `partial_matching`, each document in its own order; `None` when no pair is matched (a score, not a metric) |
| `unsupported_count` | `(X, S)` | `int` | count of statements of `X` whose nearest statement of the source `S` costs more than `SOURCE_SUPPORT_CUTOFF`; pass raw embeddings (a score, not a metric) |
| `compute_distance` | `(X, Y, *, anisotropy=False, threshold=0.77)` | `DistanceResult` | assemble the full symmetric result from embeddings |
| `compute_structural` | `(X, Y, *, anisotropy=False, threshold=0.90)` | `StructuralResult` | assemble the structural result from embeddings, with `matched_order_gap` and `n_matched` |
| `compute_source_conditioned` | `(X, Y, S, *, anisotropy=True, reranker_a=None, reranker_b=None, entail_a=None, entail_b=None)` | `SourceConditionedResult` | assemble the conditioned result; pass the grounding arrays for `grd_a`/`grd_b`/`d_grd`; the unsupported counts are read before the pre-pass |
| `grounding_residual` | `(reranker, entail)` | `float` | E03-H11 relevance-gated ungrounded mass `mean_i (1 − entail_i)·(1 − max_j R[i,j])` |
| `grounding_blend` | `(d_sel, d_grd, *, alpha=0.75)` | `float` | E03-H14 two-axis blend `alpha·d_sel + (1 − alpha)·d_grd` |

- **bound chain** - `WCD ≤ RWMD ≤ SMD`, the two cheap bounds bracket the exact distance below
- **ground cost** - `√(2 − 2cos)` on L2-normalized embeddings, a metric, so SMD is a metric too
- **structure axis** - `opw_gap` is the H55 structural distance (order-only, content cancelled); `structure_closeness = 1 − opw_gap/√2` puts it on the same 0..1 scale as `closeness` (1 = same order)
- **exact solver** - after 1.1.5 every exact solve raises `docdistance.SolverNotOptimal` when the solver stops before an optimal coupling; the iteration cap is `max(100000, n_X · n_Y)`. 1.1.5 returned the non-optimal value with only a warning from POT when the fixed cap of 100000 iterations was reached (DEF-TRANS-9)

### Matching and the matched order gap

Added after 1.1.5. `partial_matching(X, Y)` pairs the statements of two documents one to one. A pair needs a ground cost below `MATCH_CUTOFF` (0.325), and a statement without such a pair stays unmatched. The pairs are an annotation that depends on that calibrated cutoff: they are not a distance and not a metric, and they change neither the transport plan nor `smd` nor `order_gap`.

- **Where the pairs surface** - `match_index` / `match_status` on each semantic flow, `match_source_index` / `match_status` on each `targets` record, `n_kept` / `n_removed` / `n_added` in `summary`, `matched_displacement` and `moved` on each structural record, `matched_order_gap` / `n_matched` on `StructuralResult`
- **`MATCH_CUTOFF` = 0.325 (E13-H71)** - calibrated on 900 edited copies of the 10 structure-fixture documents, with the labels kept / removed / added read from the construction of each edit. Chosen on half A of the documents it reads a held-out macro-F1 of 0.9772 on half B, and 0.9674 on half A itself; the cutoff 0.3100 chosen on half B reads a held-out 0.9457 on half A. The 1.1.5 flags (`no_counterpart`, `min_incoming_cost` above 0.6) read 0.3130 and 0.3266 on the same held-out halves. Every cutoff from 0.28 to 0.44 keeps both halves at or above 0.90. With the cutoff of the other half, a kept statement is paired with its true counterpart in 9244 of 9330 cases; the heaviest target of the transport plan is the true counterpart in 9286. With no edit, 0 of 117 statements read `removed` at 0.325 (E15-H79). The value is measured: it is not the derived `changed_cost` `(1 − 0.77)·√2 = 0.3253`
- **`matched_order_gap` (E13-H70)** - with equal statement counts and every statement matched it equals `opw_gap(X, Y)` exactly. On 280 order-preserving deletions of 1 to 4 statements it reads at or below 0.005 in 280 of 280 cases, largest reading 6.101e-04, where `opw_gap` reads above 0.005 in 280 of 280. Deleted-only cases against reordered-and-deleted cases separate at AUC 0.999903, against 0.950980 for `opw_gap`. Measured on documents of 11 to 14 statements without repeated statements
- **Geometry** - the pairs are computed on the embeddings as passed in, before any anisotropy pre-pass, the geometry the cutoff is calibrated on. `n_matched`, `match_status` and `matched_displacement` therefore do not change with `anisotropy`; `matched_order_gap` is read over those pairs in the geometry `order_gap` uses
- **Rule among equal-cost matchings** - the assignment cost carries the position term `MATCH_POSITION_EPS · |i/n − j/m|` with `MATCH_POSITION_EPS` = 1e-6, so the matching is deterministic and copies of a repeated statement that cost exactly the same are ranked by position. The copies are not always paired in document order: when every tied copy shifts the same way by more than the spacing of the copies, the in-order and the crossed pairing have the same position sum and the assignment solver decides. The term changed 0 of 13223 matchings on documents without repeated statements (E15-H79)
- **Limit: repeated statements** - the position term does not read which copy of a repeated statement was deleted. On a 62-statement document with 9 repeated statements the matched statements are textually equal to the shorter document in 111 of 120 pure deletions, against a registered bar of 114 (E15-H79, Refuted on that clause; 88 of 120 without the term). The matched order gap is at or below the order gap in 120 of those 120 cases, largest reading 0.004876, and `moved` flags a kept statement in 4 of them (E15-H80)
- **Limit: inserted statement that takes a match** - a statement inserted into one document that costs less than `MATCH_CUTOFF` to a statement of the other document can be paired with that statement in place of its counterpart, for example when the inserted statement restates a statement the document already holds. Unless another pair below the cutoff exists, the counterpart then reads `added` or `removed`, and `moved` can flag the A statement of the new pair and of each matched pair it crosses. E15-H80 measured one insertion fixture, a statement without a match placed in front of a 6-statement document (0 of 6 statements flagged); the rate of the matched case is not measured
- **Limit: cutoff margin** - the largest ground cost among the 124 true reworded pairs of the fixture is 0.322901, 0.002099 below the cutoff. A reworded pair that costs 0.325 or more is not matched; unless another pair below the cutoff exists, its A statement reads `removed` and its B statement `added`
- **Limit: 5% rank step** - `moved` compares `|matched_displacement / n_matched|` with `MOVED_TOLERANCE` 0.05. A rank step of exactly 5% of the matched count (1 of 20 statements, 2 of 40) is not above the tolerance and reads `moved = false`, where the 1.1.5 form `target_index/n_b − index/n_a` exceeds 0.05 at some positions through float rounding. With two statements exchanged the two forms agree on 183 of 190 pairs at 20 statements and on 765 of 780 at 40 (E15-H80)
- **Limit: identity floor and unmatched statements** - the matched order gap of a pure deletion is the order gap of the shorter document against a copy of itself, which is not zero and grows with the statement count, and the reading covers the matched statements only; both readings are in the [structure document](solution/wmd-structure-distance-sota.md#limitations)

### Constants

Module attributes, not exported from the top-level package: import them from `docdistance.distance` (`MOVED_TOLERANCE`, `MEMO_SIZE` and `INPUT_FORMATS` from `docdistance.pipeline`).

| Constant | Value | Meaning |
| --- | --- | --- |
| `DEFAULT_THRESHOLD` | `0.77` | closeness cutoff of the semantic verdict; in-sample on the ibm summaries (E12-H63) |
| `STRUCTURE_THRESHOLD` | `0.90` | `structure_closeness` cutoff of the structural verdict; placed on the ibm ladder, no cross-article band exists (E12-H59) |
| `MATCH_CUTOFF` | `0.325` | ground cost below which `partial_matching` pairs two statements (E13-H71); added after 1.1.5 |
| `MATCH_POSITION_EPS` | `1e-6` | weight of the position term of the matching (E15-H79); added after 1.1.5 |
| `SOURCE_SUPPORT_CUTOFF` | `0.34625` | nearest-source cost above which `unsupported_count` counts a statement, valid on raw embeddings only (E15-H78); added after 1.1.5 |
| `ANISOTROPY_MIN_POOL` | `71` | minimum number of pooled statements for the all-but-the-top estimate (E12-H65); added after 1.1.5 |
| `MOVED_TOLERANCE` | `0.05` | tolerance of `moved` on `\|matched_displacement / n_matched\|`; after 1.1.5 read on matched statements (E15-H80) |
| `MEMO_SIZE` | `64` | bound, in entries, of each memo of a CLI `--pairs` batch (E12-H67, E12-H68); added after 1.1.5 |
| `INPUT_FORMATS` | `("plain", "markdown", "html", "auto")` | the values of `input_format`, of which `"auto"` is the default; added after 1.1.5 |

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
| `matched_order_gap` | `float \| None` | added after 1.1.5: the order gap over the statements both documents hold (`matched_order_gap`); `None` when no pair is matched; equals `order_gap` when the counts are equal and every statement is matched |
| `n_matched` | `int` | added after 1.1.5: number of one-to-one pairs of `partial_matching` |

`structure_closeness` and `verdict` are read from `order_gap` only. On unequal statement counts `order_gap` includes the cost of the surplus statements, so a deletion that keeps the order of the remaining statements lowers `structure_closeness` and can turn the verdict to `not similar`: 2 of 10 statements deleted read `order_gap` 0.227 (DEF-TRANS-10). `matched_order_gap` is the reading that such a deletion leaves at the identity floor.

`SourceConditionedResult`:

| Field | Type | Meaning |
| --- | --- | --- |
| `d_sel` | `float` | selection divergence: metric OT between the two coverage profiles over `S` |
| `residual_a`, `residual_b` | `float` | balanced transport cost `smd(X, S)` from each document to the whole source, a metric; it grows when a document covers only part of the source and is not the share of the document the source does not support (10 statements copied verbatim from a 70-statement source read 0.958, DEF-TRANS-8) |
| `closeness_a`, `closeness_b` | `float` | the residuals mapped to 0..1 (0.322 for that verbatim extract) |
| `n_statements_a`, `n_statements_b`, `n_statements_source` | `int` | statement counts |
| `n_unsupported_a`, `n_unsupported_b` | `int` | added after 1.1.5: count of statements with no near source statement - the nearest source statement costs more than `SOURCE_SUPPORT_CUTOFF` 0.34625 - read before the anisotropy pre-pass; a score, not a reading of entailment, for which the model-graded reading is `grd_a` / `grd_b` ([readings and limits](solution/wmd-source-conditioned-docdistance-solution-sota.md#distance-to-the-source-and-unsupported-statements)) |
| `coverage_a`, `coverage_b` | `list[float]` | each document's coverage distribution over the source statements |
| `grd_a`, `grd_b` | `float \| None` | each document's reranker x NLI grounding residual (E03-H11); `None` on the metric-only path |
| `d_grd` | `float \| None` | grounding-axis separation `|grd_a − grd_b|`; `None` without grounding |

### Detail-file fields added after 1.1.5

The dicts returned by the `*_with_details` / `*_with_map` methods and written by `--details-json` / `--source-map-json`. Every key 1.1.5 writes keeps its value under `input_format="plain"`, except `moved` and `n_moved` of the structural file and, with `markdown` on, the statements of a document that holds marker-only lines (they are dropped after 1.1.5, DEF-SEGM-15). With the default `input_format="auto"` the statements of a `.md`, `.markdown`, `.html` or `.htm` file, and so every value read from them, can differ from 1.1.5.

| File | Where | Field | Meaning |
| --- | --- | --- | --- |
| semantic | each `flows` record | `match_index` | index of the B statement the A statement is paired with one to one, `null` without a pair |
| semantic | each `flows` record | `match_status` | `kept` (in a pair) or `removed` (in no pair) |
| semantic | each `targets` record | `match_source_index` | index of the A statement the B statement is paired with, `null` without a pair |
| semantic | each `targets` record | `match_status` | `kept` or `added` (in no pair) |
| semantic | `summary` | `n_kept`, `n_removed`, `n_added` | counts of the three states; `n_kept + n_removed` is the statement count of A, `n_kept + n_added` that of B |
| semantic, structural | `thresholds` | `match_cutoff` | `MATCH_CUTOFF` in force, 0.325 |
| semantic, structural | `definitions` | `kept`, `removed`, `added` | one line each for the three states |
| structural | each `statements` record | `matched_displacement` | rank of the paired B statement among the paired B statements minus the rank of the statement among the kept A statements; `null` when not kept; 0 on every kept statement of a pure deletion |
| structural | each `statements` record | `moved` | changed definition: `matched AND matched_displacement != 0 AND abs(matched_displacement / n_matched) > moved_tolerance AND NOT no_counterpart` (E15-H80); 1.1.5 read `abs(displacement_norm) > moved_tolerance` |
| structural | `summary` | `n_moved` | count of `moved` under the changed definition |
| source map | each `a` and `b` record | `nearest_source_index`, `nearest_source_cost` | the nearest source statement and its ground cost, on the embeddings before the pre-pass |
| source map | each `a` and `b` record | `unsupported` | `nearest_source_cost` above `source_support_cutoff` |
| source map | header | `source_support_cutoff`, `n_unsupported` | `SOURCE_SUPPORT_CUTOFF` in force, 0.34625, and the unsupported counts under the keys `a` and `b` |

`moved` on the 280 order-preserving deletions of E15-H80 flags 0 statements in 280 of 280 cases, where the 1.1.5 rule flags a statement in 274 of them. On 1680 permutations of documents of 11 to 14 statements its flags equal the conjunction `displacement != 0 AND abs(displacement_norm) > moved_tolerance` statement for statement. A real record of the semantic and the structural file is in the [worked example](example-map-interpretations.md).

## CLI

`docdistance <command>`; human output is rich and coloured, `--json` is machine-readable, `--result-only` is
the bare scalar. Logs go to stderr, so stdout carries only the result.

| Command | Purpose | Key options |
| --- | --- | --- |
| `init [MODE]` | provision a mode's models from local / S3 / HuggingFace (the only command that fetches); writes `docdistance.json` | `--source`, `--backend`, `--aws-profile`, `--aws-endpoint-url`, `--region`, `--home` |
| `distance-semantic A B` | Statement Mover's Distance content closeness between two documents | `--backend`, `--gpu`, `--anisotropy`, `--threshold`, `--markdown`, `--input-format`, `--details-json`, `--pairs`, `--json`, `--result-only` |
| `distance-structural A B` | structural OPW order-gap distance between two documents | `--backend`, `--gpu`, `--anisotropy`, `--threshold`, `--markdown`, `--input-format`, `--details-json`, `--pairs`, `--json`, `--result-only` |
| `distance-wrt-source A B --source S` | source-conditioned `d(A, B | S)` | `--source/-s` (required), `--backend`, `--gpu`, `--anisotropy`, `--markdown`, `--input-format`, `--source-map-json`, `--pairs` (after 1.1.5), `--json`, `--result-only` |

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
rank shift under the crisp alignment - `0` means it stayed in place, nonzero means it moved.
`matched_displacement` is its rank shift among the statements both documents hold, and `moved` fires when the
statement is matched, `matched_displacement` is not 0, `abs(matched_displacement / n_matched)` is above
`moved_tolerance` (0.05) and the record is not `no_counterpart`; a statement without a match reads
`matched_displacement` `null` and `moved` `false`. The order gap over the matched statements is on the result
object, `result.matched_order_gap` with `result.n_matched`; the details dict does not carry it. The content
reading lives in the companion `semantic_distance_with_details`
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
docdistance distance-semantic page.html page.md                  # reader chosen per document from the file extension
docdistance distance-semantic a.md b.md --input-format plain     # the reader and the numbers of 1.1.5
docdistance distance-wrt-source sum_a.md sum_b.md -s article.md  # source-conditioned
docdistance distance-wrt-source --pairs pairs.tsv -s article.md  # batch: every pair against one source, JSONL out
```
