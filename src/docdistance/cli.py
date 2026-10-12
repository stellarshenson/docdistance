"""docdistance command-line interface.

Four subcommands - ``init`` (provision a mode's models from local / S3 / HuggingFace, the only one
that downloads), ``distance-semantic`` (symmetric SMD), ``distance-structural`` (order-gap) and
``distance-wrt-source`` (source-conditioned). A mode must be ``init``'d before its distance runs,
else the command exits 1 with a clear message.
Human output is rich and coloured; ``--json`` emits machine-readable JSON and ``--result-only`` the
bare result. ``--pairs FILE`` batches many pairs through one model load, emitting JSONL - one JSON
object per pair per line. Logs go to stderr (loguru, ``--verbose`` for DEBUG), so stdout carries
only the result.
"""

from __future__ import annotations

from enum import Enum
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import typer

from docdistance.config import configure_logging
from docdistance.distance import (
    ANISOTROPY_MIN_POOL,
    DEFAULT_THRESHOLD,
    MATCH_CUTOFF,
    SOURCE_SUPPORT_CUTOFF,
    STRUCTURE_THRESHOLD,
)

app = typer.Typer(
    rich_markup_mode="rich",
    no_args_is_help=True,
    add_completion=False,
    help="[bold]docdistance[/bold] - semantic distance between documents via Statement Mover's Distance "
    "(optimal transport over mmBERT statement embeddings).",
)

_out = Console()  # stdout, for the result
_err = Console(stderr=True)  # stderr, for errors


class Backend(str, Enum):
    openvino = "openvino"
    torch = "torch"


class Mode(str, Enum):
    wmd = "wmd"
    wmd_wrt_source = "wmd-wrt-source"


class InputFormat(str, Enum):
    plain = "plain"
    markdown = "markdown"
    html = "html"
    auto = "auto"


def _version_cb(value: bool):
    if value:
        from docdistance import __version__

        typer.echo(f"docdistance {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_cb, is_eager=True, help="show version and exit"
    ),
):
    """Semantic document distance grounded in optimal-transport theory."""


def _run(fn, machine: bool = False):
    """Call ``fn`` and turn a missing-model or un-init'd-mode error into a clean message + exit 1.

    ``machine`` marks --json / --result-only mode: a pipeline in that mode reads stdout only, so
    the error must also land there as a one-line JSON object - stderr alone is discarded by
    ``2>/dev/null`` and the empty stdout fails downstream with JSONDecodeError.
    """
    from docdistance.encoders import ModelsNotInstalled
    from docdistance.settings import NotInitializedError

    try:
        return fn()
    except (ModelsNotInstalled, NotInitializedError, ValueError, OSError) as exc:
        # ValueError covers the empty-document case (pipeline.embed_statements), which a
        # title-only markdown reaches and which otherwise escapes as a stack trace
        _err.print(f"[bold red]error:[/bold red] {exc}")
        if machine:
            if isinstance(exc, NotInitializedError):
                code = "not_initialized"  # the message already names the searched config path
            elif isinstance(exc, ModelsNotInstalled):
                code = "models_not_installed"
            elif isinstance(exc, OSError):
                code = "io_error"  # a --pairs file that cannot be read, or a --details-json / --source-map-json path that cannot be written
            else:
                code = "empty_document"
            # plain json.dumps via typer.echo, not the rich console: colour codes and
            # wrapping would corrupt the object (same constraint as the emitters)
            typer.echo(json.dumps({"error": code, "message": str(exc)}))
        raise typer.Exit(1)


def _apply_home(home: str | None) -> None:
    """Pin ``--home`` as ``$DOCDISTANCE_HOME`` before the readiness gate, mirroring ``init --home``,
    so docdistance.json and the model mirror resolve from that folder instead of cwd."""
    if home:
        import os

        from docdistance import settings

        os.environ[settings.ENV_HOME] = str(Path(home))


# segmentation options shared by the three distance commands
_MARKDOWN_OPTION = typer.Option(
    False,
    "--markdown",
    help="markdown-aware segmentation - one statement per heading / table row / list item / "
    "fenced-code line (markup stripped from the embedded text), only prose goes through SaT; "
    "equals --input-format markdown",
)
_INPUT_FORMAT_OPTION = typer.Option(
    None,
    "--input-format",
    help="reader of every document: auto (the default) chooses it per document from the file "
    "extension - .md and .markdown the markdown reader, .html and .htm the html reader, every "
    "other file and raw text the plain reader; plain gives the whole text to the prose segmenter, "
    "as 1.1.5 read every document; markdown reads as --markdown does; html converts the document "
    "to markdown lines and reads those",
)
_MIN_WORDS_OPTION = typer.Option(
    0,
    "--min-words",
    metavar="N",
    help="drop statements under N words after segmentation, before encoding - the transport "
    "marginals renormalise over the survivors, so the SMD changes",
)
_SKIP_TABLE_FRAGMENTS_OPTION = typer.Option(
    False,
    "--skip-table-fragments",
    help="drop table-fragment statements (pipes or separator punctuation) after segmentation, "
    "before encoding - the transport marginals renormalise over the survivors",
)


def _input_format(markdown: bool, input_format: InputFormat | None) -> str:
    """The ``input_format`` value the options ask for; ``--markdown`` equals ``--input-format markdown``.

    ``input_format`` is ``None`` when the option was not given, which reads ``auto``.
    ``--markdown`` beside another ``--input-format`` value, a given ``auto`` as well, names two
    readers: a usage error, exit code 2.
    """
    if not markdown:
        return input_format.value if input_format else InputFormat.auto.value
    if input_format not in (None, InputFormat.markdown):
        _err.print(
            f"[bold red]error:[/bold red] --markdown and --input-format {input_format.value} "
            "name two readers; pass one of them"
        )
        raise typer.Exit(2)
    return InputFormat.markdown.value


def _drop_extra(dd, min_words: int, skip_table_fragments: bool) -> dict:
    """The n_dropped / drop_rules keys added to machine JSON when a drop rule is active."""
    return {
        "n_dropped": dd.last_n_dropped,
        "drop_rules": {"min_words": min_words, "skip_table_fragments": skip_table_fragments},
    }


def _resolve_gpu(gpu: bool, backend: Backend) -> tuple[str, str | None]:
    """Map ``--gpu`` to (backend, device): force the torch backend on CUDA, erroring if not secured."""
    if not gpu:
        return backend.value, None
    from docdistance.encoders import GpuNotAvailable, require_gpu

    try:
        require_gpu()
    except GpuNotAvailable as exc:
        _err.print(f"[bold red]error:[/bold red] {exc}")
        raise typer.Exit(1)
    return "torch", "cuda"


def _check_pair_args(pairs: str | None, a: str | None, b: str | None) -> None:
    """Positional documents and ``--pairs`` are mutually exclusive; one of the two is required."""
    if pairs and (a is not None or b is not None):
        _err.print("[bold red]error:[/bold red] pass two documents or --pairs FILE, not both")
        raise typer.Exit(2)
    if not pairs and (a is None or b is None):
        _err.print("[bold red]error:[/bold red] two documents (or --pairs FILE) required")
        raise typer.Exit(2)


def _read_pairs(pairs_file: str) -> list[tuple[str, str, str]]:
    """Parse the batch TSV - ``a-path<TAB>b-path[<TAB>id]`` per line, blank lines skipped.

    Returns (a, b, pair) rows where ``pair`` is the given id or the 0-based row index.
    A malformed line fails loudly naming its 1-based line number.
    """
    rows: list[tuple[str, str, str]] = []
    for lineno, line in enumerate(Path(pairs_file).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) not in (2, 3) or not fields[0] or not fields[1]:
            # OSError so both _run-wrapped call sites emit the io_error object in machine mode
            raise OSError(
                f"{pairs_file} line {lineno}: "
                f"expected a-path<TAB>b-path[<TAB>id], got {len(fields)} field(s)"
            )
        pair = fields[2] if len(fields) == 3 and fields[2] else str(len(rows))
        rows.append((fields[0], fields[1], pair))
    return rows


def _details_path(details_json: str, pair: str) -> Path:
    """Resolve the per-pair details file: a ``{pair}`` template, else a directory of ``<pair>.json``."""
    if "{pair}" in details_json:
        path = Path(details_json.replace("{pair}", pair))
    else:
        path = Path(details_json) / f"{pair}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _run_batch(rows, details_json: str | None, dist, dist_details, extra=None) -> None:
    """ACC-CLI-12 batch loop: the caller built one DocDistance; emit one JSON object per pair per
    line (JSONL) to stdout, writing a per-pair details file when ``--details-json`` is given
    (``--source-map-json`` on ``distance-wrt-source``, whose source is fixed for the batch).
    The caller builds it with ``memo_size=MEMO_SIZE``, so a file that recurs across rows is
    segmented and encoded once and each (document, source) grounding is computed once.
    ``extra`` is an optional callable evaluated after each pair - its dict (e.g. the per-pair
    n_dropped of an active fragment filter) is merged into that pair's JSON line."""

    def batch():
        for a, b, pair in rows:
            if details_json:
                r, details = dist_details(a, b)
                _details_path(details_json, pair).write_text(json.dumps(details, indent=2))
            else:
                r = dist(a, b)
            typer.echo(json.dumps({"pair": pair, **r.to_dict(), **(extra() if extra else {})}))

    _run(batch, machine=True)


def _emit_distance(r, json_out: bool, result_only: bool, extra: dict | None = None) -> None:
    if result_only:
        typer.echo(str(r.smd))
        return
    if json_out:
        typer.echo(json.dumps({**r.to_dict(), **(extra or {})}, indent=2))
        return
    color = "green" if r.verdict == "similar" else "red"
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan")
    grid.add_column()
    grid.add_row("SMD (distance)", f"{r.smd:.4f}")
    grid.add_row("closeness", f"{r.closeness * 100:.1f}%")
    grid.add_row(
        "verdict", f"[{color}]{r.verdict}[/{color}]  (threshold {r.threshold:.2f} closeness)"
    )
    grid.add_row("bounds", f"WCD {r.wcd:.4f}  ≤  RWMD {r.rwmd:.4f}  ≤  SMD {r.smd:.4f}")
    grid.add_row("statements", f"{r.n_statements_a} vs {r.n_statements_b}")
    grid.add_row("anisotropy", "on" if r.anisotropy else "off")
    _out.print(
        Panel(grid, title="[bold]Document distance[/bold]", border_style=color, expand=False)
    )


def _emit_structural(r, json_out: bool, result_only: bool, extra: dict | None = None) -> None:
    # typer.echo, not _out.print: Rich colourizes and wraps, which corrupts both the
    # scalar and the JSON on any coloured or narrow terminal (the other emitters match)
    if result_only:
        typer.echo(str(r.structure_closeness))
        return
    if json_out:
        typer.echo(json.dumps({**r.to_dict(), **(extra or {})}, indent=2))
        return
    color = "green" if r.verdict == "similar" else "red"
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan")
    grid.add_column()
    grid.add_row("structure closeness", f"{r.structure_closeness * 100:.1f}%")
    grid.add_row("order gap", f"{r.order_gap:.4f}")
    grid.add_row(
        "matched order gap",
        "no statement matched" if r.matched_order_gap is None else f"{r.matched_order_gap:.4f}",
    )
    grid.add_row("matched statements", f"{r.n_matched}  (ground cost < {MATCH_CUTOFF})")
    grid.add_row("SMD (content)", f"{r.smd:.4f}")
    grid.add_row(
        "verdict", f"[{color}]{r.verdict}[/{color}]  (threshold {r.threshold:.2f} closeness)"
    )
    grid.add_row("statements", f"{r.n_statements_a} vs {r.n_statements_b}")
    grid.add_row("anisotropy", "on" if r.anisotropy else "off")
    _out.print(
        Panel(
            grid,
            title="[bold]Document structure distance[/bold]",
            border_style=color,
            expand=False,
        )
    )


def _emit_wrt_source(r, json_out: bool, result_only: bool, extra: dict | None = None) -> None:
    if result_only:
        if r.grd_a is not None:
            typer.echo(f"{r.d_sel},{r.grd_a},{r.grd_b}")
        else:
            typer.echo(f"{r.d_sel},{r.residual_a},{r.residual_b}")
        return
    if json_out:
        typer.echo(json.dumps({**r.to_dict(), **(extra or {})}, indent=2))
        return
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan")
    grid.add_column()
    grid.add_row("D_sel (selection divergence)", f"{r.d_sel:.4f}")
    if r.grd_a is not None:
        grid.add_row("A grounding D_grd (E03-H11)", f"{r.grd_a:.4f}")
        grid.add_row("B grounding D_grd (E03-H11)", f"{r.grd_b:.4f}")
        grid.add_row("D_grd separation |A-B|", f"{r.d_grd:.4f}")
    grid.add_row(
        "A → source (geometric)", f"{r.residual_a:.4f}  (closeness {r.closeness_a * 100:.1f}%)"
    )
    grid.add_row(
        "B → source (geometric)", f"{r.residual_b:.4f}  (closeness {r.closeness_b * 100:.1f}%)"
    )
    grid.add_row(
        "statements", f"A {r.n_statements_a} / B {r.n_statements_b} / S {r.n_statements_source}"
    )
    grid.add_row(
        "unsupported statements",
        f"A {r.n_unsupported_a} / B {r.n_unsupported_b}  "
        f"(nearest source cost > {SOURCE_SUPPORT_CUTOFF})",
    )
    _out.print(
        Panel(
            grid,
            title="[bold]Source-conditioned distance d(A,B|S)[/bold]",
            border_style="cyan",
            expand=False,
        )
    )
    note = (
        "D_grd = reranker x NLI relevance-gated ungrounded mass (lower = better grounded); residual "
        "= geometric distance to the source"
        if r.grd_a is not None
        else "residual = geometric distance to the source (run init wmd-wrt-source for reranker + NLI grounding)"
    )
    _out.print(f"[dim]{note}[/dim]")


@app.command(
    name="distance-semantic",
    epilog="[bold]Examples[/bold]\n\n"
    "  docdistance distance-semantic report_v1.md report_v2.md\n"
    '  docdistance distance-semantic "first text" "second text" --backend torch\n'
    "  docdistance distance-semantic a.md b.md --json\n"
    "  docdistance distance-semantic a.md b.md --details-json content.json   [dim]# statement → statement content alignment[/dim]\n"
    "  docdistance distance-structural a.md b.md --details-json order.json   [dim]# per-statement displacement / moved[/dim]\n"
    "  docdistance distance-semantic report.md brief.md --details-json map.json --relevance   [dim]# long document first, brief second[/dim]\n"
    "  docdistance distance-semantic --pairs pairs.tsv                       [dim]# batch: one model load, JSONL out[/dim]\n"
    "  docdistance distance-semantic a.md b.md --result-only",
)
def distance_semantic(
    a: str = typer.Argument(None, help="first document - a file path or raw text"),
    b: str = typer.Argument(None, help="second document - a file path or raw text"),
    backend: Backend = typer.Option(
        Backend.openvino, "--backend", help="statement encoder backend"
    ),
    gpu: bool = typer.Option(
        False,
        "--gpu",
        help="force the torch backend on CUDA; errors if GPU support is not secured",
    ),
    anisotropy: bool = typer.Option(
        False,
        "--anisotropy/--no-anisotropy",
        help="all-but-the-top anisotropy removal, estimated from the pooled statements of the "
        f"two documents; the estimate needs about {ANISOTROPY_MIN_POOL} pooled statements, and a "
        "smaller pool logs one warning and is scored all the same; off by default",
    ),
    threshold: float = typer.Option(
        DEFAULT_THRESHOLD,
        "--threshold",
        help="closeness cutoff for the similar / not-similar verdict",
    ),
    markdown: bool = _MARKDOWN_OPTION,
    input_format: InputFormat = _INPUT_FORMAT_OPTION,
    min_words: int = _MIN_WORDS_OPTION,
    skip_table_fragments: bool = _SKIP_TABLE_FRAGMENTS_OPTION,
    details_json: str = typer.Option(
        None,
        "--details-json",
        metavar="FILE",
        help="write per-statement content-alignment details (transport map) to FILE; "
        "with --pairs, a directory or a template containing {pair} - one map per pair",
    ),
    pairs: str = typer.Option(
        None,
        "--pairs",
        metavar="FILE",
        help="batch mode: TSV of a-path<TAB>b-path[<TAB>id] - one model load, "
        "one JSON object per pair per line (JSONL) to stdout",
    ),
    relevance: bool = typer.Option(
        False,
        "--relevance",
        help="add the reranker relevance of each statement of the first document over the "
        "second to the map; needs --details-json; pass the long document first and the brief "
        "second; costs one reranker pass per pair of statements",
    ),
    json_out: bool = typer.Option(False, "--json", help="machine-readable JSON to stdout"),
    result_only: bool = typer.Option(
        False, "--result-only", help="bare SMD scalar to stdout, no clutter"
    ),
    home: str = typer.Option(
        None,
        "--home",
        help="folder holding docdistance.json + the model mirror (default: $DOCDISTANCE_HOME or cwd)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="DEBUG logging to stderr"),
):
    """Symmetric distance between two documents - the exact Statement Mover's Distance."""
    configure_logging(verbose)
    _apply_home(home)
    _check_pair_args(pairs, a, b)
    if relevance and not details_json:
        _err.print("[bold red]error:[/bold red] --relevance needs --details-json")
        raise typer.Exit(2)
    requested = _input_format(markdown, input_format)

    backend_value, device = _resolve_gpu(gpu, backend)
    seg = {
        "input_format": requested,
        "min_words": min_words,
        "skip_table_fragments": skip_table_fragments,
    }
    filters_on = min_words > 0 or skip_table_fragments
    extra = None
    if pairs:
        from docdistance.pipeline import MEMO_SIZE, DocDistance

        rows = _run(lambda: _read_pairs(pairs), machine=True)
        dd = DocDistance(backend=backend_value, device=device, memo_size=MEMO_SIZE)
        _run_batch(
            rows,
            details_json,
            lambda x, y: dd.semantic_distance(
                x, y, anisotropy=anisotropy, threshold=threshold, **seg
            ),
            lambda x, y: dd.semantic_distance_with_details(
                x, y, anisotropy=anisotropy, threshold=threshold, relevance=relevance, **seg
            ),
            extra=(lambda: _drop_extra(dd, min_words, skip_table_fragments))
            if filters_on
            else None,
        )
        return
    if details_json:
        from docdistance.pipeline import DocDistance

        dd = DocDistance(backend=backend_value, device=device)

        def _details_run():
            res = dd.semantic_distance_with_details(
                a, b, anisotropy=anisotropy, threshold=threshold, relevance=relevance, **seg
            )
            Path(details_json).write_text(json.dumps(res[1], indent=2))
            return res

        result, details = _run(_details_run, machine=json_out or result_only)
        if filters_on:
            extra = _drop_extra(dd, min_words, skip_table_fragments)
        _err.print(
            f"[green]details written:[/green] {details_json} "
            f"(A {details['n_statements']['a']} → B {details['n_statements']['b']} statements)"
        )
    elif filters_on:
        from docdistance.pipeline import DocDistance

        dd = DocDistance(backend=backend_value, device=device)
        result = _run(
            lambda: dd.semantic_distance(a, b, anisotropy=anisotropy, threshold=threshold, **seg),
            machine=json_out or result_only,
        )
        extra = _drop_extra(dd, min_words, skip_table_fragments)
    else:
        from docdistance.pipeline import semantic_distance

        result = _run(
            lambda: semantic_distance(
                a,
                b,
                backend=backend_value,
                anisotropy=anisotropy,
                threshold=threshold,
                device=device,
                input_format=requested,
            ),
            machine=json_out or result_only,
        )
    _emit_distance(result, json_out, result_only, extra)


@app.command(
    name="distance-structural",
    epilog="[bold]Examples[/bold]\n\n"
    "  docdistance distance-structural a.md b.md --details-json order.json   [dim]# per-statement displacement / moved[/dim]\n"
    "  docdistance distance-structural --pairs pairs.tsv                     [dim]# batch: one model load, JSONL out[/dim]\n"
    "  docdistance distance-structural a.md b.md --result-only",
)
def distance_structural(
    a: str = typer.Argument(None, help="first document - a file path or raw text"),
    b: str = typer.Argument(None, help="second document - a file path or raw text"),
    backend: Backend = typer.Option(
        Backend.openvino, "--backend", help="statement encoder backend"
    ),
    gpu: bool = typer.Option(
        False,
        "--gpu",
        help="force the torch backend on CUDA; errors if GPU support is not secured",
    ),
    anisotropy: bool = typer.Option(
        False,
        "--anisotropy/--no-anisotropy",
        help="all-but-the-top anisotropy removal, estimated from the pooled statements of the "
        f"two documents; the estimate needs about {ANISOTROPY_MIN_POOL} pooled statements, and a "
        "smaller pool logs one warning and is scored all the same; off by default",
    ),
    threshold: float = typer.Option(
        STRUCTURE_THRESHOLD,
        "--threshold",
        help="structure_closeness cutoff for the similar / not-similar verdict",
    ),
    markdown: bool = _MARKDOWN_OPTION,
    input_format: InputFormat = _INPUT_FORMAT_OPTION,
    min_words: int = _MIN_WORDS_OPTION,
    skip_table_fragments: bool = _SKIP_TABLE_FRAGMENTS_OPTION,
    details_json: str = typer.Option(
        None,
        "--details-json",
        metavar="FILE",
        help="write per-statement structural details (displacement / moved) to FILE; "
        "with --pairs, a directory or a template containing {pair} - one map per pair",
    ),
    pairs: str = typer.Option(
        None,
        "--pairs",
        metavar="FILE",
        help="batch mode: TSV of a-path<TAB>b-path[<TAB>id] - one model load, "
        "one JSON object per pair per line (JSONL) to stdout",
    ),
    json_out: bool = typer.Option(False, "--json", help="machine-readable JSON to stdout"),
    result_only: bool = typer.Option(
        False, "--result-only", help="bare structure_closeness scalar to stdout, no clutter"
    ),
    home: str = typer.Option(
        None,
        "--home",
        help="folder holding docdistance.json + the model mirror (default: $DOCDISTANCE_HOME or cwd)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="DEBUG logging to stderr"),
):
    """Structural (order) distance between two documents - the E11-H55 OPW order-gap."""
    configure_logging(verbose)
    _apply_home(home)
    _check_pair_args(pairs, a, b)
    requested = _input_format(markdown, input_format)

    backend_value, device = _resolve_gpu(gpu, backend)
    seg = {
        "input_format": requested,
        "min_words": min_words,
        "skip_table_fragments": skip_table_fragments,
    }
    filters_on = min_words > 0 or skip_table_fragments
    extra = None
    if pairs:
        from docdistance.pipeline import MEMO_SIZE, DocDistance

        rows = _run(lambda: _read_pairs(pairs), machine=True)
        dd = DocDistance(backend=backend_value, device=device, memo_size=MEMO_SIZE)
        _run_batch(
            rows,
            details_json,
            lambda x, y: dd.structural_distance(
                x, y, anisotropy=anisotropy, threshold=threshold, **seg
            ),
            lambda x, y: dd.structural_distance_with_details(
                x, y, anisotropy=anisotropy, threshold=threshold, **seg
            ),
            extra=(lambda: _drop_extra(dd, min_words, skip_table_fragments))
            if filters_on
            else None,
        )
        return
    if details_json:
        from docdistance.pipeline import DocDistance

        dd = DocDistance(backend=backend_value, device=device)

        def _details_run():
            res = dd.structural_distance_with_details(
                a, b, anisotropy=anisotropy, threshold=threshold, **seg
            )
            Path(details_json).write_text(json.dumps(res[1], indent=2))
            return res

        result, details = _run(_details_run, machine=json_out or result_only)
        if filters_on:
            extra = _drop_extra(dd, min_words, skip_table_fragments)
        _err.print(
            f"[green]details written:[/green] {details_json} "
            f"(order_gap={details['order_gap']}, "
            f"structure_closeness={details['structure_closeness']})"
        )
    elif filters_on:
        from docdistance.pipeline import DocDistance

        dd = DocDistance(backend=backend_value, device=device)
        result = _run(
            lambda: dd.structural_distance(
                a, b, anisotropy=anisotropy, threshold=threshold, **seg
            ),
            machine=json_out or result_only,
        )
        extra = _drop_extra(dd, min_words, skip_table_fragments)
    else:
        from docdistance.pipeline import structural_distance

        result = _run(
            lambda: structural_distance(
                a,
                b,
                backend=backend_value,
                anisotropy=anisotropy,
                threshold=threshold,
                device=device,
                input_format=requested,
            ),
            machine=json_out or result_only,
        )
    _emit_structural(result, json_out, result_only, extra)


@app.command(
    name="distance-wrt-source",
    epilog="[bold]Examples[/bold]\n\n"
    "  docdistance distance-wrt-source summary_a.md summary_b.md --source article.md\n"
    "  docdistance distance-wrt-source a.md b.md -s s.md --json\n"
    "  docdistance distance-wrt-source a.md b.md -s s.md --source-map-json map.json   [dim]# statement → source map[/dim]\n"
    "  docdistance distance-wrt-source --pairs pairs.tsv -s s.md         [dim]# batch: one model load, one source, JSONL out[/dim]\n"
    "  docdistance distance-wrt-source a.md b.md -s s.md --result-only   [dim]# d_sel,grd_a,grd_b[/dim]",
)
def distance_wrt_source(
    a: str = typer.Argument(None, help="first document - a file path or raw text"),
    b: str = typer.Argument(None, help="second document - a file path or raw text"),
    source: str = typer.Option(..., "--source", "-s", help="the common source document"),
    backend: Backend = typer.Option(
        Backend.openvino, "--backend", help="statement encoder backend"
    ),
    gpu: bool = typer.Option(
        False,
        "--gpu",
        help="force the torch backend on CUDA; errors if GPU support is not secured",
    ),
    anisotropy: bool = typer.Option(
        True,
        "--anisotropy/--no-anisotropy",
        help="anisotropy removal on the conditioned selection axis - on by default (E04-H15), --no-anisotropy to opt out",
    ),
    markdown: bool = _MARKDOWN_OPTION,
    input_format: InputFormat = _INPUT_FORMAT_OPTION,
    min_words: int = _MIN_WORDS_OPTION,
    skip_table_fragments: bool = _SKIP_TABLE_FRAGMENTS_OPTION,
    source_map_json: str = typer.Option(
        None,
        "--source-map-json",
        help="also write a statement → source alignment map (which source statements each A/B statement covers) to this JSON file; "
        "with --pairs, a directory or a template containing {pair} - one map per pair",
        metavar="FILE",
    ),
    pairs: str = typer.Option(
        None,
        "--pairs",
        metavar="FILE",
        help="batch mode: TSV of a-path<TAB>b-path[<TAB>id], every pair against the one --source - "
        "one model load, one JSON object per pair per line (JSONL) to stdout",
    ),
    json_out: bool = typer.Option(False, "--json", help="machine-readable JSON to stdout"),
    result_only: bool = typer.Option(
        False, "--result-only", help="bare comma-separated d_sel,grd_a,grd_b to stdout"
    ),
    home: str = typer.Option(
        None,
        "--home",
        help="folder holding docdistance.json + the model mirror (default: $DOCDISTANCE_HOME or cwd)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="DEBUG logging to stderr"),
):
    """Source-conditioned distance d(A, B | S) - selection divergence plus each document's distance to S."""
    configure_logging(verbose)
    _apply_home(home)
    _check_pair_args(pairs, a, b)
    requested = _input_format(markdown, input_format)

    backend_value, device = _resolve_gpu(gpu, backend)
    seg = {
        "input_format": requested,
        "min_words": min_words,
        "skip_table_fragments": skip_table_fragments,
    }
    filters_on = min_words > 0 or skip_table_fragments
    extra = None
    if pairs:
        from docdistance.pipeline import MEMO_SIZE, DocDistance

        rows = _run(lambda: _read_pairs(pairs), machine=True)
        dd = DocDistance(backend=backend_value, device=device, memo_size=MEMO_SIZE)
        _run_batch(
            rows,
            source_map_json,
            lambda x, y: dd.distance_wrt_source(x, y, source, anisotropy=anisotropy, **seg),
            lambda x, y: dd.distance_wrt_source_with_map(
                x, y, source, anisotropy=anisotropy, **seg
            ),
            extra=(lambda: _drop_extra(dd, min_words, skip_table_fragments))
            if filters_on
            else None,
        )
        return
    if source_map_json:
        from docdistance.pipeline import DocDistance

        dd = DocDistance(backend=backend_value, device=device)

        def _map_run():
            res = dd.distance_wrt_source_with_map(a, b, source, anisotropy=anisotropy, **seg)
            Path(source_map_json).write_text(json.dumps(res[1], indent=2))
            return res

        result, smap = _run(_map_run, machine=json_out or result_only)
        if filters_on:
            extra = _drop_extra(dd, min_words, skip_table_fragments)
        _err.print(
            f"[green]source map written:[/green] {source_map_json} "
            f"(A {smap['n_statements']['a']} + B {smap['n_statements']['b']} statements → "
            f"top-{smap['top_k']} of {smap['n_statements']['source']} source)"
        )
    elif filters_on:
        from docdistance.pipeline import DocDistance

        dd = DocDistance(backend=backend_value, device=device)
        result = _run(
            lambda: dd.distance_wrt_source(a, b, source, anisotropy=anisotropy, **seg),
            machine=json_out or result_only,
        )
        extra = _drop_extra(dd, min_words, skip_table_fragments)
    else:
        from docdistance.pipeline import source_conditioned_distance

        result = _run(
            lambda: source_conditioned_distance(
                a,
                b,
                source,
                backend=backend_value,
                anisotropy=anisotropy,
                device=device,
                input_format=requested,
            ),
            machine=json_out or result_only,
        )
    _emit_wrt_source(result, json_out, result_only, extra)


@app.command(
    epilog="[bold]Examples[/bold]\n\n"
    "  docdistance init                                    [dim]# wmd mode, from HuggingFace[/dim]\n"
    "  docdistance init wmd-wrt-source                     [dim]# + reranker + NLI grounding models[/dim]\n"
    "  docdistance init wmd-wrt-source --source s3://your-bucket/docdistance --aws-profile NAME\n"
    "  docdistance init wmd --source /path/to/models       [dim]# from a local mirror[/dim]",
)
def init(
    mode: Mode = typer.Argument(Mode.wmd, help="which distance mode to provision models for"),
    source: str = typer.Option(
        None,
        "--source",
        help="model source base: an s3://bucket/prefix, a local dir, or omit for HuggingFace",
    ),
    backend: Backend = typer.Option(
        Backend.openvino, "--backend", help="which weights to fetch (openvino INT8 or torch)"
    ),
    aws_profile: str = typer.Option(
        None,
        "--aws-profile",
        help="AWS named profile for an s3:// source (omit in Lambda for the execution-role chain)",
    ),
    aws_endpoint_url: str = typer.Option(
        None, "--aws-endpoint-url", help="custom S3 endpoint for an S3-compatible store"
    ),
    aws_region: str = typer.Option(None, "--region", help="AWS region for an s3:// source"),
    home: str = typer.Option(
        None,
        "--home",
        help="where to write docdistance.json + the model mirror (default: $DOCDISTANCE_HOME or cwd)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="DEBUG logging to stderr"),
):
    """Provision a mode's models from local / S3 / HuggingFace and write docdistance.json."""
    configure_logging(verbose)
    from docdistance.bootstrap import init as _init
    from docdistance.encoders import ModelsNotInstalled

    try:
        summary = _init(
            mode.value,
            source=source,
            backend=backend.value,
            aws_profile=aws_profile,
            aws_endpoint_url=aws_endpoint_url,
            aws_region=aws_region,
            home=home,
        )
    # OSError covers FileNotFoundError (--source missing) and an unwritable home
    except (ModelsNotInstalled, ValueError, OSError) as exc:
        _err.print(f"[bold red]error:[/bold red] {exc}")
        raise typer.Exit(1)
    srcs = ", ".join(f"{k}:{v}" for k, v in summary["sources"].items())
    _out.print(f"[green]initialized {summary['mode']}:[/green] {srcs}")
    _out.print(f"[dim]config written: {summary['config_file']}[/dim]")


if __name__ == "__main__":
    app()
