"""docdistance command-line interface.

Four subcommands - ``init`` (provision a mode's models from local / S3 / HuggingFace, the only one
that downloads), ``distance-semantic`` (symmetric SMD), ``distance-structural`` (order-gap) and
``distance-wrt-source`` (source-conditioned). A mode must be ``init``'d before its distance runs,
else the command exits 1 with a clear message.
Human output is rich and coloured; ``--json`` emits machine-readable JSON and ``--result-only`` the
bare result. Logs go to stderr (loguru, ``--verbose`` for DEBUG), so stdout carries only the result.
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
from docdistance.distance import DEFAULT_THRESHOLD

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


def _run(fn):
    """Call ``fn`` and turn a missing-model or un-init'd-mode error into a clean message + exit 1."""
    from docdistance.encoders import ModelsNotInstalled
    from docdistance.settings import NotInitializedError

    try:
        return fn()
    except (ModelsNotInstalled, NotInitializedError) as exc:
        _err.print(f"[bold red]error:[/bold red] {exc}")
        raise typer.Exit(1)


def _resolve_gpu(gpu: bool, backend: "Backend") -> tuple[str, str | None]:
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


def _emit_distance(r, json_out: bool, result_only: bool) -> None:
    if result_only:
        typer.echo(str(r.smd))
        return
    if json_out:
        typer.echo(json.dumps(r.to_dict(), indent=2))
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


def _emit_structural(r, json_out: bool, result_only: bool) -> None:
    if result_only:
        _out.print(str(r.structure_closeness))
        return
    if json_out:
        _out.print(json.dumps(r.to_dict(), indent=2))
        return
    color = "green" if r.verdict == "similar" else "red"
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan")
    grid.add_column()
    grid.add_row("structure closeness", f"{r.structure_closeness * 100:.1f}%")
    grid.add_row("order gap", f"{r.order_gap:.4f}")
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


def _emit_wrt_source(r, json_out: bool, result_only: bool) -> None:
    if result_only:
        if r.grd_a is not None:
            typer.echo(f"{r.d_sel},{r.grd_a},{r.grd_b}")
        else:
            typer.echo(f"{r.d_sel},{r.residual_a},{r.residual_b}")
        return
    if json_out:
        typer.echo(json.dumps(r.to_dict(), indent=2))
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
    "  docdistance distance-semantic a.md b.md --result-only",
)
def distance_semantic(
    a: str = typer.Argument(..., help="first document - a file path or raw text"),
    b: str = typer.Argument(..., help="second document - a file path or raw text"),
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
        help="all-but-the-top anisotropy removal - needs a corpus, off by default for a pair",
    ),
    threshold: float = typer.Option(
        DEFAULT_THRESHOLD,
        "--threshold",
        help="closeness cutoff for the similar / not-similar verdict",
    ),
    details_json: str = typer.Option(
        None,
        "--details-json",
        metavar="FILE",
        help="write per-statement content-alignment details (transport map) to FILE",
    ),
    json_out: bool = typer.Option(False, "--json", help="machine-readable JSON to stdout"),
    result_only: bool = typer.Option(
        False, "--result-only", help="bare SMD scalar to stdout, no clutter"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="DEBUG logging to stderr"),
):
    """Symmetric distance between two documents - the exact Statement Mover's Distance."""
    configure_logging(verbose)

    backend_value, device = _resolve_gpu(gpu, backend)
    if details_json:
        from docdistance.pipeline import DocDistance

        result, details = _run(
            lambda: DocDistance(
                backend=backend_value, device=device
            ).semantic_distance_with_details(a, b, anisotropy=anisotropy, threshold=threshold)
        )
        Path(details_json).write_text(json.dumps(details, indent=2))
        _err.print(
            f"[green]details written:[/green] {details_json} "
            f"(A {details['n_statements']['a']} → B {details['n_statements']['b']} statements)"
        )
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
            )
        )
    _emit_distance(result, json_out, result_only)


@app.command(
    name="distance-structural",
    epilog="[bold]Examples[/bold]\n\n"
    "  docdistance distance-structural a.md b.md --details-json order.json   [dim]# per-statement displacement / moved[/dim]\n"
    "  docdistance distance-structural a.md b.md --result-only",
)
def distance_structural(
    a: str = typer.Argument(..., help="first document - a file path or raw text"),
    b: str = typer.Argument(..., help="second document - a file path or raw text"),
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
        help="all-but-the-top anisotropy removal - needs a corpus, off by default for a pair",
    ),
    threshold: float = typer.Option(
        DEFAULT_THRESHOLD,
        "--threshold",
        help="closeness cutoff for the similar / not-similar verdict",
    ),
    details_json: str = typer.Option(
        None,
        "--details-json",
        metavar="FILE",
        help="write per-statement structural details (displacement / moved) to FILE",
    ),
    json_out: bool = typer.Option(False, "--json", help="machine-readable JSON to stdout"),
    result_only: bool = typer.Option(
        False, "--result-only", help="bare structure_closeness scalar to stdout, no clutter"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="DEBUG logging to stderr"),
):
    """Structural (order) distance between two documents - the E11-H55 OPW order-gap."""
    configure_logging(verbose)

    backend_value, device = _resolve_gpu(gpu, backend)
    if details_json:
        from docdistance.pipeline import DocDistance

        result, details = _run(
            lambda: DocDistance(
                backend=backend_value, device=device
            ).structural_distance_with_details(a, b, anisotropy=anisotropy, threshold=threshold)
        )
        Path(details_json).write_text(json.dumps(details, indent=2))
        _err.print(
            f"[green]details written:[/green] {details_json} "
            f"(order_gap={details['order_gap']}, "
            f"structure_closeness={details['structure_closeness']})"
        )
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
            )
        )
    _emit_structural(result, json_out, result_only)


@app.command(
    name="distance-wrt-source",
    epilog="[bold]Examples[/bold]\n\n"
    "  docdistance distance-wrt-source summary_a.md summary_b.md --source article.md\n"
    "  docdistance distance-wrt-source a.md b.md -s s.md --json\n"
    "  docdistance distance-wrt-source a.md b.md -s s.md --source-map-json map.json   [dim]# statement → source map[/dim]\n"
    "  docdistance distance-wrt-source a.md b.md -s s.md --result-only   [dim]# D_sel,res_a,res_b[/dim]",
)
def distance_wrt_source(
    a: str = typer.Argument(..., help="first document - a file path or raw text"),
    b: str = typer.Argument(..., help="second document - a file path or raw text"),
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
    source_map_json: str = typer.Option(
        None,
        "--source-map-json",
        help="also write a statement → source alignment map (which source statements each A/B statement covers) to this JSON file",
        metavar="FILE",
    ),
    json_out: bool = typer.Option(False, "--json", help="machine-readable JSON to stdout"),
    result_only: bool = typer.Option(
        False, "--result-only", help="bare comma-separated D_sel,residual_a,residual_b to stdout"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="DEBUG logging to stderr"),
):
    """Source-conditioned distance d(A, B | S) - selection divergence plus each document's distance to S."""
    configure_logging(verbose)

    backend_value, device = _resolve_gpu(gpu, backend)
    if source_map_json:
        from docdistance.pipeline import DocDistance

        result, smap = _run(
            lambda: DocDistance(backend=backend_value, device=device).distance_wrt_source_with_map(
                a, b, source, anisotropy=anisotropy
            )
        )
        Path(source_map_json).write_text(json.dumps(smap, indent=2))
        _err.print(
            f"[green]source map written:[/green] {source_map_json} "
            f"(A {smap['n_statements']['a']} + B {smap['n_statements']['b']} statements → "
            f"top-{smap['top_k']} of {smap['n_statements']['source']} source)"
        )
    else:
        from docdistance.pipeline import source_conditioned_distance

        result = _run(
            lambda: source_conditioned_distance(
                a, b, source, backend=backend_value, anisotropy=anisotropy, device=device
            )
        )
    _emit_wrt_source(result, json_out, result_only)


@app.command(
    epilog="[bold]Examples[/bold]\n\n"
    "  docdistance init                                    [dim]# wmd mode, from HuggingFace[/dim]\n"
    "  docdistance init wmd-wrt-source                     [dim]# + reranker + NLI grounding models[/dim]\n"
    "  docdistance init wmd-wrt-source --source s3://general-purpose/docdistance --aws-profile stellars-tech\n"
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
    except (ModelsNotInstalled, FileNotFoundError, ValueError) as exc:
        _err.print(f"[bold red]error:[/bold red] {exc}")
        raise typer.Exit(1)
    srcs = ", ".join(f"{k}:{v}" for k, v in summary["sources"].items())
    _out.print(f"[green]initialized {summary['mode']}:[/green] {srcs}")
    if summary["config_file"]:
        _out.print(f"[dim]config written: {summary['config_file']}[/dim]")


if __name__ == "__main__":
    app()
