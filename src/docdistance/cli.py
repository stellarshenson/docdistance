"""docdistance command-line interface.

Three subcommands - ``install`` (the only one that downloads models), ``distance`` (symmetric SMD)
and ``distance-wrt-source`` (source-conditioned). Human output is rich and coloured on a capable
terminal; ``--json`` emits machine-readable JSON and ``--result-only`` emits the bare result.
Logs go to stderr (loguru, ``--verbose`` for DEBUG), so stdout carries only the result.
"""

from __future__ import annotations

from enum import Enum
import json

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


class InstallBackend(str, Enum):
    openvino = "openvino"
    torch = "torch"
    both = "both"


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
    """Call ``fn`` and turn a missing-model error into a clean message + exit code 1."""
    from docdistance.encoders import ModelsNotInstalled

    try:
        return fn()
    except ModelsNotInstalled as exc:
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


def _emit_wrt_source(r, json_out: bool, result_only: bool) -> None:
    if result_only:
        typer.echo(f"{r.d_sel},{r.residual_a},{r.residual_b}")
        return
    if json_out:
        typer.echo(json.dumps(r.to_dict(), indent=2))
        return
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan")
    grid.add_column()
    grid.add_row("D_sel (selection divergence)", f"{r.d_sel:.4f}")
    grid.add_row("A → source", f"{r.residual_a:.4f}  (closeness {r.closeness_a * 100:.1f}%)")
    grid.add_row("B → source", f"{r.residual_b:.4f}  (closeness {r.closeness_b * 100:.1f}%)")
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
    _out.print(
        "[dim]residual = geometric distance to the source; the reranker + NLI grounding grade and "
        "numeric verifier are deferred to E02[/dim]"
    )


@app.command(
    epilog="[bold]Examples[/bold]\n\n"
    "  docdistance distance report_v1.md report_v2.md\n"
    '  docdistance distance "first text" "second text" --backend torch\n'
    "  docdistance distance a.md b.md --json\n"
    "  docdistance distance a.md b.md --result-only"
)
def distance(
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
    json_out: bool = typer.Option(False, "--json", help="machine-readable JSON to stdout"),
    result_only: bool = typer.Option(
        False, "--result-only", help="bare SMD scalar to stdout, no clutter"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="DEBUG logging to stderr"),
):
    """Symmetric distance between two documents - the exact Statement Mover's Distance."""
    configure_logging(verbose)
    from docdistance.pipeline import document_distance

    backend_value, device = _resolve_gpu(gpu, backend)
    result = _run(
        lambda: document_distance(
            a, b, backend=backend_value, anisotropy=anisotropy, threshold=threshold, device=device
        )
    )
    _emit_distance(result, json_out, result_only)


@app.command(
    name="distance-wrt-source",
    epilog="[bold]Examples[/bold]\n\n"
    "  docdistance distance-wrt-source summary_a.md summary_b.md --source article.md\n"
    "  docdistance distance-wrt-source a.md b.md -s s.md --json\n"
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
        False,
        "--anisotropy/--no-anisotropy",
        help="anisotropy removal - needs a corpus, off by default",
    ),
    json_out: bool = typer.Option(False, "--json", help="machine-readable JSON to stdout"),
    result_only: bool = typer.Option(
        False, "--result-only", help="bare comma-separated D_sel,residual_a,residual_b to stdout"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="DEBUG logging to stderr"),
):
    """Source-conditioned distance d(A, B | S) - selection divergence plus each document's distance to S."""
    configure_logging(verbose)
    from docdistance.pipeline import source_conditioned_distance

    backend_value, device = _resolve_gpu(gpu, backend)
    result = _run(
        lambda: source_conditioned_distance(
            a, b, source, backend=backend_value, anisotropy=anisotropy, device=device
        )
    )
    _emit_wrt_source(result, json_out, result_only)


@app.command(
    epilog="[bold]Examples[/bold]\n\n"
    "  docdistance install               [dim]# both backends[/dim]\n"
    "  docdistance install --backend openvino",
)
def install(
    backend: InstallBackend = typer.Option(
        InstallBackend.both, "--backend", help="which encoder weights to fetch"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="DEBUG logging to stderr"),
):
    """Download and cache the models - the only command that fetches from the Hub (TQDM progress bars)."""
    configure_logging(verbose)
    from docdistance.encoders import ModelsNotInstalled, download_models

    try:
        backends = download_models(backend.value)
    except ModelsNotInstalled as exc:
        _err.print(f"[bold red]error:[/bold red] {exc}")
        raise typer.Exit(1)
    _out.print(f"[green]models ready:[/green] {', '.join(backends)}")


if __name__ == "__main__":
    app()
