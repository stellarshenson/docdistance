"""Module entry point so ``python -m docdistance`` drives the CLI (same Typer app as the console script)."""

from docdistance.cli import app

if __name__ == "__main__":
    app()
