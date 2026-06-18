"""CLI smoke tests - help, version and command structure render without loading any model.

The real end-to-end behaviour (segment -> embed -> distance) is exercised in
``notebooks/09-kj-docdistance-api-e2e.ipynb`` and by running the CLI against the fixtures.
"""

from typer.testing import CliRunner

from docdistance.cli import app

runner = CliRunner()
# widen so option / command names are not wrapped, and force plain output (no ANSI) so the
# substring assertions hold regardless of the ambient terminal - CI runners set FORCE_COLOR,
# which otherwise makes Rich colorize the help and splits flags like "--json" with escape codes
_WIDE = {"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"}


def test_app_help_lists_subcommands():
    res = runner.invoke(app, ["--help"], env=_WIDE)
    assert res.exit_code == 0
    assert "distance" in res.output
    assert "distance-wrt-source" in res.output
    assert "install" in res.output


def test_distance_help_has_flags_and_examples():
    res = runner.invoke(app, ["distance", "--help"], env=_WIDE)
    assert res.exit_code == 0
    assert "--json" in res.output
    assert "--result-only" in res.output
    assert "--backend" in res.output
    assert "--gpu" in res.output
    assert "Examples" in res.output


def test_gpu_flag_errors_when_not_secured(monkeypatch):
    """--gpu must fail loudly (exit 1) when no CUDA device is visible, never silently fall back to CPU."""
    import docdistance.encoders as enc

    monkeypatch.setattr(enc, "require_gpu", _raise_gpu_unavailable)
    res = runner.invoke(app, ["distance", "x", "y", "--gpu"], env=_WIDE)
    assert res.exit_code == 1


def _raise_gpu_unavailable():
    from docdistance.encoders import GpuNotAvailable

    raise GpuNotAvailable("no CUDA device")


def test_wrt_source_help_has_source_option():
    res = runner.invoke(app, ["distance-wrt-source", "--help"], env=_WIDE)
    assert res.exit_code == 0
    assert "--source" in res.output
    assert "--result-only" in res.output


def test_install_help_has_backend():
    res = runner.invoke(app, ["install", "--help"], env=_WIDE)
    assert res.exit_code == 0
    assert "--backend" in res.output


def test_version():
    res = runner.invoke(app, ["--version"], env=_WIDE)
    assert res.exit_code == 0
    assert "docdistance" in res.output
