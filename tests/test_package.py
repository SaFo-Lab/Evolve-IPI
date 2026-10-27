"""The installed package: console script, help text and registered pytest markers."""

from __future__ import annotations

from importlib.metadata import entry_points

import pytest

from evolve_ipi import cli


def test_console_script_points_at_cli_main() -> None:
    (script,) = entry_points(group="console_scripts", name="evolve-ipi")
    assert script.value == "evolve_ipi.cli:main"
    assert script.load() is cli.main


def test_cli_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert out.startswith("usage: evolve-ipi")
    assert "evolve-ipi: self-evolution and indirect prompt injection" in out


def test_cli_rejects_unknown_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["frobnicate"])
    assert exit_info.value.code == 2
    assert "frobnicate" in capsys.readouterr().err


def test_integration_markers_are_registered(pytestconfig: pytest.Config) -> None:
    registered = {entry.split(":")[0].strip() for entry in pytestconfig.getini("markers")}
    assert {"openclaw", "metaclaw", "docker"} <= registered
