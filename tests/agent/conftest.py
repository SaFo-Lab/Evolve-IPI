"""Fixtures for the agent tests: a stand-in for the OpenClaw CLI, and profiles to run it against."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fake_cli import GATEWAY_PORT, POLICY_PORT, PROXY_PORT, FakeCli

from evolve_ipi.agent.openclaw import (
    GATEWAY_SPAN,
    Gateway,
    Profile,
    profile_for,
    write_profile,
)
from evolve_ipi.paths import RuntimePaths
from evolve_ipi.procs import port_in_use


@pytest.fixture
def make_fake_cli(tmp_path: Path) -> Iterator[Callable[[], FakeCli]]:
    """Makes stub CLIs, each with a bin directory and a state directory of its own.

    At teardown every gateway one of them started and that is still running is killed. A test that fails half way, or
    code under test that forgets to stop a process it started, would otherwise leave the stub running.
    """
    installed: list[FakeCli] = []

    def make() -> FakeCli:
        cli = FakeCli.install(tmp_path if not installed else tmp_path / f"cli-{len(installed)}")
        installed.append(cli)
        return cli

    try:
        yield make
    finally:
        for cli in installed:
            cli.reap()


@pytest.fixture
def fake_cli(make_fake_cli: Callable[[], FakeCli], monkeypatch: pytest.MonkeyPatch) -> FakeCli:
    """``openclaw`` and ``node`` are the only programs on PATH ahead of the system directories (``lsof``, which finds
    the owner of a port on macOS, lives in /usr/sbin)."""
    cli = make_fake_cli()
    monkeypatch.setenv("PATH", f"{cli.bin_dir}:/usr/bin:/bin:/usr/sbin:/sbin")
    return cli


@pytest.fixture
def make_gateway() -> Iterator[Callable[[Profile], Gateway]]:
    """Makes gateways and stops every one of them at teardown, started or not and whatever the test did."""
    made: list[Gateway] = []

    def make(p: Profile) -> Gateway:
        gateway = Gateway(p)
        made.append(gateway)
        return gateway

    try:
        yield make
    finally:
        failures: list[Exception] = []
        for gateway in reversed(made):
            try:
                gateway.stop()
            except Exception as error:  # noqa: BLE001 -- the others are still to be stopped
                failures.append(error)
        if failures:
            raise failures[0]


@pytest.fixture
def paths(tmp_path: Path) -> RuntimePaths:
    home = tmp_path / "home"
    home.mkdir()
    return RuntimePaths(
        root=tmp_path / "runtime",
        agent_home=home,
        outputs=tmp_path / "outputs",
        metaclaw_python=tmp_path / "venv" / "bin" / "python",
    )


@pytest.fixture
def make_profile(paths: RuntimePaths) -> Callable[..., Profile]:
    def make(
        line: str = "rl", kind: str = "file", gateway_port: int = GATEWAY_PORT, provider_port: int | None = None
    ) -> Profile:
        default = POLICY_PORT if line == "rl" else PROXY_PORT
        port = default if provider_port is None else provider_port
        return profile_for(line, kind, paths, gateway_port, port)  # type: ignore[arg-type]

    return make


@pytest.fixture
def profile(make_profile: Callable[..., Profile]) -> Profile:
    """The RL line's profile for the File-Create task, with its workspace directory in place."""
    p = make_profile()
    p.workspace.mkdir(parents=True)
    return p


@pytest.fixture
def free_span(free_port: Callable[[], int]) -> Callable[[], int]:
    """A function that returns the first of ``GATEWAY_SPAN`` free consecutive local ports: a gateway takes them all."""

    def find() -> int:
        for _ in range(500):
            start = free_port()
            ports = range(start, start + GATEWAY_SPAN)
            if ports[-1] <= 65535 and not any(port_in_use(port) for port in ports):
                return start
        raise RuntimeError(f"no {GATEWAY_SPAN} consecutive free ports found")

    return find


@pytest.fixture
def port_outside(free_port: Callable[[], int]) -> Callable[[int], int]:
    """A function that returns a free port that is not among the ports of a gateway starting at the given one."""

    def find(first: int) -> int:
        while True:
            port = free_port()
            if not first <= port < first + GATEWAY_SPAN:
                return port

    return find


@pytest.fixture
def gateway_profile(
    make_profile: Callable[..., Profile],
    free_span: Callable[[], int],
    port_outside: Callable[[int], int],
    fake_cli: FakeCli,
) -> Profile:
    """A written profile whose gateway has six free ports of its own, for the stub CLI to listen on."""
    first = free_span()
    p = make_profile(gateway_port=first, provider_port=port_outside(first))
    p.workspace.mkdir(parents=True)
    write_profile(p)
    return p
