"""The gateway process: when it starts, when it refuses to, and when it is restarted.

The tests run the stub CLI (``fake_openclaw.py``), which listens on the gateway's port the way the real one does.
"""

from __future__ import annotations

import os
import signal
import socket
import time
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from fake_cli import FakeCli, wait_until

from evolve_ipi import procs
from evolve_ipi.agent import openclaw
from evolve_ipi.agent.openclaw import (
    GATEWAY_SPAN,
    OPENCLAW_VERSION,
    Gateway,
    OpenClawError,
    Profile,
    check_version,
    gateway_env,
    installed_version,
    write_profile,
)
from evolve_ipi.procs import ManagedProcess, PortInUse, ProcessError, port_in_use

SHELL_VARIABLES = {"PWD", "OLDPWD", "SHLVL", "_"}  # what a shell between the process and the stub adds


@pytest.fixture
def gateway(gateway_profile: Profile, make_gateway: Callable[[Profile], Gateway]) -> Gateway:
    return make_gateway(gateway_profile)


def gateway_calls(fake_cli: FakeCli) -> list[dict[str, Any]]:
    return [call for call in fake_cli.calls() if call["args"][0] == "gateway"]


def process_gone(pid: int, wait_s: float = 5) -> bool:
    deadline = time.monotonic() + wait_s
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def test_the_pinned_version_is_openclaw_2026_3_7() -> None:
    assert OPENCLAW_VERSION == "2026.3.7"
    assert GATEWAY_SPAN == 6


def test_the_names_other_modules_import_are_public() -> None:
    public = set(openclaw.__all__)
    assert {
        "PLUGIN_ID",
        "OPENCLAW_VERSION",
        "GATEWAY_SPAN",
        "installed_version",
        "check_version",
        "check_profile",
    } <= public
    assert all(hasattr(openclaw, name) for name in public)


# --- the installed version -------------------------------------------------------------------------


def test_installed_version_is_what_openclaw_prints(fake_cli: FakeCli) -> None:
    fake_cli.plan(version="2026.3.7")
    env = {"PATH": f"{fake_cli.bin_dir}:/usr/bin:/bin"}
    assert installed_version(env) == "2026.3.7"
    check_version(env)
    fake_cli.plan(version="2026.4.1")
    assert installed_version(env) == "2026.4.1"
    with pytest.raises(OpenClawError, match=r"'2026\.4\.1' is installed.*2026\.3\.7"):
        check_version(env)


def test_installed_version_is_looked_up_on_the_path_it_is_given(fake_cli: FakeCli, tmp_path: Path) -> None:
    with pytest.raises(OpenClawError, match="openclaw was not found on PATH"):
        installed_version({"PATH": str(tmp_path)})


def test_installed_version_asks_the_program_it_is_given(make_fake_cli: Callable[[], FakeCli]) -> None:
    first, second = make_fake_cli(), make_fake_cli()
    first.plan(version="2026.3.7")
    second.plan(version="2026.4.1")
    env = {"PATH": f"{first.bin_dir}:{second.bin_dir}"}
    assert installed_version(env) == "2026.3.7"  # the first one on PATH
    assert installed_version(env, second.bin_dir / "openclaw") == "2026.4.1"
    with pytest.raises(OpenClawError, match="2026.4.1"):
        check_version(env, second.bin_dir / "openclaw")


def test_installed_version_fails_loudly_when_openclaw_does(fake_cli: FakeCli, monkeypatch: pytest.MonkeyPatch) -> None:
    env = {"PATH": f"{fake_cli.bin_dir}:/usr/bin:/bin"}
    fake_cli.plan(version_fails="openclaw: Node.js v22.12+ is required")
    with pytest.raises(OpenClawError, match=r"exited with status 2: openclaw: Node\.js v22\.12\+ is required"):
        installed_version(env)
    monkeypatch.setattr(openclaw, "VERSION_TIMEOUT_S", 1)
    fake_cli.plan(version_sleep=30)
    with pytest.raises(OpenClawError, match=r"did not answer within 1 s"):
        installed_version(env)


@pytest.fixture
def two_openclaws(
    make_fake_cli: Callable[[], FakeCli],
    make_profile: Callable[..., Profile],
    free_span: Callable[[], int],
    port_outside: Callable[[int], int],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FakeCli, FakeCli, Profile]:
    """Two OpenClaws on PATH, ``first`` ahead of ``second``, and node only next to ``second``.

    A gateway runs the first ``openclaw`` of the PATH the run has, and gets a PATH of its own that starts with the
    directory of node, so a lookup of ``openclaw`` on the gateway's PATH finds ``second``'s instead.
    """
    first, second = make_fake_cli(), make_fake_cli()
    (first.bin_dir / "node").unlink()
    monkeypatch.setenv("PATH", f"{first.bin_dir}:{second.bin_dir}:/usr/bin:/bin:/usr/sbin:/sbin")
    span = free_span()
    p = make_profile(gateway_port=span, provider_port=port_outside(span))
    p.workspace.mkdir(parents=True)
    write_profile(p)
    assert gateway_env(p)["PATH"].startswith(f"{second.bin_dir}:{first.bin_dir}:")
    return first, second, p


def test_the_version_checked_is_the_version_that_is_launched_when_the_wrong_one_comes_first(
    two_openclaws: tuple[FakeCli, FakeCli, Profile], make_gateway: Callable[[Profile], Gateway]
) -> None:
    first, second, p = two_openclaws
    first.plan(version="2026.4.1")  # the one that would be launched
    second.plan(version=OPENCLAW_VERSION)  # the one on the gateway's PATH
    with pytest.raises(OpenClawError, match="2026.4.1"):
        make_gateway(p).start()
    assert [call["args"] for call in first.calls()] == [["--version"]]
    assert second.calls() == []


def test_the_gateway_is_launched_from_the_program_whose_version_was_checked(
    two_openclaws: tuple[FakeCli, FakeCli, Profile], make_gateway: Callable[[Profile], Gateway]
) -> None:
    first, second, p = two_openclaws
    first.plan(version=OPENCLAW_VERSION)
    second.plan(version="2026.4.1")  # not the one that is launched, so it does not matter
    gateway = make_gateway(p)
    gateway.start()
    assert gateway.alive()
    assert [call["args"][0] for call in first.calls()] == ["--version", "gateway"]
    assert second.calls() == []


# --- starting and stopping -------------------------------------------------------------------------


def test_start_waits_for_the_port_and_stop_frees_it(
    gateway: Gateway, gateway_profile: Profile, fake_cli: FakeCli
) -> None:
    port = gateway_profile.gateway_port
    assert not gateway.alive()
    gateway.start()
    assert gateway.alive() and gateway.pid is not None
    assert port_in_use(port)
    version_call, start_call = fake_cli.calls()
    assert version_call["args"] == ["--version"]
    assert start_call["args"] == ["gateway", "--port", str(port)]
    assert start_call["pid"] == gateway.pid
    assert Path(start_call["cwd"]).resolve() == gateway_profile.dir.resolve()
    env = {name: value for name, value in start_call["env"].items() if name not in SHELL_VARIABLES}
    assert env == gateway_env(gateway_profile)
    assert "listening on" in gateway.log_path.read_text()
    gateway.stop()
    assert not gateway.alive()
    assert process_gone(start_call["pid"])
    assert not any(port_in_use(p) for p in range(port, port + GATEWAY_SPAN))


def test_the_gateway_log_is_in_the_profile(gateway: Gateway, gateway_profile: Profile) -> None:
    assert gateway.log_path == gateway_profile.dir / "gateway.log"


def test_stop_does_nothing_before_start_and_after_stop(gateway: Gateway) -> None:
    gateway.stop()
    gateway.start()
    gateway.stop()
    gateway.stop()
    assert not gateway.alive()


def test_start_twice_is_refused(gateway: Gateway) -> None:
    gateway.start()
    pid = gateway.pid
    with pytest.raises(OpenClawError, match="already running"):
        gateway.start()
    assert gateway.alive() and gateway.pid == pid


def test_the_gateway_can_be_used_as_a_context_manager(
    gateway_profile: Profile, make_gateway: Callable[[Profile], Gateway]
) -> None:
    with make_gateway(gateway_profile) as running:
        assert running.alive()
        assert port_in_use(gateway_profile.gateway_port)
    assert not running.alive()
    assert not port_in_use(gateway_profile.gateway_port)


@pytest.mark.parametrize("offset", [0, 1, 3, GATEWAY_SPAN - 1])
def test_start_refuses_a_held_port_of_its_span_and_names_the_owner(
    offset: int,
    listener: int,
    make_profile: Callable[..., Profile],
    make_gateway: Callable[[Profile], Gateway],
    fake_cli: FakeCli,
) -> None:
    p = make_profile(gateway_port=listener - offset)
    p.workspace.mkdir(parents=True)
    write_profile(p)
    gateway = make_gateway(p)
    with pytest.raises(PortInUse) as raised:
        gateway.start()
    assert f"port {listener} is in use by pid {os.getpid()}" in str(raised.value)
    assert not gateway.alive()
    assert gateway_calls(fake_cli) == []
    assert not p.dir.joinpath("gateway.log").exists()


def test_start_refuses_another_openclaw_version_and_names_the_installed_one(
    gateway: Gateway, fake_cli: FakeCli
) -> None:
    fake_cli.plan(version="2026.4.1")
    with pytest.raises(OpenClawError) as raised:
        gateway.start()
    assert "2026.4.1" in str(raised.value) and OPENCLAW_VERSION in str(raised.value)
    assert not gateway.alive()
    assert gateway_calls(fake_cli) == []


def test_start_refuses_a_profile_that_is_not_what_the_run_builds(
    gateway: Gateway, gateway_profile: Profile, fake_cli: FakeCli
) -> None:
    gateway_profile.config_path.write_text("{}\n")
    with pytest.raises(OpenClawError, match=r"openclaw\.json.*differs"):
        gateway.start()
    assert fake_cli.calls() == []


def test_start_does_not_attach_to_a_server_that_is_not_its_child(
    fake_cli: FakeCli,
    listener: int,
    make_profile: Callable[..., Profile],
    make_gateway: Callable[[Profile], Gateway],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whatever listens on the port when the gateway should be up has to be the gateway: a stranger is not used."""
    p = make_profile(gateway_port=listener)  # the stranger holds the gateway's own port
    p.workspace.mkdir(parents=True)
    write_profile(p)
    fake_cli.plan(gateway={"bind": False})  # the gateway starts and runs, but never takes the port
    monkeypatch.setattr(openclaw, "require_free", lambda ports: None)  # as if the stranger came in after the check
    wait_for_port = ManagedProcess.wait_for_port

    def after_the_stub_has_noted_its_start(self: ManagedProcess, port: int, timeout_s: float) -> None:
        # The stranger is found on the first look, which can come before the stub has run its first line; a gateway
        # that is stopped by then leaves no record, and the test would read none.
        assert wait_until(lambda: len(gateway_calls(fake_cli)) == 1, 20), "the stub gateway never noted its start"
        wait_for_port(self, port, timeout_s)

    monkeypatch.setattr(ManagedProcess, "wait_for_port", after_the_stub_has_noted_its_start)
    gateway = make_gateway(p)
    with pytest.raises(ProcessError, match=rf"port {listener} is held by pid {os.getpid()}"):
        gateway.start()
    assert not gateway.alive()
    (start_call,) = gateway_calls(fake_cli)
    assert process_gone(start_call["pid"])


def test_start_reports_a_gateway_that_exits_before_it_listens(gateway: Gateway, fake_cli: FakeCli) -> None:
    fake_cli.plan(gateway={"die": "Config invalid: agents.list"})
    with pytest.raises(ProcessError, match=r"(?s)exited with status 3 before listening.*Config invalid: agents\.list"):
        gateway.start()
    assert not gateway.alive()


def test_start_refuses_a_gateway_that_did_not_load_the_plugin(
    gateway: Gateway, fake_cli: FakeCli, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without rl-training-headers no request carries its session, and nothing could be trained or evolved."""
    monkeypatch.setattr(openclaw, "GATEWAY_START_TIMEOUT_S", 3)
    fake_cli.plan(gateway={"plugin": False})
    with pytest.raises(ProcessError, match=r"did not log 'rl-training-headers: activated' within [0-9.]+ s"):
        gateway.start()
    assert not gateway.alive()
    (start_call,) = gateway_calls(fake_cli)
    assert process_gone(start_call["pid"])


def test_a_start_that_cannot_stop_what_it_started_still_knows_of_it(
    gateway: Gateway, fake_cli: FakeCli, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the clean-up of a failed start itself fails, the gateway still holds the process, so that a later stop()
    reaches it and the factory of the tests can stop it."""
    real_stop = ManagedProcess.stop
    attempts: list[int] = []

    def stop_once_refused(self: ManagedProcess, grace_s: float = 40) -> None:
        attempts.append(1)
        if len(attempts) == 1:
            raise ProcessError("cannot stop the process")
        real_stop(self, grace_s)

    monkeypatch.setattr(ManagedProcess, "stop", stop_once_refused)
    monkeypatch.setattr(openclaw, "GATEWAY_START_TIMEOUT_S", 1)
    fake_cli.plan(gateway={"bind": False})  # it never listens, whatever the speed of the machine
    with pytest.raises(ProcessError, match="cannot stop the process"):
        gateway.start()
    assert gateway.alive()  # the failed clean-up left it running, and the gateway knows
    gateway.stop()
    assert not gateway.alive()
    (start_call,) = gateway_calls(fake_cli)
    assert process_gone(start_call["pid"])


def test_the_plugin_wait_is_part_of_the_start_budget(
    gateway: Gateway, fake_cli: FakeCli, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole start gets ``GATEWAY_START_TIMEOUT_S``. A gateway that listens late and does not log the plugin's
    line is given up on when that budget is spent, not a fixed time after it."""
    monkeypatch.setattr(openclaw, "GATEWAY_START_TIMEOUT_S", 4)
    fake_cli.plan(gateway={"plugin": False, "bind_after": 1.0})
    started = time.monotonic()
    with pytest.raises(ProcessError, match="did not log 'rl-training-headers: activated'"):
        gateway.start()
    assert 3.9 <= time.monotonic() - started < 7.5  # with a fixed ten seconds after the port it would be eleven
    assert not gateway.alive()


def test_start_gives_up_when_the_port_never_comes_up(
    gateway: Gateway, fake_cli: FakeCli, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert openclaw.GATEWAY_START_TIMEOUT_S == 120
    monkeypatch.setattr(openclaw, "GATEWAY_START_TIMEOUT_S", 1)
    fake_cli.plan(gateway={"bind": False})
    with pytest.raises(ProcessError, match=r"did not listen on port \d+ within 1 s"):
        gateway.start()
    assert not gateway.alive()
    (start_call,) = gateway_calls(fake_cli)
    assert process_gone(start_call["pid"])


# --- restarting between episodes -------------------------------------------------------------------


def test_restart_if_needed_leaves_a_healthy_young_gateway_alone(gateway: Gateway) -> None:
    gateway.start()
    pid = gateway.pid
    assert gateway.restart_if_needed() is None
    assert gateway.pid == pid and gateway.alive()


def test_restart_if_needed_by_age(gateway: Gateway, gateway_profile: Profile, fake_cli: FakeCli) -> None:
    assert openclaw.GATEWAY_MAX_AGE_S == 3 * 3600
    gateway.start()
    old = gateway.pid
    assert gateway.restart_if_needed(max_age_s=3600) is None
    reason = gateway.restart_if_needed(max_age_s=0)
    assert reason is not None and "age" in reason
    assert gateway.alive() and gateway.pid != old
    assert port_in_use(gateway_profile.gateway_port)
    assert old is not None and process_gone(old)
    assert [c["args"][0] for c in fake_cli.calls()] == ["--version", "gateway", "--version", "gateway"]


@pytest.mark.parametrize(
    "line", ["FATAL ERROR: Reached heap limit Allocation failed", "heap out of memory", "Reached heap limit"]
)
def test_restart_if_needed_after_an_out_of_memory_line(line: str, gateway: Gateway) -> None:
    gateway.start()
    old = gateway.pid
    with gateway.log_path.open("a") as log:
        log.write(f"<--- Last few GCs --->\n{line}\n")
    reason = gateway.restart_if_needed()
    assert reason is not None and "log" in reason and line in reason
    assert gateway.alive() and gateway.pid != old
    # The new process starts from a log of its own: the old line does not restart it again.
    assert gateway.restart_if_needed() is None


def test_restart_if_needed_when_the_gateway_has_died(gateway: Gateway) -> None:
    gateway.start()
    old = gateway.pid
    assert old is not None
    os.killpg(old, signal.SIGKILL)
    deadline = time.monotonic() + 10
    while gateway.alive():  # alive() reaps the process once it is gone
        assert time.monotonic() < deadline, "the killed gateway is still running"
        time.sleep(0.05)
    reason = gateway.restart_if_needed()
    assert reason is not None and "exited" in reason
    assert gateway.alive() and gateway.pid != old


def test_restart_if_needed_before_any_start_is_an_error(gateway: Gateway) -> None:
    with pytest.raises(OpenClawError, match="not been started"):
        gateway.restart_if_needed()


def test_a_restart_that_is_refused_leaves_nothing_running(gateway: Gateway, fake_cli: FakeCli) -> None:
    """The version is checked again at every start; a restart onto another version stops with the old one down."""
    gateway.start()
    fake_cli.plan(version="2026.4.1")
    with pytest.raises(OpenClawError, match="2026.4.1"):
        gateway.restart_if_needed(max_age_s=0)
    assert not gateway.alive()


def let_go_of_the_port(fake_cli: FakeCli, port: int) -> None:
    """Make the stub gateway close its socket and go on running, and wait until the port is free."""
    (fake_cli.state_dir / "release-port").touch()
    assert wait_until(lambda: not port_in_use(port), 10), f"the stub kept port {port}"


def test_a_gateway_that_lost_its_port_to_another_process_is_not_passed_as_healthy(
    gateway: Gateway, gateway_profile: Profile, fake_cli: FakeCli, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gateway that is left alone is still checked to hold its port. When it does not, the caller is told, and the
    gateway is neither restarted nor stopped."""
    monkeypatch.setattr(openclaw, "PORT_OWNER_CHECK_S", 0.5)
    gateway.start()
    pid, port = gateway.pid, gateway_profile.gateway_port
    let_go_of_the_port(fake_cli, port)
    with closing(socket.socket()) as stranger:
        stranger.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        stranger.bind(("127.0.0.1", port))
        stranger.listen()
        with pytest.raises(OpenClawError, match=rf"port {port} is not held by the gateway.*held by pid {os.getpid()}"):
            gateway.restart_if_needed()
    assert gateway.alive() and gateway.pid == pid


def test_a_gateway_whose_owner_is_slow_to_show_is_still_found_healthy(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``lsof`` or ``/proc`` can fail to show the owner of a port for a moment. That is waited out, not reported as a
    gateway that lost its port."""
    gateway.start()
    real_owners = procs.port_owners
    asked: list[int] = []

    def owners_that_show_late(port: int, proc: Path | None = None) -> list[procs.PortOwner]:
        asked.append(port)
        return [] if len(asked) == 1 else real_owners(port)

    monkeypatch.setattr(procs, "port_owners", owners_that_show_late)
    assert gateway.restart_if_needed() is None
    assert len(asked) >= 2


def test_a_gateway_that_no_longer_listens_is_not_passed_as_healthy(
    gateway: Gateway, gateway_profile: Profile, fake_cli: FakeCli, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(openclaw, "PORT_OWNER_CHECK_S", 0.5)
    gateway.start()
    pid, port = gateway.pid, gateway_profile.gateway_port
    let_go_of_the_port(fake_cli, port)
    with pytest.raises(OpenClawError, match=rf"port {port} is not held by the gateway.*did not listen on port {port}"):
        gateway.restart_if_needed()
    assert gateway.alive() and gateway.pid == pid
