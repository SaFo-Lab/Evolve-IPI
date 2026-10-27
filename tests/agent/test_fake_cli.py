"""The stub CLI and its safety net: a stub gateway does not outlive the test that started it.

A stub that sleeps forever and a fixture with no teardown left a gateway running when a test failed half way, or when
the code under test forgot to stop a process it had started.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import textwrap
from collections.abc import Callable, Iterator

import fake_openclaw
import pytest
from fake_cli import FakeCli, running, wait_until

from evolve_ipi.agent import openclaw
from evolve_ipi.agent.openclaw import Gateway, Profile
from evolve_ipi.procs import ManagedProcess, ProcessError


def start_stub_gateway(fake_cli: FakeCli, port: int, stderr: int = subprocess.DEVNULL) -> subprocess.Popen[bytes]:
    """The stub gateway, started by this test the way ``ManagedProcess`` starts one: in a session of its own."""
    process = subprocess.Popen(
        [str(fake_cli.bin_dir / "openclaw"), "gateway", "--port", str(port)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=stderr,
        start_new_session=True,
    )
    assert wait_until(lambda: fake_cli.gateway_pids() == [process.pid], 20), "the stub gateway did not start"
    return process


def stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.kill()
    process.wait()


def test_the_stub_gives_up_by_itself_after_ten_minutes_whatever_happens() -> None:
    assert fake_openclaw.HARD_CAP_S == 10 * 60


def test_the_stub_gateway_stops_at_its_hard_cap(fake_cli: FakeCli, free_port: Callable[[], int]) -> None:
    fake_cli.plan(max_s=1)
    process = start_stub_gateway(fake_cli, free_port(), stderr=subprocess.PIPE)
    try:
        _, stderr = process.communicate(timeout=20)
    finally:
        stop(process)
    assert process.returncode != 0
    assert b"still running after 1 s" in stderr


def test_the_stub_gateway_stops_when_the_process_that_started_it_is_gone(
    fake_cli: FakeCli, free_port: Callable[[], int]
) -> None:
    """The stub is started by a process that exits as soon as the stub has begun: it is left an orphan."""
    starter = textwrap.dedent(
        """
        import pathlib, subprocess, sys, time
        state, wrapper, port = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
        subprocess.Popen([wrapper, "gateway", "--port", port], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
        deadline = time.monotonic() + 20
        while not (state / "calls.jsonl").exists() and time.monotonic() < deadline:  # the stub has noted its parent
            time.sleep(0.02)
        """
    )
    arguments = [str(fake_cli.state_dir), str(fake_cli.bin_dir / "openclaw"), str(free_port())]
    subprocess.run([sys.executable, "-c", starter, *arguments], check=True, timeout=60, capture_output=True)
    assert wait_until(lambda: len(fake_cli.gateway_pids()) == 1, 20)
    (pid,) = fake_cli.gateway_pids()
    assert wait_until(lambda: not running(pid), 15), f"the orphaned stub gateway {pid} is still running"


def test_reap_kills_a_gateway_that_is_still_running(fake_cli: FakeCli, free_port: Callable[[], int]) -> None:
    process = start_stub_gateway(fake_cli, free_port())
    try:
        assert running(process.pid)
        assert fake_cli.reap() == [process.pid]
        process.wait(timeout=10)
        assert process.returncode == -signal.SIGKILL
        assert fake_cli.reap() == []  # nothing is left to kill
    finally:
        stop(process)


def test_reap_only_signals_what_is_still_the_stub(fake_cli: FakeCli) -> None:
    """A gateway that ended long ago may have given its pid to another program; the reaper must not touch that."""
    bystander = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True, stdin=subprocess.DEVNULL
    )
    try:
        entry = {"args": ["gateway", "--port", "1"], "cwd": "", "env": {}, "pid": bystander.pid}
        (fake_cli.state_dir / "calls.jsonl").write_text(json.dumps(entry) + "\n")
        assert fake_cli.gateway_pids() == [bystander.pid]
        assert fake_cli.reap() == []
        assert bystander.poll() is None
    finally:
        stop(bystander)


def test_reap_goes_past_a_log_line_that_was_cut_off(fake_cli: FakeCli) -> None:
    (fake_cli.state_dir / "calls.jsonl").write_text('{"args": ["gateway", "--po')
    assert fake_cli.gateway_pids() == []
    assert fake_cli.reap() == []


@pytest.fixture
def leak_check() -> Iterator[list[Callable[[], bool]]]:
    """Each function put in it says whether something is still running, and at its teardown nothing may be.

    Fixtures are torn down in the reverse of the order a test names them. Named before ``fake_cli``, it is torn down
    after the reaper of ``fake_cli`` has killed what is left over. Named after ``fake_cli`` and before ``make_gateway``,
    it is torn down after the factory has stopped its gateways and before the reaper runs, so that only the factory
    can have stopped them."""
    still_running: list[Callable[[], bool]] = []
    yield still_running
    wait_until(lambda: not any(check() for check in still_running), 10)
    left = [check for check in still_running if check()]
    assert not left, f"{len(left)} process(es) still running when leak_check was torn down"


def test_a_gateway_whose_start_leaks_is_reaped_at_teardown(
    leak_check: list[Callable[[], bool]],
    fake_cli: FakeCli,
    gateway_profile: Profile,
    make_gateway: Callable[[Profile], Gateway],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a start fails it stops the process it started. Here that does nothing, as if a change had broken it, so the
    process is left running and nothing in the code under test knows of it any more: the teardown of ``fake_cli`` has
    to kill it."""
    # alive() polls the process, so that the one the teardown killed is also waited for
    monkeypatch.setattr(ManagedProcess, "stop", lambda self, grace_s=40: leak_check.append(self.alive))
    monkeypatch.setattr(openclaw, "GATEWAY_START_TIMEOUT_S", 1)
    fake_cli.plan(gateway={"bind": False})  # it never listens, whatever the speed of the machine
    with pytest.raises(ProcessError, match="did not listen"):
        make_gateway(gateway_profile).start()
    (leaked,) = leak_check
    assert leaked(), "the start did not leak, so this test shows nothing"
    assert wait_until(lambda: len(fake_cli.gateway_pids()) == 1, 20), "the stub never noted its start"


def test_a_gateway_that_its_test_does_not_stop_is_stopped_by_the_factory(
    fake_cli: FakeCli,
    leak_check: list[Callable[[], bool]],
    gateway_profile: Profile,
    make_gateway: Callable[[Profile], Gateway],
) -> None:
    """``leak_check`` comes after ``fake_cli`` and before ``make_gateway``: at its teardown the factory has stopped the
    gateway and the reaper has not run yet."""
    gateway = make_gateway(gateway_profile)
    gateway.start()
    pid = gateway.pid
    assert pid is not None and running(pid)
    leak_check.append(lambda: running(pid))
    # no stop() here: make_gateway stops it when the test ends, and leak_check proves that it did
