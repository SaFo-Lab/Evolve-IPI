"""Child processes and local ports: start, wait, stop, and refusing a port someone else holds."""

from __future__ import annotations

import os
import signal
import socket
import sys
import textwrap
import time
from collections.abc import Callable
from pathlib import Path
from typing import Self

import pytest

from evolve_ipi import procs
from evolve_ipi.procs import ManagedProcess, PortInUse, PortOwner, ProcessError

PYTHON = sys.executable


def python_child(tmp_path: Path, code: str, name: str = "child", **env: str) -> ManagedProcess:
    script = tmp_path / f"{name}.py"
    script.write_text(textwrap.dedent(code))
    return ManagedProcess(
        name, [PYTHON, "-u", str(script)], {"PATH": os.environ["PATH"], **env}, tmp_path / f"{name}.log"
    )


def group_alive(pgid: int, wait_s: float = 5) -> bool:
    """Whether the process group still exists ``wait_s`` from now. A killed process can stay a zombie for a moment,
    until it is reaped, and a zombie still answers a signal."""
    deadline = time.monotonic() + wait_s
    while True:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        if time.monotonic() >= deadline:
            return True
        time.sleep(0.05)


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


# --- ports -----------------------------------------------------------------------------------------


def test_a_free_port_is_not_in_use(free_port: Callable[[], int]) -> None:
    assert procs.port_in_use(free_port()) is False


def test_a_listening_port_is_in_use(listener: int) -> None:
    assert procs.port_in_use(listener) is True


def test_port_owners_finds_this_process(listener: int) -> None:
    owners = procs.port_owners(listener)
    assert [owner.pid for owner in owners] == [os.getpid()]
    assert owners[0].command


def test_require_free_names_the_owner_and_kills_nothing(listener: int, free_port: Callable[[], int]) -> None:
    with pytest.raises(PortInUse) as raised:
        procs.require_free([free_port(), listener])
    message = str(raised.value)
    assert f"port {listener}" in message
    assert f"pid {os.getpid()}" in message
    with socket.create_connection(("127.0.0.1", listener), timeout=5):
        pass  # still listening: nothing was stopped


def test_require_free_raises_when_the_owner_is_unknown(listener: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(procs, "port_owners", lambda port, proc=None: [])
    with pytest.raises(PortInUse, match=rf"port {listener} is in use by a process that cannot be identified"):
        procs.require_free([listener])


def test_require_free_passes_free_ports(free_port: Callable[[], int]) -> None:
    procs.require_free([free_port(), free_port()])


@pytest.fixture
def fake_proc(tmp_path: Path) -> Path:
    """A /proc tree: port 30077 (0x757D) listened on by pids 321 and 654, port 20000 (0x4E20) by pid 987."""
    proc = tmp_path / "proc"
    (proc / "net").mkdir(parents=True)
    header = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
    (proc / "net" / "tcp").write_text(
        header
        # 127.0.0.1:30077 LISTEN (0A), inode 4711; the same port in another state; another port listening
        + "   0: 0100007F:757D 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 4711 1\n"
        + "   1: 0100007F:757D 0100007F:9C40 06 00000000:00000000 00:00000000 00000000  1000        0 0 1\n"
        + "   2: 00000000:4E20 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 5000 1\n"
    )
    (proc / "net" / "tcp6").write_text(
        header + "   0: 00000000000000000000000000000000:757D 00000000000000000000000000000000:0000 0A "
        "00000000:00000000 00:00000000 00000000  1000        0 4712 1\n"
    )
    for pid, inode, cmdline in (
        (321, 4711, b"python\0-m\0server\0"),
        (654, 4712, b"node\0gateway\0"),
        (987, 5000, b"x\0"),
    ):
        fd = proc / str(pid) / "fd"
        fd.mkdir(parents=True)
        os.symlink("/dev/null", fd / "0")
        os.symlink(f"socket:[{inode}]", fd / "7")
        (proc / str(pid) / "cmdline").write_bytes(cmdline)
    return proc


def test_port_owners_from_proc(fake_proc: Path, tmp_path: Path) -> None:
    """The Linux path: /proc/net/tcp{,6} for the listening socket, /proc/<pid>/fd for its owner."""
    proc = fake_proc
    assert procs._proc_owners(0x757D, proc) == [
        PortOwner(321, "python -m server"),
        PortOwner(654, "node gateway"),
    ]
    assert procs._proc_owners(1234, proc) == []
    assert procs._proc_listening(0x757D, proc) is True
    assert procs._proc_listening(0x4E21, proc) is False
    assert procs._proc_owners(0x757D, tmp_path / "no-proc") is None


def test_port_owners_reads_proc_when_lsof_is_absent(fake_proc: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(procs, "_lsof_owners", lambda port: None)  # no lsof on this system
    assert procs.port_owners(0x757D, fake_proc) == [PortOwner(321, "python -m server"), PortOwner(654, "node gateway")]
    assert procs.port_owners(0x4E20, fake_proc) == [PortOwner(987, "x")]
    assert procs.port_owners(0x1234, fake_proc) == []
    assert procs.port_in_use(0x757D, fake_proc) is True


def test_port_owners_reads_proc_when_lsof_sees_nothing(fake_proc: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(procs, "_lsof_owners", lambda port: [])  # lsof cannot see another user's processes
    assert procs.port_owners(0x4E20, fake_proc) == [PortOwner(987, "x")]


def test_port_owners_takes_lsof_first(fake_proc: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(procs, "_lsof_owners", lambda port: [PortOwner(42, "seen by lsof")])
    assert procs.port_owners(0x757D, fake_proc) == [PortOwner(42, "seen by lsof")]


def test_port_owners_without_lsof_or_proc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(procs, "_lsof_owners", lambda port: None)
    assert procs.port_owners(0x757D, tmp_path / "no-proc") == []


def test_group_alive_from_proc_ignores_zombies(tmp_path: Path) -> None:
    """The Linux path: a group whose only members are zombies (never reaped by a container's first process, say)
    counts as gone."""
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self" / "stat").write_text("1 (init) S 0 1 1 0 -1 0\n")
    for pid, comm, state, pgrp in (
        (500, "python3", "Z", 500),
        (501, "sleep (child) x", "Z", 500),
        (600, "node", "S", 600),
        (601, "bash", "R", 600),
        (700, "other", "S", 701),
    ):
        (proc / str(pid)).mkdir()
        (proc / str(pid) / "stat").write_text(f"{pid} ({comm}) {state} 1 {pgrp} {pgrp} 0 -1 4194304\n")
    assert procs._group_alive(500, proc) is False
    assert procs._group_alive(600, proc) is True
    assert procs._group_alive(701, proc) is True
    assert procs._group_alive(999, proc) is False


# --- managed processes -----------------------------------------------------------------------------


def test_start_wait_for_line_and_stop(tmp_path: Path) -> None:
    child = python_child(
        tmp_path,
        """
        import time
        time.sleep(0.3)
        print("server READY on 1", flush=True)
        time.sleep(600)
        """,
    )
    child.start()
    try:
        child.wait_for_line("READY on 1", timeout_s=20)
        assert child.alive()
        pgid = child.pid
        assert os.getsid(pgid) == pgid  # a session of its own
    finally:
        child.stop(grace_s=5)
    assert not child.alive()
    assert not group_alive(pgid)


def test_wait_for_line_raises_with_the_log_tail_on_early_exit(tmp_path: Path) -> None:
    child = python_child(
        tmp_path,
        """
        import sys
        print("loading the tokenizer", flush=True)
        print("OSError: gated repo", file=sys.stderr, flush=True)
        sys.exit(3)
        """,
    )
    child.start()
    with pytest.raises(ProcessError) as raised:
        child.wait_for_line("READY", timeout_s=20)
    message = str(raised.value)
    assert "exited with status 3" in message
    assert "OSError: gated repo" in message
    assert "loading the tokenizer" in message
    child.stop()


def test_wait_for_line_times_out(tmp_path: Path) -> None:
    child = python_child(tmp_path, "import time\nprint('starting', flush=True)\ntime.sleep(600)\n")
    child.start()
    try:
        started = time.monotonic()
        with pytest.raises(ProcessError, match=r"did not log 'READY' within 0.5 s") as raised:
            child.wait_for_line("READY", timeout_s=0.5)
        assert time.monotonic() - started < 10
        assert "starting" in str(raised.value)
    finally:
        child.stop(grace_s=5)


def test_wait_for_line_ignores_what_the_log_held_before_the_start(tmp_path: Path) -> None:
    child = python_child(tmp_path, "import time\ntime.sleep(600)\n")
    child.log_path.write_text("READY from an earlier run\n")
    child.start()
    try:
        with pytest.raises(ProcessError, match="did not log"):
            child.wait_for_line("READY", timeout_s=0.5)
    finally:
        child.stop(grace_s=5)
    assert child.log_path.read_text().startswith("READY from an earlier run\n")  # appended to, not truncated


def test_wait_for_line_keeps_its_deadline_while_the_log_keeps_growing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every read finds new output: the deadline is still checked after each one."""
    child = python_child(tmp_path, "import time\ntime.sleep(600)\n")
    child.start()
    reads: list[int] = []

    class Flood:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def seek(self, offset: int) -> None:
            return None

        def read(self) -> bytes:
            reads.append(1)
            return b"x" * 4096 if len(reads) < 10_000 else b""

    monkeypatch.setattr(procs, "open", lambda *args, **kwargs: Flood(), raising=False)
    try:
        with pytest.raises(ProcessError, match="did not log 'READY' within 0 s"):
            child.wait_for_line("READY", timeout_s=0)
    finally:
        monkeypatch.undo()
        child.stop(grace_s=5)
    assert len(reads) <= 2  # one read in the loop, one for the log tail of the error


def test_wait_for_line_times_out_on_a_child_that_floods_its_log(tmp_path: Path) -> None:
    child = python_child(
        tmp_path,
        """
        import os, time
        line = b"x" * 200 + b"\\n"
        end = time.monotonic() + 30
        while time.monotonic() < end:
            os.write(1, line)
            time.sleep(0.0005)
        """,
    )
    child.start()
    try:
        started = time.monotonic()
        with pytest.raises(ProcessError, match="did not log 'READY' within 0.5 s"):
            child.wait_for_line("READY", timeout_s=0.5)
        assert time.monotonic() - started < 5
    finally:
        child.stop(grace_s=5)


def test_wait_for_line_finds_a_line_split_across_reads(tmp_path: Path) -> None:
    child = python_child(
        tmp_path,
        """
        import sys, time
        sys.stdout.write("EVAL PROXY RE"); sys.stdout.flush()
        time.sleep(0.5)
        sys.stdout.write("ADY on 127.0.0.1:1\\n"); sys.stdout.flush()
        time.sleep(600)
        """,
    )
    child.start()
    try:
        child.wait_for_line("EVAL PROXY READY on 127.0.0.1:1", timeout_s=20)
    finally:
        child.stop(grace_s=5)


def test_stdin_is_dev_null_and_the_environment_is_exactly_the_given_one(tmp_path: Path) -> None:
    child = python_child(
        tmp_path,
        """
        import os, sys
        here, null = os.fstat(0), os.stat(os.devnull)
        print("stdin_is_devnull=%r" % ((here.st_dev, here.st_ino) == (null.st_dev, null.st_ino)), flush=True)
        print("stdin=%r" % sys.stdin.read(), flush=True)
        print("parent_only=%r" % os.environ.get("EVOLVE_IPI_TEST_PARENT_ONLY"), flush=True)
        print("given=%r" % os.environ.get("GIVEN"), flush=True)
        print("DONE", flush=True)
        """,
        GIVEN="yes",
    )
    # While the child starts, this process's own stdin is a pipe, so that a child inheriting it would show.
    read_end, write_end = os.pipe()
    saved_stdin = os.dup(0)
    os.environ["EVOLVE_IPI_TEST_PARENT_ONLY"] = "secret"
    try:
        os.dup2(read_end, 0)
        child.start()
    finally:
        os.dup2(saved_stdin, 0)
        for descriptor in (saved_stdin, read_end, write_end):
            os.close(descriptor)
        del os.environ["EVOLVE_IPI_TEST_PARENT_ONLY"]
    child.wait_for_line("DONE", timeout_s=20)
    log = child.log_path.read_text()
    assert "stdin_is_devnull=True" in log
    assert "stdin=''" in log
    assert "parent_only=None" in log
    assert "given='yes'" in log
    child.stop()


def test_stop_kills_a_child_that_ignores_sigterm(tmp_path: Path) -> None:
    child = python_child(
        tmp_path,
        """
        import signal, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print("ARMED", flush=True)
        time.sleep(600)
        """,
    )
    child.start()
    child.wait_for_line("ARMED", timeout_s=20)
    started = time.monotonic()
    child.stop(grace_s=0.5)
    assert time.monotonic() - started < 15
    assert not child.alive()
    assert child.returncode == -signal.SIGKILL


def test_stop_takes_a_group_of_unreaped_zombies_for_gone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """macOS answers EPERM to a signal for a group whose only members are zombies nobody has reaped yet. With the
    leader gone and no running member left, stop() has nothing to do."""
    child = python_child(tmp_path, "print('bye', flush=True)\n")
    child.start()
    child.wait_for_line("bye", timeout_s=20)
    deadline = time.monotonic() + 20
    while child.alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    looks = iter([True])  # the first look still sees a member; every later one finds only zombies

    def refuse(pgid: int, signum: int) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(procs, "_group_alive", lambda pgid, proc=None: next(looks, False))
    monkeypatch.setattr(procs.os, "killpg", refuse)
    child.stop(grace_s=1)
    assert child.returncode == 0


def test_stop_names_eperm_when_a_member_still_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    child = python_child(tmp_path, "import time\nprint('UP', flush=True)\ntime.sleep(600)\n")
    child.start()
    child.wait_for_line("UP", timeout_s=20)

    def refuse(pgid: int, signum: int) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(procs.os, "killpg", refuse)
    try:
        with pytest.raises(
            ProcessError, match=r"cannot signal the process group \d+ of child: permission denied \(EPERM\)"
        ):
            child.stop(grace_s=1)
        assert child.alive()
    finally:
        monkeypatch.undo()
        child.stop(grace_s=5)
    assert not child.alive()


def test_stop_ends_the_whole_process_group(tmp_path: Path) -> None:
    child = python_child(
        tmp_path,
        f"""
        import subprocess, time
        grandchild = subprocess.Popen([{PYTHON!r}, "-c", "import time; time.sleep(600)"])
        print("GRANDCHILD", grandchild.pid, flush=True)
        time.sleep(600)
        """,
    )
    child.start()
    child.wait_for_line("GRANDCHILD", timeout_s=20)
    grandchild = int(child.log_path.read_text().split("GRANDCHILD")[1].split()[0])
    pgid = child.pid
    child.stop(grace_s=5)
    assert not group_alive(pgid)
    assert process_gone(grandchild)


def test_stop_before_start_and_twice_is_harmless(tmp_path: Path) -> None:
    child = python_child(tmp_path, "print('bye')\n")
    child.stop()
    child.start()
    child.wait_for_line("bye", timeout_s=20)
    child.stop()
    child.stop()
    assert child.returncode == 0


def test_a_process_starts_once(tmp_path: Path) -> None:
    child = python_child(tmp_path, "import time\ntime.sleep(600)\n")
    child.start()
    try:
        with pytest.raises(ProcessError, match="already started"):
            child.start()
    finally:
        child.stop(grace_s=5)


def test_a_missing_program_raises(tmp_path: Path) -> None:
    child = ManagedProcess("ghost", [str(tmp_path / "no-such-program")], {}, tmp_path / "ghost.log")
    with pytest.raises(ProcessError, match="cannot start ghost"):
        child.start()


def test_wait_for_port_accepts_the_childs_own_listener(tmp_path: Path, free_port: Callable[[], int]) -> None:
    port = free_port()
    child = python_child(
        tmp_path,
        f"""
        import socket, time
        sock = socket.socket(); sock.bind(("127.0.0.1", {port})); sock.listen()
        print("LISTENING", flush=True)
        time.sleep(600)
        """,
    )
    child.start()
    try:
        child.wait_for_port(port, timeout_s=20)
    finally:
        child.stop(grace_s=5)


def test_wait_for_port_waits_out_a_listener_seen_before_its_owner(
    tmp_path: Path, listener: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The child binds between two looks: the first finds no owner, a later one finds the child. That is how a
    starting server looks for a moment; it is waited out, not refused."""
    child = python_child(tmp_path, "import time\ntime.sleep(600)\n")
    child.start()
    looks: list[int] = []

    def owners(port: int) -> list[PortOwner]:
        looks.append(port)
        return [] if len(looks) == 1 else [PortOwner(child.pid, "python child")]

    monkeypatch.setattr(procs, "port_owners", owners)
    try:
        child.wait_for_port(listener, timeout_s=20)
    finally:
        child.stop(grace_s=5)
    assert len(looks) >= 2


def test_wait_for_port_refuses_a_listener_that_stays_unidentified(
    tmp_path: Path, listener: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = python_child(tmp_path, "import time\ntime.sleep(600)\n")
    child.start()
    monkeypatch.setattr(procs, "port_owners", lambda port, proc=None: [])
    monkeypatch.setattr(procs, "_UNIDENTIFIED_GRACE_S", 0.5)
    try:
        started = time.monotonic()
        with pytest.raises(ProcessError, match=rf"port {listener} is in use, but its owner cannot be identified"):
            child.wait_for_port(listener, timeout_s=20)
        assert 0.5 <= time.monotonic() - started < 10  # refused only once the grace has passed
    finally:
        child.stop(grace_s=5)


def test_wait_for_port_with_a_child_that_listens_late(tmp_path: Path, free_port: Callable[[], int]) -> None:
    port = free_port()
    child = python_child(
        tmp_path,
        f"""
        import socket, time
        time.sleep(0.3)
        sock = socket.socket(); sock.bind(("127.0.0.1", {port})); sock.listen()
        time.sleep(600)
        """,
    )
    child.start()
    try:
        child.wait_for_port(port, timeout_s=20)
    finally:
        child.stop(grace_s=5)


def test_wait_for_port_refuses_a_foreign_listener(tmp_path: Path, listener: int) -> None:
    child = python_child(tmp_path, "import time\ntime.sleep(600)\n")
    child.start()
    try:
        with pytest.raises(ProcessError, match=rf"port {listener} is held by pid {os.getpid()}"):
            child.wait_for_port(listener, timeout_s=20)
    finally:
        child.stop(grace_s=5)


def test_wait_for_port_raises_when_the_child_exits(tmp_path: Path, free_port: Callable[[], int]) -> None:
    child = python_child(tmp_path, "import sys\nprint('cannot bind', flush=True)\nsys.exit(1)\n")
    child.start()
    with pytest.raises(ProcessError, match="exited with status 1 before listening on port"):
        child.wait_for_port(free_port(), timeout_s=20)
    child.stop()


def test_context_manager_stops_the_process(tmp_path: Path) -> None:
    child = python_child(tmp_path, "import time\nprint('UP', flush=True)\ntime.sleep(600)\n")
    with child.start() as running:
        running.wait_for_line("UP", timeout_s=20)
        pgid = running.pid
    assert not child.alive()
    assert not group_alive(pgid)
