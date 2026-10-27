"""The child processes of a run, and the local ports they listen on.

A run starts its model server, trainer, MetaClaw proxy and OpenClaw gateway as ``ManagedProcess``es: each in a
session of its own, so that stopping it stops everything it started. Before a process takes a port, the port must
be free; a port held by anything else is refused with its owner named, and nothing is ever stopped to free it.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import NoReturn, Self

_PROC = Path("/proc")
_POLL_S = 0.1
_TAIL_LINES = 40
# How long a listener whose owner cannot be found is waited out before it is refused: a server that has just bound
# its port can be seen listening a moment before its owner shows in lsof or /proc.
_UNIDENTIFIED_GRACE_S = 2.0


class ProcessError(RuntimeError):
    """A child process did not start, did not get ready, exited early or could not be stopped."""


class PortInUse(RuntimeError):
    """A port this run needs is held by another process."""


# --- ports -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PortOwner:
    pid: int
    command: str | None

    def describe(self) -> str:
        return f"pid {self.pid} ({self.command})" if self.command else f"pid {self.pid}"


def _command_line(pid: int, proc: Path = _PROC) -> str | None:
    try:
        raw = (proc / str(pid) / "cmdline").read_bytes()
    except OSError:
        pass
    else:
        return raw.replace(b"\0", b" ").decode(errors="replace").strip() or None
    try:
        done = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() or None


def _lsof_owners(port: int) -> list[PortOwner] | None:
    """The processes ``lsof`` sees listening on ``port``; ``None`` when there is no usable ``lsof``."""
    try:
        done = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fpc"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode not in (0, 1):  # 1: nothing found
        return None
    owners: dict[int, str | None] = {}
    pid = None
    for line in done.stdout.splitlines():
        if line.startswith("p") and line[1:].isdigit():
            pid = int(line[1:])
            owners[pid] = None
        elif line.startswith("c") and pid is not None:
            owners[pid] = line[1:] or None
    return [PortOwner(pid, _command_line(pid) or name) for pid, name in sorted(owners.items())]


def _proc_listening_inodes(port: int, proc: Path = _PROC) -> set[str] | None:
    """Inodes of the sockets listening on ``port`` (Linux); ``None`` when there is no ``/proc/net/tcp``."""
    tables = [proc / "net" / "tcp", proc / "net" / "tcp6"]
    if not any(table.is_file() for table in tables):
        return None
    inodes = set()
    for table in tables:
        try:
            rows = table.read_text().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            fields = row.split()
            # sl local_address rem_address st ... inode; the local port is hex, state 0A is LISTEN
            if len(fields) > 9 and fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port:
                inodes.add(fields[9])
    return inodes


def _proc_listening(port: int, proc: Path = _PROC) -> bool:
    return bool(_proc_listening_inodes(port, proc))


def _proc_owners(port: int, proc: Path = _PROC) -> list[PortOwner] | None:
    """The processes whose open files include a socket listening on ``port`` (Linux).

    A process of another user hides its file descriptors, so the list can miss an owner the socket table shows.
    """
    inodes = _proc_listening_inodes(port, proc)
    if inodes is None:
        return None
    if not inodes:
        return []
    sockets = {f"socket:[{inode}]" for inode in inodes}
    owners = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            descriptors = list((entry / "fd").iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            if target in sockets:
                owners.append(PortOwner(int(entry.name), _command_line(int(entry.name), proc)))
                break
    return sorted(owners, key=lambda owner: owner.pid)


def port_owners(port: int, proc: Path = _PROC) -> list[PortOwner]:
    """The processes listening on ``port`` that can be identified: ``lsof`` where available, else ``/proc``."""
    owners = _lsof_owners(port)
    if not owners:
        owners = _proc_owners(port, proc) or owners
    return owners or []


def _accepts_connections(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def _listening(port: int, proc: Path = _PROC) -> bool:
    """Whether a listener shows in /proc or accepts a connection on 127.0.0.1, owner known or not."""
    return _proc_listening(port, proc) or _accepts_connections(port)


def port_in_use(port: int, proc: Path = _PROC) -> bool:
    """Whether anything listens on ``port``, identified or not."""
    return bool(port_owners(port, proc)) or _listening(port, proc)


def _describe_holders(port: int) -> str:
    owners = port_owners(port)
    if not owners:
        return f"port {port} is in use by a process that cannot be identified"
    return f"port {port} is in use by " + " and ".join(owner.describe() for owner in owners)


def require_free(ports: Iterable[int]) -> None:
    """Raise ``PortInUse`` naming every port in ``ports`` that is held, and its owner when it can be found."""
    held = [_describe_holders(port) for port in ports if port_in_use(port)]
    if held:
        raise PortInUse(
            "; ".join(held) + ". Nothing was started and nothing was stopped: stop that process, or give this run "
            "other ports in configs/runtime.yaml."
        )


# --- processes ---------------------------------------------------------------------------------------


def exit_without_teardown(main: Callable[[], int]) -> NoReturn:
    """Run a process entry's ``main`` and leave with its status, skipping interpreter teardown.

    The Tinker SDK's background threads can crash while the interpreter shuts down, which would turn a finished
    run into a non-zero exit with no message. Output is flushed first.
    """
    try:
        code = main()
    except SystemExit as exit_:  # argparse, after printing --help or a usage error
        if isinstance(exit_.code, str):
            print(exit_.code, file=sys.stderr)
        code = exit_.code if isinstance(exit_.code, int) else (0 if exit_.code is None else 1)
    except KeyboardInterrupt:
        code = 130
    except Exception:  # noqa: BLE001 -- anything else ends the process too, with its traceback and status 1
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def _live_group_members(pgid: int, proc: Path = _PROC) -> list[int] | None:
    """The processes of the group that are not zombies, from /proc or ``ps``; ``None`` when neither can tell.

    A zombie does not count: a process left to a parent that does not reap (a container's first process, say) can
    stay a zombie for good, and it can no longer be signalled.
    """
    members = []
    if (proc / "self" / "stat").is_file():
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text()
            except OSError:
                continue
            fields = stat[stat.rfind(")") + 2 :].split()  # after "pid (comm) ": state ppid pgrp ...
            if len(fields) > 2 and fields[2] == str(pgid) and fields[0] != "Z":
                members.append(int(entry.name))
        return members
    try:
        done = subprocess.run(
            ["ps", "-A", "-o", "pid=,pgid=,stat="], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    for line in done.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[1] == str(pgid) and not fields[2].startswith("Z"):
            members.append(int(fields[0]))
    return members


def _group_alive(pgid: int, proc: Path = _PROC) -> bool:
    """Whether any process of the group is still running (a zombie does not count)."""
    members = _live_group_members(pgid, proc)
    if members is not None:
        return bool(members)
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # it exists; whether it runs is for the signal that follows to show
    return True


class ManagedProcess:
    """A child process in a session of its own, with its output appended to ``log_path``.

    ``env`` is the child's whole environment; nothing of this process's environment is passed on.
    """

    def __init__(
        self,
        name: str,
        argv: Sequence[str | os.PathLike[str]],
        env: Mapping[str, str],
        log_path: Path,
        cwd: Path | None = None,
    ) -> None:
        self.name = name
        self.argv = [os.fspath(arg) for arg in argv]
        self.env = dict(env)
        self.log_path = Path(log_path)
        self.cwd = cwd
        self._popen: subprocess.Popen[bytes] | None = None
        self._log_start = 0
        self._group_gone = False

    def __repr__(self) -> str:
        return f"ManagedProcess({self.name!r}, pid={self.pid})"

    @property
    def pid(self) -> int | None:
        return None if self._popen is None else self._popen.pid

    @property
    def returncode(self) -> int | None:
        return None if self._popen is None else self._popen.poll()

    def start(self) -> Self:
        if self._popen is not None:
            raise ProcessError(f"{self.name} was already started (pid {self._popen.pid})")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "ab") as log:
            self._log_start = log.tell()
            try:
                self._popen = subprocess.Popen(
                    self.argv,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=self.env,
                    cwd=self.cwd,
                    start_new_session=True,
                )
            except OSError as error:
                raise ProcessError(f"cannot start {self.name} ({self.argv[0]}): {error}") from error
        return self

    def alive(self) -> bool:
        return self._popen is not None and self._popen.poll() is None

    def log_text(self) -> str:
        """What the process has written since it was started."""
        try:
            with open(self.log_path, "rb") as log:
                log.seek(self._log_start)
                return log.read().decode(errors="replace")
        except FileNotFoundError:
            return ""

    def log_tail(self, lines: int = _TAIL_LINES) -> str:
        return "\n".join(self.log_text().splitlines()[-lines:])

    def _failure(self, what: str) -> ProcessError:
        return ProcessError(f"{self.name} {what}; the last lines of {self.log_path}:\n{self.log_tail()}")

    def _exited(self) -> str:
        return f"exited with status {self.returncode}"

    def wait_for_line(self, substring: str, timeout_s: float) -> None:
        """Wait until the process has logged ``substring`` since it was started.

        Raises ``ProcessError`` with the end of the log when the process exits first or ``timeout_s`` passes.
        """
        if self._popen is None:
            raise ProcessError(f"{self.name} was never started")
        needle = substring.encode()
        if not needle:
            raise ValueError("wait_for_line needs a non-empty substring")
        deadline = time.monotonic() + timeout_s
        keep = len(needle) - 1  # the end of what was read, in case the line is split across two reads
        window = b""
        with open(self.log_path, "rb") as log:
            log.seek(self._log_start)
            while True:
                exited = not self.alive()  # checked before reading, so that its last output is read too
                chunk = log.read()
                if chunk:
                    window += chunk
                    if needle in window:
                        return
                    window = window[-keep:] if keep else b""
                elif exited:
                    raise self._failure(f"{self._exited()} before logging {substring!r}")
                # Checked after every read, so that a process that keeps writing cannot hold the wait open.
                if time.monotonic() >= deadline:
                    raise self._failure(f"did not log {substring!r} within {timeout_s:g} s")
                if not chunk:
                    time.sleep(_POLL_S)

    def owns(self, pid: int) -> bool:
        """Whether ``pid`` is this process or one it started (they share its session)."""
        try:
            return self.pid is not None and os.getsid(pid) == self.pid
        except OSError:
            return False

    def wait_for_port(self, port: int, timeout_s: float) -> None:
        """Wait until ``port`` listens and check that this process holds it, so that a client of the port talks
        to this process and never to whatever else might be listening there."""
        deadline = time.monotonic() + timeout_s
        unidentified_since: float | None = None
        while True:
            owners = port_owners(port)  # the one look that decides
            if owners:
                foreign = [owner for owner in owners if not self.owns(owner.pid)]
                if foreign:
                    held_by = " and ".join(owner.describe() for owner in foreign)
                    raise ProcessError(f"port {port} is held by {held_by}, not by {self.name} (pid {self.pid})")
                return
            if not self.alive():
                raise self._failure(f"{self._exited()} before listening on port {port}")
            now = time.monotonic()
            # A listener without a known owner is looked at again on the next poll: the child may have bound the
            # port just after the look above. Only one that stays unidentified through the grace is refused.
            if _listening(port):
                unidentified_since = now if unidentified_since is None else unidentified_since
            else:
                unidentified_since = None
            if unidentified_since is not None and (
                now - unidentified_since >= _UNIDENTIFIED_GRACE_S or now >= deadline
            ):
                raise ProcessError(
                    f"port {port} is in use, but its owner cannot be identified, so it cannot be checked to be "
                    f"{self.name} (pid {self.pid})"
                )
            if now >= deadline:
                raise self._failure(f"did not listen on port {port} within {timeout_s:g} s")
            time.sleep(_POLL_S)

    def _group_running(self) -> bool:
        if self._popen is None or self._group_gone:
            return False
        self._popen.poll()  # reap the leader, so that it no longer counts as a member
        if not _group_alive(self._popen.pid):
            self._group_gone = True  # never signal the group again: its id may be reused
        return not self._group_gone

    def _signal_group(self, signum: int) -> None:
        assert self._popen is not None
        try:
            os.killpg(self._popen.pid, signum)
        except ProcessLookupError:
            self._group_gone = True
        except PermissionError as error:
            # macOS refuses (EPERM) to signal a group whose only members are zombies that nobody has reaped yet.
            # With the leader gone and no running member left, there is nothing to stop.
            if self._popen.poll() is not None and not _group_alive(self._popen.pid):
                self._group_gone = True
            else:
                raise ProcessError(
                    f"cannot signal the process group {self._popen.pid} of {self.name}: permission denied (EPERM)"
                ) from error

    def _wait_group(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while self._group_running():
            if time.monotonic() >= deadline:
                return False
            time.sleep(_POLL_S)
        return True

    def stop(self, grace_s: float = 40) -> None:
        """SIGTERM to the process group, then SIGKILL to whatever is left after ``grace_s``."""
        if self._popen is None:
            return
        if self._group_running():
            self._signal_group(signal.SIGTERM)
            if not self._wait_group(grace_s):
                self._signal_group(signal.SIGKILL)
                if not self._wait_group(10):
                    raise ProcessError(f"could not stop {self.name} (process group {self._popen.pid})")
        try:
            self._popen.wait(timeout=10)
        except subprocess.TimeoutExpired:
            raise ProcessError(f"{self.name} (pid {self._popen.pid}) still runs outside its stopped group") from None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.stop()
