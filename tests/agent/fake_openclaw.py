"""A stand-in for the OpenClaw CLI, for the tests that must not run the real one.

``fake_cli.FakeCli`` puts a wrapper named ``openclaw`` on PATH that runs this file with ``FAKE_OPENCLAW_DIR`` set.
That directory holds what the stub does (``scenario.json``, written by the test) and what it saw (``calls.jsonl``:
the arguments, the working directory, the whole environment and the process id of every call).

Commands: ``--version``; ``agent ...``; ``gateway call sessions.delete ...``; and ``gateway --port N``, which logs
what the real gateway logs and listens on N until it is terminated. ``--version`` prints ``version``, or fails with
``version_fails``, after ``version_sleep`` seconds. The others print what their section of the scenario says
(``stdout``, ``stderr``), sleep (``sleep``) and exit with ``exit``. A gateway takes ``die`` (exit before listening),
``bind`` (false: stay up without listening), ``bind_after`` (seconds before it listens) and ``plugin`` (false: do not
log the plugin's line). Creating the file ``release-port`` makes a running gateway close its socket and carry on.

The stub does not outlive the test that started it: whenever it sleeps it also checks that the process that started it
is still there, and it stops after ``max_s`` seconds of life (default ``HARD_CAP_S``) in any case.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_VERSION = "2026.3.7"
HARD_CAP_S = 600
_STARTED = time.monotonic()
_PARENT = os.getppid()


def _state_dir() -> Path:
    return Path(os.environ["FAKE_OPENCLAW_DIR"])


def _scenario() -> dict[str, Any]:
    path = _state_dir() / "scenario.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def _record(args: list[str]) -> None:
    env = {name: value for name, value in os.environ.items() if name != "FAKE_OPENCLAW_DIR"}
    entry = {"args": args, "cwd": os.getcwd(), "env": env, "pid": os.getpid()}
    with (_state_dir() / "calls.jsonl").open("a") as log:
        log.write(json.dumps(entry) + "\n")


def _check_alive(cap_s: float) -> None:
    if os.getppid() != _PARENT:
        raise SystemExit("fake openclaw: the process that started it is gone")
    if time.monotonic() - _STARTED > cap_s:
        raise SystemExit(f"fake openclaw: still running after {cap_s:g} s")


def _pause(seconds: float, cap_s: float) -> None:
    """Sleep ``seconds``, a slice at a time, leaving at once if the stub is orphaned or has lived too long."""
    deadline = time.monotonic() + seconds
    while True:
        _check_alive(cap_s)
        left = deadline - time.monotonic()
        if left <= 0:
            return
        time.sleep(min(left, 0.2))


def _answer(section: dict[str, Any], cap_s: float) -> int:
    _pause(section.get("sleep", 0), cap_s)
    sys.stdout.write(section.get("stdout", ""))
    sys.stderr.write(section.get("stderr", ""))
    sys.stdout.flush()
    sys.stderr.flush()
    return int(section.get("exit", 0))


def _gateway(port: int, section: dict[str, Any], cap_s: float) -> int:
    if "die" in section:  # the gateway fails before it listens
        print(section["die"], file=sys.stderr, flush=True)
        return 3
    if section.get("plugin", True):  # what the rl-training-headers plugin logs when OpenClaw has loaded it
        print("[gateway] rl-training-headers: activated (fetch patched)", flush=True)
    _pause(section.get("bind_after", 0), cap_s)
    server = socket.socket()
    if section.get("bind", True):  # a gateway that stays up without taking its port is the other way to fail
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", port))
        server.listen()
    print(f"[gateway] listening on ws://127.0.0.1:{port}", flush=True)
    release = _state_dir() / "release-port"
    while True:
        _pause(0.2, cap_s)
        if release.exists():  # the port is let go, and the process goes on as if nothing had happened
            server.close()


def main(args: list[str]) -> int:
    _record(args)
    plan = _scenario()
    cap_s = float(plan.get("max_s", HARD_CAP_S))
    if args == ["--version"]:
        _pause(plan.get("version_sleep", 0), cap_s)
        if "version_fails" in plan:
            print(plan["version_fails"], file=sys.stderr)
            return 2
        print(plan.get("version", DEFAULT_VERSION))
        return 0
    if args[:1] == ["agent"]:
        return _answer(plan.get("agent", {}), cap_s)
    if args[:3] == ["gateway", "call", "sessions.delete"]:
        return _answer(plan.get("delete", {}), cap_s)
    if args[:2] == ["gateway", "--port"] and len(args) == 3:
        return _gateway(int(args[2]), plan.get("gateway", {}), cap_s)
    print(f"fake openclaw: unexpected arguments {args}", file=sys.stderr)
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
