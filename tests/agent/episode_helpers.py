"""A scripted agent for the episode tests: it stands in for the OpenClaw CLI and does what each turn's script says.

The agent's sessions behave as OpenClaw's do: every turn goes to the current session, whose transcript the gateway
appends to; ``sessions.delete`` renames the transcript to ``<id>.jsonl.deleted.<time>`` and the next turn opens a new
session. A turn can run the installed service programs (as the agent's exec tool would), write tool calls into the
transcript, create files and do anything else a test needs. Every step of an episode goes into one call log.
"""

from __future__ import annotations

import json
import os
import random
import socket
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from evolve_ipi.agent import episode
from evolve_ipi.agent.openclaw import GATEWAY_SPAN, Profile, TurnResult
from evolve_ipi.config import PortsSpec
from evolve_ipi.environment import state, verifiers, workspace
from evolve_ipi.paths import RuntimePaths
from evolve_ipi.procs import port_in_use

SYSTEM_PYTHON = Path("/usr/bin/python3")
# What the agent's exec tool offers: a bare PATH, so that `env python3` finds the system interpreter.
AGENT_ENV = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/")}
CLOCK = datetime(2026, 10, 5, 22, 0, 0, tzinfo=UTC)


def runtime_paths(tmp_path: Path) -> RuntimePaths:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return RuntimePaths(
        root=tmp_path / "runtime",
        agent_home=home,
        outputs=tmp_path / "outputs",
        metaclaw_python=tmp_path / "venv" / "bin" / "python",
    )


def _bindable(port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return not port_in_use(port)


def free_ports() -> PortsSpec:
    """Four distinct free ports, the gateway's first with the five above it free as well.

    They are taken below the ports Linux and macOS hand out for outgoing connections (from 32768 and 49152), so that no
    other program can be given one of them between this choice and the test's own check.
    """
    rng = random.Random()
    for _ in range(500):
        first = rng.randrange(20000, 32768 - GATEWAY_SPAN - 3)
        ports = range(first, first + GATEWAY_SPAN + 3)
        if all(_bindable(port) for port in ports):
            return PortsSpec(gateway=first, policy=ports[-3], proxy=ports[-2], model=ports[-1])
    raise RuntimeError("no free ports found")


@dataclass
class Turn:
    """What the agent does in one turn, and what the CLI reports about it.

    ``run`` are service programs the agent calls (``["drive", "read", "x"]``); each also appears as a tool call in the
    transcript. ``tool_calls`` are further tool calls written to the transcript, ``touch`` files the agent creates and
    ``effect`` anything else it does. ``session`` replaces the session id the CLI reports, ``no_session`` drops it.
    """

    reply: str = "Done."
    run: Sequence[Sequence[str]] = ()
    tool_calls: Sequence[dict[str, Any]] = ()
    touch: Sequence[Path] = ()
    effect: Callable[[], None] | None = None
    status: str = "ok"
    error: str | None = None
    elapsed: float = 3.0
    session: str | None = None
    no_session: bool = False


@dataclass
class FakeAgent:
    """Stands in for ``delete_session`` and ``run_turn`` of ``evolve_ipi.agent.openclaw``."""

    profile: Profile
    turns: list[Turn]
    log: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    sessions: list[str] = field(default_factory=list)
    current: str | None = None
    clock: datetime = CLOCK
    exit_codes: list[int] = field(default_factory=list)

    # --- the two CLI calls ---

    def delete_session(self, p: Profile) -> None:
        assert p == self.profile
        self.log.append("delete_session")
        if self.current is not None:
            live = self._transcript(self.current)
            if live.exists():
                stamp = self._tick(0).strftime("%Y-%m-%dT%H-%M-%S.000Z")
                live.rename(live.with_name(f"{live.name}.deleted.{stamp}"))
        self.current = None

    def run_turn(self, p: Profile, message: str, timeout_s: int) -> TurnResult:
        assert p == self.profile
        self.log.append(f"run_turn:{message}")
        self.messages.append(message)
        if not self.turns:
            raise AssertionError(f"the script has no turn left for {message!r}")
        turn = self.turns.pop(0)
        if self.current is None:
            self.current = f"00000000-0000-4000-8000-{len(self.sessions) + 1:012d}"
            self.sessions.append(self.current)
            self._write({"type": "session", "id": self.current, "cwd": str(p.workspace)}, 0)
        self._message("user", [{"type": "text", "text": message}])
        for argv in turn.run:
            self._message("assistant", [self._call("exec", {"command": " ".join(["svc/" + argv[0], *argv[1:]])})])
            done = subprocess.run(
                [str(p.workspace / "svc" / argv[0]), *argv[1:]],
                cwd=p.workspace,
                capture_output=True,
                text=True,
                env=AGENT_ENV,
                check=False,
            )
            self.exit_codes.append(done.returncode)
            self._message("toolResult", [{"type": "text", "text": done.stdout + done.stderr}])
        for call in turn.tool_calls:
            self._message("assistant", [self._call(call["name"], call["arguments"])])
        for path in turn.touch:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("created by the agent\n", encoding="utf-8")
        if turn.effect is not None:
            turn.effect()
        if turn.reply:
            self._message("assistant", [{"type": "text", "text": turn.reply}])
        session = None if turn.no_session else (turn.session or self.current)
        limit = turn.status == "ok" and turn.elapsed >= timeout_s
        return TurnResult(turn.status, turn.reply, session, turn.elapsed, turn.error, limit)

    # --- the transcript ---

    def _transcript(self, session_id: str) -> Path:
        return self.profile.sessions_dir / f"{session_id}.jsonl"

    def _tick(self, seconds: float) -> datetime:
        self.clock += timedelta(seconds=seconds)
        return self.clock

    def _write(self, entry: dict[str, Any], seconds: float = 1) -> None:
        assert self.current is not None
        entry = {**entry, "timestamp": self._tick(seconds).isoformat().replace("+00:00", "Z")}
        path = self._transcript(self.current)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def _message(self, role: str, content: list[dict[str, Any]]) -> None:
        self._write({"type": "message", "message": {"role": role, "content": content}})

    def _call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"type": "toolCall", "id": f"call-{self.clock.timestamp():.0f}", "name": name, "arguments": arguments}


def install(monkeypatch: pytest.MonkeyPatch, agent: FakeAgent) -> list[str]:
    """Put the scripted agent in place of the CLI and log every step of an episode; returns the call log."""
    monkeypatch.setattr(episode, "delete_session", agent.delete_session)
    monkeypatch.setattr(episode, "run_turn", agent.run_turn)
    log = agent.log

    def spy(owner: Any, name: str, label: str) -> None:
        original = getattr(owner, name)

        def logged(*args: Any, **kwargs: Any) -> Any:
            log.append(label)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, logged)

    spy(episode, "archive_deleted_transcripts", "archive")
    spy(episode, "transcript_tool_calls", "transcript")
    spy(workspace, "restore", "restore")
    spy(workspace, "guard", "guard")
    spy(state, "reset", "reset")
    spy(state, "read_access_log", "access_log")
    spy(verifiers, "judge_asr", "judge_asr")
    spy(verifiers, "judge_utility", "judge_utility")
    return log


def needs_system_python() -> None:
    if not SYSTEM_PYTHON.exists():
        pytest.skip("the service programs run with the system interpreter, /usr/bin/python3, which is missing")


def raw_line(request: int, raw: str, at: str = "2026-10-05T18:00:00-0400") -> str:
    return json.dumps({"at": at, "request": request, "messages": 3, "n_tokens": 7, "raw": raw}) + "\n"


REAL_PORTS = {18789, 30011, 30012, 30013}  # the ports of a real run; the integration tests use none of them


def require_openclaw(probe: Path) -> None:
    """Skip unless ``openclaw --version`` prints the pinned version, asked in a directory of its own."""
    from evolve_ipi.agent.openclaw import OPENCLAW_VERSION, OpenClawError, installed_version

    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(probe),
        "LANG": "C.UTF-8",
        "OPENCLAW_STATE_DIR": str(probe / "state"),
        "OPENCLAW_CONFIG_PATH": str(probe / "openclaw.json"),
    }
    try:
        version = installed_version(env)
    except OpenClawError as error:
        pytest.skip(f"OpenClaw cannot be run here: {error}")
    if version != OPENCLAW_VERSION:
        pytest.skip(f"this test needs OpenClaw {OPENCLAW_VERSION}; {version} is installed")
