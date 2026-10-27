"""The test side of ``fake_openclaw.py``: put the stub CLI on PATH, tell it what to do, read what it was called with."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STUB = Path(__file__).with_name("fake_openclaw.py")

# The ports of the profiles the unit tests build: ports that no run of the experiment uses (a real run uses 18789 and
# 30011 to 30013), so that a test that forgets the stub CLI cannot reach a real gateway or model server. The gateway of
# a test that starts one takes six free ports of its own instead.
GATEWAY_PORT = 810
PROXY_PORT = 820
POLICY_PORT = 830


def inspect_process(pid: int) -> tuple[str, str] | None:
    """The state (``Z`` for a zombie) and the command line of a process, or ``None`` if there is no such process."""
    if Path("/proc/self/stat").exists():
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            return None
        return stat[stat.rfind(")") + 2 :].split()[0], command.strip()
    done = subprocess.run(["ps", "-o", "stat=,command=", "-p", str(pid)], capture_output=True, text=True, check=False)
    line = done.stdout.strip()
    if done.returncode != 0 or not line:
        return None
    state, _, command = line.partition(" ")
    return state, command.strip()


def wait_until(condition: Callable[[], bool], timeout_s: float) -> bool:
    """Poll ``condition`` until it holds (true) or ``timeout_s`` has passed (false)."""
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def running(pid: int) -> bool:
    """Whether the process exists and is not a zombie, which has ended and only waits for its parent to say so."""
    seen = inspect_process(pid)
    return seen is not None and not seen[0].startswith("Z")


@dataclass
class FakeCli:
    """The stub CLI on PATH, what it is told to do, and what it was called with."""

    bin_dir: Path
    state_dir: Path

    @classmethod
    def install(cls, directory: Path) -> FakeCli:
        """Make ``openclaw`` and ``node`` programs in ``directory/fake-bin``; the caller puts that on PATH."""
        bin_dir, state_dir = directory / "fake-bin", directory / "fake-state"
        bin_dir.mkdir(parents=True)
        state_dir.mkdir()
        wrapper = bin_dir / "openclaw"
        wrapper.write_text(
            "#!/bin/sh\n"
            f"export FAKE_OPENCLAW_DIR={shlex.quote(str(state_dir))}\n"
            f'exec {shlex.quote(sys.executable)} {shlex.quote(str(STUB))} "$@"\n'
        )
        wrapper.chmod(0o755)
        node = bin_dir / "node"
        node.write_text("#!/bin/sh\nexit 0\n")
        node.chmod(0o755)
        return cls(bin_dir, state_dir)

    def plan(self, **sections: Any) -> None:
        """Set what the stub does: ``version``, and the ``agent``, ``delete`` and ``gateway`` sections."""
        (self.state_dir / "scenario.json").write_text(json.dumps(sections))

    def agent_ok(
        self,
        payloads: Sequence[Any] = ("Done.",),
        session_id: str | None = "00000000-0000-4000-8000-000000000001",
        *,
        stderr: str = "",
        sleep: float = 0.0,
        **fields: Any,
    ) -> None:
        """What ``openclaw agent --json`` prints when the gateway ran the turn.

        A payload is the text, or a dict that is used as it is; ``fields`` replace top-level keys of the reply.
        """
        meta: dict[str, Any] = {"durationMs": 12, "agentMeta": {"provider": "rl", "model": "policy"}}
        if session_id is not None:
            meta["agentMeta"]["sessionId"] = session_id
        reply: dict[str, Any] = {
            "runId": "run-1",
            "status": "ok",
            "summary": "completed",
            "result": {
                "payloads": [p if isinstance(p, dict) else {"text": p} for p in payloads],
                "meta": meta,
            },
        }
        reply.update(fields)
        self.plan(agent={"stdout": json.dumps(reply, indent=2) + "\n", "stderr": stderr, "sleep": sleep})

    def calls(self) -> list[dict[str, Any]]:
        path = self.state_dir / "calls.jsonl"
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()]

    def gateway_pids(self) -> list[int]:
        """The process ids of the gateways the stub has started, in order (a line cut short by a kill is skipped)."""
        pids: list[int] = []
        path = self.state_dir / "calls.jsonl"
        for line in path.read_text().splitlines() if path.is_file() else []:
            try:
                call = json.loads(line)
            except ValueError:
                continue
            if call["args"][:2] == ["gateway", "--port"] and call["pid"] not in pids:
                pids.append(call["pid"])
        return pids

    def reap(self) -> list[int]:
        """Kill every gateway of the stub that is still running, with its process group; returns their pids.

        This is the safety net of the tests, not a part of them: a test that stops its gateways never needs it, and one
        that fails half way, or whose code under test forgets to stop a process it started, would leave the stub
        running. The pid of a gateway that has ended may belong to another program by now, so a pid is only signalled
        while it still is this stub and leads a process group of its own.
        """
        killed: list[int] = []
        for pid in self.gateway_pids():
            seen = inspect_process(pid)
            if seen is None or seen[0].startswith("Z") or STUB.name not in seen[1]:
                continue
            try:
                if os.getpgid(pid) != pid:
                    continue
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                continue
            killed.append(pid)
        for pid in killed:
            wait_until(lambda pid=pid: not running(pid), 10)  # a kill is not the end of the process yet
        return killed
