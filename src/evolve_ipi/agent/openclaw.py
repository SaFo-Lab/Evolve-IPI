"""OpenClaw as the experiment runs it: one profile and gateway per (line, kind), and one agent turn at a time.

A profile is a directory with ``openclaw.json``, a gateway token and the state OpenClaw keeps (``state/``). It is
never shared: the RL line and the skill line, and the File-Create task and the service tasks, each have their own,
so a gateway never sees another run's sessions. Everything the OpenClaw processes need to know comes from the
environment ``gateway_env`` builds; no secret is in it, and nothing of the parent's environment is passed on.

The agent's workspace lies outside the profile. OpenClaw runs ``git init`` on a workspace it finds empty, which
would put a repository into the system prompt, so the workspace has to be filled before the gateway first sees it;
``write_profile`` never creates it.

    openclaw gateway --port <P>                  the gateway; it also takes P+1 .. P+5
    openclaw agent --agent <A> --json ...        one turn, always in the session ``agent:<A>:main``
    openclaw gateway call sessions.delete ...    the reset before each episode
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

from evolve_ipi.paths import AGENT_IDS, Kind, Line, RuntimePaths
from evolve_ipi.procs import ManagedProcess, ProcessError, require_free
from evolve_ipi.thirdparty import RL_HEADERS_PLUGIN_DIR

__all__ = [
    "GATEWAY_MAX_AGE_S",
    "GATEWAY_SPAN",
    "GATEWAY_START_TIMEOUT_S",
    "OPENCLAW_VERSION",
    "PLUGIN_ID",
    "SESSION_DELETE_TIMEOUT_S",
    "SUBPROCESS_SLACK_S",
    "Gateway",
    "OpenClawError",
    "Profile",
    "TurnResult",
    "archive_deleted_transcripts",
    "build_config",
    "check_profile",
    "check_version",
    "delete_session",
    "gateway_env",
    "installed_version",
    "profile_for",
    "run_turn",
    "transcript_timeline",
    "transcript_tool_calls",
    "write_profile",
]

# The OpenClaw version the experiment ran on. Its prompts, its tools and its CLI are part of the environment.
OPENCLAW_VERSION = "2026.3.7"

GATEWAY_SPAN = 6  # a gateway on port P also takes P+1 .. P+5
GATEWAY_START_TIMEOUT_S = 120  # the whole start: the port listens and the plugin is loaded
GATEWAY_MAX_AGE_S = 3 * 3600
PORT_OWNER_CHECK_S = 3  # how long the owner of the gateway's port may take to show when it is checked again
SESSION_DELETE_TIMEOUT_S = 120
SUBPROCESS_SLACK_S = 60  # a turn's CLI call is cut off this long after the limit it passes to OpenClaw
VERSION_TIMEOUT_S = 60

PLUGIN_ID = "rl-training-headers"  # sends X-Session-Id and X-Turn-Type with every model request
PLUGIN_VERSION = "1.0.0"
PLUGIN_ACTIVE_LINE = f"{PLUGIN_ID}: activated"  # what the plugin logs when OpenClaw has loaded it
CONTEXT_WINDOW = 32768
MAX_TOKENS = 4096

_OUT_OF_MEMORY = re.compile(r"heap out of memory|Reached heap limit|FATAL ERROR")
_FALLBACK = "falling back to embedded"  # what the CLI says when it ran the agent itself because the gateway failed
_SESSION_ID = re.compile(r"[A-Za-z0-9_-]+")
_ERROR_TAIL = 2000
_OUTPUT_HEAD = 500


class OpenClawError(RuntimeError):
    """OpenClaw is not set up as the run needs it, or something it was asked to do failed."""


# --- the profile ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class _LineSpec:
    provider: str
    model_id: str
    label: str  # the name OpenClaw shows for the model; it is not part of any prompt


_LINES: dict[Line, _LineSpec] = {
    "rl": _LineSpec("rl", "policy", "RL policy model"),
    # The id stays the same whichever model the line runs: OpenClaw prints it in the system prompt, and MetaClaw
    # replaces the model of the request before it goes on to the model server.
    "skills": _LineSpec("metaclaw", "qwen3.8-27b", "MetaClaw proxy model"),
}


@dataclass(frozen=True)
class Profile:
    """One OpenClaw profile: where it lives, the agent it runs and the model endpoint that agent talks to.

    ``provider_port`` is the policy model server of the RL line and the MetaClaw proxy of the skill line.
    ``home`` is the agent's HOME.
    """

    line: Line
    dir: Path
    workspace: Path
    gateway_port: int
    provider_port: int
    agent_id: str
    provider: str
    model_id: str
    label: str
    home: Path

    @property
    def model_ref(self) -> str:
        return f"{self.provider}/{self.model_id}"

    @property
    def config_path(self) -> Path:
        return self.dir / "openclaw.json"

    @property
    def token_path(self) -> Path:
        return self.dir / "gateway.token"

    @property
    def state_dir(self) -> Path:
        return self.dir / "state"

    @property
    def agent_dir(self) -> Path:
        return self.state_dir / "agents" / self.agent_id

    @property
    def sessions_dir(self) -> Path:
        """Where the gateway keeps the transcripts of the agent's sessions."""
        return self.agent_dir / "sessions"

    @property
    def plugin_dir(self) -> Path:
        return self.state_dir / "extensions" / PLUGIN_ID

    @property
    def session_key(self) -> str:
        """Every turn of the agent goes to this session, whatever a client asks for."""
        return f"agent:{self.agent_id}:main"


def profile_for(line: Line, kind: Kind, paths: RuntimePaths, gateway_port: int, provider_port: int) -> Profile:
    """The profile of ``line`` for the tasks of ``kind``, with its gateway on ``gateway_port``."""
    profile_dir = paths.profile(line, kind)  # refuses an unknown line or kind
    for what, path in (("the runtime directory", profile_dir), ("the agent's home", paths.agent_home)):
        if not path.is_absolute():  # OpenClaw would resolve it against its own working directory
            raise ValueError(f"{what} {path} is not an absolute path")
    workspace = paths.workspace(line, kind)  # refuses a workspace inside a git checkout
    for name, port in (("gateway", gateway_port), ("provider", provider_port)):
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError(f"the {name} port {port!r} is not a port number (1 to 65535)")
    last = gateway_port + GATEWAY_SPAN - 1
    if last > 65535:
        raise ValueError(f"the gateway takes its port and the {GATEWAY_SPAN - 1} above it: port {last} does not exist")
    if gateway_port <= provider_port <= last:
        raise ValueError(f"the provider port {provider_port} lies among the gateway's ports {gateway_port} to {last}")
    spec = _LINES[line]
    return Profile(
        line=line,
        dir=profile_dir,
        workspace=workspace,
        gateway_port=gateway_port,
        provider_port=provider_port,
        agent_id=AGENT_IDS[line],
        provider=spec.provider,
        model_id=spec.model_id,
        label=spec.label,
        home=paths.agent_home,
    )


def build_config(p: Profile) -> dict[str, Any]:
    """The contents of ``openclaw.json``.

    Sandboxing is off because the injected requests write outside the workspace: sandboxed, the agent could not carry
    them out and every episode would look like a refusal.
    """
    return {
        "models": {
            "providers": {
                p.provider: {
                    "baseUrl": f"http://127.0.0.1:{p.provider_port}/v1",
                    "apiKey": "dummy",
                    "api": "openai-completions",
                    "models": [
                        {
                            "id": p.model_id,
                            "name": p.label,
                            "api": "openai-completions",
                            "reasoning": True,
                            "input": ["text"],
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                            "contextWindow": CONTEXT_WINDOW,
                            "maxTokens": MAX_TOKENS,
                        }
                    ],
                }
            }
        },
        "agents": {
            "defaults": {"compaction": {"mode": "safeguard"}, "sandbox": {"mode": "off"}},
            "list": [
                {"id": "main", "workspace": str(p.dir / "workspace-main")},
                {
                    "id": p.agent_id,
                    "name": p.agent_id,
                    "workspace": str(p.workspace),
                    "agentDir": str(p.agent_dir),
                    "model": p.model_ref,
                },
            ],
        },
        "commands": {"native": "auto", "nativeSkills": "auto", "restart": True, "ownerDisplay": "raw"},
        "gateway": {
            "mode": "local",
            "bind": "loopback",
            "controlUi": {"allowedOrigins": [f"http://127.0.0.1:{p.gateway_port}"]},
        },
        "plugins": {
            "entries": {PLUGIN_ID: {"enabled": True}},
            "installs": {
                PLUGIN_ID: {
                    "source": "path",
                    "sourcePath": str(RL_HEADERS_PLUGIN_DIR),
                    "installPath": str(p.plugin_dir),
                    "version": PLUGIN_VERSION,
                }
            },
        },
    }


def _tree(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def _require_plugin_source() -> None:
    if not RL_HEADERS_PLUGIN_DIR.is_dir():
        raise OpenClawError(
            f"the {PLUGIN_ID} plugin is not checked out at {RL_HEADERS_PLUGIN_DIR}; "
            "run `git submodule update --init` in the repository root"
        )


def write_profile(p: Profile) -> None:
    """Write the profile's config, its gateway token and the plugin; safe to repeat.

    An existing token is kept, because a running gateway holds it. The plugin directory is replaced.
    """
    _require_plugin_source()
    p.agent_dir.mkdir(parents=True, exist_ok=True)
    (p.dir / "workspace-main").mkdir(parents=True, exist_ok=True)
    p.config_path.write_text(json.dumps(build_config(p), indent=2) + "\n", encoding="utf-8")
    if not p.token_path.exists():
        # Created with its final mode, so that the token is never readable by others, not even for a moment.
        descriptor = os.open(p.token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(secrets.token_hex(32) + "\n")
    p.token_path.chmod(0o600)
    if p.plugin_dir.exists():
        shutil.rmtree(p.plugin_dir)
    p.plugin_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(RL_HEADERS_PLUGIN_DIR, p.plugin_dir)


def check_profile(p: Profile) -> list[str]:
    """What is wrong with the profile on disk: a file that is missing or is not what ``write_profile`` writes."""
    problems = []
    if not p.home.is_dir():
        problems.append(f"the agent's home {p.home} does not exist")
    if not p.config_path.is_file():
        problems.append(f"{p.config_path} does not exist; the profile has not been written")
    else:
        try:
            on_disk = json.loads(p.config_path.read_text(encoding="utf-8"))
        except ValueError as error:
            problems.append(f"{p.config_path} is not valid JSON: {error}")
        else:
            if on_disk != build_config(p):
                problems.append(
                    f"{p.config_path} differs from the configuration this run builds for the profile "
                    "(it was edited, or written for other ports or paths)"
                )
    if not p.token_path.is_file():
        problems.append(f"{p.token_path} is missing")
    else:
        if p.token_path.stat().st_mode & 0o077:
            problems.append(f"{p.token_path} is readable by group or others; the gateway token is a secret")
        if not p.token_path.read_text(encoding="utf-8").strip():
            problems.append(f"{p.token_path} is empty")
    if not p.plugin_dir.is_dir():
        problems.append(f"the {PLUGIN_ID} plugin is missing from {p.plugin_dir}")
    elif RL_HEADERS_PLUGIN_DIR.is_dir() and _tree(p.plugin_dir) != _tree(RL_HEADERS_PLUGIN_DIR):
        problems.append(f"the {PLUGIN_ID} plugin in {p.plugin_dir} differs from {RL_HEADERS_PLUGIN_DIR}")
    return problems


# --- the environment and the version -------------------------------------------------------------


def _run(
    argv: list[str], *, env: Mapping[str, str], timeout: float, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Run an OpenClaw program to its end: output read as UTF-8, no stdin, and a non-zero exit left for the caller."""
    return subprocess.run(
        argv,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=dict(env),
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        check=False,
    )


def _tail(done: subprocess.CompletedProcess[str]) -> str:
    """What a failed program said: the end of its stderr, or of its stdout when it said nothing there."""
    return (done.stderr or done.stdout)[-_ERROR_TAIL:]


def _find(program: str, search_path: str | None = None) -> Path:
    found = shutil.which(program, path=search_path)
    if found is None:
        hint = (
            f"install openclaw@{OPENCLAW_VERSION} with npm"
            if program == "openclaw"
            else "OpenClaw needs Node.js 22.12 or later"
        )
        raise OpenClawError(f"{program} was not found on PATH; {hint}")
    return Path(found).absolute()


def _environment(p: Profile) -> tuple[dict[str, str], Path]:
    """The environment of the gateway and of every CLI call, and the ``openclaw`` program they run."""
    openclaw, node = _find("openclaw"), _find("node")
    # The directory of the node that runs openclaw comes first; /usr/bin and /bin are for the shell commands and
    # the python3 that the agent's tools use.
    path: list[str] = []
    for directory in (str(node.parent), str(openclaw.parent), "/usr/bin", "/bin"):
        if directory not in path:
            path.append(directory)
    try:
        token = p.token_path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise OpenClawError(
            f"cannot read the gateway token {p.token_path} ({error.strerror or error}); write the profile first"
        ) from None
    if not token:
        raise OpenClawError(f"the gateway token {p.token_path} is empty")
    env = {
        "PATH": ":".join(path),
        "HOME": str(p.home),
        "LANG": "C.UTF-8",
        "TZ": "America/New_York",
        "SHELL": "/bin/bash",
        "NODE_OPTIONS": "--max-old-space-size=8192",
        "OPENCLAW_CONFIG_PATH": str(p.config_path),
        "OPENCLAW_STATE_DIR": str(p.state_dir),
        "OPENCLAW_GATEWAY_PORT": str(p.gateway_port),
        "OPENCLAW_GATEWAY_URL": f"ws://127.0.0.1:{p.gateway_port}",
        "OPENCLAW_GATEWAY_TOKEN": token,
    }
    return env, openclaw


def gateway_env(p: Profile) -> dict[str, str]:
    """The whole environment of the gateway and of every ``openclaw`` call for ``p``.

    It is an allowlist: the agent's own tools run in the gateway's environment, so nothing of the parent's, and in
    particular no key, may be in it. Without ``OPENCLAW_GATEWAY_URL`` and ``_PORT`` the CLI would talk to the
    default gateway on :18789.
    """
    return _environment(p)[0]


def installed_version(env: Mapping[str, str], program: Path | None = None) -> str:
    """What ``openclaw --version`` prints when run in ``env``.

    ``program`` is the executable to ask; without it, the first ``openclaw`` on the PATH of ``env``. Whoever launches
    OpenClaw passes the path it launches, so that the version checked is the version run.
    """
    program = program or _find("openclaw", env.get("PATH"))
    try:
        done = _run([str(program), "--version"], env=env, timeout=VERSION_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise OpenClawError(f"`{program} --version` did not answer within {VERSION_TIMEOUT_S} s") from None
    except OSError as error:
        raise OpenClawError(f"cannot run {program}: {error.strerror or error}") from None
    if done.returncode != 0:
        raise OpenClawError(f"`{program} --version` exited with status {done.returncode}: {_tail(done)}")
    return done.stdout.strip()


def check_version(env: Mapping[str, str], program: Path | None = None) -> None:
    """Raise ``OpenClawError`` unless the OpenClaw that would run (see ``installed_version``) is the pinned version."""
    version = installed_version(env, program)
    if version != OPENCLAW_VERSION:
        raise OpenClawError(
            f"OpenClaw {version!r} is installed, but the experiment runs on OpenClaw {OPENCLAW_VERSION}: its prompts "
            f"and tools are part of the environment. Install openclaw@{OPENCLAW_VERSION}."
        )


# --- the gateway ----------------------------------------------------------------------------------


class Gateway:
    """The ``openclaw gateway`` process of one profile.

    It is started once and kept, and restarted only between episodes, by the caller's decision
    (``restart_if_needed``). It never attaches to a gateway someone else started: the ports are free before
    it starts, and the process that listens on its port afterwards has to be its own.
    """

    def __init__(self, p: Profile) -> None:
        self.profile = p
        self._process: ManagedProcess | None = None
        self._started_at = 0.0

    @property
    def log_path(self) -> Path:
        return self.profile.dir / "gateway.log"

    @property
    def pid(self) -> int | None:
        return None if self._process is None else self._process.pid

    def alive(self) -> bool:
        return self._process is not None and self._process.alive()

    def start(self) -> None:
        p = self.profile
        if self.alive():
            raise OpenClawError(f"the gateway of {p.dir.name} is already running (pid {self.pid})")
        problems = check_profile(p)
        if problems:
            raise OpenClawError(f"the OpenClaw profile {p.dir} is not ready:\n  - " + "\n  - ".join(problems))
        env, openclaw = _environment(p)
        check_version(env, openclaw)
        require_free(range(p.gateway_port, p.gateway_port + GATEWAY_SPAN))
        process = ManagedProcess(
            f"openclaw gateway {p.dir.name}",
            [openclaw, "gateway", "--port", str(p.gateway_port)],
            env,
            self.log_path,
            cwd=p.dir,
        )
        process.start()
        self._process = process  # from here on stop() reaches it, whatever fails next
        deadline = time.monotonic() + GATEWAY_START_TIMEOUT_S
        try:
            process.wait_for_port(p.gateway_port, GATEWAY_START_TIMEOUT_S)
            # Without the plugin no model request carries its session, so nothing could be trained or evolved. It
            # logs before the port opens, so this only reads the log; what is left of the start budget is its limit.
            process.wait_for_line(PLUGIN_ACTIVE_LINE, max(0.0, deadline - time.monotonic()))
        except BaseException:
            self.stop()
            raise
        self._started_at = time.monotonic()

    def stop(self) -> None:
        if self._process is None:
            return
        self._process.stop()
        self._process = None

    def restart_if_needed(self, max_age_s: float = GATEWAY_MAX_AGE_S) -> str | None:
        """Restart the gateway if it has died, has run for ``max_age_s`` or has logged a heap exhaustion.

        Returns the reason, or ``None`` when it was left alone. A gateway that is left alone is checked once more to
        still hold its port: if something else listens there, or nothing does, this raises and the gateway keeps
        running. Call it between episodes only: a restart during one would cut the turn off.
        """
        process = self._process
        if process is None:
            raise OpenClawError("the gateway has not been started")
        reason = self._reason_to_restart(process, max_age_s)
        if reason is None:
            self._check_port_owner(process)
            return None
        self.stop()
        self.start()
        return reason

    def _check_port_owner(self, process: ManagedProcess) -> None:
        port = self.profile.gateway_port
        try:
            process.wait_for_port(port, PORT_OWNER_CHECK_S)
        except ProcessError as error:
            raise OpenClawError(f"port {port} is not held by the gateway of {self.profile.dir.name}: {error}") from None

    def _reason_to_restart(self, process: ManagedProcess, max_age_s: float) -> str | None:
        if not process.alive():
            return f"the gateway process had exited (status {process.returncode})"
        log = process.log_text()
        found = _OUT_OF_MEMORY.search(log)
        if found:
            line = log[log.rfind("\n", 0, found.start()) + 1 :].split("\n", 1)[0].strip()
            return f"its log reports a fatal error: {line[:300]}"
        age = time.monotonic() - self._started_at
        if age >= max_age_s:
            return f"its age ({age / 3600:.1f} h) has reached {max_age_s / 3600:g} h"
        return None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.stop()


# --- sessions and transcripts ----------------------------------------------------------------------


def delete_session(p: Profile) -> None:
    """Reset the agent's session, so that the next turn starts from an empty context.

    Every turn goes to the same session whatever the CLI is told, so without the reset an episode would read the
    one before it. A failed reset raises: it is never something to continue past.
    """
    env, openclaw = _environment(p)
    argv = [str(openclaw), "gateway", "call", "sessions.delete", "--params", json.dumps({"key": p.session_key})]
    try:
        done = _run(argv, env=env, timeout=SESSION_DELETE_TIMEOUT_S, cwd=p.dir)
    except subprocess.TimeoutExpired:
        raise OpenClawError(
            f"sessions.delete for {p.session_key} did not finish within {SESSION_DELETE_TIMEOUT_S} s"
        ) from None
    if done.returncode != 0:
        raise OpenClawError(f"sessions.delete for {p.session_key} failed (exit {done.returncode}): {_tail(done)}")


def archive_deleted_transcripts(p: Profile, archive: Path) -> list[str]:
    """Move the transcripts of deleted sessions out of the profile; returns the names moved.

    ``sessions.delete`` only renames a transcript to ``<id>.jsonl.deleted.<time>``, and the agent can read the state
    directory with its tools, so an agent could find every earlier episode there. Each move is logged in
    ``moves.jsonl`` in ``archive``; nothing is deleted or overwritten.
    """
    destination = Path(archive)
    resolved = destination.resolve()
    for guarded, what in ((p.dir, "the OpenClaw profile"), (p.workspace, "the agent's workspace")):
        root = guarded.resolve()
        if resolved == root or root in resolved.parents:
            raise OpenClawError(
                f"the transcript archive {destination} is inside {what} {guarded}, where the agent can read it"
            )
    sessions = p.sessions_dir
    sources = sorted(sessions.glob("*.jsonl.deleted.*"))
    present = [source.name for source in sources if (destination / source.name).exists()]
    if present:
        raise OpenClawError(f"{present[0]} is already in the archive {destination}; it is not overwritten")
    destination.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    try:
        for source in sources:
            shutil.move(str(source), str(destination / source.name))
            moved.append(source.name)
    finally:
        if moved:
            entry = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "from": str(sessions), "files": moved}
            with (destination / "moves.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(entry) + "\n")
    left = sorted(path.name for path in sessions.glob("*.jsonl.deleted.*"))
    if left:
        raise OpenClawError(f"deleted transcripts are still in {sessions}: {left[:5]}")
    return moved


def _transcript(p: Profile, session_id: str) -> tuple[Path, list[tuple[int, dict[str, Any]]]]:
    """The entries of a session's transcript, with their line numbers; the deleted copy if there is no live one."""
    if not _SESSION_ID.fullmatch(session_id):
        raise ValueError(f"{session_id!r} is not a session id")
    sessions = p.sessions_dir
    candidates = [sessions / f"{session_id}.jsonl", *sorted(sessions.glob(f"{session_id}.jsonl.deleted.*"))]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise OpenClawError(f"no transcript for session {session_id} under {sessions}")
    entries: list[tuple[int, dict[str, Any]]] = []
    # Split on newlines only: a JSON string may hold characters that str.splitlines() would break a line at.
    for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").split("\n"), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            raise OpenClawError(f"{path.name} line {number} is not JSON: {line[:200]!r}") from None
        if not isinstance(entry, dict):
            raise OpenClawError(f"{path.name} line {number} is not a JSON object: {line[:200]!r}")
        entries.append((number, entry))
    return path, entries


def _content_calls(message: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [item for item in content if isinstance(item, dict) and item.get("type") == "toolCall"]


def transcript_tool_calls(p: Profile, session_id: str) -> list[dict[str, Any]]:
    """The tool calls the agent made in the session, in order, as ``{"name", "arguments"}``."""
    _, entries = _transcript(p, session_id)
    calls: list[dict[str, Any]] = []
    for _, entry in entries:
        message = entry.get("message")
        if isinstance(message, dict) and message.get("role") == "assistant":
            calls.extend(
                {"name": call.get("name"), "arguments": call.get("arguments")} for call in _content_calls(message)
            )
    return calls


def _epoch(path: Path, number: int, stamp: Any) -> float:
    if isinstance(stamp, str):
        try:
            return datetime.fromisoformat(stamp).timestamp()
        except ValueError:
            pass
    raise OpenClawError(f"{path.name} line {number} has a timestamp that is not a time: {stamp!r}")


def transcript_timeline(p: Profile, session_id: str) -> list[tuple[float, str]]:
    """When the user spoke and when the agent called tools, as ``(epoch seconds, "user" | "tool_call")``.

    A message of the agent that carries several tool calls is one ``"tool_call"`` entry. A turn is judged to have
    looped, rather than stalled, from the gaps in this timeline.
    """
    path, entries = _transcript(p, session_id)
    marks: list[tuple[float, str]] = []
    for number, entry in entries:
        message = entry.get("message")
        if entry.get("type") != "message" or not isinstance(message, dict):
            continue
        if message.get("role") == "user":
            kind = "user"
        elif message.get("role") == "assistant" and _content_calls(message):
            kind = "tool_call"
        else:
            continue
        marks.append((_epoch(path, number, entry.get("timestamp")), kind))
    return marks


# --- one turn --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnResult:
    """What one turn came to.

    ``status`` is ``agent_error`` when the CLI failed, did not print what a gateway turn prints, or ran the agent
    itself because the gateway failed; ``timeout`` when it did not return in time. ``limit_reached`` is set on an
    ``ok`` turn that took as long as the limit it was given: OpenClaw ends such a turn itself, with a notice as the
    reply, and the notice is not an answer.
    """

    status: Literal["ok", "agent_error", "timeout"]
    reply: str
    session_id: str | None
    elapsed: float
    error: str | None
    limit_reached: bool = False


def _unexpected(what: str, elapsed: float) -> TurnResult:
    return TurnResult("agent_error", "", None, elapsed, f"unexpected JSON from the CLI: {what}")


def _parse_turn(done: subprocess.CompletedProcess[str], elapsed: float, timeout_s: int) -> TurnResult:
    if done.returncode != 0:
        return TurnResult("agent_error", "", None, elapsed, f"exit {done.returncode}: {_tail(done)}")
    if _FALLBACK in done.stderr:
        error = f"the gateway could not run the turn ({_FALLBACK}): {done.stderr[-_ERROR_TAIL:]}"
        return TurnResult("agent_error", "", None, elapsed, error)
    try:
        data = json.loads(done.stdout)
    except ValueError:
        return TurnResult("agent_error", "", None, elapsed, f"non-JSON output: {done.stdout[:_OUTPUT_HEAD]!r}")
    if not isinstance(data, dict):
        return _unexpected(f"not an object: {done.stdout[:_OUTPUT_HEAD]!r}", elapsed)
    result = data.get("result") or {}
    if not isinstance(result, dict) or not isinstance(result.get("payloads") or [], list):
        return _unexpected("result.payloads is not a list", elapsed)
    payloads = result.get("payloads") or []
    if not all(isinstance(payload, dict) for payload in payloads):
        return _unexpected("a payload is not an object", elapsed)
    texts = []
    for payload in payloads:
        text = payload.get("text") or ""
        if not isinstance(text, str):
            return _unexpected("a payload's text is not a string", elapsed)
        if text.strip():
            texts.append(text.strip())
    reply = "\n\n".join(texts)
    meta = result.get("meta") or {}
    agent_meta = (meta.get("agentMeta") or {}) if isinstance(meta, dict) else None
    if not isinstance(agent_meta, dict) or not isinstance(agent_meta.get("sessionId") or "", str):
        return _unexpected("result.meta.agentMeta is not an object with a text sessionId", elapsed)
    session_id = agent_meta.get("sessionId") or None
    if session_id is None:
        error = "no gateway session id in the reply (CLI embedded-agent fallback?)"
        return TurnResult("agent_error", reply, None, elapsed, error)
    status = data.get("status") or "ok"
    if status != "ok":
        return TurnResult("agent_error", reply, session_id, elapsed, f"agent status {status!r}")
    return TurnResult("ok", reply, session_id, elapsed, None, limit_reached=elapsed >= timeout_s)


def run_turn(p: Profile, message: str, timeout_s: int) -> TurnResult:
    """Send ``message`` to the agent and wait for its turn to end.

    OpenClaw is told to end the turn after ``timeout_s`` seconds; the CLI call itself is cut off ``SUBPROCESS_SLACK_S``
    later. The turn runs in the agent's workspace. A failure of the agent or of the CLI is a result, not an exception.
    """
    if timeout_s < 1:
        raise ValueError(f"timeout_s must be at least 1, not {timeout_s}: OpenClaw reads 0 as no limit at all")
    if not p.workspace.is_dir():
        raise OpenClawError(f"the agent's workspace {p.workspace} does not exist; build it before the first turn")
    env, openclaw = _environment(p)
    argv = [str(openclaw), "agent", "--agent", p.agent_id, "--json", "--timeout", str(timeout_s), "-m", message]
    limit = timeout_s + SUBPROCESS_SLACK_S
    started = time.monotonic()
    try:
        done = _run(argv, env=env, timeout=limit, cwd=p.workspace)
    except subprocess.TimeoutExpired:
        return TurnResult("timeout", "", None, round(time.monotonic() - started, 1), f"no result within {limit:g} s")
    return _parse_turn(done, round(time.monotonic() - started, 1), timeout_s)
