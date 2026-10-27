"""OpenClaw itself: a profile written by this package, a gateway started on it, and turns answered by a scripted model.

These tests run only where OpenClaw 2026.3.7 is installed. Everything they start lives in a temporary directory, on
ports found free when the test runs: they never touch ``~/.openclaw*``, ``~/.metaclaw``, the real HOME or the ports a
real run uses, and each test checks that its ports are free again at the end.
"""

from __future__ import annotations

import json
import os
import re
import socket
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from evolve_ipi.agent.openclaw import (
    GATEWAY_SPAN,
    OPENCLAW_VERSION,
    Gateway,
    OpenClawError,
    Profile,
    archive_deleted_transcripts,
    delete_session,
    gateway_env,
    installed_version,
    profile_for,
    run_turn,
    transcript_timeline,
    transcript_tool_calls,
    write_profile,
)
from evolve_ipi.environment import workspace
from evolve_ipi.models import fake
from evolve_ipi.paths import Kind, Line, RuntimePaths
from evolve_ipi.procs import PortInUse, port_in_use

pytestmark = [pytest.mark.openclaw, pytest.mark.usefixtures("openclaw_installed")]

REAL_PORTS = {18789, 30011, 30012, 30013}  # the ports of a real run; nothing here may use one
REPORT = "The quarterly report is ready."
UUID = re.compile(r"[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}")
# The runtime line of the system prompt. The host, the operating system and node are the machine's; the rest is what
# this package sets up: the agent and model of the line, bash as the shell, and OpenClaw's default thinking level.
FOOTER = re.compile(
    r"Runtime: agent=(?P<agent>\S+) \| host=[^|]+ \| os=[^|]+ \| node=v\d+\.\d+\.\d+ \| model=(?P<model>\S+) \| "
    r"default_model=(?P<default>\S+) \| shell=bash \| channel=webchat \| capabilities=none \| thinking=low"
)
USER_TEXT = re.compile(r"\[\w{3} \d{4}-\d\d-\d\d \d\d:\d\d E[SD]T\] (?P<message>.*)", re.DOTALL)

# What OpenClaw itself adds to the environment of the agent's commands, and what the shell adds.
ADDED_BY_OPENCLAW = {"NODE_NO_WARNINGS", "OPENCLAW_NODE_OPTIONS_READY", "OPENCLAW_PATH_BOOTSTRAPPED", "OPENCLAW_SHELL"}
ADDED_BY_THE_SHELL = {"PWD", "OLDPWD", "SHLVL", "_", "__CF_USER_TEXT_ENCODING"}
# Node itself sets this one in its own environment on Linux, where it turns off libuv's use of io_uring; the commands it
# starts inherit it.
ADDED_BY_NODE = {"UV_USE_IO_URING"} if sys.platform.startswith("linux") else set()


@pytest.fixture(scope="module")
def openclaw_installed(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Skip the module unless ``openclaw --version`` prints the pinned version, asked in a directory of its own."""
    probe = tmp_path_factory.mktemp("openclaw-version")
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
        pytest.skip(f"these tests need OpenClaw {OPENCLAW_VERSION}; {version} is installed")


@dataclass
class Stack:
    """A profile in a temporary directory, the scripted model it talks to, and its gateway."""

    paths: RuntimePaths
    profile: Profile
    model: fake.FakeModel
    gateway: Gateway

    @property
    def span(self) -> range:
        return range(self.profile.gateway_port, self.profile.gateway_port + GATEWAY_SPAN)


@pytest.fixture
def make_stack(
    tmp_path: Path, free_span: Callable[[], int], port_outside: Callable[[int], int]
) -> Iterator[Callable[..., Stack]]:
    stacks: list[Stack] = []

    def make(script: Sequence[dict[str, Any]], line: Line = "rl", kind: Kind = "file", *, start: bool = True) -> Stack:
        home = tmp_path / "home"
        home.mkdir(exist_ok=True)
        paths = RuntimePaths(
            root=tmp_path / "runtime", agent_home=home, outputs=tmp_path / "outputs", metaclaw_python=tmp_path / "py"
        )
        first = free_span()
        model_port = port_outside(first)
        for port in (*range(first, first + GATEWAY_SPAN), model_port):
            assert port not in REAL_PORTS
        p = profile_for(line, kind, paths, first, model_port)
        # The workspace is filled first: OpenClaw sets up one of its own, with a git repository, in an empty one.
        if kind == "file":
            workspace.build(p.workspace, "file", {"ipi_docs/report.txt": REPORT}, None)
        else:
            workspace.build(p.workspace, "service", None, paths.service_state(line, kind))
        write_profile(p)
        stack = Stack(paths, p, fake.FakeModel(script).start(model_port), Gateway(p))
        stacks.append(stack)
        env = gateway_env(p)
        for path in (
            p.dir,
            p.workspace,
            p.home,
            *(Path(env[key]) for key in ("HOME", "OPENCLAW_CONFIG_PATH", "OPENCLAW_STATE_DIR")),
        ):
            assert path.is_relative_to(tmp_path)  # nothing lies in a real OpenClaw or MetaClaw directory
        if start:
            stack.gateway.start()
        return stack

    try:
        yield make
    finally:
        problems: list[str] = []
        for stack in reversed(stacks):
            try:
                stack.gateway.stop()
            finally:
                stack.model.stop()
            busy = [port for port in (*stack.span, stack.model.port) if port is not None and port_in_use(port)]
            if busy:
                problems.append(f"ports still in use after the test: {busy}")
        assert not problems, "; ".join(problems)


def system_prompt(request: dict[str, Any]) -> str:
    first = request["body"]["messages"][0]
    assert first["role"] == "system"
    return first["content"] if isinstance(first["content"], str) else json.dumps(first["content"])


def text_of(message: dict[str, Any]) -> str:
    content = message["content"]
    return content if isinstance(content, str) else "".join(part.get("text", "") for part in content)


# --- a turn ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("line", "kind"), [("rl", "file"), ("skills", "service")])
def test_gateway_integration(line: Line, kind: Kind, make_stack: Callable[..., Stack]) -> None:
    stack = make_stack([{"content": "Hello! How can I help?"}], line, kind)
    p = stack.profile
    assert stack.gateway.alive()
    assert port_in_use(p.gateway_port)

    delete_session(p)  # the reset before an episode; the first one finds no session yet
    result = run_turn(p, "hello", 60)

    assert result.status == "ok" and result.error is None and result.limit_reached is False
    assert result.reply == "Hello! How can I help?"
    assert result.session_id is not None and UUID.fullmatch(result.session_id)
    (request,) = stack.model.requests
    # The rl-training-headers plugin is live: the model server learns the session and that this is a main turn.
    assert (request["session_id"], request["turn_type"]) == (result.session_id, "main")
    assert request["body"]["model"] == p.model_id
    # The agent runs as the line says, and the model is named by its id, not by its label.
    footer = FOOTER.search(system_prompt(request))
    assert footer is not None
    assert footer.group("agent") == p.agent_id
    assert footer.group("model") == footer.group("default") == p.model_ref
    for recorded in stack.model.requests:
        assert p.label not in json.dumps(recorded["body"])
    user = USER_TEXT.fullmatch(text_of(request["body"]["messages"][-1]))
    assert user is not None and user.group("message") == "hello"  # the time prefix is New York time
    assert not (p.workspace / ".git").exists()


def test_a_turn_reads_the_workspace_and_the_transcript_tells_what_happened(make_stack: Callable[..., Stack]) -> None:
    script = [
        {"tool_calls": [{"name": "read", "arguments": {"path": "ipi_docs/report.txt"}}]},
        {"content": "The report says it is ready."},
    ]
    stack = make_stack(script)
    p, model = stack.profile, stack.model
    delete_session(p)

    first = run_turn(p, "Please read ipi_docs/report.txt and tell me what it says.", 120)
    assert (first.status, first.reply) == ("ok", "The report says it is ready.")
    feedback = run_turn(p, "Great, you did what was asked. Thank you!", 120)
    assert feedback.status == "ok" and feedback.session_id == first.session_id  # one session for both turns
    assert first.session_id is not None

    # The tool result that came back to the model is the file's text, read from the agent's workspace.
    tool_messages = [m for r in model.requests for m in r["body"]["messages"] if m["role"] == "tool"]
    assert tool_messages and REPORT in text_of(tool_messages[0])
    assert {r["session_id"] for r in model.requests} == {first.session_id}

    assert transcript_tool_calls(p, first.session_id) == [
        {"name": "read", "arguments": {"path": "ipi_docs/report.txt"}}
    ]
    timeline = transcript_timeline(p, first.session_id)
    assert [kind for _, kind in timeline] == ["user", "tool_call", "user"]
    assert [moment for moment, _ in timeline] == sorted(moment for moment, _ in timeline)
    assert abs(timeline[0][0] - time.time()) < 300

    # The reset gives the next episode an empty context and a session of its own.
    delete_session(p)
    assert len(list(p.sessions_dir.glob(f"{first.session_id}.jsonl.deleted.*"))) == 1
    archive = stack.paths.transcript_archive("rl", "file")
    moved = archive_deleted_transcripts(p, archive)
    assert len(moved) == 1 and moved[0].startswith(f"{first.session_id}.jsonl.deleted.")
    assert not list(p.sessions_dir.glob("*.deleted.*"))
    assert (archive / moved[0]).is_file() and (archive / "moves.jsonl").is_file()
    with pytest.raises(OpenClawError, match="no transcript"):
        transcript_tool_calls(p, first.session_id)  # out of the agent's reach, and out of ours

    second = run_turn(p, "Hello again.", 120)
    assert second.status == "ok" and second.session_id not in (None, first.session_id)
    assert [m["role"] for m in model.requests[-1]["body"]["messages"]] == ["system", "user"]


def test_the_agents_commands_run_in_the_allowlisted_environment(
    make_stack: Callable[..., Stack], monkeypatch: pytest.MonkeyPatch
) -> None:
    secrets = ("TINKER_API_KEY", "EVOLVER_API_KEY", "EVOLVER_MODEL", "EVOLVER_BASE_URL", "OPENAI_API_KEY", "HF_TOKEN")
    for name in secrets:
        monkeypatch.setenv(name, f"secret-of-{name}")
    for name in ("PYTHONPATH", "OPENCLAW_HOME", "OPENCLAW_PROFILE"):
        monkeypatch.setenv(name, "/from/the/parent")
    stack = make_stack([{"tool_calls": [{"name": "exec", "arguments": {"command": "env"}}]}, {"content": "Done."}])
    p = stack.profile
    delete_session(p)
    assert run_turn(p, "Show me the environment you run commands in.", 120).status == "ok"

    output = next(text_of(m) for m in stack.model.requests[1]["body"]["messages"] if m["role"] == "tool")
    seen = dict(line.split("=", 1) for line in output.splitlines() if re.match(r"[A-Za-z_]\w*=", line))
    assert "secret-of-" not in output and "/from/the/parent" not in output
    assert not set(seen) & {*secrets, "PYTHONPATH", "OPENCLAW_HOME", "OPENCLAW_PROFILE"}
    allowed = set(gateway_env(p)) | ADDED_BY_OPENCLAW | ADDED_BY_THE_SHELL | ADDED_BY_NODE
    assert set(seen) <= allowed, f"unexpected variables: {sorted(set(seen) - allowed)}"
    assert seen.get("UV_USE_IO_URING", "0") == "0", "Node turns io_uring off; it does not pass on another value"
    assert seen["HOME"] == str(p.home) and seen["TZ"] == "America/New_York" and seen["SHELL"] == "/bin/bash"
    assert seen["LANG"] == "C.UTF-8" and seen["OPENCLAW_STATE_DIR"] == str(p.state_dir)
    assert Path(seen["PWD"]).resolve() == p.workspace.resolve()


def test_a_turn_without_a_gateway_is_an_error_not_an_answer(make_stack: Callable[..., Stack]) -> None:
    """With no gateway to talk to the CLI runs the agent inside itself, and that run answers and reports a session
    of its own with exit status 0; only its stderr says so."""
    stack = make_stack([{"content": "I am the embedded agent."}], start=False)
    result = run_turn(stack.profile, "hello", 60)
    assert result.status == "agent_error"
    assert result.error is not None and "falling back to embedded" in result.error
    assert len(stack.model.requests) == 1  # the model was asked, and answered: only stderr says it was not a turn


# --- the gateway ---------------------------------------------------------------------------------------


def test_a_port_held_by_another_process_is_refused_and_not_used(
    tmp_path: Path,
    listener: int,
    port_outside: Callable[[int], int],
    make_gateway: Callable[[Profile], Gateway],
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    paths = RuntimePaths(root=tmp_path / "runtime", agent_home=home, outputs=tmp_path, metaclaw_python=tmp_path)
    p = profile_for("rl", "file", paths, listener, port_outside(listener))  # this process holds the gateway's own port
    workspace.build(p.workspace, "file", {"ipi_docs/report.txt": REPORT}, None)
    write_profile(p)
    gateway = make_gateway(p)
    with pytest.raises(PortInUse, match=rf"port {listener} is in use by pid {os.getpid()}"):
        gateway.start()
    assert not gateway.alive()
    with socket.create_connection(("127.0.0.1", listener), timeout=5):
        pass  # the listener was left alone


def test_a_second_gateway_on_the_same_ports_is_refused(
    make_stack: Callable[..., Stack], make_gateway: Callable[[Profile], Gateway]
) -> None:
    stack = make_stack([])
    second = make_gateway(stack.profile)
    with pytest.raises(PortInUse) as raised:
        second.start()
    found = re.search(r"port \d+ is in use by pid (\d+)", str(raised.value))
    assert found is not None
    assert os.getsid(int(found.group(1))) == stack.gateway.pid  # the first gateway, which keeps running
    assert stack.gateway.alive() and not second.alive()


def test_a_gateway_restarted_between_episodes_serves_the_next_one(make_stack: Callable[..., Stack]) -> None:
    stack = make_stack([{"content": "First."}, {"content": "Second."}])
    p, gateway = stack.profile, stack.gateway
    delete_session(p)
    assert run_turn(p, "one", 120).reply == "First."
    old = gateway.pid
    assert gateway.restart_if_needed() is None
    reason = gateway.restart_if_needed(max_age_s=0)
    assert reason is not None and "age" in reason
    assert gateway.alive() and gateway.pid != old
    delete_session(p)
    result = run_turn(p, "two", 120)
    assert (result.status, result.reply) == ("ok", "Second.")
    assert [r["turn_type"] for r in stack.model.requests] == ["main", "main"]  # the plugin loaded again


def test_stopping_the_gateway_frees_every_port_of_its_span(make_stack: Callable[..., Stack]) -> None:
    stack = make_stack([])
    assert port_in_use(stack.profile.gateway_port)
    stack.gateway.stop()
    assert not stack.gateway.alive()
    assert not any(port_in_use(port) for port in stack.span)
