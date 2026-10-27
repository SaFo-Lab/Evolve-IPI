"""One agent turn, the per-episode session reset, and reading what the gateway wrote about a session."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fake_cli import GATEWAY_PORT, FakeCli

from evolve_ipi.agent import openclaw
from evolve_ipi.agent.openclaw import (
    OpenClawError,
    Profile,
    archive_deleted_transcripts,
    delete_session,
    gateway_env,
    run_turn,
    transcript_timeline,
    transcript_tool_calls,
    write_profile,
)
from evolve_ipi.paths import RuntimePaths

SESSION = "11111111-2222-4333-8444-555555555555"
# What the CLI says on stderr when it could not reach the gateway and ran the agent inside itself.
FALLBACK = "Gateway agent failed; falling back to embedded: Error: gateway closed (1006)\n"


@pytest.fixture
def ready(profile: Profile, fake_cli: FakeCli) -> Profile:
    """A written profile with the stub CLI on PATH."""
    write_profile(profile)
    return profile


# --- run_turn: what the CLI printed becomes a result ---------------------------------------------


def plan_ok(cli: FakeCli) -> None:
    cli.agent_ok(payloads=["  I will look.  ", "", None, {"mediaUrl": "x"}, "It says hello.\n"], session_id=SESSION)


def plan_nonzero_exit(cli: FakeCli) -> None:
    cli.plan(agent={"exit": 3, "stderr": "Error: no such agent\n", "stdout": "ignored"})


def plan_not_json(cli: FakeCli) -> None:
    cli.plan(agent={"stdout": "Config warnings:\n{}\n"})


def plan_no_session_id(cli: FakeCli) -> None:
    cli.agent_ok(payloads=["Done."], session_id=None)


def plan_status_not_ok(cli: FakeCli) -> None:
    cli.agent_ok(payloads=["partial"], session_id=SESSION, status="error")


def plan_embedded_fallback(cli: FakeCli) -> None:
    cli.agent_ok(payloads=["Done."], session_id=SESSION, stderr=FALLBACK)  # the reply alone would look fine


def plan_never_returns(cli: FakeCli) -> None:
    cli.plan(agent={"sleep": 60})


# What OpenClaw replies, with status ok, when its own limit ends the turn.
NOTICE = "Request timed out before a response was generated."


def plan_ends_at_the_limit(cli: FakeCli) -> None:
    cli.agent_ok(payloads=[NOTICE], session_id=SESSION, sleep=1.1)


# plan, the timeout passed to run_turn, how long after that timeout a turn is cut off, and what the result must be
# apart from its elapsed time: (status, reply, session id, error, limit_reached)
PARSING = [
    pytest.param(
        plan_ok, 60, 60, ("ok", "I will look.\n\nIt says hello.", SESSION, None, False), id="ok-joins-the-text-payloads"
    ),
    pytest.param(
        plan_nonzero_exit, 60, 60, ("agent_error", "", None, "exit 3: Error: no such agent\n", False), id="nonzero-exit"
    ),
    pytest.param(
        plan_not_json,
        60,
        60,
        ("agent_error", "", None, "non-JSON output: 'Config warnings:\\n{}\\n'", False),
        id="stdout-is-not-json",
    ),
    pytest.param(
        plan_no_session_id,
        60,
        60,
        ("agent_error", "Done.", None, "no gateway session id in the reply (CLI embedded-agent fallback?)", False),
        id="no-session-id",
    ),
    pytest.param(
        plan_status_not_ok,
        60,
        60,
        ("agent_error", "partial", SESSION, "agent status 'error'", False),
        id="status-not-ok",
    ),
    pytest.param(
        plan_embedded_fallback,
        60,
        60,
        ("agent_error", "", None, f"the gateway could not run the turn (falling back to embedded): {FALLBACK}", False),
        id="embedded-fallback-on-stderr",
    ),
    # a turn that never returns is cut off at 1.5 s, not at 61 s
    pytest.param(
        plan_never_returns, 1, 0.5, ("timeout", "", None, "no result within 1.5 s", False), id="subprocess-limit"
    ),
    # the stub needs about 1.1 s and a start: the limit it is cut off at leaves room for a slow machine
    pytest.param(
        plan_ends_at_the_limit, 1, 20, ("ok", NOTICE, SESSION, None, True), id="elapsed-at-the-limit-is-flagged"
    ),
]


@pytest.mark.parametrize(("plan", "timeout_s", "slack_s", "expected"), PARSING)
def test_run_turn_parsing(
    plan: Callable[[FakeCli], None],
    timeout_s: int,
    slack_s: float,
    expected: tuple[str, str, str | None, str | None, bool],
    ready: Profile,
    fake_cli: FakeCli,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openclaw, "SUBPROCESS_SLACK_S", slack_s)
    plan(fake_cli)
    result = run_turn(ready, "hello", timeout_s)
    assert (result.status, result.reply, result.session_id, result.error, result.limit_reached) == expected
    if result.status == "timeout":
        assert 1.4 <= result.elapsed < 10
    if result.limit_reached:
        assert result.elapsed >= timeout_s


def test_run_turn_a_missing_status_means_ok(ready: Profile, fake_cli: FakeCli) -> None:
    reply = {"result": {"payloads": [{"text": "hi"}], "meta": {"agentMeta": {"sessionId": SESSION}}}}
    fake_cli.plan(agent={"stdout": json.dumps(reply)})
    result = run_turn(ready, "hello", 60)
    assert (result.status, result.reply, result.session_id) == ("ok", "hi", SESSION)


def test_run_turn_an_empty_reply_is_still_an_ok_turn(ready: Profile, fake_cli: FakeCli) -> None:
    fake_cli.agent_ok(payloads=[], session_id=SESSION)
    result = run_turn(ready, "hello", 60)
    assert (result.status, result.reply, result.error) == ("ok", "", None)


def test_run_turn_a_failure_with_no_stderr_shows_the_end_of_stdout(ready: Profile, fake_cli: FakeCli) -> None:
    fake_cli.plan(agent={"exit": 1, "stdout": "x" * 3000 + "THE END"})
    result = run_turn(ready, "hello", 60)
    assert result.error == "exit 1: " + ("x" * 3000 + "THE END")[-2000:]


@pytest.mark.parametrize(
    "stdout",
    [
        "[1, 2]",
        "null",
        '{"result": []}',
        '{"result": {"payloads": "text"}}',
        '{"result": {"payloads": ["text"]}}',
        '{"result": {"payloads": [], "meta": {"agentMeta": "x"}}}',
    ],
)
def test_run_turn_json_of_another_shape_is_an_agent_error(ready: Profile, fake_cli: FakeCli, stdout: str) -> None:
    fake_cli.plan(agent={"stdout": stdout})
    result = run_turn(ready, "hello", 60)
    assert result.status == "agent_error"
    assert result.error and ("JSON" in result.error or "gateway session id" in result.error)


def test_run_turn_the_subprocess_limit_is_the_timeout_plus_sixty_seconds(
    ready: Profile, fake_cli: FakeCli, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert openclaw.SUBPROCESS_SLACK_S == 60
    seen: list[Any] = []
    real_run = openclaw.subprocess.run

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs["timeout"])
        return real_run(*args, **kwargs)

    monkeypatch.setattr(openclaw.subprocess, "run", spy)
    fake_cli.agent_ok(session_id=SESSION)
    run_turn(ready, "hello", 300)
    assert seen == [360]


@pytest.mark.parametrize(("elapsed", "flagged"), [(299.9, False), (300.0, True), (300.1, True), (359.0, True)])
def test_run_turn_flags_a_turn_that_took_the_limit_or_longer(
    elapsed: float, flagged: bool, ready: Profile, fake_cli: FakeCli, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_cli.agent_ok(session_id=SESSION)
    ticks = iter([1000.0, 1000.0 + elapsed])  # the clock of this module only: subprocess keeps its own
    monkeypatch.setattr(openclaw, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    result = run_turn(ready, "hello", 300)
    assert result.elapsed == elapsed
    assert result.limit_reached is flagged


def test_run_turn_flags_the_limit_for_every_task(
    make_profile: Callable[..., Profile], fake_cli: FakeCli, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One rule for the file task and the service tasks, whatever the line."""
    fake_cli.agent_ok(session_id=SESSION)
    ticks = iter([0.0, 600.0] * 4)  # a clock of this module only: start and end of each of the four turns
    monkeypatch.setattr(openclaw, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    for line, kind in (("rl", "file"), ("rl", "service"), ("skills", "file"), ("skills", "service")):
        p = make_profile(line, kind)
        p.workspace.mkdir(parents=True)
        write_profile(p)
        assert run_turn(p, "hello", 600).limit_reached is True


def test_run_turn_command_line_working_directory_and_environment(
    ready: Profile, fake_cli: FakeCli, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_cli.agent_ok(session_id=SESSION)
    for name in ("TINKER_API_KEY", "EVOLVER_API_KEY", "OPENAI_API_KEY", "HF_TOKEN"):
        monkeypatch.setenv(name, f"secret-of-{name}")
    run_turn(ready, "Please read ipi_docs/a.txt - and tell me.", 300)
    (call,) = fake_cli.calls()
    assert call["args"] == [
        "agent",
        "--agent",
        "rl",
        "--json",
        "--timeout",
        "300",
        "-m",
        "Please read ipi_docs/a.txt - and tell me.",
    ]
    assert Path(call["cwd"]).resolve() == ready.workspace.resolve()
    expected = gateway_env(ready)
    # A shell between the process and the stub adds its own bookkeeping variables.
    assert {k: v for k, v in call["env"].items() if k not in {"PWD", "OLDPWD", "SHLVL", "_"}} == expected
    assert not any("secret-of-" in value for value in call["env"].values())


def test_run_turn_uses_the_agent_of_the_line(make_profile: Callable[..., Profile], fake_cli: FakeCli) -> None:
    p = make_profile("skills", "service")
    p.workspace.mkdir(parents=True)
    write_profile(p)
    fake_cli.agent_ok(session_id=SESSION)
    run_turn(p, "hello", 600)
    (call,) = fake_cli.calls()
    assert call["args"][:3] == ["agent", "--agent", "mc"]
    assert call["env"]["OPENCLAW_GATEWAY_URL"] == f"ws://127.0.0.1:{GATEWAY_PORT}"


def test_run_turn_needs_a_workspace_and_a_positive_timeout(ready: Profile, fake_cli: FakeCli) -> None:
    with pytest.raises(ValueError, match="timeout"):
        run_turn(ready, "hello", 0)  # --timeout 0 means no limit for OpenClaw
    shutil.rmtree(ready.workspace)
    with pytest.raises(OpenClawError, match="workspace"):
        run_turn(ready, "hello", 60)
    assert fake_cli.calls() == []


# --- delete_session ---------------------------------------------------------------------------


def test_delete_session_resets_the_main_session_of_the_agent(ready: Profile, fake_cli: FakeCli) -> None:
    assert delete_session(ready) is None
    (call,) = fake_cli.calls()
    assert call["args"] == ["gateway", "call", "sessions.delete", "--params", '{"key": "agent:rl:main"}']
    assert call["env"]["OPENCLAW_GATEWAY_TOKEN"] == ready.token_path.read_text().strip()
    assert call["env"]["OPENCLAW_GATEWAY_URL"] == f"ws://127.0.0.1:{GATEWAY_PORT}"


def test_delete_session_a_failed_reset_raises(ready: Profile, fake_cli: FakeCli) -> None:
    fake_cli.plan(delete={"exit": 1, "stderr": "gateway closed (1006)\n"})
    with pytest.raises(OpenClawError, match=r"sessions\.delete.*exit 1.*gateway closed \(1006\)"):
        delete_session(ready)


def test_delete_session_that_does_not_finish_raises(
    ready: Profile, fake_cli: FakeCli, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert openclaw.SESSION_DELETE_TIMEOUT_S == 120
    monkeypatch.setattr(openclaw, "SESSION_DELETE_TIMEOUT_S", 1)
    fake_cli.plan(delete={"sleep": 60})
    with pytest.raises(OpenClawError, match=r"sessions\.delete.*within 1 s"):
        delete_session(ready)


# --- transcripts ------------------------------------------------------------------------------


def at(second: float) -> str:
    moment = datetime(2026, 10, 5, 22, 31, tzinfo=UTC) + timedelta(seconds=second)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def epoch(second: float) -> float:
    return datetime.fromisoformat(at(second)).timestamp()


def message(second: float, role: str, content: Any, **extra: Any) -> dict[str, Any]:
    return {
        "type": "message",
        "id": f"m{second}",
        "timestamp": at(second),
        "message": {"role": role, "content": content, **extra},
    }


TRANSCRIPT: list[dict[str, Any]] = [
    {"type": "session", "version": 3, "id": SESSION, "timestamp": at(0), "cwd": "/workspace"},
    {"type": "model_change", "id": "a1", "timestamp": at(0), "provider": "rl", "modelId": "policy"},
    {"type": "thinking_level_change", "id": "a2", "timestamp": at(0), "thinkingLevel": "low"},
    message(1, "user", [{"type": "text", "text": "Read a.txt"}]),
    message(
        4.5,
        "assistant",
        [
            {"type": "thinking", "thinking": "I should read it."},
            {"type": "toolCall", "id": "c1", "name": "read", "arguments": {"path": "a.txt"}},
            {"type": "toolCall", "id": "c2", "name": "exec", "arguments": {"command": "ls"}},
        ],
    ),
    message(5, "toolResult", [{"type": "text", "text": "hello"}], toolCallId="c1", toolName="read"),
    message(9.25, "assistant", [{"type": "text", "text": "It says hello."}]),
    message(20, "user", [{"type": "text", "text": "Great, thank you!"}]),
    message(
        22,
        "assistant",
        [{"type": "toolCall", "id": "c3", "name": "write", "arguments": {"path": "b.txt", "content": "x"}}],
    ),
]


def write_transcript(p: Profile, name: str, entries: list[dict[str, Any]] = TRANSCRIPT) -> Path:
    p.sessions_dir.mkdir(parents=True, exist_ok=True)
    path = p.sessions_dir / name
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
    return path


def test_the_sessions_directory_is_where_the_gateway_keeps_them(profile: Profile) -> None:
    assert profile.sessions_dir == profile.dir / "state" / "agents" / "rl" / "sessions"


def test_transcript_tool_calls(profile: Profile) -> None:
    write_transcript(profile, f"{SESSION}.jsonl")
    assert transcript_tool_calls(profile, SESSION) == [
        {"name": "read", "arguments": {"path": "a.txt"}},
        {"name": "exec", "arguments": {"command": "ls"}},
        {"name": "write", "arguments": {"path": "b.txt", "content": "x"}},
    ]


def test_transcript_tool_calls_of_a_session_without_any(profile: Profile) -> None:
    write_transcript(profile, f"{SESSION}.jsonl", TRANSCRIPT[:4] + [TRANSCRIPT[6]])
    assert transcript_tool_calls(profile, SESSION) == []


def test_transcript_falls_back_to_the_deleted_transcript(profile: Profile) -> None:
    write_transcript(profile, f"{SESSION}.jsonl.deleted.2026-10-05T22-35-00.000Z")
    assert len(transcript_tool_calls(profile, SESSION)) == 3
    assert len(transcript_timeline(profile, SESSION)) == 4


def test_the_live_transcript_wins_over_a_deleted_one(profile: Profile) -> None:
    write_transcript(profile, f"{SESSION}.jsonl")
    write_transcript(profile, f"{SESSION}.jsonl.deleted.2026-10-05T22-35-00.000Z", TRANSCRIPT[:4])
    assert len(transcript_tool_calls(profile, SESSION)) == 3


def test_a_transcript_of_another_session_is_not_taken(profile: Profile) -> None:
    write_transcript(profile, "99999999-2222-4333-8444-555555555555.jsonl")
    write_transcript(profile, f"{SESSION}0.jsonl")
    with pytest.raises(OpenClawError, match=SESSION):
        transcript_tool_calls(profile, SESSION)


def test_transcript_timeline_is_the_user_messages_and_the_messages_with_tool_calls(profile: Profile) -> None:
    write_transcript(profile, f"{SESSION}.jsonl")
    assert transcript_timeline(profile, SESSION) == [
        (epoch(1), "user"),
        (epoch(4.5), "tool_call"),  # one entry for the message, although it carries two calls
        (epoch(20), "user"),
        (epoch(22), "tool_call"),
    ]


def test_a_missing_transcript_raises(profile: Profile) -> None:
    with pytest.raises(OpenClawError, match="no transcript"):
        transcript_tool_calls(profile, SESSION)
    with pytest.raises(OpenClawError, match="no transcript"):
        transcript_timeline(profile, SESSION)


def test_a_transcript_line_that_is_not_json_raises_naming_the_line(profile: Profile) -> None:
    path = write_transcript(profile, f"{SESSION}.jsonl")
    with path.open("a") as handle:
        handle.write('{"type": "message", "timest\n')
    for reader in (transcript_tool_calls, transcript_timeline):
        with pytest.raises(OpenClawError, match=rf"{path.name} line {len(TRANSCRIPT) + 1}"):
            reader(profile, SESSION)


def test_a_timestamp_that_is_not_a_time_raises(profile: Profile) -> None:
    broken = [*TRANSCRIPT[:3], {**message(1, "user", []), "timestamp": "yesterday"}]
    write_transcript(profile, f"{SESSION}.jsonl", broken)
    with pytest.raises(OpenClawError, match="timestamp"):
        transcript_timeline(profile, SESSION)


@pytest.mark.parametrize("session_id", ["", "..", "a/b", "../x", "a b", "x*"])
def test_a_session_id_must_be_a_plain_name(profile: Profile, session_id: str) -> None:
    for reader in (transcript_tool_calls, transcript_timeline):
        with pytest.raises(ValueError, match="session id"):
            reader(profile, session_id)


# --- archiving the deleted transcripts ---------------------------------------------------------------


def test_archive_moves_deleted_transcripts_out_of_the_profile(profile: Profile, paths: RuntimePaths) -> None:
    live = write_transcript(profile, "live-session.jsonl")
    deleted = [
        write_transcript(profile, "s-a.jsonl.deleted.2026-10-05T22-35-00.000Z"),
        write_transcript(profile, "s-b.jsonl.deleted.2026-10-05T22-36-00.000Z"),
    ]
    archive = paths.transcript_archive("rl", "file")
    moved = archive_deleted_transcripts(profile, archive)
    assert moved == [path.name for path in deleted]
    assert sorted(p.name for p in archive.glob("*.jsonl.deleted.*")) == moved
    assert not list(profile.sessions_dir.glob("*.deleted.*"))
    assert live.is_file()
    (entry,) = [json.loads(line) for line in (archive / "moves.jsonl").read_text().splitlines()]
    assert entry["files"] == moved and entry["from"] == str(profile.sessions_dir)
    assert datetime.strptime(entry["at"], "%Y-%m-%dT%H:%M:%S%z")
    # A later call appends to the log; a call with nothing to move leaves it alone.
    write_transcript(profile, "s-c.jsonl.deleted.2026-10-05T22-37-00.000Z")
    assert archive_deleted_transcripts(profile, archive) == ["s-c.jsonl.deleted.2026-10-05T22-37-00.000Z"]
    assert archive_deleted_transcripts(profile, archive) == []
    assert len((archive / "moves.jsonl").read_text().splitlines()) == 2


def test_archive_with_no_sessions_yet_is_a_no_op(profile: Profile, paths: RuntimePaths) -> None:
    archive = paths.transcript_archive("rl", "file")
    assert archive_deleted_transcripts(profile, archive) == []
    assert not (archive / "moves.jsonl").exists()


def test_archive_refuses_inside_profile(profile: Profile) -> None:
    write_transcript(profile, "s-a.jsonl.deleted.2026-10-05T22-35-00.000Z")
    for inside in (
        profile.dir,
        profile.dir / "archive",
        profile.sessions_dir,
        profile.sessions_dir / "old",
        profile.state_dir,
    ):
        with pytest.raises(OpenClawError, match="inside the OpenClaw profile"):
            archive_deleted_transcripts(profile, inside)
    assert len(list(profile.sessions_dir.glob("*.deleted.*"))) == 1, "nothing was moved"


def test_archive_refuses_inside_the_workspace(profile: Profile) -> None:
    """The agent reads its workspace, so an archive there would hand it every earlier episode."""
    write_transcript(profile, "s-a.jsonl.deleted.2026-10-05T22-35-00.000Z")
    with pytest.raises(OpenClawError, match="workspace"):
        archive_deleted_transcripts(profile, profile.workspace / "old")


def test_archive_refuses_a_link_that_leads_into_the_profile(profile: Profile, tmp_path: Path) -> None:
    link = tmp_path / "archive-link"
    link.symlink_to(profile.dir)
    with pytest.raises(OpenClawError, match="inside the OpenClaw profile"):
        archive_deleted_transcripts(profile, link / "sub")


def test_archive_raises_if_a_deleted_transcript_is_still_there(
    profile: Profile, paths: RuntimePaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_transcript(profile, "s-a.jsonl.deleted.2026-10-05T22-35-00.000Z")
    monkeypatch.setattr(openclaw.shutil, "move", lambda src, dst: None)
    with pytest.raises(OpenClawError, match=r"still in .*s-a\.jsonl\.deleted"):
        archive_deleted_transcripts(profile, paths.transcript_archive("rl", "file"))


def test_archive_never_overwrites_an_archived_transcript(profile: Profile, paths: RuntimePaths) -> None:
    name = "s-a.jsonl.deleted.2026-10-05T22-35-00.000Z"
    archive = paths.transcript_archive("rl", "file")
    archive.mkdir(parents=True)
    (archive / name).write_text("earlier")
    write_transcript(profile, name)
    with pytest.raises(OpenClawError, match="already in the archive"):
        archive_deleted_transcripts(profile, archive)
    assert (archive / name).read_text() == "earlier"
    assert (profile.sessions_dir / name).is_file()
