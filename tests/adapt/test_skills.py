"""Skill evolution: the library hash, closing a conversation, the drain, the proxy's start and environment, the guards,
the Inkling-Small audit, and ``adapt_skills`` over a scripted world (no model server, proxy or gateway is started)."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import shutil
import signal
import sys
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest
from metaclaw_helpers import GUARDS, event, write_jsonl
from skills_world import SECRETS, SHARED_AUDIT, Script, World, abort_at, episodes, voids, why

from evolve_ipi import children, thirdparty
from evolve_ipi.adapt import AdaptAbort, skills
from evolve_ipi.agent.anomalies import TrainingPolicy
from evolve_ipi.agent.episode import EpisodeAbort, LeftoverState, RawDumpError
from evolve_ipi.agent.openclaw import OpenClawError
from evolve_ipi.config import ConfigError, Configs, ModelSpec, Secrets, load_configs
from evolve_ipi.paths import RuntimePaths
from evolve_ipi.procs import PortInUse, ProcessError, port_in_use
from evolve_ipi.tasks import load_skill_order, load_task

REPO = Path(__file__).resolve().parents[2]
FAKE_PROXY = Path(__file__).with_name("skills_fake_proxy.py")


@pytest.fixture(scope="module")
def cfgs() -> Configs:
    return load_configs(REPO / "configs")


# --- the library hash ------------------------------------------------------------------------------


def write_skill(library: Path, name: str, text: str) -> None:
    (library / name).mkdir(parents=True, exist_ok=True)
    (library / name / "SKILL.md").write_text(text, encoding="utf-8")


def test_lib_sha_hashes_each_skill_file_with_its_path_in_path_order(tmp_path: Path) -> None:
    library = tmp_path / "library"
    write_skill(library, "a-b", "second\n")
    write_skill(library, "a", "first\n")
    write_skill(library, "b", "third\n")
    (library / "evolution_history.jsonl").write_text('{"num_failures_analyzed": 1}\n', encoding="utf-8")
    (library / "a" / "notes.txt").write_text("not a skill", encoding="utf-8")
    # Path order compares the components: "a" < "a-b". Comparing the strings would put "a-b/SKILL.md" first.
    expected = hashlib.sha256(b"a/SKILL.md\0first\na-b/SKILL.md\0second\nb/SKILL.md\0third\n").hexdigest()
    assert skills.lib_sha(library) == expected
    string_order = hashlib.sha256(b"a-b/SKILL.md\0second\na/SKILL.md\0first\nb/SKILL.md\0third\n").hexdigest()
    assert skills.lib_sha(library) != string_order


def test_lib_sha_of_an_empty_library_and_of_a_nested_skill(tmp_path: Path) -> None:
    library = tmp_path / "library"
    library.mkdir()
    assert skills.lib_sha(library) == hashlib.sha256(b"").hexdigest()
    write_skill(library / "group", "inner", "x")
    assert skills.lib_sha(library) == hashlib.sha256(b"group/inner/SKILL.md\0x").hexdigest()


def test_lib_sha_refuses_a_library_that_does_not_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no skill library"):
        skills.lib_sha(tmp_path / "missing")


def test_lib_sha_changes_with_any_byte_and_any_name(tmp_path: Path) -> None:
    library = tmp_path / "library"
    write_skill(library, "verify-first", "## Verify\n")
    before = skills.lib_sha(library)
    (library / "verify-first" / "SKILL.md").write_text("## Verify \n", encoding="utf-8")
    assert skills.lib_sha(library) != before
    (library / "verify-first" / "SKILL.md").write_text("## Verify\n", encoding="utf-8")
    assert skills.lib_sha(library) == before
    (library / "verify-first").rename(library / "verify-later")
    assert skills.lib_sha(library) != before


# --- the history and the drain ------------------------------------------------------------------------


def test_history_lines_counts_non_blank_lines_and_a_missing_file_as_zero(tmp_path: Path) -> None:
    history = tmp_path / "evolution_history.jsonl"
    assert skills.history_lines(history) == 0
    history.write_text('{"a": 1}\n\n  \n{"b": 2}\n{"c": 3}', encoding="utf-8")
    assert skills.history_lines(history) == 3


class Writer:
    """Writes the history as the evolver would: ``growth[k]`` lines before poll ``k`` (counted from 0)."""

    def __init__(self, path: Path, growth: dict[int, int]) -> None:
        self.path, self.growth, self.polls, self.sleeps = path, growth, 0, []

    def write_for_next_poll(self) -> None:
        for _ in range(self.growth.get(self.polls, 0)):
            with open(self.path, "a", encoding="utf-8") as history:
                history.write("{}\n")

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.polls += 1
        self.write_for_next_poll()


def drain_with(tmp_path: Path, growth: dict[int, int], max_polls: int = 300) -> tuple[int, Writer]:
    writer = Writer(tmp_path / "evolution_history.jsonl", growth)
    writer.write_for_next_poll()
    count = skills.drain(writer.path, 12, poll_s=3, stable_polls=5, max_polls=max_polls, sleep=writer.sleep)
    return count, writer


def test_drain_stops_after_five_unchanged_polls_once_twelve_landed(tmp_path: Path) -> None:
    count, writer = drain_with(tmp_path, {0: 10, 2: 2})
    assert count == 12
    # polls 0..1 see 10, poll 2 sees 12, polls 3..7 see it unchanged: the eighth poll ends the drain without a sleep
    assert writer.sleeps == [3] * 7


def test_drain_waits_out_its_polls_when_too_few_land(tmp_path: Path) -> None:
    count, writer = drain_with(tmp_path, {0: 11}, max_polls=20)
    assert count == 11
    assert len(writer.sleeps) == 20  # it sleeps after every poll that does not end it, the last one too


def test_drain_returns_an_overshoot_for_the_caller_to_refuse(tmp_path: Path) -> None:
    count, writer = drain_with(tmp_path, {0: 12, 1: 1})
    assert count == 13
    assert len(writer.sleeps) == 6


def test_drain_checks_on_every_poll_and_stops_at_the_first_failure(tmp_path: Path) -> None:
    writer = Writer(tmp_path / "evolution_history.jsonl", {0: 11})
    writer.write_for_next_poll()
    checked: list[int] = []

    def check() -> None:
        checked.append(len(writer.sleeps))
        if len(checked) == 3:
            raise AdaptAbort("the proxy exited")

    with pytest.raises(AdaptAbort, match="the proxy exited"):
        skills.drain(writer.path, 12, poll_s=3, stable_polls=5, max_polls=300, sleep=writer.sleep, check=check)
    assert checked == [0, 1, 2]  # before each poll, the third one failing
    assert len(writer.sleeps) == 2


def test_drain_on_a_missing_history_counts_zero(tmp_path: Path) -> None:
    count, writer = drain_with(tmp_path, {}, max_polls=3)
    assert (count, len(writer.sleeps)) == (0, 3)


def test_the_configured_drain_is_the_experiments(cfgs: Configs) -> None:
    spec = cfgs.skills
    assert (spec.drain_poll_s, spec.drain_stable_polls, spec.drain_max_polls, spec.conversations) == (3, 5, 300, 12)


# --- closing a conversation --------------------------------------------------------------------------


class Stub:
    """An HTTP server that records every request and answers with ``status``."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.status = 200
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                stub.requests.append(
                    {"method": "POST", "path": self.path, "headers": dict(self.headers.items()), "body": body}
                )
                self.send_response(stub.status)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub() -> Iterator[Stub]:
    server = Stub()
    try:
        yield server
    finally:
        server.close()


def test_close_session_sends_the_exact_request(stub: Stub, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []
    real_post = httpx.post

    def recording_post(*args: Any, **kwargs: Any) -> httpx.Response:
        seen.append(kwargs)
        return real_post(*args, **kwargs)

    monkeypatch.setattr(skills.httpx, "post", recording_post)
    skills.close_session(stub.port, "openai-gpt-oss-120b", "3f1c-session")
    (request,) = stub.requests
    assert (request["method"], request["path"]) == ("POST", "/v1/chat/completions")
    assert request["body"] == (
        b'{"model":"openai-gpt-oss-120b","messages":[{"role":"user","content":"done"}],"max_tokens":1}'
    )
    headers = {name.lower(): value for name, value in request["headers"].items()}
    assert headers["x-session-id"] == "3f1c-session"
    assert headers["x-turn-type"] == "side"
    assert headers["x-session-done"] == "true"
    assert headers["content-type"] == "application/json"
    assert seen[0]["timeout"] == 120


def test_close_session_raises_for_an_error_status(stub: Stub) -> None:
    stub.status = 502
    with pytest.raises(httpx.HTTPStatusError):
        skills.close_session(stub.port, "qwen3.8-27b", "s")


def test_close_session_refuses_an_empty_session_id(stub: Stub) -> None:
    for empty in ("", "   "):
        with pytest.raises(ValueError, match="empty session id"):
            skills.close_session(stub.port, "qwen3.8-27b", empty)
    assert stub.requests == []


def test_close_session_ignores_proxy_settings_of_the_environment(stub: Stub, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:9")
    skills.close_session(stub.port, "qwen3.8-27b", "s")
    assert len(stub.requests) == 1


# --- the proxy's environment and start ---------------------------------------------------------------


@pytest.fixture
def paths(tmp_path: Path) -> RuntimePaths:
    home = tmp_path / "home"
    home.mkdir()
    return RuntimePaths(
        root=tmp_path / "runtime", agent_home=home, outputs=tmp_path / "outputs", metaclaw_python=Path(sys.executable)
    )


def make_env(model: ModelSpec, tmp_path: Path, cfgs: Configs, paths: RuntimePaths, evolve: bool) -> dict[str, str]:
    return skills.proxy_env(
        model,
        tmp_path / "library",
        evolve,
        True,
        tmp_path / "rec",
        cfgs.ports,
        SECRETS,
        skills=cfgs.skills,
        paths=paths,
    )


def test_proxy_env_of_an_evolving_proxy(tmp_path: Path, cfgs: Configs, paths: RuntimePaths) -> None:
    env = make_env(cfgs.models["gpt-oss-120b"], tmp_path, cfgs, paths, True)
    sums = thirdparty.verify_patches()
    assert {key: env[key] for key in env if key not in ("PATH", "LANG", "TZ")} == {
        "HOME": str(paths.service_home),
        "PYTHONUNBUFFERED": "1",
        "PYTHONSAFEPATH": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "SKILLS_DIR": str(tmp_path / "library"),
        "AUTO_EVOLVE": "1",
        "REQUIRE_EMPTY": "1",
        "LLM_BASE": f"http://127.0.0.1:{cfgs.ports.model}/v1",
        "LLM_MODEL": "openai-gpt-oss-120b",
        "PROXY_PORT": str(cfgs.ports.proxy),
        "RECORD_DIR": str(tmp_path / "rec"),
        "EXPECT_SHA256_SKILL_EVOLVER": sums["MetaClaw/metaclaw/skill_evolver.py"],
        "EXPECT_SHA256_API_SERVER": sums["MetaClaw/metaclaw/api_server.py"],
        "SKILL_TOP_K": "6",
        "TASK_SPECIFIC_TOP_K": "10",
        "EVOLVE_EVERY": "1000",
        "MAX_NEW_SKILLS": "3",
        "EVOLVER_MAX_COMPLETION_TOKENS": "3000",
        "OPENAI_BASE_URL": "https://evolver.invalid/v1",
        "OPENAI_API_KEY": "evolver-test-key",
        "SKILL_EVOLVER_MODEL": "evolver-test-model",
    }


def test_proxy_env_passes_on_only_path_lang_and_tz(
    tmp_path: Path, cfgs: Configs, paths: RuntimePaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in {
        "TINKER_API_KEY": "t",
        "OPENCLAW_GATEWAY_TOKEN": "g",
        "HF_TOKEN": "h",
        "EVOLVER_API_KEY": "e",
        "PYTHONPATH": "/somewhere",
        "LANG": "C.UTF-8",
        "TZ": "America/New_York",
    }.items():
        monkeypatch.setenv(name, value)
    env = make_env(cfgs.models["qwen3.8-27b"], tmp_path, cfgs, paths, True)
    assert (env["PATH"], env["LANG"], env["TZ"]) == (os.environ["PATH"], "C.UTF-8", "America/New_York")
    for name in ("TINKER_API_KEY", "OPENCLAW_GATEWAY_TOKEN", "HF_TOKEN", "EVOLVER_API_KEY", "PYTHONPATH"):
        assert name not in env


def test_proxy_env_of_an_evaluation_proxy_has_no_evolver(tmp_path: Path, cfgs: Configs, paths: RuntimePaths) -> None:
    env = make_env(cfgs.models["qwen3.8-27b"], tmp_path, cfgs, paths, False)
    assert env["AUTO_EVOLVE"] == "0"
    assert not {"OPENAI_BASE_URL", "OPENAI_API_KEY", "SKILL_EVOLVER_MODEL"} & env.keys()
    no_secrets = skills.proxy_env(
        cfgs.models["qwen3.8-27b"], tmp_path / "l", False, False, tmp_path / "r", cfgs.ports, Secrets(),
        skills=cfgs.skills, paths=paths,
    )  # fmt: skip
    assert no_secrets["REQUIRE_EMPTY"] == "0"


@pytest.mark.parametrize("missing", ["evolver_base_url", "evolver_api_key", "evolver_model"])
def test_an_evolving_proxy_needs_every_evolver_secret(
    tmp_path: Path, cfgs: Configs, paths: RuntimePaths, missing: str
) -> None:
    secrets = Secrets(**{**SECRETS.__dict__, missing: None})
    with pytest.raises(ConfigError, match=missing.upper()):
        skills.proxy_env(
            cfgs.models["qwen3.8-27b"], tmp_path / "l", True, True, tmp_path / "r", cfgs.ports, secrets,
            skills=cfgs.skills, paths=paths,
        )  # fmt: skip


def test_proxy_env_refuses_unpatched_frameworks(
    tmp_path: Path, cfgs: Configs, paths: RuntimePaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unpatched() -> dict[str, str]:
        raise thirdparty.PatchError("MetaClaw/metaclaw/api_server.py has sha256 0, SHA256SUMS says 1")

    monkeypatch.setattr(skills, "verify_patches", unpatched)
    with pytest.raises(thirdparty.PatchError):
        make_env(cfgs.models["qwen3.8-27b"], tmp_path, cfgs, paths, False)


@pytest.fixture
def fake_proxy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """The stand-in proxy in place of the real one; at teardown any stand-in still running is killed, whatever the
    test did (each writes its pid, and runs in a process group of its own)."""
    monkeypatch.setattr(skills, "PROXY_SCRIPT", FAKE_PROXY)
    yield
    for pid_file in tmp_path.rglob("pid"):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(int(pid_file.read_text()), signal.SIGKILL)


def behave(paths: RuntimePaths, **behaviour: Any) -> None:
    paths.service_home.mkdir(parents=True, exist_ok=True)
    (paths.service_home / "fake_proxy.json").write_text(json.dumps(behaviour), encoding="utf-8")


def start(
    tmp_path: Path, cfgs: Configs, paths: RuntimePaths, port: int, evolve: bool = True, record: str = "rec"
) -> Any:
    return skills.start_proxy(
        cfgs.models["qwen3.8-27b"], tmp_path / "library", evolve, True, tmp_path / record,
        cfgs.ports.model_copy(update={"proxy": port}), SECRETS, skills=cfgs.skills, paths=paths,
    )  # fmt: skip


@pytest.mark.usefixtures("fake_proxy")
def test_start_proxy_waits_for_both_lines_and_the_port(
    tmp_path: Path, cfgs: Configs, paths: RuntimePaths, free_port: Callable[[], int]
) -> None:
    port = free_port()
    proxy = start(tmp_path, cfgs, paths, port)
    try:
        assert proxy.alive() and port_in_use(port)
        assert proxy.log_path == tmp_path / "rec" / "proxy.log"
        env = json.loads((tmp_path / "rec" / "env.json").read_text(encoding="utf-8"))
        assert env["HOME"] == str(paths.service_home)
        assert env["OPENAI_API_KEY"] == "evolver-test-key"
        assert "TINKER_API_KEY" not in env
        assert (tmp_path / "library").is_dir()
        skills.close_session(port, "qwen3.8-27b", "first-session")
        history = tmp_path / "library" / "evolution_history.jsonl"
        assert skills.drain(history, 1, poll_s=0.01, stable_polls=3, max_polls=500) == 1
    finally:
        proxy.stop()
    assert not port_in_use(port)


@pytest.mark.usefixtures("fake_proxy")
def test_start_proxy_refuses_a_record_directory_that_exists(
    tmp_path: Path, cfgs: Configs, paths: RuntimePaths, free_port: Callable[[], int]
) -> None:
    (tmp_path / "rec").mkdir()
    with pytest.raises(ValueError, match="fresh record directory"):
        start(tmp_path, cfgs, paths, free_port())


@pytest.mark.usefixtures("fake_proxy")
def test_start_proxy_refuses_a_port_in_use(tmp_path: Path, cfgs: Configs, paths: RuntimePaths, listener: int) -> None:
    with pytest.raises(PortInUse, match=f"port {listener}"):
        start(tmp_path, cfgs, paths, listener)
    assert not (tmp_path / "rec").exists()


@pytest.mark.usefixtures("fake_proxy")
def test_start_proxy_reports_a_proxy_that_exits_before_it_is_ready(
    tmp_path: Path, cfgs: Configs, paths: RuntimePaths, free_port: Callable[[], int]
) -> None:
    behave(paths, exit_before_ready=3)
    with pytest.raises(ProcessError, match="exited with status 3 before logging 'METACLAW patched '") as error:
        start(tmp_path, cfgs, paths, free_port())
    assert "fake proxy starting" in str(error.value)


@pytest.mark.usefixtures("fake_proxy")
def test_start_proxy_stops_a_proxy_that_never_gets_ready(
    tmp_path: Path, cfgs: Configs, paths: RuntimePaths, free_port: Callable[[], int], monkeypatch: pytest.MonkeyPatch
) -> None:
    behave(paths, hang_before_ready=True)
    monkeypatch.setattr(skills, "READY_TIMEOUT_S", 1)
    with pytest.raises(ProcessError, match="did not log 'skills_only proxy READY on 127.0.0.1:"):
        start(tmp_path, cfgs, paths, free_port())
    pid = int((tmp_path / "rec" / "pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # stopped, and reaped


@pytest.mark.usefixtures("fake_proxy")
def test_start_proxy_needs_the_port_held_by_the_proxy(
    tmp_path: Path, cfgs: Configs, paths: RuntimePaths, free_port: Callable[[], int], monkeypatch: pytest.MonkeyPatch
) -> None:
    behave(paths, no_listen=True)
    monkeypatch.setattr(skills, "PORT_TIMEOUT_S", 1)
    with pytest.raises(ProcessError, match="did not listen on port"):
        start(tmp_path, cfgs, paths, free_port())


def test_start_proxy_needs_the_metaclaw_python(tmp_path: Path, cfgs: Configs, paths: RuntimePaths) -> None:
    missing = RuntimePaths(root=paths.root, agent_home=paths.agent_home, metaclaw_python=tmp_path / "no" / "python")
    with pytest.raises(ConfigError, match="EVOLVE_IPI_METACLAW_PYTHON"):
        skills.start_proxy(
            cfgs.models["qwen3.8-27b"], tmp_path / "library", False, True, tmp_path / "rec", cfgs.ports, SECRETS,
            skills=cfgs.skills, paths=missing,
        )  # fmt: skip


# --- the guards ---------------------------------------------------------------------------------------


def test_run_guards_writes_both_logs_and_passes(
    tmp_path: Path, paths: RuntimePaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = tmp_path / "guards.py"
    fake.write_text("import sys\nprint('checked', sys.argv[1:])\n", encoding="utf-8")
    monkeypatch.setattr(skills, "GUARDS_SCRIPT", fake)
    history, records = tmp_path / "h.jsonl", tmp_path / "c.jsonl"
    skills.run_guards(history, records, tmp_path / "guards", paths=paths)
    assert (tmp_path / "guards" / "no-turn-drop.log").read_text() == f"checked ['no-turn-drop', '{history}']\n"
    assert (tmp_path / "guards" / "saw-conversation.log").read_text() == (
        f"checked ['saw-conversation', '{history}', '{records}']\n"
    )


def test_run_guards_aborts_on_a_failed_guard(tmp_path: Path, paths: RuntimePaths) -> None:
    history = write_jsonl(tmp_path / "h.jsonl", [event(3, prompts=["a"])])
    with pytest.raises(AdaptAbort, match=r"^the guard no-turn-drop failed \(exit 1\): .* 3 turns were fed") as error:
        skills.run_guards(history, tmp_path / "c.jsonl", tmp_path / "guards", paths=paths)
    assert "saw-conversation" not in str(error.value)
    assert not (tmp_path / "guards" / "saw-conversation.log").exists()
    assert "3 turns were fed" in (tmp_path / "guards" / "no-turn-drop.log").read_text()


def test_run_guards_keeps_what_a_guard_wrote_before_it_timed_out(
    tmp_path: Path, paths: RuntimePaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = tmp_path / "guards.py"
    fake.write_text(
        "import sys, time\nprint('checking', sys.argv[1], flush=True)\n"
        "print('half way', file=sys.stderr, flush=True)\ntime.sleep(60)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(skills, "GUARDS_SCRIPT", fake)
    monkeypatch.setattr(skills, "GUARD_TIMEOUT_S", 5)
    with pytest.raises(AdaptAbort, match=r"^the guard no-turn-drop did not finish within 5 s"):
        skills.run_guards(tmp_path / "h.jsonl", tmp_path / "c.jsonl", tmp_path / "guards", paths=paths)
    log = (tmp_path / "guards" / "no-turn-drop.log").read_text(encoding="utf-8")
    assert "checking no-turn-drop" in log and "half way" in log
    assert not (tmp_path / "guards" / "saw-conversation.log").exists()


def test_run_guards_logs_a_vacuous_pass_as_a_warning(
    tmp_path: Path, paths: RuntimePaths, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake = tmp_path / "guards.py"
    fake.write_text(
        "import sys\nprint('  evolver input: 1 event(s)')\n"
        "if sys.argv[1] == 'saw-conversation':\n"
        "    print('                 WARNING: no recorded turn had a conversation of 300 characters or fewer, so this'"
        " ' run could not distinguish a patched evolver from an unpatched one. The pass above is vacuous.')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(skills, "GUARDS_SCRIPT", fake)
    with caplog.at_level("WARNING", logger="evolve_ipi.adapt.skills"):
        skills.run_guards(tmp_path / "h.jsonl", tmp_path / "c.jsonl", tmp_path / "guards", paths=paths)
    (record,) = [r for r in caplog.records if r.name == "evolve_ipi.adapt.skills"]
    assert record.levelname == "WARNING"
    assert "saw-conversation" in record.getMessage() and "The pass above is vacuous." in record.getMessage()


def test_run_guards_logs_nothing_for_a_pass_with_teeth(
    tmp_path: Path, paths: RuntimePaths, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake = tmp_path / "guards.py"
    fake.write_text("print('  evolver input: 1 event(s)')\nprint('   (3 recorded turn(s) ... has teeth)')\n")
    monkeypatch.setattr(skills, "GUARDS_SCRIPT", fake)
    with caplog.at_level("WARNING", logger="evolve_ipi.adapt.skills"):
        skills.run_guards(tmp_path / "h.jsonl", tmp_path / "c.jsonl", tmp_path / "guards", paths=paths)
    assert [r for r in caplog.records if r.name == "evolve_ipi.adapt.skills"] == []


def stand_in_evolver_check(tmp_path: Path, body: str) -> Path:
    """A stand-in for ``metaclaw_guards.py evolver``: it records what it was given, then runs ``body``."""
    script = tmp_path / "guards.py"
    script.write_text(
        "import json, os, sys, time\n"
        f"open({str(tmp_path / 'seen.json')!r}, 'w').write(json.dumps({{'argv': sys.argv[1:], 'env': dict(os.environ)}}))\n"
        + body,
        encoding="utf-8",
    )
    return script


def test_evolver_preflight_runs_the_evolver_check_with_metaclaws_python(
    tmp_path: Path, paths: RuntimePaths, monkeypatch: pytest.MonkeyPatch, cfgs: Configs
) -> None:
    monkeypatch.setattr(skills, "GUARDS_SCRIPT", stand_in_evolver_check(tmp_path, "print('answered')\n"))
    monkeypatch.setenv("TINKER_API_KEY", "tinker-test-key")
    skills.evolver_preflight(SECRETS, cfgs.skills, paths=paths)
    seen = json.loads((tmp_path / "seen.json").read_text(encoding="utf-8"))
    assert seen["argv"] == ["evolver", "3000"]  # the key travels in the environment only, never in argv
    env = seen["env"]
    assert (env["OPENAI_BASE_URL"], env["OPENAI_API_KEY"], env["SKILL_EVOLVER_MODEL"]) == (
        "https://evolver.invalid/v1",
        "evolver-test-key",
        "evolver-test-model",
    )
    assert env["HOME"] == str(paths.service_home)
    assert "TINKER_API_KEY" not in env


def test_evolver_preflight_names_the_endpoint_and_the_model_and_never_the_key(
    tmp_path: Path, paths: RuntimePaths, monkeypatch: pytest.MonkeyPatch, cfgs: Configs
) -> None:
    body = "print('evolver: refused, key ' + os.environ['OPENAI_API_KEY'], file=sys.stderr)\nsys.exit(1)\n"
    monkeypatch.setattr(skills, "GUARDS_SCRIPT", stand_in_evolver_check(tmp_path, body))
    with pytest.raises(ConfigError) as error:
        skills.evolver_preflight(SECRETS, cfgs.skills, paths=paths)
    message = str(error.value)
    assert message.startswith(
        "the skill evolver at https://evolver.invalid/v1 (model evolver-test-model) cannot be used"
    )
    assert "evolver-test-key" not in message
    assert "<the evolver key>" in message


def test_the_evolver_key_is_taken_out_before_the_message_is_cut(
    tmp_path: Path, paths: RuntimePaths, monkeypatch: pytest.MonkeyPatch, cfgs: Configs
) -> None:
    """The message keeps its last 800 characters; a key that straddles the cut must not leave a piece behind."""
    body = "sys.stderr.write('x' * 1000 + os.environ['OPENAI_API_KEY'] + 'y' * 792)\nsys.exit(1)\n"
    monkeypatch.setattr(skills, "GUARDS_SCRIPT", stand_in_evolver_check(tmp_path, body))
    with pytest.raises(ConfigError) as error:
        skills.evolver_preflight(SECRETS, cfgs.skills, paths=paths)
    message = str(error.value)
    assert message.endswith("y" * 792) and "test-key" not in message and "-key" not in message.replace(" key>", "")


def test_evolver_preflight_names_the_endpoint_and_the_model_once(
    tmp_path: Path, paths: RuntimePaths, monkeypatch: pytest.MonkeyPatch, cfgs: Configs
) -> None:
    said = "evolver: refused the request: AuthenticationError (HTTP 401, invalid_api_key)"
    body = f"print({said!r}, file=sys.stderr)\nsys.exit(1)\n"
    monkeypatch.setattr(skills, "GUARDS_SCRIPT", stand_in_evolver_check(tmp_path, body))
    with pytest.raises(ConfigError) as error:
        skills.evolver_preflight(SECRETS, cfgs.skills, paths=paths)
    message = str(error.value)
    where = "the skill evolver at https://evolver.invalid/v1 (model evolver-test-model)"
    assert message == f"{where} cannot be used: {said}"


def test_evolver_preflight_says_once_that_the_evolver_answered(
    tmp_path: Path,
    paths: RuntimePaths,
    monkeypatch: pytest.MonkeyPatch,
    cfgs: Configs,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(skills, "GUARDS_SCRIPT", stand_in_evolver_check(tmp_path, "print('answered')\n"))
    with caplog.at_level("INFO", logger="evolve_ipi.adapt.skills"):
        skills.evolver_preflight(SECRETS, cfgs.skills, paths=paths)
    said = [r.getMessage() for r in caplog.records if r.name == "evolve_ipi.adapt.skills"]
    assert said == ["the skill evolver at https://evolver.invalid/v1 (model evolver-test-model) answered"]


def test_evolver_preflight_that_gets_no_answer(
    tmp_path: Path, paths: RuntimePaths, monkeypatch: pytest.MonkeyPatch, cfgs: Configs
) -> None:
    monkeypatch.setattr(skills, "GUARDS_SCRIPT", stand_in_evolver_check(tmp_path, "time.sleep(60)\n"))
    monkeypatch.setattr(skills, "EVOLVER_PREFLIGHT_TIMEOUT_S", 1)
    with pytest.raises(ConfigError, match=r"\(model evolver-test-model\) did not answer within 1 s"):
        skills.evolver_preflight(SECRETS, cfgs.skills, paths=paths)


def test_run_guards_runs_the_second_guard_with_the_metaclaw_python(tmp_path: Path, paths: RuntimePaths) -> None:
    # sys.executable stands in for MetaClaw's Python and cannot import metaclaw: the second guard must fail loudly
    history = write_jsonl(tmp_path / "h.jsonl", [event(1)])
    with pytest.raises(AdaptAbort, match="saw-conversation failed .exit 1.") as error:
        skills.run_guards(history, tmp_path / "c.jsonl", tmp_path / "guards", paths=paths)
    assert "metaclaw" in str(error.value)
    assert GUARDS == skills.GUARDS_SCRIPT


# --- adapt_skills ------------------------------------------------------------------------------------


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    return World(tmp_path, monkeypatch)


def test_twelve_conversations_in_the_skill_order(world: World) -> None:
    result = world.adapt("qwen3.8-27b", "drive-create")
    run = world.run_dir("qwen3.8-27b", "drive-create")
    order = load_skill_order("drive-create")
    assert world.named("episode") == order
    assert world.named("close") == [f"session-1-{n}" for n in range(1, 13)]
    # the closing of each conversation comes after its feedback turn, within the episode
    for item_id in order:
        position = world.calls.index(("feedback", item_id))
        assert world.calls[position + 1][0] == "close"
    assert {(a["mode"], a["authorized"], a["policy"]) for a in world.episode_args} == {
        ("train", True, TrainingPolicy("abort", 0))
    }
    records = episodes(run)
    assert [r["id"] for r in records] == order
    assert [r["conversation"] for r in records] == list(range(1, 13))
    assert [r["closure"]["history_before"] for r in records] == list(range(12))
    assert all(set(r["closure"]) == {"at", "history_before"} for r in records)
    library = run.adapt_dir / "library"
    assert result == {"library": library, "sha": skills.lib_sha(library), "evolutions": 12, "skills": 12}
    marker = json.loads((run.adapt_dir / "library.json").read_text(encoding="utf-8"))
    assert set(marker) == {"library", "sha", "evolutions", "skills", "zero_skill_events", "at"}
    assert (marker["library"], marker["sha"]) == ("library", result["sha"])
    assert (marker["evolutions"], marker["skills"], marker["zero_skill_events"]) == (12, 12, 0)
    stats = json.loads((run.adapt_dir / "stats.json").read_text(encoding="utf-8"))
    assert stats == {
        "conversations": order,
        "executions": 4,
        "praised": 4,
        "attempt": 1,
        "evolver_model": "evolver-test-model",
    }
    assert voids(run) == []


def test_the_order_of_the_steps(world: World) -> None:
    world.adapt()
    steps = [(call[0], call[1] if isinstance(call[1], str) else "") for call in world.calls]
    steps = [step for step in steps if step[0] in ("prepare", "start", "stop", "guards")]
    assert steps == [
        ("prepare", "drive-create"),
        ("start", "gateway"),  # first, as in an evaluation: a gateway it refuses fails before a model loads
        ("start", "model server"),
        ("start", "proxy"),
        ("stop", "gateway"),  # after the last conversation; the drain needs the proxy alone
        ("stop", "proxy"),
        ("stop", "model server"),
        ("guards", ""),
    ]


def test_the_servers_and_the_proxy_are_started_as_the_experiment_ran_them(world: World) -> None:
    world.adapt("gpt-oss-120b", "chat-message")
    run = world.run_dir("gpt-oss-120b", "chat-message")
    (server,) = world.server_args
    assert (server["family"], server["base_model"], server["served_name"]) == (
        "harmony",
        "openai/gpt-oss-120b",
        "openai-gpt-oss-120b",
    )
    assert (server["port"], server["checkpoint"], server["temperature"]) == (world.cfgs.ports.model, None, 0.6)
    assert server["out_dir"] == run.adapt_dir / "model_server"
    env = server["env"]
    assert env["TINKER_API_KEY"] == "tinker-test-key"
    assert (env["HOME"], env["HF_HOME"]) == (str(world.paths.service_home), str(world.paths.hf_home))
    assert not {k for k in env if k.startswith(("EVOLVER", "OPENAI"))}
    (proxy,) = world.proxy_args
    assert (proxy["evolve"], proxy["require_empty"]) == (True, True)
    assert (proxy["skills_dir"], proxy["record_dir"]) == (run.adapt_dir / "library", run.adapt_dir / "proxy")
    assert (proxy["ports"], proxy["skills"], proxy["secrets"]) == (world.cfgs.ports, world.cfgs.skills, SECRETS)
    prepare = next(call for call in world.calls if call[0] == "prepare")
    assert prepare == ("prepare", "chat-message", "skills", world.cfgs.ports.proxy, 600)
    assert {a["policy"] for a in world.episode_args} == {TrainingPolicy("abort", 4)}
    closes = [call for call in world.calls if call[0] == "close"]
    assert {(c[2], c[3]) for c in closes} == {(world.cfgs.ports.proxy, "openai-gpt-oss-120b")}


def test_file_create_continues_when_the_model_does_not_engage(world: World) -> None:
    result = world.adapt("gpt-oss-20b", "file-create")
    assert {a["policy"] for a in world.episode_args} == {TrainingPolicy("continue", 0)}
    assert world.calls[1] == ("prepare", "file-create", "skills", world.cfgs.ports.proxy, 300)  # after the preflight
    stats = json.loads((result["library"].parent / "stats.json").read_text(encoding="utf-8"))
    assert (stats["executions"], stats["praised"]) == (3, 3)


@pytest.mark.parametrize(
    ("missing", "variable"),
    [
        ("evolver_model", "EVOLVER_MODEL"),
        ("evolver_api_key", "EVOLVER_API_KEY"),
        ("evolver_base_url", "EVOLVER_BASE_URL"),
        ("tinker_api_key", "TINKER_API_KEY"),
    ],
)
def test_a_missing_secret_fails_before_anything_starts(world: World, missing: str, variable: str) -> None:
    secrets = Secrets(**{**SECRETS.__dict__, missing: None})
    with pytest.raises(ConfigError, match=variable):
        world.adapt(secrets=secrets)
    assert world.calls == []
    assert not world.run_dir("qwen3.8-27b", "drive-create").adapt_dir.exists()


def test_a_model_outside_skill_evolution_is_refused(world: World) -> None:
    model = world.cfgs.models["gpt-oss-120b"].model_copy(update={"mechanisms": ("rl",)})
    run = world.run_dir("gpt-oss-120b", "drive-create")
    with pytest.raises(ConfigError, match="does not take part in skill evolution"):
        skills.adapt_skills(model, load_task("drive-create"), run, world.cfgs, SECRETS, paths=world.paths)
    assert world.calls == []


def test_a_missing_metaclaw_python_fails_before_anything_starts(world: World, tmp_path: Path) -> None:
    world.paths = RuntimePaths(
        root=world.paths.root, agent_home=world.paths.agent_home, metaclaw_python=tmp_path / "nowhere" / "python"
    )
    with pytest.raises(ConfigError, match="EVOLVE_IPI_METACLAW_PYTHON"):
        world.adapt()
    assert world.calls == []


def test_unpatched_frameworks_fail_before_anything_starts(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    def unpatched() -> dict[str, str]:
        raise thirdparty.PatchError("not patched")

    monkeypatch.setattr(skills, "verify_patches", unpatched)
    with pytest.raises(thirdparty.PatchError):
        world.adapt()
    assert world.calls == []


def test_an_aborted_episode_restarts_the_evolution_from_an_empty_library(world: World) -> None:
    world.scripts = [abort_at(5, lambda: EpisodeAbort("drive-create-adapt-007: no_calls: access.log is empty"))]
    result = world.adapt()
    run = world.run_dir("qwen3.8-27b", "drive-create")
    (void,) = voids(run)
    assert why(void)["reason"] == "EpisodeAbort: drive-create-adapt-007: no_calls: access.log is empty"
    assert why(void)["attempt"] == 1
    assert len(episodes(run)) == 12  # the run's own episodes are the second attempt's
    voided = [json.loads(line)["id"] for line in (void / "episodes.jsonl").read_text().splitlines()]
    assert voided == world.named("episode")[:4]
    assert skills.history_lines(void / "library" / "evolution_history.jsonl") == 4
    assert skills.history_lines(result["library"] / "evolution_history.jsonl") == 12
    assert result["skills"] == 12
    # the first attempt's proxy and model server were stopped before the second attempt began
    second = [i for i, call in enumerate(world.calls) if call[0] == "prepare"][1]
    assert world.calls.index(("stop", "proxy")) < world.calls.index(("stop", "model server")) < second
    assert json.loads((run.adapt_dir / "stats.json").read_text(encoding="utf-8"))["attempt"] == 2


def test_an_aborted_episode_keeps_its_record_in_why(world: World) -> None:
    record = {"id": "drive-create-adapt-005", "anomalies": [{"type": "no_calls", "detail": "access.log is empty"}]}
    world.scripts = [abort_at(2, lambda: EpisodeAbort("drive-create-adapt-005: no_calls", record, [record]))]
    world.adapt()
    (void,) = voids(world.run_dir("qwen3.8-27b", "drive-create"))
    assert why(void)["episode"] == record
    assert why(void)["retried_attempts"] == [record]


def test_the_restart_limit(world: World) -> None:
    limit = world.cfgs.skills.max_restarts
    world.scripts = [abort_at(1, lambda: AdaptAbort("the proxy log has a fatal line")) for _ in range(limit + 1)]
    with pytest.raises(AdaptAbort, match=f"failed {limit + 1} times") as error:
        world.adapt()
    assert "skills.max_restarts" in str(error.value)
    run = world.run_dir("qwen3.8-27b", "drive-create")
    assert len(voids(run)) == limit + 1
    assert world.attempt == limit + 1
    assert not (run.adapt_dir / "library.json").exists()
    assert world.calls[-3:] == [("stop", "proxy"), ("stop", "model server"), ("stop", "gateway")]


def test_leftover_state_stops_at_once(world: World) -> None:
    world.scripts = [Script(prepare_raises=LeftoverState("/home/.metaclaw holds a.txt"))]
    with pytest.raises(LeftoverState):
        world.adapt()
    assert world.attempt == 1
    assert voids(world.run_dir("qwen3.8-27b", "drive-create")) == []
    assert [call[1] for call in world.calls if call[0] == "start"] == []


def test_leftover_state_in_an_episode_stops_at_once_and_voids_the_attempt(world: World) -> None:
    world.scripts = [abort_at(3, lambda: LeftoverState("the target directory holds b.txt"))]
    with pytest.raises(LeftoverState):
        world.adapt()
    (void,) = voids(world.run_dir("qwen3.8-27b", "drive-create"))
    assert why(void)["reason"] == "LeftoverState: the target directory holds b.txt"
    assert world.attempt == 1
    assert ("stop", "proxy") in world.calls and ("stop", "model server") in world.calls


def test_a_port_in_use_stops_at_once(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    def busy(*args: Any, **kwargs: Any) -> Any:
        raise PortInUse("port 30013 is in use by pid 1 (python)")

    monkeypatch.setattr(skills, "start_model_server", busy)
    with pytest.raises(PortInUse):
        world.adapt()
    assert world.attempt == 1


def test_a_finished_evolution_is_never_run_again(world: World) -> None:
    first = world.adapt()
    calls = len(world.calls)
    assert world.adapt() == first
    assert len(world.calls) == calls


def test_a_finished_library_that_changed_is_refused(world: World) -> None:
    first = world.adapt()
    write_skill(first["library"], "added-by-hand", "x")
    calls = len(world.calls)
    with pytest.raises(skills.LibraryError, match="changed since its evolution finished"):
        world.adapt()
    assert len(world.calls) == calls


def test_a_finished_library_that_is_gone_is_refused(world: World) -> None:
    first = world.adapt()
    shutil.rmtree(first["library"])
    with pytest.raises(skills.LibraryError, match="is gone"):
        world.adapt()


def test_an_interrupted_evolution_restarts_from_an_empty_library(world: World) -> None:
    run = world.run_dir("qwen3.8-27b", "drive-create")
    write_skill(run.adapt_dir / "library", "from-the-interrupted-run", "x")
    (run.adapt_dir / "episodes.jsonl").write_text('{"id": "drive-create-adapt-003"}\n', encoding="utf-8")
    result = world.adapt()
    (void,) = voids(run)
    assert why(void)["reason"].startswith("an earlier run stopped before its evolution finished")
    assert (void / "library" / "from-the-interrupted-run" / "SKILL.md").is_file()
    assert not (result["library"] / "from-the-interrupted-run").exists()
    assert len(episodes(run)) == 12


def test_ctrl_c_stops_everything_and_the_next_run_starts_over(world: World) -> None:
    world.scripts = [abort_at(4, KeyboardInterrupt)]
    with pytest.raises(KeyboardInterrupt):
        world.adapt()
    assert [call for call in world.calls if call[0] == "stop"] == [
        ("stop", "proxy"),
        ("stop", "model server"),
        ("stop", "gateway"),
    ]
    run = world.run_dir("qwen3.8-27b", "drive-create")
    assert voids(run) == []  # left in place; the next run moves it aside
    assert len(episodes(run)) == 3
    result = world.adapt()
    (void,) = voids(run)
    assert len((void / "episodes.jsonl").read_text().splitlines()) == 3
    assert result["evolutions"] == 12


def test_a_fatal_line_in_the_proxy_log_restarts(world: World) -> None:
    world.scripts = [Script(proxy_log_at={2: "12:00:01 | ERROR | x | Traceback (most recent call last)\n"})]
    world.adapt()
    (void,) = voids(world.run_dir("qwen3.8-27b", "drive-create"))
    assert why(void)["reason"].startswith("AdaptAbort: a watched log has a fatal line: proxy.log: 12:00:01 | ERROR")
    assert len((void / "episodes.jsonl").read_text().splitlines()) == 2


def test_a_fatal_line_in_the_model_server_log_restarts(world: World) -> None:
    world.scripts = [Script(model_log_at={7: 'INFO: "POST /v1/chat/completions HTTP/1.1" 503 Service Unavailable\n'})]
    world.adapt()
    (void,) = voids(world.run_dir("qwen3.8-27b", "drive-create"))
    assert "model_server.log: INFO" in why(void)["reason"]


def test_agent_content_transcribed_into_the_proxy_log_is_not_fatal(world: World) -> None:
    content = (
        "12:00:01 | INFO | metaclaw.api_server | [OpenClaw] session=s turn=1 next_state role=tool len=40: "
        'Traceback (most recent call last)\n  File "x.py", line 1\nZeroDivisionError: division by zero\n'
        "12:00:02 | INFO | metaclaw.api_server | next request\n"
    )
    world.scripts = [Script(proxy_log_at={3: content})]
    world.adapt()
    assert voids(world.run_dir("qwen3.8-27b", "drive-create")) == []


def test_lines_written_before_the_first_conversation_are_not_watched(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    start_proxy = world.start_proxy

    def proxy_with_startup_noise(*args: Any, **kwargs: Any) -> Any:
        process = start_proxy(*args, **kwargs)
        process.write("Traceback (most recent call last)\n")
        return process

    monkeypatch.setattr(skills, "start_proxy", proxy_with_startup_noise)
    world.adapt()
    assert voids(world.run_dir("qwen3.8-27b", "drive-create")) == []


@pytest.mark.parametrize(
    ("what", "error"),
    [
        ("model server", ProcessError("model server exited with status 1 before logging 'EVAL PROXY READY'")),
        ("proxy", ProcessError("MetaClaw proxy did not log 'METACLAW patched ' within 180 s")),
        ("gateway", OpenClawError("openclaw 2026.4.0 is installed; this experiment runs 2026.3.7")),
    ],
)
def test_a_process_that_does_not_start_stops_the_evolution_at_once(world: World, what: str, error: Exception) -> None:
    world.scripts = [Script(start_error={what: error})]
    with pytest.raises(skills.StartFailed, match=f"^the {what} did not start: {type(error).__name__}: ") as stop:
        world.adapt()
    assert stop.value.__cause__ is error
    assert world.attempt == 1  # deterministic: starting it again would fail the same way
    (void,) = voids(world.run_dir("qwen3.8-27b", "drive-create"))
    assert why(void)["reason"].startswith(f"StartFailed: the {what} did not start")
    stopped = [call[1] for call in world.calls if call[0] == "stop"]
    started = [call[1] for call in world.calls if call[0] == "start" and call[1] != what]
    assert set(started) <= set(stopped) <= {*started, "gateway"}  # whatever had started is stopped again
    assert stopped[-1] == "gateway"


def test_a_gateway_that_cannot_go_on_between_conversations_restarts(world: World) -> None:
    world.scripts = [Script(gateway_restart_at={5: OpenClawError("port 18789 is not held by the gateway")})]
    world.adapt()
    (void,) = voids(world.run_dir("qwen3.8-27b", "drive-create"))
    assert why(void)["reason"] == "OpenClawError: port 18789 is not held by the gateway"
    assert world.attempt == 2


def test_a_proxy_that_dies_during_the_drain_stops_it_at_once(world: World) -> None:
    # one evolution never lands, so the drain would poll 300 times; the proxy dies at its third poll
    world.scripts = [Script(evolutions={12: 0}, proxy_alive_checks=12 + 2)]
    world.adapt()
    (void,) = voids(world.run_dir("qwen3.8-27b", "drive-create"))
    assert why(void)["reason"].startswith("AdaptAbort: the proxy exited (status 1); the end of its log:")
    assert "killed" in why(void)["reason"]
    assert world.proxies[0].asked == 12 + 3  # once after each conversation, then once per poll


def test_ctrl_c_wins_over_a_process_that_will_not_stop(world: World) -> None:
    """A failure to stop while an interrupt propagates is a note on it, never another attempt."""
    stop = ProcessError("could not stop MetaClaw proxy (process group 4242)")
    world.scripts = [Script(raise_at={4: KeyboardInterrupt()}, stop_error={"proxy": stop})]
    with pytest.raises(KeyboardInterrupt) as raised:
        world.adapt()
    assert world.attempt == 1, "no attempt starts after Ctrl-C"
    notes = getattr(raised.value, "__notes__", ())
    assert any("ProcessError: could not stop MetaClaw proxy (process group 4242)" in note for note in notes)
    stops = [call for call in world.calls if call[0] == "stop"]
    assert stops == [("stop", "proxy"), ("stop", "model server"), ("stop", "gateway")], "the others are stopped"
    assert voids(world.run_dir("qwen3.8-27b", "drive-create")) == []  # left for the next run, as after any Ctrl-C


def test_a_failure_to_stop_after_an_aborted_attempt_is_kept_with_the_void(world: World) -> None:
    stop = ProcessError("could not stop model server (process group 4242)")
    aborted = AdaptAbort("the proxy log has a fatal line")
    world.scripts = [Script(raise_at={2: aborted}, stop_error={"model server": stop})]
    world.adapt()
    (void,) = voids(world.run_dir("qwen3.8-27b", "drive-create"))
    assert why(void)["reason"] == "AdaptAbort: the proxy log has a fatal line"
    assert why(void)["stop_failures"] == ["ProcessError: could not stop model server (process group 4242)"]
    assert world.attempt == 2


def test_a_process_that_will_not_stop_after_the_conversations_is_raised(world: World) -> None:
    """With nothing propagating, a failure to stop is the failure; the next attempt meets the port it holds."""
    world.scripts = [Script(stop_error={"proxy": ProcessError("could not stop MetaClaw proxy")})]
    world.scripts += [Script(start_error={"gateway": OpenClawError("port 18789 is in use")})]
    with pytest.raises(skills.StartFailed, match="the gateway did not start"):
        world.adapt()
    first, _ = voids(world.run_dir("qwen3.8-27b", "drive-create"))
    assert why(first)["reason"] == "ProcessError: could not stop MetaClaw proxy"
    second = [i for i, call in enumerate(world.calls) if call[0] == "prepare"][1]
    assert world.calls.index(("stop", "proxy")) < world.calls.index(("stop", "model server")) < second
    assert "guards" not in [call[0] for call in world.calls[:second]], "the attempt stopped before its guards"


def test_a_fatal_line_logged_during_the_drain_stops_it_at_once(world: World) -> None:
    # one evolution never lands, so the drain would poll 300 times; the evolver's request fails at its second poll
    fatal = "12:00:02 | ERROR | metaclaw.skill_evolver | Traceback (most recent call last)\n"
    world.scripts = [Script(evolutions={12: 0}, proxy_log_at_check={12 + 2: fatal})]
    world.adapt()
    (void,) = voids(world.run_dir("qwen3.8-27b", "drive-create"))
    assert why(void)["reason"].startswith("AdaptAbort: a watched log has a fatal line: proxy.log: 12:00:02 | ERROR")
    assert world.proxies[0].asked == 12 + 2, "the next poll found the line: once after each conversation, then per poll"


def test_the_model_server_of_an_evolution_gets_what_every_model_server_gets(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/src")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    world.adapt()
    (server,) = world.server_args
    assert server["env"] == children.model_server_env(world.paths, SECRETS)
    assert server["env"]["PYTHONPATH"] == "/src", "found as the evaluation's model server finds this package"
    assert server["env"]["HTTPS_PROXY"] == "http://proxy.example:3128"
    assert server["env"]["no_proxy"] == server["env"]["NO_PROXY"] == "127.0.0.1,localhost"


def test_an_inkling_dump_cut_off_mid_line_is_said_once(world: World, caplog: pytest.LogCaptureFixture) -> None:
    world.scripts = [Script(audit_unfinished=True)]
    with caplog.at_level("WARNING", logger="evolve_ipi.adapt.skills"):
        world.adapt("inkling-small", "drive-create")
    warnings = [r.getMessage() for r in caplog.records if r.name == "evolve_ipi.adapt.skills"]
    raw = world.run_dir("inkling-small", "drive-create").adapt_dir / "model_server" / "raw.jsonl"
    assert len(warnings) == 1 and str(raw) in warnings[0] and "cut off" in warnings[0], warnings


def test_a_caller_that_asked_the_evolver_already_makes_no_request(world: World) -> None:
    world.adapt(preflight=False)
    assert [call[0] for call in world.calls].count("preflight") == 0
    assert (world.run_dir("qwen3.8-27b", "drive-create").adapt_dir / "library.json").is_file()


def test_the_evolver_preflight_comes_first_and_once(world: World) -> None:
    world.scripts = [abort_at(3, lambda: AdaptAbort("the proxy log has a fatal line"))]
    world.adapt("gpt-oss-120b", "chat-invite")
    assert world.calls[0] == ("preflight", "evolver-test-model", 3000)
    assert [call[0] for call in world.calls].count("preflight") == 1  # not again for the restart


def test_a_failing_evolver_preflight_stops_before_anything_is_made(world: World) -> None:
    world.preflight_error = ConfigError("the skill evolver at https://evolver.invalid/v1 (model m) cannot be used")
    with pytest.raises(ConfigError, match="cannot be used"):
        world.adapt()
    assert world.calls == [("preflight", "evolver-test-model", 3000)]
    assert not world.run_dir("qwen3.8-27b", "drive-create").adapt_dir.exists()


def test_a_finished_evolution_needs_no_preflight(world: World) -> None:
    first = world.adapt()
    world.preflight_error = ConfigError("the evolver is unreachable today")
    assert world.adapt() == first
    assert [call[0] for call in world.calls].count("preflight") == 1


def test_a_proxy_that_died_restarts(world: World) -> None:
    world.scripts = [Script(proxy_dies_at=6)]
    world.adapt()
    (void,) = voids(world.run_dir("qwen3.8-27b", "drive-create"))
    assert why(void)["reason"].startswith("AdaptAbort: the proxy exited (status 1); the end of its log:")


@pytest.mark.parametrize(("evolutions", "count"), [({5: 0}, 11), ({5: 2}, 13)])
def test_a_drain_that_does_not_count_exactly_twelve_restarts(
    world: World, evolutions: dict[int, int], count: int
) -> None:
    world.scripts = [Script(evolutions=evolutions)]
    world.adapt()
    (void,) = voids(world.run_dir("qwen3.8-27b", "drive-create"))
    assert why(void)["reason"] == (
        f"AdaptAbort: 12 conversations were closed but {count} evolutions landed in "
        f"{world.run_dir('qwen3.8-27b', 'drive-create').adapt_dir / 'library' / 'evolution_history.jsonl'}"
    )


def test_a_failed_guard_restarts(world: World) -> None:
    world.scripts = [Script(guard_fails="saw-conversation")]
    result = world.adapt()
    (void,) = voids(world.run_dir("qwen3.8-27b", "drive-create"))
    assert why(void)["reason"] == "AdaptAbort: the saw-conversation guard failed"
    run = world.run_dir("qwen3.8-27b", "drive-create")
    assert world.named("guards") == [run.adapt_dir / "library" / "evolution_history.jsonl"] * 2
    assert result["evolutions"] == 12


def test_zero_skill_events_are_counted_not_refused(world: World) -> None:
    world.scripts = [Script(zero_skill_at={2, 9})]
    result = world.adapt()
    marker = json.loads((result["library"].parent / "library.json").read_text(encoding="utf-8"))
    assert (marker["zero_skill_events"], marker["skills"], result["skills"]) == (2, 10, 10)


def test_a_gateway_restart_between_conversations_is_recorded(world: World) -> None:
    world.scripts = [Script(gateway_restart_at={4: "its age (3.0 h) has reached 3 h"})]
    world.adapt()
    records = episodes(world.run_dir("qwen3.8-27b", "drive-create"))
    assert records[3]["gateway_restarted"] == "its age (3.0 h) has reached 3 h"
    assert [r.get("gateway_restarted") for r in records].count(None) == 11
    assert world.named("restart_if_needed")[:1] == [None]  # asked before every conversation but the first
    assert len(world.named("restart_if_needed")) == 11


def test_inkling_small_audits_the_raw_dump_with_the_shared_audit(world: World) -> None:
    world.scripts = [
        Script(audit_problems=["raw.jsonl: 1 generation(s) with tool-call markup that parsed to nothing"]),
        Script(audit_problems=["raw.jsonl: no raw generation"]),
        Script(audit_raises=RawDumpError("line 3 of the raw dump is not a generation")),
    ]
    result = world.adapt("inkling-small", "chat-invite")
    run = world.run_dir("inkling-small", "chat-invite")
    first, second, third = voids(run)
    assert why(first)["reason"] == (
        "AdaptAbort: the raw audit of the model server's generations failed: raw.jsonl: 1 generation(s) with "
        "tool-call markup that parsed to nothing"
    )
    assert why(second)["reason"] == (
        "AdaptAbort: the raw audit of the model server's generations failed: raw.jsonl: no raw generation"
    )
    assert why(third)["reason"] == (
        f"AdaptAbort: the raw dump {third.parent.parent / 'model_server' / 'raw.jsonl'} of the model server cannot be "
        "read: line 3 of the raw dump is not a generation"
    )
    assert json.loads((first / "raw_audit.json").read_text())["generations"] == 1
    raw = run.adapt_dir / "model_server" / "raw.jsonl"
    assert world.audits == [(raw, None)] * 4  # every attempt, on its own model server's dump, as a server dump
    audit = json.loads((run.adapt_dir / "raw_audit.json").read_text(encoding="utf-8"))
    assert (audit["where"], audit["generations"]) == (str(raw), 1)
    assert result["evolutions"] == 12
    # in every attempt the audit comes after the drain and both guards
    assert [call[0] for call in world.calls if call[0] in ("guards", "audit")] == ["guards", "audit"] * 4


def test_inkling_small_with_the_real_shared_audit(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, SHARED_AUDIT)  # the module itself, not the world's stand-in
    unparsed = '<|message_model|>exec<|content_invoke_tool_json|>{"name":"exec"}<|content_model_end_sampling|>'
    world.scripts = [Script(raw_text=unparsed), Script(raw_text=None)]
    result = world.adapt("inkling-small", "drive-create")
    run = world.run_dir("inkling-small", "drive-create")
    first, second = voids(run)
    assert why(first)["reason"].startswith("AdaptAbort: the raw audit of the model server's generations failed: ")
    assert "1 generation(s) with tool-call markup that parsed to nothing" in why(first)["reason"]
    assert json.loads((first / "raw_audit.json").read_text())["unparsed_markup"] == 1
    empty = second.parent.parent / "model_server" / "raw.jsonl"
    assert why(second)["reason"] == (
        f"AdaptAbort: the raw audit of the model server's generations failed: {empty}: no raw generation"
    )
    audit = json.loads((run.adapt_dir / "raw_audit.json").read_text(encoding="utf-8"))
    assert (audit["generations"], audit["unparsed_markup"]) == (1, 0)
    assert result["evolutions"] == 12
    assert world.audits == []


def test_inkling_small_needs_the_shared_audit_before_anything_starts(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, SHARED_AUDIT, None)  # the module cannot be imported
    with pytest.raises(ModuleNotFoundError, match="inkling_audit"):
        world.adapt("inkling-small", "drive-create")
    assert world.calls == []
    assert world.adapt("qwen3.8-27b", "drive-create")["evolutions"] == 12  # the other models never import it


def test_other_models_are_not_audited(world: World) -> None:
    world.adapt("qwen3.8-27b")
    assert world.audits == []
    assert not (world.run_dir("qwen3.8-27b", "drive-create").adapt_dir / "raw_audit.json").exists()


def test_the_skill_order_must_hold_the_configured_number_of_conversations(world: World) -> None:
    world.cfgs = dataclasses.replace(world.cfgs, skills=world.cfgs.skills.model_copy(update={"conversations": 11}))
    with pytest.raises(ConfigError, match="12 conversations, but skills.yaml says 11"):
        world.adapt()
    assert world.calls == []


def test_the_shipped_skill_orders_hold_twelve_known_items() -> None:
    for name in ("file-create", "drive-create", "email-invite", "chat-message", "chat-invite"):
        order = load_skill_order(name)
        assert len(order) == len(set(order)) == 12
        assert set(order) <= set(load_task(name).adaptation)
