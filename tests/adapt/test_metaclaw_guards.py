"""The two checks run on a skill library's evolution: no turn dropped from the evolver's prompt, and the evolver's
excerpts taken from the conversation rather than from the system prompt."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from metaclaw_helpers import (
    GUARDS,
    WINDOW,
    OpenAIStub,
    conversation_of,
    event,
    flatten,
    long_turn,
    metaclaw_env,
    metaclaw_python,
    prompt_of,
    short_turn,
    turn_record,
    write_jsonl,
)

from evolve_ipi.adapt import metaclaw_guards, skills
from evolve_ipi.config import ConfigError, Secrets, load_configs
from evolve_ipi.paths import RuntimePaths


def run_main(argv: list[str]) -> tuple[int, str]:
    """``main`` in this process: the exit status and the message of a failure."""
    try:
        return metaclaw_guards.main(argv), ""
    except SystemExit as stop:
        assert isinstance(stop.code, str), stop.code
        return 1, stop.code


# --- no-turn-drop ------------------------------------------------------------------------------


def test_no_turn_drop_passes_and_reports_the_longest_episode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    history = write_jsonl(tmp_path / "h.jsonl", [event(2), event(5), event(3)])
    assert run_main(["no-turn-drop", str(history)]) == (0, "")
    assert capsys.readouterr().out == (
        "  evolver input: 3 event(s), every turn rendered, longest episode 5 turn(s) (~5500 chars of evolver prompt)\n"
    )


def test_no_turn_drop_adds_up_several_histories_and_skips_blank_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = write_jsonl(tmp_path / "a.jsonl", [event(2)])
    first.write_text(first.read_text() + "\n   \n", encoding="utf-8")
    second = write_jsonl(tmp_path / "b.jsonl", [event(7), event(1)])
    assert run_main(["no-turn-drop", str(first), str(second)]) == (0, "")
    assert "3 event(s), every turn rendered, longest episode 7 turn(s) (~7700 chars" in capsys.readouterr().out


def test_no_turn_drop_fails_on_a_dropped_turn(tmp_path: Path) -> None:
    history = write_jsonl(tmp_path / "h.jsonl", [event(3), event(7, prompts=["x"] * 6)])
    status, message = run_main(["no-turn-drop", str(history)])
    assert status == 1
    assert message.startswith(
        f"no-turn-drop: {history} line 2 -- 7 turns were fed to the evolver but only 6 were rendered into its "
        "prompt, so 1 turn(s) were silently dropped.\n"
    )
    assert "third_party/patches/metaclaw-skill-evolver.patch" in message


def test_no_turn_drop_counts_missing_excerpts_as_none_rendered(tmp_path: Path) -> None:
    line = event(0)
    del line["failure_prompts"]
    assert run_main(["no-turn-drop", str(write_jsonl(tmp_path / "h.jsonl", [line]))])[0] == 0
    line = event(2)
    line["failure_prompts"] = None
    status, message = run_main(["no-turn-drop", str(write_jsonl(tmp_path / "g.jsonl", [line]))])
    assert status == 1 and "2 turns were fed to the evolver but only 0 were rendered" in message


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ([{"failure_prompts": []}], "line 1 has no num_failures_analyzed; this is not a MetaClaw evolution history"),
        ([], "holds no evolution events"),
        (["not a json object"], "line 1 is not a JSON object"),
    ],
)
def test_no_turn_drop_refuses_what_is_not_a_history(tmp_path: Path, content: list[object], expected: str) -> None:
    history = write_jsonl(tmp_path / "h.jsonl", content)
    status, message = run_main(["no-turn-drop", str(history)])
    assert status == 1
    assert message == f"no-turn-drop: {history} {expected}"


def test_no_turn_drop_refuses_a_line_that_is_not_json(tmp_path: Path) -> None:
    history = tmp_path / "h.jsonl"
    history.write_text(json.dumps(event(1)) + "\n{broken\n", encoding="utf-8")
    status, message = run_main(["no-turn-drop", str(history)])
    assert status == 1
    assert message.startswith(f"no-turn-drop: {history} line 2 is not JSON")


def test_no_turn_drop_refuses_a_missing_history(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    assert run_main(["no-turn-drop", str(missing)]) == (1, f"no-turn-drop: no such history: {missing}")


@pytest.mark.parametrize("argv", [[], ["no-turn-drop"], ["frobnicate", "x"], ["saw-conversation", "only-one"]])
def test_wrong_usage_exits_1_with_the_usage(argv: list[str]) -> None:
    status, message = run_main(argv)
    assert status == 1
    assert message.startswith("usage:")


def test_no_turn_drop_as_a_script_needs_nothing_but_the_standard_library(tmp_path: Path) -> None:
    good = write_jsonl(tmp_path / "good.jsonl", [event(4)])
    bad = write_jsonl(tmp_path / "bad.jsonl", [event(4, prompts=["a"])])
    env = metaclaw_env(tmp_path)
    passed = subprocess.run(
        [sys.executable, str(GUARDS), "no-turn-drop", str(good)], capture_output=True, text=True, env=env, check=False
    )
    assert (passed.returncode, passed.stderr) == (0, "")
    assert "1 event(s), every turn rendered, longest episode 4 turn(s)" in passed.stdout
    failed = subprocess.run(
        [sys.executable, str(GUARDS), "no-turn-drop", str(bad)], capture_output=True, text=True, env=env, check=False
    )
    assert failed.returncode == 1
    assert failed.stderr.startswith(f"no-turn-drop: {bad} line 1 -- 4 turns were fed")


# --- saw-conversation --------------------------------------------------------------------------


def records_and_history(tmp_path: Path, excerpts: list[str]) -> tuple[Path, Path]:
    turns = [short_turn(), short_turn("What is on my calendar?"), long_turn()]
    records = write_jsonl(tmp_path / "conversations.jsonl", [turn_record(m, turn=i) for i, m in enumerate(turns, 1)])
    history = write_jsonl(tmp_path / "history.jsonl", [event(len(excerpts), prompts=excerpts)])
    return records, history


def test_saw_conversation_passes_with_teeth(tmp_path: Path) -> None:
    excerpts = [conversation_of(short_turn())[-WINDOW:], conversation_of(long_turn())[-WINDOW:]]
    records, history = records_and_history(tmp_path, excerpts)
    assert metaclaw_guards.saw_conversation(history, records, flatten) == [
        "  evolver input: 1 event(s), 2 turn excerpt(s), all from the conversation and not the system prompt",
        (
            "                 (2 recorded turn(s) had a conversation short enough for the two sources to differ, so "
            "the check has teeth)"
        ),
    ]


def test_saw_conversation_warns_when_no_turn_could_tell_the_sources_apart(tmp_path: Path) -> None:
    records = write_jsonl(tmp_path / "conversations.jsonl", [turn_record(long_turn())])
    history = write_jsonl(tmp_path / "history.jsonl", [event(1, prompts=[conversation_of(long_turn())[-WINDOW:]])])
    lines = metaclaw_guards.saw_conversation(history, records, flatten)
    assert lines[0].startswith("  evolver input: 1 event(s), 1 turn excerpt(s)")
    assert lines[1] == (
        "                 WARNING: no recorded turn had a conversation of 300 characters or fewer, so this run could "
        "not distinguish a patched evolver from an unpatched one. The pass above is vacuous."
    )


def test_saw_conversation_fails_on_an_excerpt_of_the_system_prompt(tmp_path: Path) -> None:
    from_prompt = prompt_of(short_turn())[-WINDOW:]  # what an unpatched evolver recorded
    records, history = records_and_history(tmp_path, [conversation_of(long_turn())[-WINDOW:], from_prompt])
    with pytest.raises(SystemExit) as stop:
        metaclaw_guards.saw_conversation(history, records, flatten)
    message = str(stop.value.code)
    assert message.startswith(
        f"saw-conversation: {history} line 1, failure 2 was NOT the tail of any recorded conversation. It BEGINS:"
        f"\n\n  {from_prompt[:200]!r}\n\n"
    )
    assert "third_party/patches/" in message


@pytest.mark.parametrize(
    ("records_rows", "history_rows", "expected"),
    [
        ([{"turn": 1}], [event(0)], "{records} line 1 has no `messages`"),
        ([], [event(0)], "{records} holds no turns"),
        ([turn_record(short_turn())], [{"num_failures_analyzed": 1}], "{history} line 1 has no failure_prompts"),
        ([turn_record(short_turn())], [], "{history} holds no evolution events"),
        (["no object"], [event(0)], "{records} line 1 is not a JSON object"),
    ],
)
def test_saw_conversation_refuses_what_it_cannot_check(
    tmp_path: Path, records_rows: list[object], history_rows: list[object], expected: str
) -> None:
    records = write_jsonl(tmp_path / "conversations.jsonl", records_rows)
    history = write_jsonl(tmp_path / "history.jsonl", history_rows)
    with pytest.raises(SystemExit) as stop:
        metaclaw_guards.saw_conversation(history, records, flatten)
    assert str(stop.value.code).startswith("saw-conversation: " + expected.format(records=records, history=history))


def test_saw_conversation_reads_the_records_before_the_history(tmp_path: Path) -> None:
    records = tmp_path / "missing-records.jsonl"
    history = tmp_path / "missing-history.jsonl"
    with pytest.raises(SystemExit) as stop:
        metaclaw_guards.saw_conversation(history, records, flatten)
    assert str(stop.value.code) == f"saw-conversation: no such file: {records}"


@pytest.mark.metaclaw
def test_saw_conversation_with_metaclaws_own_flattening(tmp_path: Path) -> None:
    python = metaclaw_python()
    good = [conversation_of(long_turn())[-WINDOW:], conversation_of(short_turn())[-WINDOW:]]
    records, history = records_and_history(tmp_path, good)
    env = metaclaw_env(tmp_path)
    argv = [str(python), str(GUARDS), "saw-conversation", str(history), str(records)]
    passed = subprocess.run(argv, capture_output=True, text=True, env=env, check=False)
    assert (passed.returncode, passed.stderr) == (0, "")
    assert "1 event(s), 2 turn excerpt(s), all from the conversation" in passed.stdout
    assert "check has teeth" in passed.stdout

    write_jsonl(history, [event(1, prompts=[prompt_of(short_turn())[-WINDOW:]])])
    failed = subprocess.run(argv, capture_output=True, text=True, env=env, check=False)
    assert failed.returncode == 1
    assert failed.stderr.startswith(f"saw-conversation: {history} line 1, failure 1 was NOT the tail")


# --- the evolver, before an evolution ------------------------------------------------------------------

KEY = "evolver-test-key"
REFUSED = {
    "error": {
        "message": f"Incorrect API key provided: {KEY}. You can find your API key at the usual place.",
        "type": "invalid_request_error",
        "param": None,
        "code": "invalid_api_key",
    }
}
NO_DEPLOYMENT = {
    "error": {"code": "DeploymentNotFound", "message": "The API deployment for this resource does not exist."}
}
# A provider that quotes the key in the code itself, alone or in a sentence.
KEY_AS_CODE = {"error": {"code": KEY, "message": "refused"}}
KEY_IN_CODE = {"error": {"code": f"invalid key {KEY}", "message": "refused"}}


def evolver_check(tmp_path: Path, base_url: str) -> subprocess.CompletedProcess[str]:
    env = metaclaw_env(tmp_path, OPENAI_BASE_URL=base_url, OPENAI_API_KEY=KEY, SKILL_EVOLVER_MODEL="stub-model")
    argv = [str(metaclaw_python()), str(GUARDS), "evolver", "3000"]
    return subprocess.run(argv, capture_output=True, text=True, env=env, timeout=300, check=False)


@pytest.mark.metaclaw
def test_the_evolver_check_sends_the_evolvers_own_request(tmp_path: Path) -> None:
    with OpenAIStub() as stub:
        done = evolver_check(tmp_path, stub.base_url)
    assert done.returncode == 0, done.stderr
    assert done.stdout == f"  evolver: {stub.base_url} (model stub-model) answered\n"
    (request,) = stub.requests
    assert request["path"] == "/v1/chat/completions"
    assert request["body"] == {
        "model": "stub-model",
        "messages": [{"role": "user", "content": "Reply with the single word OK."}],
        "max_completion_tokens": 3000,
    }
    assert request["headers"]["authorization"] == f"Bearer {KEY}"


ANSWER_EMPTY = {
    "id": "chatcmpl-2",
    "object": "chat.completion",
    "created": 0,
    "model": "stub-model",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": None}, "finish_reason": "length"}],
}


@pytest.mark.metaclaw
def test_the_evolver_check_accepts_a_reply_without_text(tmp_path: Path) -> None:
    # a reasoning model can spend the budget on reasoning: an answer without text still proves endpoint, key and model
    with OpenAIStub(reply=ANSWER_EMPTY) as stub:
        done = evolver_check(tmp_path, stub.base_url)
    assert done.returncode == 0, done.stderr


@pytest.mark.metaclaw
@pytest.mark.parametrize(
    ("status", "reply", "named"),
    [
        (401, REFUSED, "AuthenticationError (HTTP 401, invalid_api_key)"),
        (404, NO_DEPLOYMENT, "NotFoundError (HTTP 404, DeploymentNotFound)"),
        (401, KEY_AS_CODE, "AuthenticationError (HTTP 401, <code withheld>)"),
        (401, KEY_IN_CODE, "AuthenticationError (HTTP 401, <code withheld>)"),
    ],
)
def test_the_evolver_check_names_what_failed_and_not_the_key(
    tmp_path: Path, status: int, reply: dict[str, object], named: str
) -> None:
    """The endpoint and the model are named once, by the caller (``skills.evolver_preflight``)."""
    with OpenAIStub(status, reply) as stub:
        done = evolver_check(tmp_path, stub.base_url)
    assert done.returncode == 1
    assert done.stderr == f"evolver: refused the request: {named}\n"
    assert KEY not in done.stdout + done.stderr


@pytest.mark.metaclaw
def test_the_evolver_check_on_an_endpoint_nobody_listens_on(tmp_path: Path) -> None:
    with OpenAIStub() as stub:
        base_url = stub.base_url  # closed when the block ends: nothing listens there any more
    done = evolver_check(tmp_path, base_url)
    assert done.returncode == 1
    assert done.stderr == "evolver: could not be reached: APIConnectionError\n"


@pytest.mark.metaclaw
def test_evolver_preflight_with_metaclaws_python(tmp_path: Path) -> None:
    spec = load_configs(Path(__file__).resolve().parents[2] / "configs").skills
    paths = RuntimePaths(root=tmp_path / "runtime", agent_home=tmp_path / "home", metaclaw_python=metaclaw_python())
    with OpenAIStub() as stub:
        secrets = Secrets(evolver_base_url=stub.base_url, evolver_api_key=KEY, evolver_model="stub-model")
        skills.evolver_preflight(secrets, spec, paths=paths)
        assert len(stub.requests) == 1
    with OpenAIStub(401, REFUSED) as stub:
        secrets = Secrets(evolver_base_url=stub.base_url, evolver_api_key=KEY, evolver_model="stub-model")
        with pytest.raises(ConfigError) as error:
            skills.evolver_preflight(secrets, spec, paths=paths)
    message = str(error.value)
    assert message == (
        f"the skill evolver at {stub.base_url} (model stub-model) cannot be used: evolver: refused the request: "
        "AuthenticationError (HTTP 401, invalid_api_key)"
    )
    assert KEY not in message


@pytest.mark.parametrize(
    ("code", "key", "printed"),
    [
        ("invalid_api_key", KEY, "invalid_api_key"),
        ("DeploymentNotFound", KEY, "DeploymentNotFound"),
        (429, KEY, "429"),
        ("rate.limit-exceeded_2", "", "rate.limit-exceeded_2"),
        (KEY, KEY, "<code withheld>"),  # an identifier, but the key
        (f"x{KEY}x", KEY, "<code withheld>"),
        (f"Incorrect API key provided: {KEY}", KEY, "<code withheld>"),
        ("a" * 65, KEY, "<code withheld>"),
        ("", KEY, "<code withheld>"),
        ("line\nbreak", KEY, "<code withheld>"),
    ],
)
def test_an_error_code_is_printed_only_as_a_short_identifier_without_the_key(
    code: object, key: str, printed: str
) -> None:
    assert metaclaw_guards.error_code(code, key) == printed
