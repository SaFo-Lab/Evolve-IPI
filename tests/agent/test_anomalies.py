"""The anomaly rules of one episode, the validity of an evaluation pass, the training policies and the watched logs."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from evolve_ipi.agent import anomalies
from evolve_ipi.agent.anomalies import PassVerdict, TrainingPolicy, classify, detect
from evolve_ipi.config import ModelSpec
from evolve_ipi.tasks import Task, load_task

READ_ONLY = "drive-create"  # judge_utility "none": utility is a non-empty reply
ACTION = "chat-invite"  # judge_utility "dm_to"

T = 600


@pytest.fixture(scope="module")
def tasks() -> dict[str, Task]:
    return {name: load_task(name) for name in ("file-create", READ_ONLY, ACTION)}


def item_of(tasks: dict[str, Task], name: str) -> dict[str, Any]:
    return copy.deepcopy(next(iter(tasks[name].test.values())))


def service_record(**fields: Any) -> dict[str, Any]:
    """What a service episode recorded: a turn that went well and one successful read of the carrier."""
    record = {
        "id": "x",
        "status": "ok",
        "error": None,
        "elapsed": 12.3,
        "reply": "It says hello.",
        "encountered": True,
        "access_log": [{"time": "t", "program": "drive", "argv": ["read", "x"], "rc": 0}],
    }
    record.update(fields)
    return record


def file_record(**fields: Any) -> dict[str, Any]:
    record = {"id": "file-create-test-200", "status": "ok", "error": None, "elapsed": 8.4, "reply": "It says hello."}
    record.update(encountered=True, tool_calls=[{"name": "read", "arguments": {"path": "x"}}])
    record.update(fields)
    return record


def types(found: list[dict[str, str]]) -> list[str]:
    return [a["type"] for a in found]


# --- the type sets ---------------------------------------------------------------------------


def test_the_pipeline_and_behaviour_types() -> None:
    assert anomalies.PIPELINE == {
        "timeout",
        "agent_error",
        "state_error",
        "ids_mismatch",
        "reasoning_orphans",
        "aborted",
    }
    assert anomalies.BEHAVIOR == {"no_calls", "empty_reply", "not_encountered"}
    assert anomalies.NONENGAGEMENT == ("no_calls", "empty_reply")
    assert anomalies.FATAL_IN_TRAINING == ("timeout", "agent_error", "state_error")


# --- detect: service tasks ---------------------------------------------------------------------


def test_a_clean_service_episode_has_no_anomaly(tasks: dict[str, Task]) -> None:
    for name in (READ_ONLY, ACTION):
        assert detect(service_record(), "service", item=item_of(tasks, name), timeout_s=T) == []


def test_a_failed_turn_is_recorded_with_its_error_in_full(tasks: dict[str, Task]) -> None:
    error = "exit 1: " + "x" * 3000
    found = detect(
        service_record(status="agent_error", error=error, reply="", access_log=[], encountered=False),
        "service",
        item=item_of(tasks, READ_ONLY),
        timeout_s=T,
    )
    assert found == [
        {"type": "agent_error", "detail": error},
        {"type": "no_calls", "detail": "access.log is empty: no service program ran"},
        {"type": "empty_reply", "detail": "read-only group: the reply is empty"},
    ]


def test_a_cut_off_turn_is_a_timeout(tasks: dict[str, Task]) -> None:
    found = detect(
        service_record(status="timeout", error="no result within 660 s", reply="", session_id=None),
        "service",
        item=item_of(tasks, ACTION),
        timeout_s=T,
    )
    assert found == [{"type": "timeout", "detail": "no result within 660 s"}]


@pytest.mark.parametrize(("elapsed", "flagged"), [(599.9, False), (600.0, True), (605.0, True)])
def test_a_turn_that_took_the_limit_is_a_timeout(tasks: dict[str, Task], elapsed: float, flagged: bool) -> None:
    found = detect(service_record(elapsed=elapsed), "service", item=item_of(tasks, ACTION), timeout_s=T)
    expected = [{"type": "timeout", "detail": f"the turn took {elapsed:.1f}s, at or beyond the 600s limit"}]
    assert found == (expected if flagged else [])


def test_no_calls_not_encountered_and_empty_reply(tasks: dict[str, Task]) -> None:
    item = item_of(tasks, READ_ONLY)
    program, prefix = item["encounter"]["program"], " ".join(item["encounter"]["argv_prefix"])
    assert types(detect(service_record(access_log=[], encountered=False), "service", item=item, timeout_s=T)) == [
        "no_calls"
    ]
    other_call = [{"time": "t", "program": "drive", "argv": ["list"], "rc": 0}]
    assert detect(service_record(access_log=other_call, encountered=False), "service", item=item, timeout_s=T) == [
        {"type": "not_encountered", "detail": f"no successful `{program} {prefix}` call in access.log"}
    ]
    assert detect(service_record(reply="  \n "), "service", item=item, timeout_s=T) == [
        {"type": "empty_reply", "detail": "read-only group: the reply is empty"}
    ]


def test_an_empty_reply_is_no_anomaly_when_the_task_has_an_action(tasks: dict[str, Task]) -> None:
    assert detect(service_record(reply=""), "service", item=item_of(tasks, ACTION), timeout_s=T) == []


def test_state_errors_are_pipeline_anomalies_and_hide_no_calls(tasks: dict[str, Task]) -> None:
    item = item_of(tasks, ACTION)
    found = detect(
        service_record(access_log=[], encountered=False),
        "service",
        item=item,
        timeout_s=T,
        state_errors=["access.log line 1 is not JSON", "state file slack.json is missing"],
    )
    assert found == [
        {"type": "state_error", "detail": "access.log line 1 is not JSON"},
        {"type": "state_error", "detail": "state file slack.json is missing"},
    ]


def test_the_order_of_the_anomalies(tasks: dict[str, Task]) -> None:
    found = detect(
        service_record(
            elapsed=650.0, reply="", encountered=False, access_log=[{"program": "drive", "argv": ["list"], "rc": 0}]
        ),
        "service",
        item=item_of(tasks, READ_ONLY),
        timeout_s=T,
        state_errors=["broken"],
    )
    assert types(found) == ["timeout", "state_error", "not_encountered", "empty_reply"]


def test_turn_only_gives_the_anomaly_of_the_turn_alone(tasks: dict[str, Task]) -> None:
    record = service_record(status="agent_error", error="boom", reply="", access_log=[], encountered=False)
    found = detect(record, "service", item=item_of(tasks, READ_ONLY), timeout_s=T, turn_only=True)
    assert found == [{"type": "agent_error", "detail": "boom"}]


# --- detect: File-Create ---------------------------------------------------------------------


def test_a_clean_file_episode_has_no_anomaly(tasks: dict[str, Task]) -> None:
    item = tasks["file-create"].test["file-create-test-200"]
    assert detect(file_record(), "file", item=item, timeout_s=300) == []


def test_file_anomalies_name_the_episode(tasks: dict[str, Task]) -> None:
    item = tasks["file-create"].test["file-create-test-200"]
    name = item["src_relpath"].rsplit("/", 1)[1]
    error = "exit 1: " + "y" * 900
    found = detect(
        file_record(status="agent_error", error=error, reply="", encountered=False, session_id=None),
        "file",
        item=item,
        timeout_s=300,
    )
    rid = "file-create-test-200"
    assert found == [
        {"id": rid, "type": "agent_error", "detail": error[:500]},
        {"id": rid, "type": "empty_reply", "detail": "read-only request: the reply is empty"},
        {"id": rid, "type": "not_encountered", "detail": f"no tool call names {name}"},
    ]


@pytest.mark.parametrize(("elapsed", "flagged"), [(299.9, False), (300.0, True), (303.4, True)])
def test_elapsed_at_the_limit_on_file_create_is_a_timeout(
    tasks: dict[str, Task], elapsed: float, flagged: bool
) -> None:
    item = tasks["file-create"].test["file-create-test-200"]
    found = detect(file_record(elapsed=elapsed), "file", item=item, timeout_s=300)
    expected = [
        {
            "id": "file-create-test-200",
            "type": "timeout",
            "detail": f"the turn took {elapsed:.1f}s, at or beyond the 300s limit",
        }
    ]
    assert found == (expected if flagged else [])


def test_detect_refuses_an_unknown_kind(tasks: dict[str, Task]) -> None:
    with pytest.raises(ValueError, match="kind"):
        detect(service_record(), "web", item=item_of(tasks, ACTION), timeout_s=T)  # type: ignore[arg-type]


# --- loops -----------------------------------------------------------------------------------


def timeline(start: float, calls: list[float], *, users: tuple[float, ...] = ()) -> list[tuple[float, str]]:
    marks = [(start, "user"), *((t, "user") for t in users)]
    marks += [(t, "tool_call") for t in calls]
    return sorted(marks)


def evenly(n: int, first: float, last: float) -> list[float]:
    if n == 1:
        return [first]
    step = (last - first) / (n - 1)
    return [first + i * step for i in range(n)]


@pytest.mark.parametrize(("n", "looped"), [(19, False), (20, True)])
def test_a_loop_needs_twenty_tool_calls(n: int, looped: bool) -> None:
    calls = evenly(n, 5.0, 595.0)  # gaps of at most 31 s
    assert anomalies.loop_excuse(timeline(0.0, calls), 600.0) is looped


@pytest.mark.parametrize(("gap", "looped"), [(120.0, True), (121.0, False)])
def test_a_loop_has_no_silence_longer_than_120_seconds(gap: float, looped: bool) -> None:
    calls = [*evenly(20, 1.0, 20.0), 20.0 + gap]  # the last call ends the silence; the end follows 1 s later
    assert anomalies.loop_excuse(timeline(0.0, calls), 20.0 + gap + 1.0) is looped


def test_the_loop_is_decided_on_the_gap_before_it_is_rounded() -> None:
    """120.04 s is reported as 120.0 but is a silence longer than the 120 s a loop allows."""
    calls = [*evenly(20, 1.0, 20.0), 20.0 + 120.04]
    evidence = anomalies.loop_evidence(timeline(0.0, calls), 20.0 + 120.04 + 1.0)
    assert evidence["max_gap_s"] == 120.0 and evidence["loop"] is False


@pytest.mark.parametrize(
    ("start_gap", "end_gap", "looped"), [(120.0, 120.0, True), (121.0, 1.0, False), (1.0, 121.0, False)]
)
def test_the_silences_before_the_first_and_after_the_last_call_count(
    start_gap: float, end_gap: float, looped: bool
) -> None:
    calls = evenly(25, start_gap, start_gap + 240.0)
    assert anomalies.loop_excuse(timeline(0.0, calls), start_gap + 240.0 + end_gap) is looped


def test_loop_evidence_reports_the_timeline() -> None:
    calls = evenly(21, 4.5, 597.9)
    evidence = anomalies.loop_evidence(timeline(1000.0, [1000.0 + c for c in calls]), 605.0)
    assert evidence == {
        "loop": True,
        "tool_calls": 21,
        "max_gap_s": round(max(b - a for a, b in zip([0.0, *calls, 605.0], [*calls, 605.0])), 1),
        "first_call_after_s": 4.5,
        "last_call_before_end_s": 7.1,
    }


def test_every_tool_call_counts_but_only_those_after_the_start_bound_the_gaps() -> None:
    # 19 calls inside the turn and one before its first user message: twenty calls, and the gaps are those of the turn.
    inside = evenly(19, 10.0, 590.0)
    marks = [(-50.0, "tool_call"), (0.0, "user"), *((t, "tool_call") for t in inside)]
    evidence = anomalies.loop_evidence(marks, 600.0)
    assert evidence["tool_calls"] == 20
    assert evidence["loop"] is True
    assert evidence["first_call_after_s"] == -50.0


def test_calls_after_the_end_are_not_cut_off() -> None:
    # The marks are the start, every call at or after it in transcript order, and the end; a call after the end only
    # adds a negative step. This is the rule the recorded verdicts were made with.
    calls = evenly(20, 5.0, 100.0)
    evidence = anomalies.loop_evidence(timeline(0.0, calls), 50.0)
    assert evidence["loop"] is True
    assert evidence["max_gap_s"] == 5.0
    assert evidence["last_call_before_end_s"] == -50.0


def test_the_start_is_the_first_user_message() -> None:
    calls = evenly(20, 130.0, 590.0)
    # A second user message (feedback) does not move the start: the silence before the first call is 130 s.
    assert anomalies.loop_excuse(timeline(0.0, calls, users=(125.0,)), 600.0) is False


def test_no_loop_without_a_user_message_or_an_elapsed_time() -> None:
    calls = [(float(t), "tool_call") for t in range(30)]
    assert anomalies.loop_evidence(calls, 600.0) == {
        "loop": False,
        "reason": "no user message or no elapsed time",
        "tool_calls": 30,
    }
    assert anomalies.loop_excuse(timeline(0.0, evenly(30, 1.0, 590.0)), None) is False


def test_a_turn_with_no_call_is_no_loop() -> None:
    assert anomalies.loop_evidence([(0.0, "user")], 600.0) == {
        "loop": False,
        "tool_calls": 0,
        "max_gap_s": 600.0,
        "first_call_after_s": None,
        "last_call_before_end_s": None,
    }


# --- classify --------------------------------------------------------------------------------


def eval_record(rid: str, *found: str, **fields: Any) -> dict[str, Any]:
    record = {
        "id": rid,
        "status": "ok",
        "elapsed": 20.0,
        "session_id": f"s-{rid}",
        "anomalies": [{"type": t, "detail": "d"} for t in found],
    }
    record.update(fields)
    return record


def looping(start: float = 0.0, elapsed: float = 605.0) -> list[tuple[float, str]]:
    return timeline(start, [start + t for t in evenly(40, 3.0, elapsed - 4.0)])


def stalling(start: float = 0.0) -> list[tuple[float, str]]:
    return timeline(start, [start + t for t in evenly(25, 3.0, 300.0)])  # silent for the last 305 s


def test_a_pass_with_behaviour_anomalies_only_is_valid() -> None:
    records = [eval_record("a"), eval_record("b", "no_calls", "empty_reply"), eval_record("c", "not_encountered")]
    verdict = classify(records, {}, None)
    assert verdict == PassVerdict(valid=True, reasons=[], evidence=verdict.evidence)
    assert verdict.valid


@pytest.mark.parametrize("kind", sorted(anomalies.PIPELINE))
def test_a_pipeline_anomaly_invalidates_the_pass(kind: str) -> None:
    verdict = classify([eval_record("a"), eval_record("b", kind)], {}, None)
    assert not verdict.valid
    assert verdict.reasons == [f"pipeline anomaly {kind} at b"]


def test_an_unknown_anomaly_type_invalidates_the_pass() -> None:
    verdict = classify([eval_record("a", "something_new")], {}, None)
    assert not verdict.valid
    assert verdict.reasons == ["pipeline anomaly something_new at a"]


def test_a_timeout_in_a_loop_counts_as_behaviour() -> None:
    records = [eval_record("a", "timeout", elapsed=605.0), eval_record("b")]
    verdict = classify(records, {"a": looping()}, None)
    assert verdict.valid, verdict.reasons
    (row,) = verdict.evidence["loops"]
    assert row["id"] == "a" and row["loop"] is True and row["excused"] == ["timeout"] and row["limits"] == ["time"]


def test_a_timeout_without_a_loop_or_a_transcript_invalidates_the_pass() -> None:
    records = [eval_record("a", "timeout", elapsed=605.0), eval_record("b", "timeout", elapsed=605.0)]
    verdict = classify(records, {"a": stalling()}, None)
    assert not verdict.valid
    assert verdict.reasons == ["pipeline anomaly timeout at a", "pipeline anomaly timeout at b"]
    rows = {row["id"]: row for row in verdict.evidence["loops"]}
    assert rows["a"]["loop"] is False and rows["a"]["max_gap_s"] > 120
    assert rows["b"] == {
        "id": "b",
        "session_id": "s-b",
        "elapsed": 605.0,
        "limits": ["time"],
        "excused": [],
        "loop": False,
        "reason": "no transcript found",
    }


def test_a_cut_off_turn_is_looked_at_too_but_has_no_transcript() -> None:
    record = eval_record("a", "timeout", status="timeout", session_id=None, elapsed=660.0)
    verdict = classify([record], {}, None)
    assert not verdict.valid
    assert verdict.evidence["loops"][0]["reason"] == "no transcript found"


def test_a_loop_excuses_a_timeout_only() -> None:
    verdict = classify([eval_record("a", "timeout", "agent_error", elapsed=605.0)], {"a": looping()}, None)
    assert verdict.reasons == ["pipeline anomaly agent_error at a"]


def test_duplicate_episode_ids_are_refused() -> None:
    with pytest.raises(ValueError, match="twice"):
        classify([eval_record("a"), eval_record("a")], {}, None)


# --- classify: generations outside every episode ---------------------------------------------


def generation(request: int, at: str) -> dict[str, Any]:
    return {"at": at, "request": request, "messages": 3, "n_tokens": 10, "raw": "x"}


def owned(record: dict[str, Any], *requests: int) -> dict[str, Any]:
    record["reasoning"] = [{"request": r, "n_tokens": 10, "thinking": None, "output": "x"} for r in requests]
    return record


def at(second: int) -> str:
    return f"2026-10-05T03:59:{second:02d}-0400"


def test_every_generation_inside_an_episode_window() -> None:
    records = [owned(eval_record("a"), 1, 2), owned(eval_record("b"), 3)]
    raw = [generation(1, at(0)), generation(2, at(1)), generation(3, at(2))]
    verdict = classify(records, {}, raw)
    assert verdict.valid and verdict.evidence["orphans"] == []


@pytest.mark.parametrize(("lag", "valid"), [(30, True), (31, False)])
def test_an_orphan_right_after_a_loop_is_excused(lag: int, valid: bool) -> None:
    records = [owned(eval_record("a", "timeout", elapsed=605.0), 1, 2), owned(eval_record("b"), 4)]
    raw = [generation(1, at(0)), generation(2, at(10)), generation(3, at(10 + lag)), generation(4, at(50))]
    verdict = classify(records, {"a": looping()}, raw)
    assert verdict.valid is valid
    (row,) = verdict.evidence["orphans"]
    assert row == {
        "raw_line": 2,
        "request": 3,
        "at": at(10 + lag),
        "after_episode": "a",
        "next_episode": "b",
        "lag_s": float(lag),
        "explained": valid,
    }
    if not valid:
        assert verdict.reasons == [
            "1 of the 4 generations in the raw dump lie outside every episode's window and no loop explains them"
        ]


def test_an_orphan_written_before_the_last_generation_of_the_loop_is_not_excused() -> None:
    """The lag is bounded below as well: a generation stamped before the one it follows is no in-flight request."""
    records = [owned(eval_record("a", "timeout", elapsed=605.0), 1, 2), owned(eval_record("b"), 4)]
    raw = [generation(1, at(0)), generation(2, at(10)), generation(3, at(5)), generation(4, at(50))]
    verdict = classify(records, {"a": looping()}, raw)
    (row,) = verdict.evidence["orphans"]
    assert (row["after_episode"], row["next_episode"], row["lag_s"], row["explained"]) == ("a", "b", -5.0, False)
    assert not verdict.valid


def test_an_orphan_after_an_episode_that_did_not_loop_invalidates_the_pass() -> None:
    records = [owned(eval_record("a"), 1), owned(eval_record("b"), 3)]
    verdict = classify(records, {}, [generation(1, at(0)), generation(2, at(2)), generation(3, at(9))])
    assert not verdict.valid


def test_an_orphan_at_the_end_after_a_loop_is_excused() -> None:
    records = [owned(eval_record("a"), 1), owned(eval_record("b", "timeout", elapsed=605.0), 2)]
    verdict = classify(records, {"b": looping()}, [generation(1, at(0)), generation(2, at(5)), generation(3, at(7))])
    assert verdict.valid, verdict.reasons
    assert verdict.evidence["orphans"][0]["next_episode"] is None


def test_an_orphan_before_the_first_episode_or_between_non_neighbours_is_not_excused() -> None:
    records = [owned(eval_record("a", "timeout", elapsed=605.0), 2), owned(eval_record("b"), 4)]
    raw = [generation(1, at(0)), generation(2, at(1)), generation(3, at(2)), generation(4, at(3))]
    verdict = classify(records, {"a": looping()}, raw)
    explained = {row["request"]: row["explained"] for row in verdict.evidence["orphans"]}
    assert explained == {1: False, 3: True}
    assert not verdict.valid
    # The next owned generation belongs to an episode two places on: the orphan is not between neighbours.
    records = [
        owned(eval_record("a", "timeout", elapsed=605.0), 1),
        eval_record("b"),
        owned(eval_record("c"), 3),
    ]
    verdict = classify(records, {"a": looping()}, [generation(1, at(0)), generation(2, at(1)), generation(3, at(2))])
    assert verdict.evidence["orphans"][0]["explained"] is False


def test_without_a_raw_dump_there_is_nothing_to_account_for() -> None:
    verdict = classify([eval_record("a")], {}, None)
    assert verdict.valid and verdict.evidence["orphans"] is None


# --- classify: a context overflow inside a loop ------------------------------------------------


def incident(episode: str | None) -> dict[str, Any]:
    return {"source": "model_server.log", "line": "... exceeds the model's context window ...", "episode": episode}


def test_an_overflow_inside_a_looping_episode_excuses_its_failure() -> None:
    records = [eval_record("a", "agent_error", elapsed=320.0), eval_record("b")]
    verdict = classify(records, {"a": looping(elapsed=320.0)}, None, overflow_incidents=[incident("a")])
    assert verdict.valid, verdict.reasons
    (row,) = verdict.evidence["loops"]
    assert row["limits"] == ["context"] and row["excused"] == ["agent_error", "timeout"]


@pytest.mark.parametrize("where", [None, "b", "zz"])
def test_an_overflow_outside_a_looping_episode_invalidates_the_pass(where: str | None) -> None:
    records = [eval_record("a", "agent_error", elapsed=320.0), eval_record("b", elapsed=605.0)]
    verdict = classify(
        records, {"a": looping(elapsed=320.0), "b": stalling()}, None, overflow_incidents=[incident(where)]
    )
    assert not verdict.valid
    assert any(reason.startswith("context-overflow incident") for reason in verdict.reasons)
    assert "pipeline anomaly agent_error at a" in verdict.reasons


# --- training policies -----------------------------------------------------------------------


def model(retries: int) -> ModelSpec:
    return ModelSpec(
        name="m",
        tinker_id="org/m",
        family="harmony",
        skill_served_name="m",
        mechanisms=("rl", "skills"),
        skill_train_retries=retries,
    )


def test_what_is_fatal_under_each_policy() -> None:
    assert TrainingPolicy("continue", 0).fatal == ("timeout", "agent_error", "state_error")
    assert TrainingPolicy("abort", 4).fatal == ("timeout", "agent_error", "state_error", "no_calls", "empty_reply")


def test_a_policy_is_checked() -> None:
    with pytest.raises(ValueError, match="continue"):
        TrainingPolicy("continue", 2)
    with pytest.raises(ValueError, match="nonengagement"):
        TrainingPolicy("skip", 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="retries"):
        TrainingPolicy("abort", -1)


def test_the_policy_of_each_combination() -> None:
    for kind in ("file", "service"):
        assert anomalies.policy_for("rl", kind, model(4)) == TrainingPolicy("continue", 0)
    assert anomalies.policy_for("skills", "file", model(4)) == TrainingPolicy("continue", 0)
    assert anomalies.policy_for("skills", "service", model(4)) == TrainingPolicy("abort", 4)
    assert anomalies.policy_for("skills", "service", model(0)) == TrainingPolicy("abort", 0)
    with pytest.raises(ValueError, match="mechanism"):
        anomalies.policy_for("dpo", "service", model(0))  # type: ignore[arg-type]


def test_fatal_anomalies_and_when_a_retry_is_allowed() -> None:
    abort = TrainingPolicy("abort", 4)
    found = [{"type": "not_encountered", "detail": ""}, {"type": "no_calls", "detail": ""}]
    assert abort.fatal_anomalies(found) == [found[1]]
    assert abort.retryable(found)
    assert not abort.retryable([*found, {"type": "timeout", "detail": ""}])
    assert not abort.retryable([{"type": "not_encountered", "detail": ""}])
    keep_going = TrainingPolicy("continue", 0)
    assert keep_going.fatal_anomalies(found) == []
    assert not keep_going.retryable(found)


# --- watched logs ----------------------------------------------------------------------------

FATAL_LINES = [
    'INFO:     127.0.0.1:50532 - "POST /v1/chat/completions HTTP/1.1" 429 Too Many Requests',
    'INFO:     127.0.0.1:50532 - "POST /v1/chat/completions HTTP/1.1" 503 Service Unavailable',
    "12:00:01 | WARNING | rate limit exceeded, backing off",
    "openai.RateLimitError: slow down",
    "Error code: 500 - {'error': 'internal'}",
    "error code: 429",
    "prompt of 32775 tokens exceeds the model's context window of 32768",
    "Traceback (most recent call last):",
    "12:00:02 | ERROR | Exception logged for session ID 1234",
]


@pytest.mark.parametrize("line", FATAL_LINES)
def test_each_fatal_pattern(line: str) -> None:
    assert anomalies.fatal_log_lines(f"12:00:00 | INFO | fine\n{line}\n", "model_server.log") == [
        f"model_server.log: {line.strip()[:300]}"
    ]


def test_ordinary_lines_are_not_fatal() -> None:
    text = (
        'INFO:     127.0.0.1:1 - "POST /v1/chat/completions HTTP/1.1" 200 OK\n'
        'INFO:     127.0.0.1:1 - "POST /v1/chat/completions HTTP/1.1" 4290 odd\n'
        "12:00:00 | INFO | error code: 4290 is not a status\n"
    )
    assert anomalies.fatal_log_lines(text, "x.log") == []


def test_a_fatal_line_is_cut_at_300_characters_and_every_one_is_reported() -> None:
    long = "Traceback (most recent call last): " + "z" * 400
    found = anomalies.fatal_log_lines(f"{long}\nRateLimitError\n", "trainer.log")
    assert found == [f"trainer.log: {long[:300]}", "trainer.log: RateLimitError"]


def test_the_last_line_counts_without_a_newline() -> None:
    assert anomalies.fatal_log_lines("ok\nRateLimitError", "x.log") == ["x.log: RateLimitError"]


def test_agent_content_the_proxy_transcribes_is_skipped() -> None:
    text = (
        "2026-10-04 17:40:01 INFO next_state role=tool len=812: python3 svc/drive list\n"
        "Traceback (most recent call last):\n"
        '  File "svc/drive", line 3, in <module>\n'
        "RateLimitError: this is the agent's own command output\n"
        "2026-10-04 17:40:02 INFO [OpenClaw] tool_calls: [{'name': 'exec', 'arguments': 'Too Many Requests'}]\n"
        "    Error code: 500 inside the arguments\n"
        "2026-10-04 17:40:03 INFO request done\n"
    )
    assert anomalies.fatal_log_lines(text, "proxy.log") == []


def test_a_new_log_record_ends_the_transcribed_content() -> None:
    text = (
        "17:40:01 | INFO | next_state role=tool len=12: hello\n"
        "  still the tool's text: Traceback (most recent call last)\n"
        "ERROR:    Exception in ASGI application\n"
        "Traceback (most recent call last):\n"
    )
    assert anomalies.fatal_log_lines(text, "proxy.log") == ["proxy.log: Traceback (most recent call last):"]


def test_the_text_before_a_content_mark_is_still_read() -> None:
    line = "17:40:01 | ERROR | RateLimitError next_state role=tool len=3: abc"
    assert anomalies.fatal_log_lines(line + "\n", "proxy.log") == [f"proxy.log: {line}"]


OVERFLOW_RECORD = (
    "21:20:00 | INFO | sampling\n"
    "21:20:01 | ERROR | prompt of 32775 tokens exceeds the model's context window of 32768\n"
    "21:20:01 | ERROR | Exception logged for session ID abc\n"
    'INFO:     127.0.0.1:5 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error\n'
    "ERROR:    Exception in ASGI application\n"
    "Traceback (most recent call last):\n"
    '  File "api_server.py", line 1, in handle\n'
    "\n"
    "During handling of the above exception, another exception occurred:\n"
    "tinker.BadRequestError: prompt exceeds the model's context window\n"
    "21:20:05 | INFO | next request\n"
)


def test_an_overflow_stops_a_training_run() -> None:
    found = anomalies.fatal_log_lines(OVERFLOW_RECORD, "trainer.log")
    assert found == [
        "trainer.log: 21:20:01 | ERROR | prompt of 32775 tokens exceeds the model's context window of 32768",
        "trainer.log: 21:20:01 | ERROR | Exception logged for session ID abc",
        'trainer.log: INFO:     127.0.0.1:5 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error',
        "trainer.log: Traceback (most recent call last):",
        "trainer.log: tinker.BadRequestError: prompt exceeds the model's context window",
    ]


def test_in_an_evaluation_an_overflow_record_is_an_incident() -> None:
    scan = anomalies.scan_log(OVERFLOW_RECORD, "model_server.log", defer_overflow=True)
    assert scan.fatal == []
    assert scan.incidents == [
        {
            "source": "model_server.log",
            "line": "21:20:01 | ERROR | prompt of 32775 tokens exceeds the model's context window of 32768",
            "line_time": "21:20:01",
        }
    ]


def test_another_fatal_line_inside_an_overflow_record_still_stops_the_run() -> None:
    text = OVERFLOW_RECORD.replace("21:20:05 | INFO | next request\n", "") + "RateLimitError\n"
    scan = anomalies.scan_log(text, "model_server.log", defer_overflow=True)
    assert len(scan.incidents) == 1
    assert scan.fatal == ["model_server.log: RateLimitError"]


def test_a_traceback_that_ends_in_another_error_inside_an_overflow_record_is_fatal() -> None:
    text = OVERFLOW_RECORD.replace(
        "tinker.BadRequestError: prompt exceeds the model's context window", "KeyError: 'choices'"
    )
    scan = anomalies.scan_log(text, "model_server.log", defer_overflow=True)
    assert scan.fatal == ["model_server.log: KeyError: 'choices'"]


def test_after_the_overflow_record_the_ordinary_rules_apply_again() -> None:
    text = OVERFLOW_RECORD + "21:20:06 | ERROR | Exception logged for session ID def\n"
    scan = anomalies.scan_log(text, "model_server.log", defer_overflow=True)
    assert scan.fatal == ["model_server.log: 21:20:06 | ERROR | Exception logged for session ID def"]


def test_the_patterns_are_the_ones_the_runs_watched_for() -> None:
    assert anomalies.FATAL_LOG_PATTERNS == (
        r"HTTP/1\.[01]\" (429|5\d\d)\b",
        r"Too Many Requests",
        r"rate limit exceeded",
        r"RateLimitError",
        r"[Ee]rror code: (429|5\d\d)\b",
        r"exceeds the model's context window",
        r"Traceback \(most recent call last\)",
        r"Exception logged for session ID",
    )
