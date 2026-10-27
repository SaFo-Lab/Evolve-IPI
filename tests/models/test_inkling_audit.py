"""The audit of Inkling-Small's raw generations, in both forms: training rows that carry the bridge's count of calls,
and the model server's dump, whose rows are parsed again."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evolve_ipi.models import inkling_audit

CALL = (
    '<|content_thinking|>ok<|end_message|><|message_model|>read<|content_invoke_tool_json|>{"name": "read", '
    '"args": {"path": "a.txt"}}<|end_message|><|content_model_end_sampling|>'
)
TEXT = "<|content_text|>Here it is.<|end_message|><|content_model_end_sampling|>"
UNPARSED = "<|message_model|>read<|content_invoke_tool_text|>read a.txt<|end_message|><|content_model_end_sampling|>"
TRUNCATED = '<|message_model|>read<|content_invoke_tool_json|>{"name": "read", "args": {"pa'
UNCLOSED = "<|content_thinking|>still thinking about it"
IN_REASONING = (
    "<|content_thinking|>maybe <|message_model|>read<|content_invoke_tool_json|>{}<|end_message|>"
    "<|content_text|>No.<|end_message|><|content_model_end_sampling|>"
)
UNMARKED = "<|message_model|>a header with no content type<|end_message|><|content_model_end_sampling|>"

COUNTERS = (
    "generations",
    "with_calls",
    "unparsed_markup",
    "truncated_calls",
    "reasoning_unclosed",
    "closure_one_token",
    "markup_in_reasoning",
    "unmarked_text",
    "parse_mismatch",
)


def training_row(raw: str, parsed: int) -> dict[str, Any]:
    return {"at": "t", "n_tokens": 10, "stop": ["<|content_model_end_sampling|>"], "finish_reason": "stop",
            "parsed_calls": parsed, "raw": raw}  # fmt: skip


def server_row(raw: str, request: int, messages: int = 3, n_tokens: int = 10) -> dict[str, Any]:
    return {"at": "t", "request": request, "messages": messages, "n_tokens": n_tokens, "raw": raw}


def dump(path: Path, rows: list[dict[str, Any]], tail: str = "") -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows) + tail, encoding="utf-8")
    return path


def counters(audit: dict[str, Any]) -> dict[str, int]:
    return {key: audit[key] for key in COUNTERS}


def test_a_clean_training_dump(tmp_path: Path) -> None:
    path = dump(
        tmp_path / "raw_train.jsonl", [training_row(CALL, 1), training_row(TEXT, 0), training_row(TRUNCATED, 0)]
    )
    audit = inkling_audit.audit_raw_dump(path, parsed_key="parsed_calls")
    assert audit["where"] == str(path)
    assert counters(audit) == {
        "generations": 3,
        "with_calls": 1,
        "unparsed_markup": 0,
        "truncated_calls": 1,  # cut off at the token limit: the model's doing, reported only
        "reasoning_unclosed": 0,
        "closure_one_token": 0,
        "markup_in_reasoning": 0,
        "unmarked_text": 1,  # the cut-off call's text lies outside every content segment: reported only
        "parse_mismatch": 0,
    }
    assert audit["examples"] == [
        {"kind": "unmarked_text", "text": 'read {"name": "read", "args": {"pa', "tail": TRUNCATED[-300:]}
    ]
    assert audit["unfinished_last_line"] is False
    assert inkling_audit.audit_problems(audit) == []


def test_every_counter_of_a_model_server_dump(tmp_path: Path) -> None:
    rows = [
        server_row(CALL, 1),
        server_row(UNPARSED, 2),
        server_row(TRUNCATED, 3),
        server_row(UNCLOSED, 4),
        server_row(IN_REASONING, 5),
        server_row(UNMARKED, 6),
        server_row("<|content_thinking|>", 7, messages=1, n_tokens=1),  # a session's one-token closing request
        server_row(TEXT, 8, messages=1, n_tokens=10),  # a one-message request that got a full answer
        server_row(TEXT, 9, messages=3, n_tokens=1),  # a one-token answer in a conversation
    ]
    audit = inkling_audit.audit_raw_dump(dump(tmp_path / "raw.jsonl", rows), parsed_key=None)
    assert counters(audit) == {
        "generations": 9,
        "with_calls": 1,
        "unparsed_markup": 1,
        "truncated_calls": 1,
        "reasoning_unclosed": 2,
        "closure_one_token": 1,
        "markup_in_reasoning": 1,
        "unmarked_text": 3,  # the text-form call, the cut-off call and the header without a content type
        "parse_mismatch": 0,  # the model server's rows carry no count of their own
    }
    assert [example["kind"] for example in audit["examples"]] == ["unparsed", "unmarked_text", "unmarked_text"]
    assert audit["examples"][1]["text"] == "read read a.txt"
    (problem,) = inkling_audit.audit_problems(audit)
    assert "1 generation(s) with tool-call markup that parsed to nothing" in problem
    assert "0 parse mismatch(es)" in problem


def test_a_count_of_calls_that_differs_from_the_audit_is_a_mismatch_only_in_training_rows(tmp_path: Path) -> None:
    path = dump(tmp_path / "raw.jsonl", [training_row(CALL, 0), training_row(TEXT, 1)])
    training = inkling_audit.audit_raw_dump(path, parsed_key="parsed_calls")
    assert training["parse_mismatch"] == 2
    assert [(e["kind"], e["bridge"], e["audit"]) for e in training["examples"]] == [
        ("parse_mismatch", 0, 1),
        ("parse_mismatch", 1, 0),
    ]
    assert "2 parse mismatch(es)" in inkling_audit.audit_problems(training)[0]
    assert inkling_audit.audit_raw_dump(path, parsed_key=None)["parse_mismatch"] == 0


def test_a_row_without_the_count_is_a_mismatch(tmp_path: Path) -> None:
    path = dump(tmp_path / "raw.jsonl", [server_row(TEXT, 1)])
    assert inkling_audit.audit_raw_dump(path, parsed_key="parsed_calls")["parse_mismatch"] == 1


def test_at_most_three_examples_in_row_order(tmp_path: Path) -> None:
    rows = [server_row(UNMARKED, 1), server_row(UNPARSED, 2), server_row(UNMARKED, 3), server_row(UNPARSED, 4)]
    audit = inkling_audit.audit_raw_dump(dump(tmp_path / "raw.jsonl", rows), parsed_key=None)
    assert [example["kind"] for example in audit["examples"]] == ["unmarked_text", "unparsed", "unmarked_text"]
    assert audit["unparsed_markup"] == 2 and audit["unmarked_text"] == 4
    assert audit["examples"][1]["tail"] == UNPARSED[-1500:]


@pytest.mark.parametrize("parsed_key", ["parsed_calls", None])
def test_no_generation_is_a_problem(tmp_path: Path, parsed_key: str | None) -> None:
    empty = inkling_audit.audit_raw_dump(dump(tmp_path / "raw.jsonl", []), parsed_key=parsed_key)
    missing = inkling_audit.audit_raw_dump(tmp_path / "absent.jsonl", parsed_key=parsed_key)
    for audit in (empty, missing):
        assert audit["generations"] == 0
        assert inkling_audit.audit_problems(audit) == [f"{audit['where']}: no raw generation"]


def test_an_unfinished_last_line_is_not_a_generation_yet(tmp_path: Path) -> None:
    path = dump(tmp_path / "raw.jsonl", [training_row(TEXT, 0)], tail='{"raw": "cut off by a stop')
    audit = inkling_audit.audit_raw_dump(path, parsed_key="parsed_calls")
    assert audit["generations"] == 1 and audit["unfinished_last_line"] is True
    assert inkling_audit.audit_problems(audit) == []


@pytest.mark.parametrize("line", ["not json", '{"n_tokens": 3}', "[1, 2]"])
def test_a_line_that_is_not_a_generation_raises(tmp_path: Path, line: str) -> None:
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps(training_row(TEXT, 0)) + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        inkling_audit.audit_raw_dump(path, parsed_key="parsed_calls")


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    path.write_text("\n" + json.dumps(training_row(TEXT, 0)) + "\n\n", encoding="utf-8")
    assert inkling_audit.audit_raw_dump(path, parsed_key="parsed_calls")["generations"] == 1


def test_the_text_outside_every_segment_is_read_by_the_formatter_itself() -> None:
    """The audit reads it through the formatter's public helper, which knows the markup."""
    from evolve_ipi.models.formats.inkling import InklingFormatter

    formatter = InklingFormatter()
    assert formatter.unmarked_text("<|content_text|>Done.<|end_message|>") == ""
    assert formatter.unmarked_text("<|content_thinking|>hm<|end_message|>stray<|content_model_end_sampling|>") == (
        "stray"
    )
    raw = '<|message_model|>read{"name": "read"}<|end_message|>'  # a header without a content type
    assert formatter.unmarked_text(raw) == 'read{"name": "read"}'
