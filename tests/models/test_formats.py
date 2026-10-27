"""The tool-call format of each model family: reading a generation, its visible content, the history fix."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, get_args

import pytest

from evolve_ipi.config import ModelSpec, load_configs
from evolve_ipi.models.formats import Family, Formatter, get_formatter, new_call_id
from evolve_ipi.models.formats.harmony import HarmonyFormatter
from evolve_ipi.models.formats.inkling import InklingFormatter
from evolve_ipi.models.formats.qwen import QwenFormatter

FIXTURES = Path(__file__).parent / "fixtures" / "generations.jsonl"
FAMILIES: tuple[Family, ...] = ("qwen", "harmony", "inkling")
CALL_ID = re.compile(r"call[0-9a-f]{12}")


def read_generations() -> list[dict[str, Any]]:
    lines = FIXTURES.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


GENERATIONS = read_generations()
NOT_QWEN = [row for row in GENERATIONS if row["family"] != "qwen"]
WITH_AUDIT = [row for row in GENERATIONS if "audit" in row]


def functions(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Name and arguments of each call. The id is random, so only its shape is checked."""
    for call in calls:
        assert set(call) == {"id", "type", "function"}
        assert call["type"] == "function"
        assert CALL_ID.fullmatch(call["id"]), call["id"]
        assert set(call["function"]) == {"name", "arguments"}
    return [call["function"] for call in calls]


def names(calls: list[dict[str, Any]]) -> list[str]:
    return [call["function"]["name"] for call in calls]


EXEC_LS = {"name": "exec", "arguments": '{"command": "ls"}'}


# --- choosing and describing a format ---------------------------------------------------------


@pytest.mark.parametrize(
    ("family", "cls"),
    [("qwen", QwenFormatter), ("harmony", HarmonyFormatter), ("inkling", InklingFormatter)],
)
def test_get_formatter_returns_the_formatter_of_the_family(family: Family, cls: type) -> None:
    formatter = get_formatter(family)
    assert type(formatter) is cls
    assert isinstance(formatter, Formatter)


@pytest.mark.parametrize(
    "name", ["", "Qwen", "QWEN", "gpt-oss", "openai/gpt-oss-120b", "qwen3.8-27b", "quokka", "inkling-small"]
)
def test_get_formatter_takes_a_family_and_nothing_else(name: Any) -> None:
    with pytest.raises(ValueError) as err:
        get_formatter(name)
    message = str(err.value)
    assert repr(name) in message
    assert all(family in message for family in FAMILIES)


def test_the_families_are_the_ones_the_configuration_allows(configs_dir: Path) -> None:
    assert set(get_args(ModelSpec.model_fields["family"].annotation)) == set(get_args(Family)) == set(FAMILIES)
    for spec in load_configs(configs_dir).models.values():
        assert isinstance(get_formatter(spec.family), Formatter)


@pytest.mark.parametrize(
    ("family", "stops", "needs_raw_decode"),
    [
        ("qwen", None, False),
        ("harmony", ["<|call|>", "<|return|>"], True),
        ("inkling", ["<|content_model_end_sampling|>"], True),
    ],
)
def test_stop_sequences_and_raw_decoding(family: Family, stops: list[str] | None, needs_raw_decode: bool) -> None:
    formatter = get_formatter(family)
    assert formatter.stop_sequences == stops
    assert formatter.needs_raw_decode is needs_raw_decode


def test_call_ids_are_call_and_twelve_hex_digits() -> None:
    ids = {new_call_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(CALL_ID.fullmatch(one) for one in ids)


# --- one synthetic generation for each branch of each parser ----------------------------------


def test_the_fixtures_cover_every_family_and_each_has_its_own_id() -> None:
    ids = [row["id"] for row in GENERATIONS]
    assert len(ids) == len(set(ids))
    assert {row["family"] for row in GENERATIONS} == set(FAMILIES)
    assert {row["id"] for row in WITH_AUDIT} == {row["id"] for row in GENERATIONS if row["family"] == "inkling"}


@pytest.mark.parametrize("row", GENERATIONS, ids=lambda row: row["id"])
def test_parse_output_reads_each_generation(row: dict[str, Any]) -> None:
    result = get_formatter(row["family"]).parse_output(row["raw"])
    assert isinstance(result, tuple)
    calls, content = result
    assert functions(calls) == row["calls"]
    assert content == row["content"]


@pytest.mark.parametrize("row", NOT_QWEN, ids=lambda row: row["id"])
def test_harmony_and_inkling_read_the_whole_generation_in_both_parsers(row: dict[str, Any]) -> None:
    formatter = get_formatter(row["family"])
    assert functions(formatter.parse_tool_calls(row["raw"])) == row["calls"]
    assert formatter.clean_content(row["raw"]) == row["content"]


@pytest.mark.parametrize("row", WITH_AUDIT, ids=lambda row: row["id"])
def test_inkling_audit_helpers(row: dict[str, Any]) -> None:
    formatter = InklingFormatter()
    raw = row["raw"]
    assert row["audit"] == {
        "call_markup": formatter.call_markup(raw),
        "reasoning_closed": formatter.reasoning_closed(raw),
        "markup_in_reasoning": formatter.markup_in_reasoning(raw),
        "truncated_call": formatter.truncated_call(raw),
    }


@pytest.mark.parametrize("family", FAMILIES)
def test_text_without_markup_is_no_call_and_no_crash(family: Family) -> None:
    calls, content = get_formatter(family).parse_output("just some words")
    assert calls == []
    assert isinstance(content, str)


# --- Qwen ---------------------------------------------------------------------------------------


def test_qwen_reads_the_json_and_the_xml_form() -> None:
    formatter = get_formatter("qwen")
    json_form = '<tool_call>{"name": "exec", "arguments": {"command": "ls"}}</tool_call>'
    xml_form = "<tool_call><function=exec><parameter=command>ls</parameter></function></tool_call>"
    for text in (json_form, xml_form):
        assert functions(formatter.parse_tool_calls(text)) == [EXEC_LS]
        assert functions(formatter.parse_output(text)[0]) == [EXEC_LS]


def test_qwen_gives_every_call_a_fresh_id() -> None:
    text = '<tool_call>{"name": "a"}</tool_call><tool_call>{"name": "b"}</tool_call>'
    first, second = get_formatter("qwen").parse_tool_calls(text)
    assert first["id"] != second["id"]


def test_a_qwen_call_inside_the_reasoning_is_not_served_but_is_read_by_the_training_parser() -> None:
    text = 'Maybe <tool_call>{"name": "exec", "arguments": {"command": "ls"}}</tool_call>\n</think>\nNo.'
    formatter = get_formatter("qwen")
    assert formatter.parse_output(text) == ([], "No.")
    assert functions(formatter.parse_tool_calls(text)) == [EXEC_LS]


def test_qwen_serves_calls_from_after_the_last_closing_tag_and_the_whole_text_without_one() -> None:
    formatter = get_formatter("qwen")
    early = '<tool_call>{"name": "early"}</tool_call>'
    late = '<tool_call>{"name": "late"}</tool_call>'
    assert names(formatter.parse_output(f"{early}</think>{early}</think>{late}")[0]) == ["late"]
    assert names(formatter.parse_output(f"{early}{late}")[0]) == ["early", "late"]


def test_a_qwen_call_without_its_closing_tag_is_ignored() -> None:
    text = '</think>\n<tool_call>{"name": "exec", "arguments": {"command": "ls"}}'
    formatter = get_formatter("qwen")
    assert formatter.parse_tool_calls(text) == []
    assert formatter.parse_output(text)[0] == []


def test_qwen_content_drops_the_reasoning_the_calls_and_the_end_token() -> None:
    formatter = get_formatter("qwen")
    text = 'thinking\n</think>\n\n  Looking now.\n<tool_call>{"name": "exec"}</tool_call><|im_end|>\n'
    assert formatter.clean_content(text) == "Looking now."
    assert formatter.clean_content("no reasoning here <|im_end|>") == "no reasoning here"


# --- harmony ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "header",
    ["<|channel|>commentary to=functions.exec <|constrain|>json", "<|channel|>analysis to=functions.exec"],
    ids=["commentary-channel", "analysis-channel"],
)
def test_harmony_reads_a_call_on_the_commentary_and_on_the_analysis_channel(header: str) -> None:
    text = header + '<|message|>{"command": "ls"}<|call|>'
    calls, content = get_formatter("harmony").parse_output(text)
    assert functions(calls) == [EXEC_LS]
    assert content == ""


def test_harmony_content_is_the_final_channel() -> None:
    text = "<|channel|>analysis<|message|>hidden<|end|><|start|>assistant<|channel|>final<|message|>Visible.<|return|>"
    assert get_formatter("harmony").parse_output(text) == ([], "Visible.")


# --- Inkling ------------------------------------------------------------------------------------


def test_inkling_reads_a_call_and_leaves_the_reasoning_out() -> None:
    text = (
        "<|content_thinking|>private<|end_message|><|message_model|>exec<|content_invoke_tool_json|>"
        '{"name":"exec","args":{"command":"ls"}}<|end_message|><|content_model_end_sampling|>'
    )
    calls, content = get_formatter("inkling").parse_output(text)
    assert functions(calls) == [EXEC_LS]
    assert content == ""


def test_inkling_answer_is_its_text_messages_without_the_reasoning() -> None:
    text = (
        "<|content_thinking|>private<|end_message|><|message_model|><|content_text|>Visible.<|end_message|>"
        "<|content_model_end_sampling|>"
    )
    assert get_formatter("inkling").parse_output(text) == ([], "Visible.")


def test_inkling_does_not_parse_the_text_form_of_a_call_but_reports_its_markup() -> None:
    text = "<|message_model|>exec<|content_invoke_tool_text|>ls<|end_message|>"
    formatter = InklingFormatter()
    assert formatter.parse_output(text) == ([], "")
    assert formatter.call_markup(text)


def test_inkling_names_the_markup_that_must_not_reach_a_reply() -> None:
    tokens = InklingFormatter().markup_tokens
    assert "<|content_thinking|>" in tokens
    assert "<|content_invoke_tool_json|>" in tokens
    assert "<|content_model_end_sampling|>" in tokens


# --- the history fix ----------------------------------------------------------------------------

HISTORY = [
    {"role": "user", "content": "list the folder"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call1", "type": "function", "function": {"name": "exec", "arguments": '{"command": "ls"}'}}
        ],
    },
    {"role": "tool", "tool_call_id": "call1", "content": "a.txt"},
]


def arguments_of(messages: list[dict[str, Any]]) -> Any:
    return messages[1]["tool_calls"][0]["function"]["arguments"]


@pytest.mark.parametrize("family", FAMILIES)
def test_string_arguments_become_dicts(family: Family) -> None:
    out = get_formatter(family).normalize_tool_call_arguments(HISTORY)
    assert arguments_of(out) == {"command": "ls"}


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("text", ["", "   ", "{not json"])
def test_empty_or_invalid_arguments_become_an_empty_dict(family: Family, text: str) -> None:
    history = copy.deepcopy(HISTORY)
    history[1]["tool_calls"][0]["function"]["arguments"] = text
    out = get_formatter(family).normalize_tool_call_arguments(history)
    assert arguments_of(out) == {}


@pytest.mark.parametrize("family", FAMILIES)
def test_dict_arguments_and_foreign_entries_are_left_alone(family: Family) -> None:
    history: list[Any] = copy.deepcopy(HISTORY)
    history[1]["tool_calls"][0]["function"]["arguments"] = {"command": "ls"}
    history += ["not a message", 3, {"role": "assistant", "tool_calls": ["not a call", {"function": 7}]}]
    out = get_formatter(family).normalize_tool_call_arguments(history)
    assert arguments_of(out) == {"command": "ls"}
    assert out[3:] == history[3:]


@pytest.mark.parametrize("family", FAMILIES)
def test_a_tool_call_without_a_function_key_is_read_as_the_function_itself(family: Family) -> None:
    history = [{"role": "assistant", "tool_calls": [{"name": "exec", "arguments": '{"command": "ls"}'}]}]
    out = get_formatter(family).normalize_tool_call_arguments(history)
    assert out[0]["tool_calls"][0]["arguments"] == {"command": "ls"}


@pytest.mark.parametrize("family", FAMILIES)
def test_normalizing_returns_a_copy_and_leaves_the_input_alone(family: Family) -> None:
    history = copy.deepcopy(HISTORY)
    out = get_formatter(family).normalize_tool_call_arguments(history)
    assert out is not history
    assert history == HISTORY
    assert out[1]["tool_calls"][0] is not history[1]["tool_calls"][0]


def test_qwen_keeps_a_content_that_is_none() -> None:
    assert get_formatter("qwen").normalize_tool_call_arguments(HISTORY)[1]["content"] is None


def test_harmony_turns_a_content_that_is_none_into_an_empty_text() -> None:
    history = HISTORY + [{"role": "assistant", "content": None}, {"role": "assistant"}]
    out = get_formatter("harmony").normalize_tool_call_arguments(history)
    assert out[1]["content"] == ""
    assert out[3]["content"] == ""
    assert "content" not in out[4]


def test_inkling_drops_the_empty_content_of_an_assistant_turn_that_calls_a_tool() -> None:
    call = HISTORY[1]["tool_calls"]
    history = [
        {"role": "assistant", "content": "", "tool_calls": call},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": "kept", "tool_calls": call},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": None, "tool_calls": call},
    ]
    out = get_formatter("inkling").normalize_tool_call_arguments(history)
    assert [message["content"] for message in out] == [None, "", "kept", "", None]
